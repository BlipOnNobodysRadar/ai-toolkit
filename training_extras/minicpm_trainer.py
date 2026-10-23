#!/usr/bin/env python3
"""Feasibility PEFT trainer for MiniCPM-o 4.5 (native audiovisual, teacher-forced caption loss).

Scope: bounded worker runs, round-robin batch-1 rows, frozen encoders, LoRA/DoRA/LoHa/LoKr on
llm.model.layers.*.self_attn.{q_proj,v_proj}. Not a quality proof. No ASR/TTS/quant/server/monitor.
Losses via training_extras.caption_loss; optimizers/schedulers via training_extras.optim.
"""

import argparse, json, math, os, sys, time, hashlib, random, re
from .toolkit_validation import fixed_rng

os.environ.setdefault("HF_HOME", r"E:/ai_cache/video-captioning/huggingface")
os.environ.setdefault("HF_MODULES_CACHE", r"E:/ai_cache/video-captioning/huggingface/modules")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FFMPEG_BIN = r"C:/Users/someb/AI/tools/ffmpeg-cinematics12/ffmpeg-2026-09-10-git-fd7c73d01e-essentials_build/bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import torch
import transformers

from transformers import AutoModel
from minicpmo.utils import get_video_frame_audio_segments
from training_extras.caption_loss import CaptionLoss

SYSTEM_TEXT = "You are a helpful assistant capable of perceiving video and audio."
MODEL_ROOT = r"C:/Users/someb/AI/video-captioning/minicpm-o45/model"
PROMPT_PATH = r"C:/Users/someb/AI/video-captioning/scripts/audio-first-prompt.txt"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "training-record.json")
TARGET_REGEX = r"llm\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
OPT_KEYS = {"optimizer", "scheduler", "caption_loss", "lora_plus_ratio"}


def log(m):
    print(m, flush=True)


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def video_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path, find_videos):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v, c = r.get("video"), r.get("caption")
            if not v or not isinstance(v, str) or not os.path.isabs(v):
                raise SystemExit(f"row {i}: 'video' must be an absolute path")
            if not os.path.exists(v):
                raise SystemExit(f"row {i}: video not found: {v}")
            if not isinstance(c, str) or not c.strip():
                raise SystemExit(f"row {i}: missing/empty 'caption'")
            g = r.get("source_group")
            if g is not None and (not isinstance(g, str) or not g.strip()):
                raise SystemExit(f"row {i}: 'source_group' must be a non-empty string")
            rows.append({"video": v, "caption": c, "source_group": g,
                         "vsha": video_sha256(v) if find_videos else None})
    if not rows:
        raise SystemExit("no data rows")
    return rows


def leak_key(r):
    return r["source_group"] if r.get("source_group") else r["vsha"]


def check_leakage(train_rows, val_rows, label):
    tr_res = {os.path.realpath(r["video"]) for r in train_rows}
    tr_sha = {r["vsha"] for r in train_rows}
    tr_grp = {r["source_group"] for r in train_rows if r.get("source_group")}
    for r in val_rows:
        hit = (os.path.realpath(r["video"]) in tr_res) or (r["vsha"] in tr_sha) \
              or (r.get("source_group") and r["source_group"] in tr_grp)
        if hit:
            raise SystemExit(f"{label} leakage: row {r['video']} overlaps train by "
                             f"resolved video / sha256 / source_group")


def load_options(path):
    if not path:
        return {"optimizer": {"name": "adamw", "args": {}},
                "scheduler": {"name": "constant", "args": {}},
                "caption_loss": {}, "lora_plus_ratio": 1}
    with open(path, "r", encoding="utf-8") as f:
        o = json.load(f)
    bad = set(o) - OPT_KEYS
    if bad:
        raise SystemExit(f"options: unsupported keys rejected: {sorted(bad)} "
                         f"(edm2/spatial cannot apply to caption)")
    for k in ("optimizer", "scheduler"):
        if k in o:
            if not isinstance(o[k], dict) or set(o[k]) - {"name", "args"}:
                raise SystemExit(f"options.{k} must be {{'name':..,'args':{{}}}}")
    if set(o.get("caption_loss", {})) & {"diffusion", "spatial", "edm", "edm2"}:
        raise SystemExit("options.caption_loss: diffusion/spatial keys rejected")
    o.setdefault("optimizer", {"name": "adamw", "args": {}})
    o.setdefault("scheduler", {"name": "constant", "args": {}})
    o.setdefault("caption_loss", {})
    ratio = o.setdefault("lora_plus_ratio", 1)
    if not isinstance(ratio, (int, float)) or ratio <= 0:
        raise SystemExit("lora_plus_ratio must be positive")
    return o


def prepare_out(out, resume):
    os.makedirs(out, exist_ok=True)
    for name in ADAPTER_FILES:
        p = os.path.join(out, name)
        if os.path.exists(p):
            if resume:
                log(f"warning: overwriting existing {p} (explicit --resume)")
            else:
                raise SystemExit(f"refusing to overwrite existing adapter file: {p}")
    return out


def build_native(base, video, prompt, caption):
    frames, audios, _stacked = get_video_frame_audio_segments(
        video, stack_frames=1, use_ffmpeg=True, use_audio=True)
    if not frames:
        raise SystemExit(f"no frames decoded: {video}")
    if not audios:
        raise SystemExit(f"no audio segments decoded: {video}")
    rms_values = []
    for n, a in enumerate(audios):
        arr = a[0] if isinstance(a, (list, tuple)) else a
        rms = float(torch.as_tensor(arr).float().pow(2).mean().sqrt())
        if not math.isfinite(rms):
            rms = -1.0
        log(f"  audio seg {n}: rms={rms:.6f}")
        rms_values.append(rms)
    if len(frames) != len(audios):
        raise SystemExit("native aligned frame/audio count mismatch")
    placeholders = "".join("<image>./</image><audio>./</audio>" for _ in frames)
    messages = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": placeholders + prompt}]
    prefix = base.processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        use_tts_template=True, enable_thinking=False)
    full_text = prefix + caption + "<|im_end|>"
    kw = dict(max_slice_nums=1, use_image_id=None, stream_input=False,
              return_tensors="pt", max_length=32768)
    parts = [[1] * len(audios)]
    pre = base.processor([prefix], [frames], [audios], parts, **kw).to("cuda")
    full = base.processor([full_text], [frames], [audios], parts, **kw).to("cuda")
    full.pop("image_sizes", None)
    pre.pop("image_sizes", None)
    pn = int(pre.input_ids.shape[1])
    fn = int(full.input_ids.shape[1])
    if fn <= pn or not torch.equal(full.input_ids[:, :pn], pre.input_ids):
        raise SystemExit("prefix mismatch / truncation: full does not begin with exact prefix")
    target = full.input_ids[:, pn:].long()
    if target.numel() == 0:
        raise SystemExit("empty target")
    eos = base.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if int(target[0, -1]) != eos:
        raise SystemExit("caption EOS missing; target may have been truncated")
    del pre
    return full, pn, fn, target, len(frames), len(audios), rms_values


def encode(base, full):
    with torch.no_grad():
        embeds, _ = base.get_vllm_embedding(full)
        embeds = base.get_omni_embedding(
            full, input_embeddings=embeds, chunk_length=base.config.audio_chunk_length)
    pos = torch.arange(full.input_ids.shape[1], device=embeds.device).unsqueeze(0)
    mask = full.attention_mask
    return embeds.detach(), pos, mask


def make_cache(base, row, prompt):
    """Native prep + embed caching for exactly one row; logs native audio RMS once."""
    full, pn, fn, target, nf, na, rms_values = build_native(base, row["video"], prompt, row["caption"])
    embeds, pos, mask = encode(base, full)
    del full
    torch.cuda.empty_cache()
    entry = {"embeds": embeds, "pos": pos, "mask": mask, "target": target, "pn": pn, "fn": fn,
             "frames": nf, "audio_segments": na, "audio_segment_rms": rms_values}
    del target
    return entry


def adapter_config(adapter_type, rank, alpha):
    from peft import LoraConfig, LoHaConfig, LoKrConfig
    if adapter_type in ("lora", "dora"):
        return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                          target_modules=TARGET_REGEX, use_dora=(adapter_type == "dora"))
    if adapter_type == "loha":
        return LoHaConfig(r=rank, alpha=alpha, target_modules=TARGET_REGEX)
    if adapter_type == "lokr":
        return LoKrConfig(r=rank, alpha=alpha, target_modules=TARGET_REGEX)
    raise SystemExit(f"unknown adapter type: {adapter_type}")


def adapter_marker(adapter_type):
    return "lora_" if adapter_type in ("lora", "dora") else ("hada_" if adapter_type == "loha" else "lokr_")


def peek_adapter_meta(adir):
    with open(os.path.join(adir, "adapter_config.json"), "r", encoding="utf-8") as f:
        c = json.load(f)
    return c


def save_checkpoint(model, opt, sched, step, out, options, dataset_info, prompt_hash):
    import torch as _t
    cd = os.path.join(out, f"checkpoint-step{step:06d}")
    os.makedirs(cd, exist_ok=True)
    model.save_pretrained(cd, safe_serialization=True)
    state = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict() if sched is not None else None,
             "step": step, "cpu_rng": _t.get_rng_state(),
             "options": options, "dataset": dataset_info, "prompt_hash": prompt_hash, "schedule_steps":sched._extras_horizon}
    if _t.cuda.is_available():
        state["cuda_rng"] = _t.cuda.get_rng_state_all()
    _t.save(state, os.path.join(cd, "trainer_state.pt.tmp"))
    os.replace(os.path.join(cd, "trainer_state.pt.tmp"),os.path.join(cd, "trainer_state.pt"))
    log(f"checkpoint saved: {cd}")
    return cd


def load_trainer_state(resume, out, options, dataset_info, prompt_hash):
    with open(os.path.join(resume, "trainer_state.pt"), "rb") as f:
        st = torch.load(f, map_location="cpu", weights_only=True)
    if st.get("prompt_hash") != prompt_hash:
        raise SystemExit("resume: prompt hash mismatch")
    got, want = st.get("dataset"), dataset_info
    for k in ("train_sha", "train_count", "val_sha", "val_count"):
        if got.get(k) != want.get(k):
            raise SystemExit(f"resume: dataset compatibility mismatch on {k}")
    if got.get("rows_hash") != want.get("rows_hash"):
        raise SystemExit("resume: dataset rows hash mismatch")
    if st.get("options") != options:
        raise SystemExit("resume: options hash mismatch")
    return st


def set_rng_from_state(st):
    if "cpu_rng" in st:
        torch.set_rng_state(st["cpu_rng"])
    if "cuda_rng" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda_rng"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--verify-reload", action="store_true")
    ap.add_argument("--options")
    ap.add_argument("--validate-every", type=int, default=25)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume")
    ap.add_argument("--adapter-type", choices=["lora", "dora", "loha", "lokr"], default="lora")
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()
    if a.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if a.rank <= 0 or a.alpha <= 0 or not math.isfinite(a.lr) or a.lr <= 0:
        raise SystemExit("rank, alpha and lr must be positive")
    if a.validate_every <= 0 or a.grad_accum <= 0 or a.checkpoint_every <= 0:
        raise SystemExit("validate-every / grad-accum / checkpoint-every must be positive")
    options = load_options(a.options)
    ratio = float(options.get("lora_plus_ratio", 1))
    if ratio != 1 and a.adapter_type not in ("lora", "dora"):
        raise SystemExit("lora_plus_ratio != 1 only supported for lora/dora")

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    rows = read_rows(a.data, find_videos=True)
    val_rows = read_rows(a.val_data, find_videos=True) if a.val_data else []
    if val_rows:
        check_leakage(rows, val_rows, "val/train")
        if {os.path.realpath(r["video"]) for r in rows} & {os.path.realpath(r["video"]) for r in val_rows}:
            raise SystemExit("overlapping train/val datasets")
    out = prepare_out(a.out, a.resume)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    rows_hash = hashlib.sha256(json.dumps([r["vsha"] for r in rows] + [r["vsha"] for r in val_rows]).encode()).hexdigest()
    dataset_info = {"train_sha": sha256_file(a.data), "train_count": len(rows),
                    "val_sha": sha256_file(a.val_data) if a.val_data else None,
                    "val_count": len(val_rows), "rows_hash": rows_hash}

    options["_trainer"]={"adapter_type":a.adapter_type,"rank":a.rank,"alpha":a.alpha,"lr":a.lr,"seed":a.seed,"grad_accum":a.grad_accum}
    resume_state=load_trainer_state(a.resume,out,options,dataset_info,prompt_hash) if a.resume else None

    log(f"torch {torch.__version__} cuda {torch.version.cuda} devices {torch.cuda.device_count()}")
    t0 = time.time()
    base = AutoModel.from_pretrained(
        MODEL_ROOT, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        init_vision=True, init_audio=True, init_tts=False).eval().cuda()
    base.prepare_processor()
    for p in base.parameters():
        p.requires_grad_(False)

    from peft import get_peft_model, PeftModel, get_peft_model_state_dict
    if a.resume:
        model = PeftModel.from_pretrained(base, a.resume, is_trainable=True)
        meta=peek_adapter_meta(a.resume)
        expected={"lora":"LORA","dora":"LORA","loha":"LOHA","lokr":"LOKR"}[a.adapter_type]
        if meta["peft_type"]!=expected or (a.adapter_type in ("lora","dora") and bool(meta.get("use_dora"))!=(a.adapter_type=="dora")):
            raise SystemExit("resume adapter type mismatch")
    else:
        model = get_peft_model(base, adapter_config(a.adapter_type,a.rank,a.alpha))
    model.eval()
    llm = model.get_base_model().llm
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(llm, "config"):
        llm.config.use_cache = False

    marker = adapter_marker(a.adapter_type)
    trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainables or not all(marker in n for n, _ in trainables):
        raise SystemExit(f"unexpected trainable set (all must contain '{marker}')")
    targeted = sorted({re.split(r"\.(?:lora_|hada_|lokr_)",n)[0] for n,_ in trainables})
    log(f"adapter {a.adapter_type} trainable tensors: {len(trainables)}")
    for n, _ in trainables[:6]:
        log(f"  {n}")
    log(f"exact targeted modules: {len(targeted)}")

    vocab = llm.get_output_embeddings().weight.shape[0]

    from training_extras.optim import apply_run_length_defaults, create_optimizer, create_scheduler, evaluation_optimizer

    if ratio != 1:
        groups=[{"params":[p for n,p in trainables if ".lora_B." not in n],"lr":a.lr},
                {"params":[p for n,p in trainables if ".lora_B." in n],"lr":a.lr*ratio}]
    else:groups=[p for _,p in trainables]
    horizon=resume_state["schedule_steps"] if resume_state else a.steps
    opt_args=apply_run_length_defaults(options["optimizer"]["name"],options["optimizer"].get("args",{}),horizon)
    opt=create_optimizer(groups,options["optimizer"]["name"],a.lr,opt_args)
    sched=create_scheduler(opt,options["scheduler"]["name"],horizon,options["scheduler"].get("args",{}))
    sched._extras_horizon=horizon
    start_step=0
    if resume_state:
        start_step=int(resume_state["step"])
        if start_step>=a.steps:raise SystemExit("resume checkpoint already reached requested steps")
        sched.load_state_dict(resume_state["scheduler"])
        opt.load_state_dict(resume_state["optimizer"])
        set_rng_from_state(resume_state)
        log(f"resumed at step {start_step}")
    criterion = CaptionLoss(options.get("caption_loss", {}))
    use_kl = float(options.get("caption_loss", {}).get("kl_anchor", 0) or 0) > 0

    rec = {"status": "init", "steps": a.steps, "start_step": start_step, "rank": a.rank,
           "alpha": a.alpha, "lr": a.lr, "adapter_type": a.adapter_type,
           "lora_plus_ratio": ratio, "options": options,
           "trainable_tensors": len(trainables), "targeted_modules": len(targeted),
           "losses": [], "records": [], "validation_history": [],
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "transformers": transformers.__version__,
                        "peft": __import__("peft").__version__},
           "trainable_parameters": sum(p.numel() for _, p in trainables),
           "target_names": targeted, "quality_training": False, "base_model": MODEL_ROOT,
           "base_revision": "503e754207c94da6bb26850b4469f367c9ea3582",
           "prompt": prompt, "data_sha256": dataset_info["train_sha"],
           "dataset": dataset_info, "prompt_sha256": prompt_hash, "seed": a.seed}
    for n, t in (trainables[0], trainables[len(trainables) // 2]):
        rec.setdefault("initial_digests", {})[n] = hashlib.sha256(
            t.detach().to("cpu").float().contiguous().numpy().tobytes()).hexdigest()[:16]

    train_cache = {}
    eval_cache = {}

    def get_entry(store, row):
        key = (row["video"],row["caption"])
        if key not in store:
            base.eval()
            for k in list(store):
                if k != key:
                    del store[k]
                    torch.cuda.empty_cache()
            store[key] = make_cache(base, row, prompt)
            log(f"  cached one row ({'train' if store is train_cache else 'val'}): {key}")
        return store[key]

    def forward_logits(entry):
        m = int(entry["target"].shape[1])
        assert entry["pn"] >= 1 and entry["fn"] - entry["pn"] == m
        o = llm(inputs_embeds=entry["embeds"], position_ids=entry["pos"],
                attention_mask=entry["mask"], use_cache=False, logits_to_keep=m + 1)
        return o.logits[:, -m-1:-1, :]

    def frozen_reference(entry):
        if "reference" not in entry:
            training=llm.training
            try:
                with model.disable_adapter(),torch.no_grad():
                    llm.eval()
                    entry["reference"]=forward_logits(entry).float().detach().cpu()
            finally:llm.train(training)
        return entry["reference"].to("cuda")

    def evaluate_rows(selected, label, step):
        llm.eval()
        sums = {"raw_ce": 0.0, "nll": 0.0, "correct": 0.0, "tokens": 0.0, "examples": 0.0}
        obj = 0.0
        with torch.no_grad(), evaluation_optimizer(opt), fixed_rng(a.seed,torch.device("cuda")):
            for r in selected:
                e = get_entry(eval_cache, r)
                logits = forward_logits(e)
                ref = frozen_reference(e) if use_kl else None
                loss, metrics = criterion(logits, e["target"], reference_logits=ref)
                tk = float(e["target"].numel())
                ce = float(metrics.get("raw_ce", metrics.get("ce", float("nan"))))
                sums["raw_ce"] += ce * tk
                sums["nll"] += float(loss.detach()) * tk
                sums["correct"] += float(metrics.get("token_accuracy", 0.0)) * tk
                sums["tokens"] += tk
                sums["examples"] += 1.0
                obj = obj + float(loss.detach())
                del logits, loss, metrics, ref
        t = max(sums["tokens"], 1.0)
        raw_metrics = {"raw_ce": sums["raw_ce"] / t, "nll": sums["raw_ce"] / t,
                       "perplexity": math.exp(min(sums["raw_ce"] / t, 80.0)),
                       "token_accuracy": sums["correct"] / t}
        entry = {"step": step, "split": label, "examples": int(sums["examples"]),
                 "tokens": int(sums["tokens"]), "objective": sums["nll"] / t,
                 "raw_metrics": raw_metrics}
        log(f"[validate {label}] step {step} raw_ce {raw_metrics['raw_ce']:.6f} "
            f"ppl {raw_metrics['perplexity']:.3f} acc {raw_metrics['token_accuracy']:.4f}")
        return entry

    def checkpoint(step):
        return save_checkpoint(model, opt, sched, step, out, options, dataset_info, prompt_hash)

    def train_eval(step):
        e = get_entry(train_cache, rows[0])
        llm.eval()
        with torch.no_grad():
            logits = forward_logits(e)
            ref = frozen_reference(e) if use_kl else None
            loss, metrics = criterion(logits, e["target"], reference_logits=ref)
        return float(loss.detach())

    if not a.resume:
        rec["initial_loss"] = train_eval(0)
    if val_rows:
        rec["validation_history"].append(evaluate_rows(val_rows, "val", start_step))

    accum = a.grad_accum
    changed_any = False
    for gstep in range(start_step, a.steps):
        if callable(getattr(opt,"train",None)):opt.train()
        for micro in range(accum):
            idx = (gstep * accum + micro) % len(rows)
            row = rows[idx]
            e = get_entry(train_cache, row)
            llm.train()
            logits = forward_logits(e)
            ref = frozen_reference(e) if use_kl else None
            loss, metrics = criterion(logits, e["target"], reference_logits=ref)
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise SystemExit(f"non-finite loss at step {gstep} micro {micro}")
            (loss / accum).backward()
            rec["records"].append({
                "step": gstep, "micro": micro, "video": row["video"],
                "prefix_tokens": e["pn"], "full_tokens": e["fn"],
                "target_tokens": int(e["target"].shape[1]), "frames": e["frames"],
                "audio_segments": e["audio_segments"],
                "audio_segment_rms": e["audio_segment_rms"],
                "training_objective": lv, "raw_ce": float(metrics.get("raw_ce", lv))})
            del logits, loss, metrics, ref
        gn = 0.0
        avgn = []
        for n, p in trainables:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise SystemExit(f"non-finite grad at step {gstep}")
                gn += float(p.grad.detach().float().pow(2).sum())
                avgn.append(float(p.grad.detach().float().abs().mean()))
        if gn <= 0.0 or not math.isfinite(gn):
            raise SystemExit(f"zero/non-finite trainable grad norm at step {gstep}")
        for n, p in model.get_base_model().named_parameters():
            if not p.requires_grad and p.grad is not None:
                raise SystemExit(f"base grad not None: {n}")
        before = {n: p.detach().clone() for n, p in trainables}
        opt.step()
        sched.step()
        changed = [n for n, p in trainables if not torch.equal(before[n], p.detach())]
        # Some optimizers initialize their moments before their first update.
        # Require a real update across the run rather than on every single step.
        changed_any = changed_any or bool(changed)
        opt.zero_grad(set_to_none=True)
        lrs = sorted({float(g.get("lr", a.lr)) for g in opt.param_groups})
        rec["losses"].append({"step": gstep, "grad_sq_norm": gn, "lrs": lrs,
                              "updated": len(changed), "mean_abs_grad": avgn[:1]})
        log(f"[train] step {gstep} gradsq {gn:.4e} lrs {lrs} updated {len(changed)}")
        del before
        if val_rows and a.validate_every > 0 and ((gstep + 1) % a.validate_every == 0):
            rec["validation_history"].append(evaluate_rows(val_rows, "val", gstep + 1))
            llm.train()
        if a.checkpoint_every > 0 and ((gstep + 1) % a.checkpoint_every == 0):
            checkpoint(gstep + 1)

    if not changed_any:
        raise SystemExit('No adapter tensor changed across this run; try enough steps to pass optimizer initialization, or inspect the configuration')
    if callable(getattr(opt,"eval",None)):opt.eval()
    rec["final_train_loss"] = train_eval(a.steps)
    if val_rows and (not rec["validation_history"] or rec["validation_history"][-1]["step"] != a.steps):
        rec["validation_history"].append(evaluate_rows(val_rows, "val", a.steps))
    rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    rec["runtime_s"] = time.time() - t0

    if callable(getattr(opt,"train",None)):opt.train()
    checkpoint(a.steps)
    if callable(getattr(opt,"eval",None)):opt.eval()
    model.save_pretrained(out, safe_serialization=True)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    if not sd or any(not torch.isfinite(v).all() for v in sd.values()):
        raise SystemExit("saved adapter not finite")
    adapter_factor_names = [k for k, v in sd.items() if any(t in k for t in
                           ("lora_", "hada_", "lokr_")) and torch.any(v != 0)]
    if not adapter_factor_names:
        raise SystemExit("no nonzero saved adapter factor tensors")
    rec["saved_tensors"] = len(sd)
    rec["saved_nonzero_adapter_tensors"] = len(adapter_factor_names)
    rec["weight_change_verified"] = True

    if a.verify_reload:
        saved_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        llm.gradient_checkpointing_disable()
        for p in opt.param_groups:
            p["params"] = []
        opt.state.clear()
        del opt, sched
        torch.cuda.empty_cache()
        clean = model.unload()
        del model, llm, trainables
        if hasattr(clean, "peft_config"):
            del clean.peft_config
        torch.cuda.empty_cache()
        for p in clean.parameters():
            p.requires_grad_(False)
        reloaded = PeftModel.from_pretrained(clean, out, is_trainable=False).eval()
        rstate = get_peft_model_state_dict(reloaded)
        if set(saved_state) != set(rstate):
            raise SystemExit("reloaded adapter key mismatch")
        for k, v in saved_state.items():
            if not torch.equal(v.detach().cpu(), rstate[k].detach().cpu()):
                raise SystemExit(f"reloaded adapter mismatch: {k}")
        cllm = reloaded.get_base_model().llm
        cllm.eval()
        e = get_entry(eval_cache, rows[0])
        with torch.no_grad():
            m = int(e["target"].shape[1])
            o = cllm(inputs_embeds=e["embeds"], position_ids=e["pos"], attention_mask=e["mask"],
                     use_cache=False, logits_to_keep=m + 1)
            reference=None
            if use_kl:
                with reloaded.disable_adapter():
                    reference=cllm(inputs_embeds=e["embeds"],position_ids=e["pos"],attention_mask=e["mask"],use_cache=False,logits_to_keep=m+1).logits[:,:-1,:]
            rloss, rmetrics = criterion(o.logits[:, :-1, :], e["target"],reference_logits=reference)
        rec["reload_loss"] = float(rloss)
        rec["reload_raw_ce"] = float(rmetrics.get("raw_ce", float(rloss)))
        if abs(float(rloss) - rec["final_train_loss"]) > 1e-5:
            raise SystemExit("reloaded loss mismatch")
        rec["reload_verified"]=True
    rec["status"]="passed"
    with open(os.path.join(out,"training-record.json"),"w",encoding="utf-8") as f:json.dump(rec,f,indent=2)
    log(f"status: {rec['status']}")

if __name__=="__main__":main()
