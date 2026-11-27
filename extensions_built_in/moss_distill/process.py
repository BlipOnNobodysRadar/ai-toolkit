"""Offline teacher-trajectory distillation for the native MOSS T2I transformer."""
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from jobs.process.BaseExtensionProcess import BaseExtensionProcess
from toolkit import save_root_lease
from toolkit.config_modules import ModelConfig
from toolkit.diffusion_dpo import prediction_errors, preference_logits
from toolkit.ui_job_status import UIJobStatus
from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.pipeline import moss_sigmas
from .recipe import check_grid, child, digest, integer, ranked_pairs, records, solver_step, teacher_anchors, validate


def atomic(path, write):
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, obj):
    atomic(path, lambda f: f.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8"))


def bind(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"{path.name} differs from this run. Use a new job name.")
    else:
        save_json(path, value)


class LowRank(torch.nn.Module):
    def __init__(self, backbone, rank):
        super().__init__()
        self.active = True
        self.handles = []
        self.maps = torch.nn.ModuleDict()
        self.names = {}
        for i in range(max(0, len(backbone.layers) - 18), len(backbone.layers)):
            for name, layer in backbone.layers[i].proj.items():
                key = f"b{i}_{name}"
                pair = torch.nn.Sequential(torch.nn.Linear(layer.in_features, rank, bias=False),
                                           torch.nn.Linear(rank, layer.out_features, bias=False)).to(layer.weight.device)
                torch.nn.init.zeros_(pair[1].weight)
                self.maps[key] = pair
                self.names[key] = f"transformer.layers.{i}.proj.{name}"
                def add(_, args, out, adapter=pair):
                    if self.active:
                        # BF16 GEMMs, FP32 trainable master weights (autocast also during checkpoint replay).
                        with torch.autocast(out.device.type, dtype=torch.bfloat16):
                            delta = adapter(args[0])
                        return out + delta.to(out.dtype)
                    return out
                self.handles.append(layer.register_forward_hook(add))

    def export(self, model):
        weights = {}
        for key, pair in self.maps.items():
            name = self.names[key]
            weights[name + ".lora_down.weight"] = pair[0].weight.detach().cpu()
            weights[name + ".lora_up.weight"] = pair[1].weight.detach().cpu()
            weights[name + ".alpha"] = torch.tensor(float(pair[0].out_features))
        # Q/K/V conversions may share backing storage. Safetensors requires independent tensors.
        return {k: v.contiguous().clone() for k, v in model.convert_lora_weights_before_save(weights).items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def velocity(net, ctx, skip=()):
    def predict(x, sigma):
        return net(x, torch.as_tensor(sigma * 1000, device=x.device).reshape(1),
                   ctx, skip_layers=skip).float()
    return predict


def grids(r):
    teacher = r.get("teacher_sigmas") or moss_sigmas(r["teacher_steps"], shift=r["shift"]).tolist()
    n = r["steps"]
    if r["schedule"] == "moss":
        target = moss_sigmas(n, shift=r["shift"]).tolist()
    elif r["schedule"] == "flow_uniform":
        ts = [1 - i/n for i in range(n + 1)]
        target = [r["shift"]*t/(1+(r["shift"]-1)*t) for t in ts]
    else:
        target = r["sigmas"]
    return teacher, check_grid(target, n)


def read_anchor(cache, key, i, device):
    with safe_open(cache / (key + ".safetensors"), framework="pt", device="cpu") as f:
        return f.get_tensor(str(i)).to(device)


def flow_preference(net, adapters, ctx, good, bad, sigma, eps, beta):
    clean = torch.cat((good, bad))
    eps = eps.expand_as(clean)
    target = eps - clean
    noisy = clean + sigma * target
    pred = velocity(net, ctx.expand(2, -1, -1))
    try:
        adapters.active = False
        with torch.no_grad():
            ref = prediction_errors(pred(noisy, sigma), target)
    finally:
        adapters.active = True
    policy = prediction_errors(pred(noisy, sigma), target)
    return -F.logsigmoid(preference_logits(policy, ref, beta)).mean()


class Halt(Exception):
    pass


class MossDistill(BaseExtensionProcess):
    def run(self):
        status = UIJobStatus(self.get_conf("sqlite_db_path", "./aitk_db.db"))
        home = Path(self.get_conf("training_folder", "output"))
        out = child(home, self.name)
        lease = save_root_lease.acquire(str(out), self.job, self)
        source_lease = None
        def tick(message, step=None):
            if status.should_stop() or status.should_return_to_queue():
                raise Halt()
            status.update_info(message)
            if step is not None:
                status.update_key("step", step)
            print(message, flush=True)
        try:
            if self.get_conf("stage", "all") == "student":
                source = child(home, self.get_conf("teacher_job", required=True))
                if source == out:
                    raise ValueError("The reviewed student needs its own output job")
                # The trainer reads anchors lazily. Prevent UI deletion or regeneration mid-job.
                source_lease = save_root_lease.acquire(str(source), self.job, self)
            out.mkdir(parents=True, exist_ok=True)
            self.execute(home, out, tick, status)
            status.update_status("completed", "Completed")
        except Halt:
            status.update_status("queued" if status.should_return_to_queue() else "stopped", "Paused")
        except Exception as e:
            status.update_status("error", str(e))
            raise
        finally:
            save_root_lease.release(source_lease, self)
            save_root_lease.release(lease, self)

    def execute(self, home, out, tick, status):
        r = self.get_conf("recipe", required=True)
        validate(r)
        stage = self.get_conf("stage", "all")
        if stage not in ("prepare", "student", "all"):
            raise ValueError("Unknown distillation stage")
        opts = self.get_conf("student", {})
        updates = integer(opts.get("updates", 1000), "updates")
        rank = integer(opts.get("rank", 8), "rank")
        save_every = integer(opts.get("save_every", 50), "save_every")
        lr = float(opts.get("lr", 0.0001))
        pref = float(opts.get("dpo_weight", 0))
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(pref) or pref < 0:
            raise ValueError("Invalid learning rate or DPO weight")
        source = child(home, self.get_conf("teacher_job", required=True)) if stage == "student" else out
        cache = child(source, "teacher-cache")
        if stage == "student" and (source == out or not (cache / "ready.json").is_file()):
            raise ValueError("Select a completed teacher cache from another job")
        cache.mkdir(exist_ok=True)
        cache_id = cache / "cache-id.json"
        if stage != "student" and not cache_id.exists():
            save_json(cache_id, uuid.uuid4().hex)
        rows = records(r)
        votes = self.get_conf("votes", [])
        excluded = self.get_conf("excluded", [])
        pairs = ranked_pairs(rows, votes, excluded)
        if pref and not pairs:
            raise ValueError("DPO needs at least one unequal within-prompt preference. Vote or turn DPO off.")
        excluded_set = set(excluded)
        usable = [row for row in rows if row["id"] not in excluded_set]
        if not usable:
            raise ValueError("No teacher targets left after exclusions")
        total = (0 if stage == "student" else len(rows)) + (0 if stage == "prepare" else updates)
        status.begin_process("Loading MOSS", total)
        dev = torch.device(self.get_conf("device", "cuda:0"))
        model = MossImageModel(dev, ModelConfig(arch="moss_image", name_or_path=r["model"], dtype="bf16",
                    low_vram=True, vae_dtype="bf16", model_kwargs={"fused_pointwise": False}), dtype="bf16")
        # Load/encode the TE before moving the DiT to CUDA. Do not keep both on the card.
        tg, sg = grids(r)
        ctx_path = cache / "contexts.safetensors"
        if stage != "student":
            bind(cache / "recipe.json", r)
            if not ctx_path.exists():
                model.text_encoder = model._load_text_encoder()
                model.tokenizer = model._load_tokenizer()
                contexts = {}
                for i, prompt in enumerate([*r["prompts"], r.get("negative", "")]):
                    tick(f"Encoding prompt {i+1}/{len(r['prompts'])+1}")
                    contexts[str(i)] = model.get_prompt_embeds(prompt).text_embeds[0].detach().cpu().contiguous()
                atomic(ctx_path, lambda f: save_file(contexts, str(f)))
                model.text_encoder = model.tokenizer = None
                torch.cuda.empty_cache()
        contexts = load_file(str(ctx_path))
        model.model = net = model._load_transformer().eval().requires_grad_(False).to(dev)
        # Detect in-place replacement of local transformer weights across generation/resume.
        files = {}
        for k, filename in model.feedback_component_sources.items():
            if k.startswith("transformer"):
                p = Path(filename).resolve()
                files[k] = [str(p), p.stat().st_size, p.stat().st_mtime_ns]
        meta = {"version": 1, "cache": json.loads(cache_id.read_text()), "recipe": r,
                "weights": files, "teacher_grid": tg, "target_grid": sg}
        key = digest(meta)
        if stage == "student":
            if json.loads((cache/"ready.json").read_text())["fingerprint"] != key or self.get_conf("fingerprint") != key:
                raise ValueError("Teacher cache, model or review fingerprint changed")
        else:
            bind(cache / "identity.json", meta)
            self.prepare(model, net, r, tg, sg, cache, contexts, rows, tick, dev, stage == "prepare")
            save_json(cache/"ready.json", {"fingerprint": key, "rows": rows, "recipe": r, "teacher_job": self.name})
        if stage == "prepare":
            return
        identity = {"teacher": key, "rank": rank, "lr": lr, "votes": votes, "excluded": excluded, "dpo_weight": pref}
        bind(out/"student-recipe.json", identity)
        save_json(out/"target-sampler.json", {"sampler": r["sampler"], "sigmas": sg, "cfg": 1, "steps": r["steps"]})
        groups = {row["id"]: row["group"] for row in rows}
        save_json(out/"preference-pairs.json", [{"chosen": a, "rejected": b,
                   "prompt": r["prompts"][groups[a]]} for a,b in pairs])
        for a, b in pairs:
            # Matching stems are the fork's native offline DPO folder convention.
            for sub, item in (("chosen", a), ("rejected", b)):
                folder = out/"preferences-export"/sub
                folder.mkdir(parents=True, exist_ok=True)
                stem = a + "__" + b
                atomic(folder/(stem+".png"), lambda f: shutil.copyfile(cache/(item+".png"), f))
                atomic(folder/(stem+".txt"), lambda f: f.write_text(r["prompts"][groups[a]], encoding="utf-8"))
        self.fit(model, net, cache, out, contexts, rows, usable, pairs, sg, r["sampler"], rank, lr, pref,
                 updates, save_every, digest(identity), tick, dev, len(rows) if stage == "all" else 0)

    @staticmethod
    @torch.no_grad()
    def prepare(model, net, r, tg, sg, cache, contexts, rows, tick, dev, previews):
        if previews:
            model.vae = model._load_vae().to(dev)
        neg = velocity(net, contexts[str(len(r["prompts"]))][None].to(dev), (22,))
        for j, row in enumerate(rows):
            key = row["id"]
            path = cache / (key + ".safetensors")
            png = cache / (key + ".png")
            tick(f"Teacher {j+1}/{len(rows)} · {key}", j)
            if not path.exists():
                ctx = contexts[str(row["group"])][None].to(dev)
                cond = velocity(net, ctx)
                def guided(x, s):
                    v = cond(x, s)
                    return v if r["cfg"] == 1 else v + (r["cfg"] - 1) * (v - neg(x, s))
                gen = torch.Generator(device="cpu").manual_seed(row["seed"])
                x = torch.randn((1, net.params.latent_channels, r["height"]//16, r["width"]//16),
                                generator=gen, dtype=torch.float32).to(dev)
                anchors = {str(i): y.cpu().contiguous() for i, y in enumerate(teacher_anchors(x, tg, sg, guided))}
                atomic(path, lambda f: save_file(anchors, str(f)))
            if previews and not png.exists():
                end = read_anchor(cache, key, len(sg)-1, dev)
                im = model.decode_latents(end, dev, torch.float32)[0].clamp(-1,1)
                im = ((im+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
                atomic(png, lambda f: Image.fromarray(im).save(f, format="PNG"))
        tick("Teacher cache complete", len(rows))

    @staticmethod
    def fit(model, net, cache, out, contexts, rows, usable, pairs, sg, sampler, rank, lr, pref,
            updates, save_every, fingerprint, tick, dev, offset):
        net.enable_gradient_checkpointing()
        # Initialize reproducibly without relying on the order of downloads/teacher generation.
        torch.manual_seed(12345)
        adapters = LowRank(net, rank)
        opt = torch.optim.AdamW(adapters.parameters(), lr=lr, weight_decay=0)
        group = {row["id"]: row["group"] for row in rows}
        checkpoint = out/"distill-state.pt"
        start = 0
        try:
            if checkpoint.is_file():
                saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
                if saved["fingerprint"] != fingerprint:
                    raise ValueError("Checkpoint does not match this student")
                adapters.load_state_dict(saved["adapter"])
                opt.load_state_dict(saved["optimizer"])
                start = saved["step"]
                del saved
            if updates < start:
                raise ValueError("Requested fewer updates than already completed")
            def export():
                atomic(out/"moss-speed.safetensors", lambda f: save_file(adapters.export(model), str(f),
                       metadata={"sampler": sampler, "steps": str(len(sg)-1), "sigmas": json.dumps(sg), "cfg": "1"}))
            for step in range(start+1, updates+1):
                rng = random.Random(step)
                if step == start+1 or step % 10 == 0:
                    tick(f"Student update {step}/{updates}", offset+step-1)
                row = rng.choice(usable)
                i = rng.randrange(len(sg)-1)
                x = read_anchor(cache, row["id"], i, dev)
                y = read_anchor(cache, row["id"], i+1, dev)
                ctx = contexts[str(row["group"])][None].to(dev)
                a,b = sg[i:i+2]
                end = solver_step(velocity(net, ctx), x, a, b, sampler)
                # Match average velocity, not raw displacements (which vanish on narrow intervals).
                loss = ((end - y)/(a-b)).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite trajectory loss at {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if pref:
                    win, lose = rng.choice(pairs)
                    ctx = contexts[str(group[win])][None].to(dev)
                    good, bad = (read_anchor(cache, k, len(sg)-1, dev) for k in (win, lose))
                    gen = torch.Generator(device=dev).manual_seed(step)
                    eps = torch.randn(good.shape, generator=gen, device=dev)
                    dpo = flow_preference(net, adapters, ctx, good, bad, rng.uniform(.02,.98), eps, 100)
                    if not torch.isfinite(dpo):
                        raise FloatingPointError(f"Non-finite preference loss at {step}")
                    (pref*dpo).backward()
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                if step % save_every == 0 or step == updates:
                    atomic(checkpoint, lambda f: torch.save({"fingerprint": fingerprint, "adapter": adapters.state_dict(),
                           "optimizer": opt.state_dict(), "step": step}, f))
                    export()
                    tick(f"Saved update {step}; trajectory MSE {loss.item():.5g}", offset+step)
            if start == updates:
                export()  # recover a crash between publishing the checkpoint and adapter
        finally:
            adapters.close()
