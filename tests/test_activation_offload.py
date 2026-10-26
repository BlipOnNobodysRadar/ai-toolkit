"""model.activation_offloading on MiniMax-H3: checkpointed block inputs move to CPU RAM after each block's forward
and come back right before that block's recompute (toolkit/memory_management/activation_offload.py).

The CPU tests run a tiny H3 transformer in float32, where the offload path still frees and refills the storages
(host buffers are plain CPU tensors there), and require outputs and every gradient to be bit-identical to the same
model without offloading. The CUDA test adds the pinned, non-blocking path and checks the peak-memory saving.
"""
import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.memory_management.activation_offload import ActivationOffloader

TEXT, VIDEO, AUDIO = 5, 12, 3


def tiny_params(**kw):
    base = dict(
        hidden_size=32,
        num_layers=4,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=48,
        latents_dim=4,
        audio_latents_dim=4,
        text_dim=24,
        timestep_input_dim=16,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=2,  # rotates 12 of the 16 head channels
    )
    base.update(kw)
    return MiniMaxH3TransformerParams(**base)


def build(device="cpu", dtype=torch.float32, **kw):
    torch.manual_seed(0)
    model = MiniMaxH3Transformer(tiny_params(**kw)).to(device=device, dtype=dtype)
    model.enable_gradient_checkpointing()
    return model


def make_inputs(p, batch=1, text=TEXT, video=VIDEO, audio=AUDIO, pad_rows=0, device="cpu", dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    seq = text + video + audio
    video_dim = p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
    tags = torch.empty(batch, seq, dtype=torch.long)
    tags[:, :text] = 1
    tags[:, text : text + video] = 0
    tags[:, text + video :] = 2
    if pad_rows:
        tags[-1, text - pad_rows : text] = -1  # the last item's prompt is shorter
    times = torch.empty(batch, seq)
    for b in range(batch):
        times[b, :text] = 1.0
        times[b, text : text + video] = 0.37 + 0.4 * b
        times[b, text + video :] = 0.61
    return dict(
        hidden_states=torch.randn(batch, video, video_dim, generator=g).to(device, dtype),
        audio_hidden_states=torch.randn(batch, audio, p.audio_latents_dim, generator=g).to(device, dtype),
        encoder_hidden_states=torch.randn(batch, text, p.text_dim, generator=g).to(device, dtype),
        row_timesteps=times.to(device),
        token_tags=tags.to(device),
        position_ids=(torch.rand(batch, seq, 3, generator=g) * 8).to(device),
        video_indices=torch.arange(text, text + video, device=device),
        audio_indices=torch.arange(text + video, seq, device=device),
        text_indices=torch.arange(0, text, device=device),
    )


def loss_of(video_out, audio_out):
    g = torch.Generator().manual_seed(7)
    wv = torch.randn(video_out.shape, generator=g).to(video_out.device, video_out.dtype)
    wa = torch.randn(audio_out.shape, generator=g).to(audio_out.device, audio_out.dtype)
    return (video_out * wv).sum() + (audio_out * wa).sum()


def run(model, batches, offload, retain_twice=False):
    model.activation_offloading = offload
    model.zero_grad(set_to_none=True)
    outs = []
    for inputs in batches:
        video_out, audio_out = model(**inputs)
        loss = loss_of(video_out, audio_out)
        if retain_twice:
            loss.backward(retain_graph=True)
        loss.backward()
        outs.append((video_out.detach().clone(), audio_out.detach().clone()))
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return outs, grads


def assert_identical(a, b):
    outs_a, grads_a = a
    outs_b, grads_b = b
    for (va, aa), (vb, ab) in zip(outs_a, outs_b):
        assert torch.equal(va, vb) and torch.equal(aa, ab)
    assert grads_a.keys() == grads_b.keys() and len(grads_a) > 0
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name


@pytest.fixture
def stash_log(monkeypatch):
    """Record every block input actually offloaded (stash returns early for the ones it skips)."""
    log = []
    orig = ActivationOffloader.stash

    def recording(self, block_input, block_output):
        before = self.num_offloaded
        orig(self, block_input, block_output)
        if self.num_offloaded > before:
            log.append(block_input)

    monkeypatch.setattr(ActivationOffloader, "stash", recording)
    return log


def test_offload_matches_plain_cpu(stash_log):
    model = build()
    batches = [make_inputs(model.params)]
    plain = run(model, batches, offload=False)
    assert stash_log == []
    offloaded = run(model, batches, offload=True)
    assert len(stash_log) == model.params.num_layers
    assert_identical(plain, offloaded)


def test_storages_freed_after_forward_and_refilled_in_backward(stash_log):
    model = build()
    model.activation_offloading = True
    video_out, audio_out = model(**make_inputs(model.params))
    assert len(stash_log) == model.params.num_layers
    sizes = [t.numel() * t.element_size() for t in stash_log]
    assert all(t.untyped_storage().nbytes() == 0 for t in stash_log)
    loss_of(video_out, audio_out).backward()
    # the log holds the tensors, so their refilled storages are still here to inspect
    assert [t.untyped_storage().nbytes() for t in stash_log] == sizes


def test_gradient_accumulation_two_microbatches():
    model = build()
    batches = [make_inputs(model.params, seed=1), make_inputs(model.params, seed=2)]
    assert_identical(run(model, batches, offload=False), run(model, batches, offload=True))


def test_retain_graph_second_backward():
    model = build()
    batches = [make_inputs(model.params)]
    assert_identical(
        run(model, batches, offload=False, retain_twice=True),
        run(model, batches, offload=True, retain_twice=True),
    )


def test_batch_two_with_padding_rows():
    model = build()
    batches = [make_inputs(model.params, batch=2, pad_rows=2)]
    assert_identical(run(model, batches, offload=False), run(model, batches, offload=True))


def test_no_grad_forward_is_untouched(stash_log):
    model = build()
    inputs = make_inputs(model.params)
    with torch.no_grad():
        model.activation_offloading = False
        ref = model(**inputs)
        model.activation_offloading = True
        out = model(**inputs)
    assert stash_log == []
    assert all(torch.equal(a, b) for a, b in zip(ref, out))


def test_frozen_model_offloads_nothing(stash_log):
    model = build()
    model.requires_grad_(False)
    inputs = make_inputs(model.params)
    model.activation_offloading = False
    ref = model(**inputs)
    model.activation_offloading = True
    out = model(**inputs)
    assert stash_log == []
    assert all(torch.equal(a, b) for a, b in zip(ref, out))


def test_without_checkpointing_flag_is_ignored(stash_log):
    model = build()
    model.disable_gradient_checkpointing()
    batches = [make_inputs(model.params)]
    plain = run(model, batches, offload=False)
    offloaded = run(model, batches, offload=True)
    assert stash_log == []
    assert_identical(plain, offloaded)


def test_skips_inputs_that_do_not_own_their_storage():
    base = torch.randn(4, 8, requires_grad=True)
    view = base[:, :4]  # a view into a larger storage
    out = (view * 2).sum(dim=1, keepdim=True)
    offloader = ActivationOffloader()
    offloader.stash(view, out)
    assert offloader.num_offloaded == 0
    assert base.untyped_storage().nbytes() == base.numel() * base.element_size()


def test_model_config_flag_defaults_off():
    from toolkit.config_modules import ModelConfig

    assert ModelConfig(name_or_path="x").activation_offloading is False
    assert ModelConfig(name_or_path="x", activation_offloading=True).activation_offloading is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned non-blocking path and peak memory need CUDA")
def test_cuda_pinned_offload_identical_and_saves_memory(monkeypatch):
    """GPU smoke: float32 with the math SDPA kernel so both runs are deterministic, a sequence long enough that the
    block inputs dominate, and a peak-memory comparison."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    hosts = []
    orig_host = ActivationOffloader._host_buffer

    def recording_host(self, t):
        buf, pinned = orig_host(self, t)
        hosts.append((buf, pinned))
        return buf, pinned

    monkeypatch.setattr(ActivationOffloader, "_host_buffer", recording_host)

    dev = torch.device("cuda")
    layers = 12
    model = build(device=dev, hidden_size=256, num_layers=layers, num_attention_heads=2, attention_head_dim=128,
                  ffn_hidden_size=512, rope_inv_freq_len=16)
    # like LoRA training: a frozen base with a small trainable part, so gradients don't rival the block inputs at
    # the end of backward (with every weight trainable the peak moves there and hides the saving)
    model.requires_grad_(False)
    for block in model.blocks:
        block.mlp.fc2.weight.requires_grad_(True)
    batches = [make_inputs(model.params, video=4096, device=dev)]
    boundary_bytes = (TEXT + 4096 + AUDIO) * 256 * 4

    def measured(offload, backend):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        with sdpa_kernel(backend):
            result = run(model, batches, offload=offload)
        torch.cuda.synchronize()
        return result, torch.cuda.max_memory_allocated() - start

    # bit-identity under the deterministic math kernel
    plain, _ = measured(False, SDPBackend.MATH)
    offloaded, _ = measured(True, SDPBackend.MATH)
    assert_identical(plain, offloaded)
    assert hosts and all(pinned and buf.is_pinned() for buf, pinned in hosts)

    # memory under the memory-efficient kernel, whose O(S) footprint doesn't swamp the block inputs
    _, peak_plain = measured(False, SDPBackend.EFFICIENT_ATTENTION)
    _, peak_offload = measured(True, SDPBackend.EFFICIENT_ATTENTION)
    saved = peak_plain - peak_offload
    print(f"peak plain {peak_plain / 2**20:.1f} MiB, offloaded {peak_offload / 2**20:.1f} MiB, "
          f"saved {saved / 2**20:.1f} MiB; block inputs {layers} x {boundary_bytes / 2**20:.1f} MiB")
    # all but about two block inputs (the one in flight and the one being produced) should be gone at the peak
    assert saved >= (layers - 3) * boundary_bytes
