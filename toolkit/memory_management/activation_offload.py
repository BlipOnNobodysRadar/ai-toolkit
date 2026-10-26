"""
Activation offloading for checkpointed block loops (model.activation_offloading).

Under gradient checkpointing each block keeps only its input for the backward recompute, but it keeps it on the
GPU: for a block loop that is one hidden-state tensor per block, all alive from the end of forward until that
block's backward. On long video sequences that is the largest single VRAM cost that scales with tokens (MiniMax-H3:
50 blocks x 5376 x bf16 = 0.51 MiB per token).

ActivationOffloader moves each block input to CPU RAM once the block's forward is done and frees its GPU storage;
a hook on the block's output gradient copies it back just before that block's backward recompute. The tensor
object itself never changes (same shape, strides, version counter), so torch.utils.checkpoint (non-reentrant)
unpacks it as if nothing happened and the math is bit-identical. Gradients never pass through the CPU.

Ordering: both copies are queued on the current stream, the one the block computes on, so the D2H is ordered
before any reuse of the freed block (the caching allocator recycles it only within that stream) and the H2D is
ordered before the recompute. The host buffer is pinned on CUDA, so neither copy blocks the host and the layer
offloading ring keeps running ahead. Pinned buffers come from torch's caching host allocator and are reused
across steps.

Only plain, fully owned, contiguous inputs whose block output requires grad are offloaded; anything else is left
on the device (frozen block ranges, inference, views into a larger storage).
"""

from typing import List, Optional

import torch

from toolkit.print import print_acc


def _owns_whole_storage(t: torch.Tensor) -> bool:
    """True when ``t`` is the only view of a storage sized exactly to it, so freeing that storage frees ``t`` and
    nothing else, and refilling it restores ``t`` exactly."""
    if t.is_sparse or not t.is_contiguous() or t.storage_offset() != 0:
        return False
    try:
        storage = t.untyped_storage()
    except (RuntimeError, NotImplementedError):
        return False
    return storage.nbytes() == t.numel() * t.element_size() and storage.nbytes() > 0


class ActivationOffloader:
    """One instance per forward pass of a block loop. Call ``stash(block_input, block_output)`` after each
    checkpointed block; ``block_input`` must not be read again in that forward pass."""

    _logged = False
    _warned_pin = False

    def __init__(self):
        self.num_offloaded = 0
        self.bytes_offloaded = 0

    def _host_buffer(self, t: torch.Tensor):
        pin = t.is_cuda
        if pin:
            try:
                return torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True), True
            except RuntimeError as e:
                if not ActivationOffloader._warned_pin:
                    ActivationOffloader._warned_pin = True
                    print_acc(
                        f"activation offloading: pinned host allocation failed ({e}); using pageable memory, "
                        "so each copy blocks the host"
                    )
        return torch.empty(t.shape, dtype=t.dtype, device="cpu"), False

    def stash(self, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        if not (
            torch.is_grad_enabled()
            and isinstance(block_output, torch.Tensor)
            and block_output.requires_grad
            and isinstance(block_input, torch.Tensor)
            and _owns_whole_storage(block_input)
        ):
            return
        x = block_input
        host, pinned = self._host_buffer(x)
        host.copy_(x.detach(), non_blocking=pinned)

        storage = x.untyped_storage()
        nbytes = storage.nbytes()
        storage.resize_(0)

        # everything the refill needs, dropped right after the refill: holding the storage past it would keep
        # every refilled input alive until the end of backward instead of until its own block's backward is done
        # (the checkpoint frame releases its saved input then)
        state: List[Optional[object]] = [
            storage, host, pinned, nbytes, x.dtype, x.device, x.size(), x.stride()
        ]

        def refill(grad):
            if not state:
                return None
            storage, host, pinned, nbytes, dtype, device, size, stride = state
            state.clear()
            with torch.no_grad():
                storage.resize_(nbytes)
                # write through an alias with its own version counter, so the saved input's version (checked when
                # the checkpoint unpacks it) is untouched
                alias = torch.empty(0, dtype=dtype, device=device).set_(storage, 0, size, stride)
                alias.copy_(host, non_blocking=pinned)
            return None

        block_output.register_hook(refill)
        self.num_offloaded += 1
        self.bytes_offloaded += nbytes

    def log_once(self) -> None:
        if self.num_offloaded and not ActivationOffloader._logged:
            ActivationOffloader._logged = True
            print_acc(
                f"activation offloading: {self.num_offloaded} block inputs, "
                f"{self.bytes_offloaded / 2**30:.2f} GiB per forward kept in CPU RAM"
            )
