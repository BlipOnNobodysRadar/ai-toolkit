"""Model state that sample generation borrows from training and gives back, also when generation fails.

Shared by BaseModel.generate_images and the legacy StableDiffusion.generate_images (docs/STATE_MACHINES.md, M6):
assistant/inference LoRA activation and placement, the trained network's mode, multiplier, activation and (when
merging is allowed) its merge, the device-state preset and the CPU/CUDA RNG states.

Ownership rules: entry values are recorded before anything changes, and return_ undoes only what borrow did. An
entry state that cannot be given back exactly is refused before any mutation: an auxiliary LoRA spread over several
devices/dtypes, or one sampling would have to cast to another dtype (casting back would not restore its values).
Every give-back step is attempted even if an earlier one fails; a give-back failure never replaces the error that
ended generation (it is attached to it as a note, best effort).

Merging the network into the base weights for speed is lossy: merge_in/merge_out round-trip every touched weight
through its dtype (bf16/fp16), so W -> W + d - d is not W. A holder that is being trained sets
``merge_network_for_sampling = False`` and samples through the LoRA hooks instead; inference-only holders keep the
merge.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from toolkit.accelerator import unwrap_model
from toolkit.print import print_acc


def settle(steps, primary=None, what='borrowed state'):
    """Run every cleanup step. With a primary error in flight, failures are attached to it and it stays the error;
    otherwise the first failure is raised (later ones attached)."""
    errors = []
    for step in steps:
        try:
            step()
        except BaseException as e:  # a KeyboardInterrupt here must not skip the remaining steps either
            errors.append(e)
    if not errors:
        return
    if primary is not None:
        for e in errors:
            try:  # best effort: nothing here may replace the primary error
                detail = ''.join(f"; {note}" for note in getattr(e, '__notes__', ()))
                primary.add_note(f"while giving back {what} after this error: {e!r}{detail}")
                print_acc(f"Could not give back {what}: {e!r}")
            except BaseException:
                pass
        return
    for e in errors[1:]:
        errors[0].add_note(f"also failed while giving back {what}: {e!r}")
    raise errors[0]


def enter_device_preset(holder, preset):
    """Save the device state, then apply a preset; returns the saved state for leave_device_preset. If applying the
    preset fails part-way (some modules moved), the saved state is put back before the error propagates."""
    holder.save_device_state()  # nothing has moved if this fails
    saved = getattr(holder, 'device_state', None)
    try:
        holder.set_device_state_preset(preset)
    except BaseException as primary:
        settle([lambda: leave_device_preset(holder, saved)], primary, 'the device state')
        raise
    return saved


def leave_device_preset(holder, saved):
    # set_device_state_preset re-saves before moving anything; put back the snapshot taken before any move, in case
    # that re-save failed half way
    if saved is not None:
        holder.device_state = saved
    holder.restore_device_state()


def _placement(lora):
    """(device, dtype) shared by all of an auxiliary LoRA's floating tensors (parameters and buffers: force_to moves
    both), None if it has none. A LoRA spread over several devices or dtypes cannot be put back by force_to, so it is
    refused before sampling changes anything."""
    modules = [lora] + list(getattr(lora, 'unet_loras', None) or []) + list(getattr(lora, 'text_encoder_loras', None) or [])
    places = set()
    for module in modules:
        for t in list(module.parameters()) + list(module.buffers()):
            if t.is_floating_point():
                places.add((t.device, t.dtype))
    if len(places) > 1:
        raise ValueError(f"The assistant/inference LoRA is spread over {sorted(map(str, places))}; sampling can only "
                         f"borrow and give back one placement")
    return next(iter(places), None)


def _module_device(module):
    device = getattr(module, 'device', None)
    if device is None:
        p = next(module.parameters(), None)
        device = p.device if p is not None else None
    return device


def borrow(holder, image_configs, blank_network, move_unet_before_merge=False):
    """Switch holder to sampling; returns the record return_() needs. Undoes its own steps if one fails."""
    state = SimpleNamespace(network=blank_network, owns_network=False, network_training=None, start_multiplier=1.0,
                            network_was_active=None, merge_multiplier=1.0, merge_started=False, merged=False,
                            unet_entry_device=None, aux=None, aux_was_active=None, aux_placement=None,
                            aux_moved=False, device_state=False, saved_device_state=None, rng=None, cuda_rng=None)
    # entry state, recorded (and validated) before anything changes
    assistant = holder.model_config.assistant_lora_path is not None
    inference = holder.model_config.inference_lora_path is not None
    if assistant or inference:
        state.aux_placement = _placement(holder.assistant_lora)
        moves = inference or holder.invert_assistant_lora
        if moves and state.aux_placement is not None and state.aux_placement[1] != holder.torch_dtype:
            # every loader casts it to the holder dtype; a round trip through another dtype would change its values
            raise ValueError(f"The assistant/inference LoRA is {state.aux_placement[1]} but samples run in "
                             f"{holder.torch_dtype}; sampling cannot borrow it without changing its weights")
    network = unwrap_model(holder.network) if holder.network is not None else None
    try:
        state.rng = torch.get_rng_state()
        state.cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        if assistant or inference:
            aux = state.aux = holder.assistant_lora
            state.aux_was_active = aux.is_active
            # an inverted assistant and an inference lora are off the device while training and switched on here;
            # a plain assistant is switched off
            active, move = (True, True) if (inference or holder.invert_assistant_lora) else (False, False)
            print_acc("Loading inference lora" if inference else "Unloading assistant lora")
            aux.is_active = active
            if move:
                state.aux_moved = True  # set first: a force_to that fails part-way still gets moved back
                aux.force_to(holder.device_torch, holder.torch_dtype)

        if network is not None:
            state.network, state.owns_network = network, True
            state.network_training = network.training
            state.start_multiplier = network.multiplier
            state.network_was_active = network.is_active
            network.eval()
            # check if we have the same network weight for all samples. If we do, we can merge in
            # the network to drastically speed up inference (not while training: see the module docstring).
            # A network that is already merged stays as it is: merging again would add the delta twice.
            unique_network_weights = set([x.network_multiplier for x in image_configs])
            if (len(unique_network_weights) == 1 and network.can_merge_in and not network.is_merged_in
                    and getattr(holder, 'merge_network_for_sampling', True)):
                if move_unet_before_merge:
                    # make sure it is on device before merging.
                    state.unet_entry_device = _module_device(holder.unet)
                    holder.unet.to(holder.device_torch)
                state.merge_multiplier = unique_network_weights.pop()
                state.merge_started = True
                network.merge_in(merge_weight=state.merge_multiplier)
                state.merged = True
        else:
            state.start_multiplier = blank_network.multiplier

        state.saved_device_state = enter_device_preset(holder, 'generate')
        state.device_state = True
    except BaseException as primary:
        return_(holder, state, primary)
        raise
    return state


def return_(holder, state, primary=None):
    """Give back what borrow took, in the original teardown order; safe on a partly borrowed state."""
    steps = []
    if state.rng is not None:
        steps.append(lambda: torch.set_rng_state(state.rng))
    if state.cuda_rng is not None:
        steps.append(lambda: torch.cuda.set_rng_state(state.cuda_rng))
    if state.device_state:
        steps.append(lambda: leave_device_preset(holder, state.saved_device_state))
    network = state.network
    if state.owns_network:
        def network_flags():
            if state.network_training:
                network.train()
            else:
                network.eval()
            network.multiplier = state.start_multiplier
            network.is_active = state.network_was_active
        steps.append(network_flags)
    if state.merged:
        steps.append(lambda: network.merge_out(state.merge_multiplier))
    elif state.merge_started:
        def partial_merge():
            # merge_in failed part-way: merging out would subtract deltas that were never added
            raise RuntimeError("Merging the network for sampling failed part-way; the base weights are partly "
                               "merged and must be reloaded")
        steps.append(partial_merge)
    if state.unet_entry_device is not None:
        steps.append(lambda: holder.unet.to(state.unet_entry_device))
    if state.aux is not None:
        aux = state.aux

        def aux_back():
            print_acc("Unloading inference lora" if holder.model_config.inference_lora_path is not None
                      else "Loading assistant lora")
            aux.is_active = state.aux_was_active
            if state.aux_moved and state.aux_placement is not None:
                # back where it was (the loaders park an inactive one on the CPU), same dtype: values unchanged
                aux.force_to(*state.aux_placement)
        steps.append(aux_back)
    settle(steps, primary, 'the sampling state')


@contextmanager
def borrowed(holder, image_configs, blank_network, move_unet_before_merge=False):
    """borrow ... return_ around a generation body; a body error stays the error even if giving back also fails."""
    state = borrow(holder, image_configs, blank_network, move_unet_before_merge)
    try:
        yield state
    except BaseException as primary:
        return_(holder, state, primary)
        raise
    return_(holder, state)
