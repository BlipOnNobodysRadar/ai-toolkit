"""H3 AdaLN projections bypass the lora hook (F.linear on the raw weight), so
LoRASpecialNetwork must not create (and export) loras for them. CPU only."""
import torch

from toolkit.lora_special import LoRASpecialNetwork
from toolkit.config_modules import NetworkConfig


def _h3_cls():
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
    return MinimaxH3Model


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(8, 24)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from extensions_built_in.diffusion_models.minimax_h3.src.transformer import MiniMaxH3AdalnProj
        self.adaln_proj = MiniMaxH3AdalnProj(8, 8, expand=2, modalities=1, apply_silu=False)
        self.attn = _Attn()


class ToyH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])


class _StubModel:
    use_old_lokr_format = False

    def __init__(self, arch, ignores=None):
        self.arch = arch
        if ignores is not None:
            self.lora_ignore_if_contains = ignores

    def get_transformer_block_names(self):
        return ["blocks"]


def _names(base_model, ignore=None):
    toy = ToyH3()
    kw = {}
    if ignore is not None:
        kw['ignore_if_contains'] = ignore
    n = LoRASpecialNetwork(
        None, toy, lora_dim=4, alpha=4, train_text_encoder=False,
        target_lin_modules=['ToyH3'], is_transformer=True, transformer_only=True,
        base_model=base_model, network_config=NetworkConfig(type='lora', linear=4, linear_alpha=4),
        **kw)
    return [m.lora_name for m in n.unet_loras]


def test_h3_declares_adaln_ignore():
    cls = _h3_cls()
    assert 'adaln_proj' in cls.lora_ignore_if_contains
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
        MinimaxH3Ref2VAModel, MinimaxH3FastModel, MinimaxH3OmniModel)
    for sub in (MinimaxH3Ref2VAModel, MinimaxH3FastModel, MinimaxH3OmniModel):
        assert 'adaln_proj' in sub.lora_ignore_if_contains


def test_h3_skips_adaln_even_with_explicit_empty_ignore():
    stub = _StubModel('minimax_h3', list(_h3_cls().lora_ignore_if_contains))
    cfg_list = []
    names = _names(stub, cfg_list)
    assert names and not any('adaln_proj' in n for n in names)
    assert sum('qkv_proj' in n for n in names) == 2
    assert cfg_list == []  # config list is not mutated
    assert not any('adaln_proj' in n for n in _names(stub))


def test_models_without_declared_ignores_keep_adaln():
    # e.g. Ideogram4, whose adaln_proj is a normally-called, trainable linear
    names = _names(_StubModel('ideogram4'), [])
    assert sum('adaln_proj' in n for n in names) == 2
    assert sum('qkv_proj' in n for n in names) == 2


def test_merge_keeps_config_ignores():
    stub = _StubModel('minimax_h3', ['adaln_proj'])
    names = _names(stub, ['blocks.1.'])
    assert names and all('blocks$$0$$attn' in n for n in names)
