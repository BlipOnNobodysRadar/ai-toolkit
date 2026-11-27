"""Native CPU integration fixture: no downloads, no GPU, run in the normal training environment."""
import pytest
import torch
from safetensors.torch import load_file
from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageParams, MossImageTransformer
from extensions_built_in.moss_distill.process import MossDistill, read_anchor
from extensions_built_in.moss_distill.recipe import records


@pytest.mark.parametrize("sampler,pref", [("euler",0.), ("heun",.1)])
def test_prepare_update_resume_and_export(tmp_path, sampler, pref):
    torch.manual_seed(10)
    device = torch.device("cpu")
    net = MossImageTransformer(MossImageParams(dim=64, ffn_dim=128, num_heads=2, num_layers=2,
        text_dim=32, freq_dim=16, latent_channels=8)).to(torch.bfloat16).eval().requires_grad_(False)
    model = MossImageModel.__new__(MossImageModel)
    model.lora_keys_use_comfy_prefix = True
    ctx = {"0": torch.randn(3,32,dtype=torch.bfloat16), "1": torch.randn(2,32,dtype=torch.bfloat16)}
    r = dict(prompts=["sign"], copies=2, seed=5, height=32, width=32, cfg=3)
    rows = records(r)
    sg = [1.,.4,0.]
    tg = [1.,.7,.3,0.]
    tick = lambda *args: None
    MossDistill.prepare(model,net,r,tg,sg,tmp_path,ctx,rows,tick,device,False)
    assert not torch.equal(read_anchor(tmp_path,rows[0]["id"],0,device),read_anchor(tmp_path,rows[1]["id"],0,device))
    before = {k:p.clone() for k,p in net.named_parameters()}
    pairs = [(rows[0]["id"],rows[1]["id"])] if pref else []
    def fit(to):
        MossDistill.fit(model,net,tmp_path,tmp_path,ctx,rows,rows,pairs,sg,sampler,2,.001,pref,to,1,"tiny",tick,device,0)
    fit(1)
    first = load_file(str(tmp_path/"moss-speed.safetensors"))
    assert any(t.abs().sum()>0 for k,t in first.items() if k.endswith("lora_up.weight"))
    fit(2)
    last = load_file(str(tmp_path/"moss-speed.safetensors"))
    assert any(not torch.equal(v,last[k]) for k,v in first.items())
    assert all(torch.equal(before[k],v) for k,v in net.named_parameters())
    assert torch.load(tmp_path/"distill-state.pt",weights_only=True)["step"]==2
    assert "diffusion_model.layers.0.proj.q.lora_down.weight" in last
