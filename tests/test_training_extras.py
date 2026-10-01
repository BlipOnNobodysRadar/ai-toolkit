import math
import copy
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training_extras import runtime as rt
from training_extras.loss_bridge import LossBridge
from training_extras.optim import coerce_args, create_optimizer, create_scheduler
from training_extras.toolkit_validation import fixed_rng

VALID_LOSSES = [
    "l2", "l1", "standard_pseudo_huber", "standard_huber", "standard_smooth_l1",
    "huber", "smooth_l1", "x_sigmoid", "log_cosh", "squared_logarithmic",
    "soft_welsch", "scaled_quadratic", "standard_deviation_loss", "psnr_loss",
    "geman_mcclure_loss", "smooth_l2_log", "mse_pyramid_2d",
]


def _sched(num=1000, with_alphas=True):
    cfg = SimpleNamespace(num_train_timesteps=num)
    if with_alphas:
        cfg.alphas_cumprod = torch.linspace(0.99, 0.001, num)
    scheduler=SimpleNamespace(config=cfg)
    if with_alphas:scheduler.alphas_cumprod=cfg.alphas_cumprod
    return scheduler


def _bridge(loss="l2", is_flow=False, pred_type="epsilon", num=1000, with_alphas=True):
    return LossBridge({"loss": loss}, _sched(num, with_alphas), is_flow, pred_type)


def test_create_optimizer_adamw_and_coerce():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = create_optimizer(p, "adamw", 0.01)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    args = coerce_args({"a": "true", "b": "false", "c": "(1, 2)"})
    assert args["a"] is True and args["b"] is False
    assert isinstance(args["c"], tuple)
    assert coerce_args({"s": "__import__('os').system('x')"})["s"].startswith("__import__")
    with pytest.raises(TypeError):
        create_optimizer(p, "adamw", 0.01, {"not_a_real_kwarg": 1})


@pytest.mark.parametrize("name,kwargs", [("easy:WarpAINO", {"compile_step":False,"meta_lr":0.0}), ("easy:SCORN", {}), ("easy:SimplifiedAdEMAMixExM", {"torch_compile":False})])
def test_custom_optimizer_changes_weights(name,kwargs):
    p=torch.nn.Parameter(torch.randn(8,8))
    before=p.detach().clone()
    opt=create_optimizer([p],name,0.01,kwargs)
    
    for _ in range(3):
        opt.zero_grad();p.square().mean().backward();opt.step()
    assert torch.isfinite(p).all() and not torch.equal(p,before)


def test_adamw_native_update_changes_weights():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    x = torch.randn(4, 8)
    before = model.weight.detach().clone()
    for _ in range(4):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    assert torch.isfinite(model.weight).all()
    assert not torch.allclose(before, model.weight.detach())


def test_roundtrip_state_continue_identical():
    def run(seed):
        torch.manual_seed(seed)
        m = torch.nn.Linear(8, 8)
        o = torch.optim.AdamW(m.parameters(), lr=0.01)
        x = torch.randn(4, 8)
        for _ in range(2):
            o.zero_grad(); m(x).square().mean().backward(); o.step()
        return m, o, x

    m1, o1, x = run(0)
    m2, o2, _ = run(0)
    s = m1.state_dict()
    m2.load_state_dict(s)
    o2.load_state_dict(copy.deepcopy(o1.state_dict()))
    o1.zero_grad();o2.zero_grad()
    m1(x).square().mean().backward()
    m2(x).square().mean().backward()
    o1.step(); o2.step()
    assert torch.allclose(m1.weight, m2.weight, atol=1e-6)


def test_rawr_schedule_lr_and_restore():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = create_optimizer(p, "adamw", 0.01)
    sch = create_scheduler(opt, "easy:RAWR", 20, {"d": 0.45, "warmup_steps": 2, "min_lr": 1e-6, "gamma": 0.9})
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    assert lrs[0] < lrs[1] <= max(lrs) + 1e-9
    st = sch.state_dict()
    sch2 = create_scheduler(opt, "easy:RAWR", 20, {"d": 0.45, "warmup_steps": 2, "min_lr": 1e-6, "gamma": 0.9})
    sch2.load_state_dict(st)
    sch.step(); sch2.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(sch2.optimizer.param_groups[0]["lr"])


def test_loss_bridge_all_valid_losses():
    torch.manual_seed(0)
    for name in VALID_LOSSES:
        b = _bridge(name)
        pred = torch.rand(2, 4, 8, 8) + 1.0
        target = torch.rand(2, 4, 8, 8) + 1.0
        if name == "squared_logarithmic":
            pred = pred.clamp(min=0); target = target.clamp(min=0)
        loss = b.element_loss(pred, target)
        assert torch.isfinite(loss).all()
        assert loss.shape == (2, 4, 8, 8)
        assert loss.sum().item() >= 0 or name == "psnr_loss"


def test_loss_bridge_grad_flows():
    for name in VALID_LOSSES:
        b = _bridge(name)
        pred = (torch.rand(1, 1, 8, 8) + 1.0).requires_grad_(True)
        target = torch.rand(1, 1, 8, 8) + 1.0
        out = b.element_loss(pred, target).mean()
        out.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_pyramid_small_and_large():
    for size in (8, 32):
        b = _bridge("mse_pyramid_2d")
        pred = torch.rand(2, 4, size, size)
        target = torch.rand(2, 4, size, size)
        loss = b.element_loss(pred, target)
        assert torch.isfinite(loss).all()


def test_invalid_loss_raises():
    with pytest.raises(ValueError):
        _bridge("not_a_loss")


def test_standard_deviation_loss_keeps_grad():
    b = _bridge("standard_deviation_loss")
    pred = torch.rand(2, 4, 8, 8).requires_grad_(True)
    target = torch.rand(2, 4, 8, 8)
    b.element_loss(pred, target).mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("video",[False,True])
def test_flow_x0_auxiliary_perfect_and_perturbed(video):
    shape=(2,4,3,16,16) if video else (2,4,16,16)
    clean=torch.randn(shape);noise=torch.randn_like(clean);t=torch.tensor([100.,500.])
    sigma=(t/1000).reshape(2,*([1]*(len(shape)-1)))
    noisy=(1-sigma)*clean+sigma*noise
    bridge=LossBridge({k:{"scale":.01} for k in ("anchor","hf","frequency","topology")},_sched(),True)
    v,_=bridge.augment(torch.zeros(2),noise-clean,clean,noisy,t)
    assert v.shape==(2,) and torch.allclose(v,torch.zeros_like(v),atol=1e-6)
    pred=(noise-clean+.1*torch.randn(shape)).requires_grad_()
    v,_=bridge.augment(torch.zeros(2),pred,clean,noisy,t)
    v.mean().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


def test_edm_discrete_one_and_resume(tmp_path):
    cfg={"edm2":{"enabled":True,"channels":16,"optimizer":"adamw"}}
    ex=rt.TrainingExtras(cfg,_sched(with_alphas=False),True,"epsilon",torch.device("cpu"),10)
    assert ex.edm.model._normalize_timesteps(torch.tensor([1])).item()==1
    loss=torch.tensor([1.,2.],requires_grad=True)
    value,_=ex.augment(loss,torch.empty(0),torch.empty(0),torch.empty(0),torch.tensor([1.,500.]))
    value.mean().backward();ex.step()
    path=tmp_path/"extras.pt";ex.checkpoint(path)
    fresh=rt.TrainingExtras(cfg,_sched(with_alphas=False),True,"epsilon",torch.device("cpu"),10);fresh.restore(path)
    for a,b in zip(ex.parameters(),fresh.parameters()):assert torch.equal(a,b)
    a,_=ex.edm.weight(loss,torch.tensor([1.,500.]));b,_=fresh.edm.weight(loss,torch.tensor([1.,500.]))
    assert torch.equal(a,b)


def test_caption_shift_mask_and_anchor():
    from training_extras.caption_loss import CaptionLoss
    logits=torch.randn(2,3,5,requires_grad=True);targets=torch.tensor([[1,2,-100],[3,4,0]])
    expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
    got,metrics=CaptionLoss({})(logits,targets)
    assert torch.allclose(got,expected) and metrics["raw_ce"]==pytest.approx(float(expected.detach()))
    got2,_=CaptionLoss({"kl_anchor":.1})(logits,targets,logits.detach())
    assert torch.allclose(got2,got,atol=1e-6)
    got.backward();assert logits.grad[0,2].abs().sum()==0
    with pytest.raises(ValueError,match="spatial/diffusion"):CaptionLoss({"edm2":True})


def test_fixed_rng_restores_state():
    torch.manual_seed(0)
    py_before = __import__("random").getstate()
    np_before = np.random.get_state()[1].copy()
    t_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with fixed_rng(1234, torch.device("cpu")):
            _ = torch.randn(4)
            raise RuntimeError("boom")
    assert torch.equal(torch.get_rng_state(), t_before)
    assert np.array_equal(np.random.get_state()[1], np_before)
    assert __import__("random").getstate() == py_before


@pytest.mark.parametrize("channels",[1,4])
def test_wavelet_channel_order_and_video(channels):
    from training_extras.element_weighting import ElementWeighting
    cfg={"wavelet":{"enabled":True},"token_mining":{"enabled":True}}
    e=ElementWeighting(cfg,_sched(with_alphas=False),True)
    x=torch.randn(2,channels,3,16,16);loss=x.square().requires_grad_();t=torch.tensor([1.,900.])
    out,logs=e.apply(loss,x,t)
    assert out.shape==loss.shape and 0 < float(logs["wavelet_mask_fraction"]) < 1
    out.mean().backward();assert torch.isfinite(loss.grad).all()
    baseline=ElementWeighting({"wavelet":{"enabled":True}},_sched(with_alphas=False),True)
    a,_=baseline.apply(torch.ones_like(x),x,t)
    b=torch.stack([baseline.apply(torch.ones_like(x[:,:,i]),x[:,:,i],t)[0] for i in range(3)],dim=2)
    assert torch.equal(a,b)
    c,_=baseline.apply(torch.ones_like(x),x.flip(1),t)
    assert torch.equal(a,c)


def test_lora_plus_preserves_blocks_and_factors():
    from training_extras.optim import lora_plus_groups
    m=torch.nn.Module();m.lora_unet_test=torch.nn.Module();m.lora_unet_test.lora_up=torch.nn.Linear(2,2,bias=False);m.lora_unet_test.lora_down=torch.nn.Linear(2,2,bias=False)
    groups=lora_plus_groups([{"params":list(m.parameters()),"lr":.01}],m,.02,4,6)
    assert {g['lr'] for g in groups}=={.01,.04}
    assert sum(len(g['params']) for g in groups)==2
    up=m.lora_unet_test.lora_up.weight
    assert [g['lr'] for g in groups if any(p is up for p in g['params'])]==[.04]


@pytest.mark.parametrize("mode",["dwa","gradnorm"])
def test_dynamic_auxiliary_weighting(mode):
    ex=rt.TrainingExtras({"anchor":{"scale":.01},"dynamic":{"mode":mode}},_sched(with_alphas=False),True,"epsilon","cpu",3)
    p=torch.nn.Parameter(torch.randn(2,4,8,8));clean=torch.randn_like(p);noise=torch.randn_like(p);t=torch.tensor([200.,700.]);sig=(t/1000).reshape(2,1,1,1);noisy=(1-sig)*clean+sig*noise
    for _ in range(3):
        p.grad=None;base=(p-noise+clean).square().flatten(1).mean(1);out,logs=ex.augment(base,p,clean,noisy,t,[p]);out.mean().backward()
        assert torch.isfinite(p.grad).all() and logs['loss/dynamic_weight']>=0


def test_scheduled_huber_and_profile_isolation():
    from training_extras.profiles import generate
    bridge=LossBridge({"loss":"standard_huber","delta":.01,"delta_schedule":"exponential"},_sched(with_alphas=False),True)
    pred=torch.ones(2,4,8,8);v=bridge.element_loss(pred,torch.zeros_like(pred),torch.tensor([0.,1000.]))
    assert v[0].mean()>v[1].mean()
    base={"config":{"name":"case","process":[{"model":{},"datasets":[{"folder_path":"train"}],"network":{"type":"lora","linear":16,"linear_alpha":32},"train":{"steps":20,"lr":.00005,"seed":42},"training_folder":"out"}]}}
    before=copy.deepcopy(base);profiles=dict(generate(base))
    assert base==before and len(profiles)==16
    assert profiles['edm2']['config']['process'][0]['train']['lr']==.00005
    assert profiles['lowalpha']['config']['process'][0]['network']['linear_alpha']==4
    assert profiles['rex45']['config']['process'][0]['train']['optimizer']=='adamw8bit'
    assert profiles['usual_mapped']['config']['process'][0]['train']['training_extras']['edm2']['optimizer']=='easy:SCORN'


@pytest.mark.parametrize("zeros",[False,True])
def test_signed_and_zero_regression_domains(zeros):
    for name in VALID_LOSSES:
        p=(torch.zeros(2,4,8,8) if zeros else torch.randn(2,4,8,8)).requires_grad_()
        target=torch.zeros_like(p) if zeros else torch.randn_like(p)
        if name=='squared_logarithmic' and not zeros:
            with pytest.raises(ValueError,match='greater than -1'):_bridge(name).element_loss(p,target)
            continue
        out=_bridge(name).element_loss(p,target);out.mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(p.grad).all(),name


def test_host_accumulation_keeps_all_microbatch_gradients():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from contextlib import nullcontext
    p=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.SGD([p],lr=.1)
    def train_one(target,accum_scale):
        loss=(p-target).square();(loss*accum_scale).backward();return loss.detach()
    f=SimpleNamespace(sd=SimpleNamespace(is_multistage=False),train_config=SimpleNamespace(gradient_accumulation_steps=2,optimizer='sgd',max_grad_norm=100.),params=[p],optimizer=opt,steps_this_boundary=0,train_single_accumulation=train_one,model_config=SimpleNamespace(low_vram=False),accelerator=SimpleNamespace(clip_grad_norm_=torch.nn.utils.clip_grad_norm_),timer=lambda _:nullcontext(),adapter=None,ema=None,embedding=None,lr_scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1),end_of_training_loop=lambda:None,is_grad_accumulation_step=True)
    SDTrainer.hook_train_loop(f,[torch.tensor(1.)]);assert p.item()==0
    f.is_grad_accumulation_step=False;SDTrainer.hook_train_loop(f,[torch.tensor(3.)])
    assert p.item()==pytest.approx(.4)


def test_cached_audio_dto_deepcopy():
    from toolkit.dto import DTO
    source=DTO(torch.randn(1,4,3,8,8),audio=torch.randn(1,8,16),metadata={'frames':56})
    a,b=copy.deepcopy([source,source])
    assert a is b and a is not source
    assert torch.equal(a.tensor,source.tensor) and torch.equal(a.audio,source.audio)
    assert a.tensor.data_ptr()!=source.tensor.data_ptr() and a.audio.data_ptr()!=source.audio.data_ptr()
    a.metadata['frames']=0
    assert source.metadata['frames']==56


def test_transformer_alpha_export_and_resume(tmp_path):
    from toolkit.lora_special import LoRASpecialNetwork
    from toolkit.config_modules import NetworkConfig
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.proj=torch.nn.Linear(4,4,bias=False)
    def network():
        return LoRASpecialNetwork(None,Toy(),lora_dim=2,alpha=.5,train_text_encoder=False,
            target_lin_modules=['Toy'],is_transformer=True,peft_format=True,
            network_config=NetworkConfig(type='lora',linear=2,linear_alpha=.5,respect_alpha=True))
    n=network();n.apply_to(None,None,False,True)
    with torch.no_grad():
        for key,p in n.named_parameters():
            if key.endswith('lora_up.weight'):p.fill_(.1)
    raw={k:v.clone() for k,v in n.state_dict().items()}
    exported=n.get_state_dict(dtype=torch.float32)
    factors=[v for k,v in exported.items() if k.endswith('lora_B.weight')]
    assert factors and all(torch.allclose(v,torch.full_like(v,.025)) for v in factors)
    path=str(tmp_path/'alpha.safetensors');n.save_weights(path,dtype=torch.float32)
    n2=network();n2.apply_to(None,None,False,True);n2.load_weights(path)
    assert all(torch.equal(v,n2.state_dict()[k]) for k,v in raw.items())
