"""Generate matched, independent ablations without starting training."""
import argparse, copy, json
from pathlib import Path
import yaml

WARP = dict(compile_step=False,weight_decay=.1,relative_wd=True,aino_mix_alpha=.05,
            meta_lr=0,came_confidence=True,betas=[.95,.95,.99],came_beta=.999)
USUAL = dict(alpha=1,betas=[.99,.997],min_beta1=.95,  # beta1_warmup is set per run to the step count below
             amsgrad_max_decay_rate=.96,amsgrad_min_decay_rate=.96,
             use_adabelief=True,torch_compile=False,update_strategy='cautious')

def generate(document, steps=None):
    base=copy.deepcopy(document);p=base['config']['process'][0];t=p['train'];n=p['network']
    if n['type'] not in ('lora','dora','locon'):raise ValueError('These factor ablations require LoRA, LoCon or DoRA')
    if not p.get('datasets'):raise ValueError('A training dataset is required')
    count=int(steps if steps is not None else t['steps'])
    if count<1:raise ValueError('steps must be positive')
    t.update(steps=count,optimizer='adamw8bit',optimizer_params={'weight_decay':.0001},
             lr_scheduler='constant',lr_scheduler_params={},loss_type='mse')
    t.pop('training_extras',None);n.update(lora_plus_ratio=1,lora_plus_text_encoder_ratio=1)
    if t.get('validation_config',{}).get('datasets'):t['validation_config'].setdefault('validation_items',[])
    rex=dict(d=.45,gamma=.9,min_lr=1e-6,warmup_steps=round(.05*count))
    edm=dict(enabled=True,channels=448,float64=True,optimizer='easy:SCORN',lr=.015,
             optimizer_args={'betas':[.95,.997]},scheduler='constant_with_warmup',
             scheduler_args={'warmup_steps':round(.1*count)})
    recipes={'baseline':{},'warp':{'optimizer':'easy:WarpAINO','optimizer_params':WARP},
        'warp_compiled':{'optimizer':'easy:WarpAINO','optimizer_params':dict(WARP,compile_step=True)},
        'rex45':{'lr_scheduler':'easy:RAWR','lr_scheduler_params':rex},
        'lowalpha':{},'lora_plus':{},'edm2':{'training_extras':{'edm2':edm}},
        'logcosh':{'training_extras':{'loss':'log_cosh'}},
        'huber':{'training_extras':{'loss':'standard_pseudo_huber','delta':.01,'delta_schedule':'exponential'}},
        'wavelet':{'training_extras':{'wavelet':{'enabled':True}}},
        'token_mining':{'training_extras':{'token_mining':{'enabled':True}}},
        'usual_mapped':{'optimizer':'easy:SimplifiedAdEMAMixExM','optimizer_params':dict(USUAL,beta1_warmup=count),
             'lr_scheduler':'easy:RAWR','lr_scheduler_params':dict(rex,d=.9),'training_extras':{'edm2':edm}}}
    for name in ('anchor','hf','frequency','topology'):recipes[name]={'training_extras':{name:{'scale':.01}}}
    for name,patch in recipes.items():
        cfg=copy.deepcopy(base);q=cfg['config']['process'][0];q['train'].update(copy.deepcopy(patch))
        if name=='lowalpha':q['network'].update(linear_alpha=q['network']['linear']/4,respect_alpha=True)
        if name in ('usual_mapped','lora_plus'):q['network'].update(lora_plus_ratio=4,lora_plus_text_encoder_ratio=6)
        cfg['config']['name']=base['config']['name']+'_'+name
        cfg.setdefault('meta',{})['comparison']='Same data, seed, model LR and step budget. Hypothesis, not a quality recommendation.'
        yield name,cfg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int);ap.add_argument('--overwrite',action='store_true');a=ap.parse_args()
    configs=list(generate(yaml.safe_load(Path(a.base).read_text(encoding='utf-8')),a.steps));root=Path(a.out)
    targets=[root/(name+'.yaml') for name,_ in configs]+[root/'index.json']
    if not a.overwrite and any(p.exists() for p in targets):raise FileExistsError('Comparison output already exists; choose a new directory or --overwrite')
    root.mkdir(parents=True,exist_ok=True)
    for name,cfg in configs:(root/(name+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (root/'index.json').write_text(json.dumps([{'profile':n,'file':n+'.yaml','job':c['config']['name']} for n,c in configs],indent=2))
    print(f'Wrote {len(configs)} comparisons to {root}')

if __name__=='__main__':main()
