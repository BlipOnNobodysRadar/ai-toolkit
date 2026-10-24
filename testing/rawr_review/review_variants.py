"""Serial independent probes: report observed equivalence without hiding drift."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
for key, value in {
    'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
    'AITK_HF_DOWNLOADER': 'default', 'PYTHONDONTWRITEBYTECODE': '1',
}.items():
    os.environ[key] = value
sys.path[:0] = [str(REPO), str(REPO / 'tests')]
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def no_cuda(*args, **kwargs):
    raise AssertionError('CUDA initialization forbidden in independent review')
torch.cuda._lazy_init = no_cuda
torch.cuda.init = no_cuda
torch.cuda.is_available = lambda: False
torch.accelerator.current_accelerator = lambda *a, **kw: None
from test_rawr_trainer_continuity import equal

parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=('fp16-export', 'learnable-snr'))
parser.add_argument('--evidence-root', type=Path, default=ROOT / 'evidence/variants')
args = parser.parse_args()
evidence, variant = args.evidence_root.resolve(), args.variant
evidence.mkdir(parents=True, exist_ok=True)
if any((evidence / f'{variant}-{suffix}').exists() for suffix in ('control', 'resume')):
    raise SystemExit('Existing evidence preserved; pass a fresh --evidence-root directory.')

def run(name, *extra):
    suffix = len(list(evidence.glob(f'{name}-*.log')))
    log = evidence / f'{name}-{suffix}.log'
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run([
            sys.executable, '-B', '-u', str(ROOT / 'review_variant_probe.py'),
            variant, name, '--fixtures', os.environ['AITK_RAWR_CPU_FIXTURES'],
            '--evidence-root', str(evidence), '--gas', '2', *extra],
            stdout=out, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text(encoding='utf-8')[-6000:]
    return evidence / name

baseline = run(f'{variant}-control')
resumed = run(f'{variant}-resume', '--stop', '2')
out = resumed / 'output' / resumed.name
manifest_path = out / 'rawr-resume.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
optimizer_path = (out / '.rawr-checkpoints' / manifest['slot'] / 'optimizer.pt'
                  if manifest else out / 'optimizer.pt')
checkpoint = torch.load(optimizer_path, weights_only=True, map_location='cpu')
stochastic = checkpoint['_aitk_rawr_checkpoint']['training_state']['stochastic']
run(resumed.name)

def load(path, prefix, step, before=False):
    stem = f'{prefix}-before-{step}' if before else f'{prefix}-step-{step}'
    return torch.load(path / f'{stem}.pt', weights_only=True, map_location='cpu')

def trace(path, prefix):
    return {(row['kind'], row['step']): row for row in
            map(json.loads, (path / f'{prefix}.jsonl').read_text().splitlines())}

a, b = trace(baseline, 'trace'), trace(resumed, 'resume')
rows = []
for step in range(3, 8):
    expected, actual = load(baseline, 'trace', step), load(resumed, 'resume', step)
    before_a, before_b = load(baseline, 'trace', step, True), load(resumed, 'resume', step, True)
    rows.append({
        'step': step,
        'parameter_max_abs_error_before': max((p - q).abs().max().item()
                                              for p, q in zip(before_a['params'], before_b['params'])),
        'parameter_max_abs_error_after': max((p - q).abs().max().item()
                                             for p, q in zip(expected['params'], actual['params'])),
        'optimizer_exact': equal(expected['optimizer'], actual['optimizer']),
        'scheduler_exact': equal(expected['scheduler'], actual['scheduler']),
        'torch_rng_exact': a['before', step]['rng'] == b['before', step]['rng'],
        'pending_gradient_norm_exact': a['before', step]['grad_norm'] == b['before', step]['grad_norm'],
        'snr_state_exact_before_and_after': equal(before_a['snr'], before_b['snr']) and equal(expected['snr'], actual['snr']),
        'all_cpu_rng_exact_before_and_after': equal(before_a['rng'], before_b['rng']) and equal(expected['rng'], actual['rng']),
        'gradient_tensors_exact_before_and_after': equal(before_a['gradients'], before_b['gradients']) and equal(expected['gradients'], actual['gradients']),
    })
result = {
    'variant': variant, 'checkpoint_stochastic_supported': stochastic['supported'],
    'committed_pair': manifest is not None,
    'checkpoint_step': checkpoint['_aitk_rawr_checkpoint']['trainer_step'],
    'rows': rows,
    'cuda_initialized': any(json.loads((path / f'{prefix}-result.json').read_text())['cuda_initialized']
                            for path, prefix in [(baseline, 'trace'), (resumed, 'trace'), (resumed, 'resume')]),
}
(evidence / f'{variant}-comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
