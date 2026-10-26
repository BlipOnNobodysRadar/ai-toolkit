"""Opt-in actual tiny diffusion trainer regression, with no downloads or GPU.

Set AITK_RAWR_CPU_FIXTURES to a directory containing template.yaml, tiny-sd/
and tiny-data/. The model must already be local. All outputs use pytest's
temporary directory. Each child runs serially, on one CPU thread.

Each case runs in two modes. 'recorded' turns train.reseed_per_step off, so the recorded stochastic state carries
the RNG and data position. 'reseeded' leaves it on (the default with a training_seed), so per-step reseeding does
and the recorded path stands aside; SNR state and pending gradients still travel with the committed pair.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_rawr_trainer_continuity import equal
from training_extras.rawr_stochastic import RESEEDED

FIXTURES = os.environ.get('AITK_RAWR_CPU_FIXTURES')
pytestmark = pytest.mark.skipif(not FIXTURES, reason='local tiny diffusion fixtures not supplied')
PROBE = Path(__file__).resolve().parents[1] / 'testing/rawr_cpu_probe.py'
MODES = {'recorded': 'off', 'reseeded': 'on'}


def run(root, name, mode, *args, expected=0):
    root.mkdir(parents=True, exist_ok=True)
    suffix = len(list(root.glob(f'{name}-*.log')))
    with (root / f'{name}-{suffix}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-B', '-u', str(PROBE), name,
                                 '--fixtures', str(Path(FIXTURES).resolve()),
                                 '--evidence-root', str(root), '--gas', '2', '--reseed', MODES[mode], *args],
                                stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == expected, (root / f'{name}-{suffix}.log').read_text(encoding='utf-8')[-6000:]
    return root / name


def trace(path, prefix):
    return {(row['kind'], row['step']): row for row in
            map(json.loads, (path / f'{prefix}.jsonl').read_text().splitlines())}


@pytest.fixture(scope='module', params=list(MODES))
def control(request, tmp_path_factory):
    mode = request.param
    root = tmp_path_factory.mktemp(f'actual-rawr-cpu-{mode}')
    run(root, 'control', mode)
    result = json.loads((root / 'control/trace-result.json').read_text())
    assert result['status'] == 'completed' and result['cuda_initialized'] is False
    return root, root / 'control', mode


def check_mode(training_state, mode):
    stochastic = training_state['stochastic']
    if mode == 'recorded':
        assert stochastic['supported']
    else:
        assert not stochastic['supported'] and stochastic['reason'] == RESEEDED
        assert 'snr' in training_state and 'dataset_identity' in training_state


def compare(control, resumed, start):
    a, b = trace(control, 'trace'), trace(resumed, 'resume')
    rows = []
    for step in range(start, 8):
        expected = torch.load(control / f'trace-step-{step}.pt', map_location='cpu', weights_only=True)
        actual = torch.load(resumed / f'resume-step-{step}.pt', map_location='cpu', weights_only=True)
        assert equal(expected, actual), f'parameters/optimizer/scheduler diverged at loop {step}'
        expected_before = torch.load(control / f'trace-before-{step}.pt', map_location='cpu', weights_only=True)
        actual_before = torch.load(resumed / f'resume-before-{step}.pt', map_location='cpu', weights_only=True)
        assert equal(expected_before, actual_before), f'pre-update state diverged at loop {step}'
        before_a, before_b = a['before', step], b['before', step]
        assert before_a['rng'] == before_b['rng']
        assert before_a['counter'] == before_b['counter']
        assert before_a['grad_norm'] == before_b['grad_norm']
        rows.append({'step': step, 'parameters_optimizer_scheduler_exact': True,
                     'rng_exact': True, 'pending_gradients_exact': True,
                     'all_cpu_rng_and_gradient_tensors_exact_before_and_after': True})
    result = json.loads((resumed / 'resume-result.json').read_text())
    assert result['status'] == 'completed' and result['cuda_initialized'] is False
    (resumed / 'comparison.json').write_text(json.dumps(rows, indent=2))


@pytest.mark.parametrize('stop', [2, 3])
def test_actual_stochastic_resume_at_partial_and_complete_accumulation(control, stop):
    root, baseline, mode = control
    # A periodic save at loop 3 is enabled for the boundary case; the underlying
    # run remains eight loops with the same scheduler and optimizer budget.
    interval = '3' if stop == 3 else '2'
    name = f'resume-{stop}'
    saved = run(root, name, mode, '--save-every', interval, '--stop', str(stop))
    manifest = json.loads((saved / 'output' / name / 'rawr-resume.json').read_text())
    assert manifest['step'] == stop + 1
    checkpoint = torch.load(saved / 'output' / name / '.rawr-checkpoints' / manifest['slot'] / 'optimizer.pt',
                            map_location='cpu', weights_only=True)
    check_mode(checkpoint['_aitk_rawr_checkpoint']['training_state'], mode)
    run(root, name, mode, '--save-every', interval)
    compare(baseline, saved, stop + 1)


def test_actual_process_exit_before_pair_commit_recovers_previous_generation(control):
    root, baseline, mode = control
    name = 'interrupted-pair'
    saved = run(root, name, mode, '--crash-before-commit', '4', expected=87)
    output = saved / 'output' / name
    manifest = json.loads((output / 'rawr-resume.json').read_text())
    assert manifest['step'] == 3, 'interrupted loop 4 must not replace the loop-2 save'
    assert 'INJECTED_PROCESS_EXIT_BEFORE_MANIFEST' in (root / f'{name}-0.log').read_text(encoding='utf-8')
    run(root, name, mode)
    assert min(step for kind, step in trace(saved, 'resume') if kind == 'before') == 3
    compare(baseline, saved, 3)


@pytest.mark.parametrize('mode', list(MODES))
def test_actual_snr_interrupted_pair_ignores_truncated_legacy_json(tmp_path, mode):
    flags = ('--learnable-snr', '--default-export-dtype')
    baseline = run(tmp_path, 'snr-control', mode, *flags)
    name = 'snr-interrupted'
    saved = run(tmp_path, name, mode, *flags, '--crash-before-commit', '4', expected=87)
    output = saved / 'output' / name
    manifest = json.loads((output / 'rawr-resume.json').read_text())
    assert manifest['step'] == 3
    (output / 'learnable_snr.json').write_text('{"interrupted":')
    run(tmp_path, name, mode, *flags)
    assert min(step for kind, step in trace(saved, 'resume') if kind == 'before') == 3
    compare(baseline, saved, 3)
