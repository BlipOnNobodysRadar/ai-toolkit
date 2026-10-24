"""Opt-in actual tiny diffusion trainer regression, with no downloads or GPU.

Set AITK_RAWR_CPU_FIXTURES to a directory containing template.yaml, tiny-sd/
and tiny-data/. The model must already be local. All outputs use pytest's
temporary directory. Each child runs serially, on one CPU thread.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_rawr_trainer_continuity import equal

FIXTURES = os.environ.get('AITK_RAWR_CPU_FIXTURES')
pytestmark = pytest.mark.skipif(not FIXTURES, reason='local tiny diffusion fixtures not supplied')
PROBE = Path(__file__).resolve().parents[1] / 'testing/rawr_cpu_probe.py'


def run(root, name, *args, expected=0):
    root.mkdir(parents=True, exist_ok=True)
    suffix = len(list(root.glob(f'{name}-*.log')))
    with (root / f'{name}-{suffix}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-B', '-u', str(PROBE), name,
                                 '--fixtures', str(Path(FIXTURES).resolve()),
                                 '--evidence-root', str(root), '--gas', '2', *args],
                                stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == expected, (root / f'{name}-{suffix}.log').read_text(encoding='utf-8')[-6000:]
    return root / name


def trace(path, prefix):
    return {(row['kind'], row['step']): row for row in
            map(json.loads, (path / f'{prefix}.jsonl').read_text().splitlines())}


@pytest.fixture(scope='module')
def control(tmp_path_factory):
    root = tmp_path_factory.mktemp('actual-rawr-cpu')
    run(root, 'control')
    result = json.loads((root / 'control/trace-result.json').read_text())
    assert result['status'] == 'completed' and result['cuda_initialized'] is False
    return root, root / 'control'


def compare(control, resumed, start):
    a, b = trace(control, 'trace'), trace(resumed, 'resume')
    rows = []
    for step in range(start, 8):
        expected = torch.load(control / f'trace-step-{step}.pt', map_location='cpu', weights_only=True)
        actual = torch.load(resumed / f'resume-step-{step}.pt', map_location='cpu', weights_only=True)
        assert equal(expected, actual), f'parameters/optimizer/scheduler diverged at loop {step}'
        before_a, before_b = a['before', step], b['before', step]
        assert before_a['rng'] == before_b['rng']
        assert before_a['counter'] == before_b['counter']
        assert before_a['grad_norm'] == before_b['grad_norm']
        rows.append({'step': step, 'parameters_optimizer_scheduler_exact': True,
                     'rng_exact': True, 'pending_gradients_exact': True})
    result = json.loads((resumed / 'resume-result.json').read_text())
    assert result['status'] == 'completed' and result['cuda_initialized'] is False
    (resumed / 'comparison.json').write_text(json.dumps(rows, indent=2))


@pytest.mark.parametrize('stop', [2, 3])
def test_actual_stochastic_resume_at_partial_and_complete_accumulation(control, stop):
    root, baseline = control
    # A periodic save at loop 3 is enabled for the boundary case; the underlying
    # run remains eight loops with the same scheduler and optimizer budget.
    interval = '3' if stop == 3 else '2'
    name = f'resume-{stop}'
    saved = run(root, name, '--save-every', interval, '--stop', str(stop))
    manifest = json.loads((saved / 'output' / name / 'rawr-resume.json').read_text())
    assert manifest['step'] == stop + 1
    checkpoint = torch.load(saved / 'output' / name / '.rawr-checkpoints' / manifest['slot'] / 'optimizer.pt',
                            map_location='cpu', weights_only=True)
    assert checkpoint['_aitk_rawr_checkpoint']['training_state']['stochastic']['supported']
    run(root, name, '--save-every', interval)
    compare(baseline, saved, stop + 1)


def test_actual_process_exit_before_pair_commit_recovers_previous_generation(control):
    root, baseline = control
    name = 'interrupted-pair'
    saved = run(root, name, '--crash-before-commit', '4', expected=87)
    output = saved / 'output' / name
    manifest = json.loads((output / 'rawr-resume.json').read_text())
    assert manifest['step'] == 3, 'interrupted loop 4 must not replace the loop-2 save'
    assert 'INJECTED_PROCESS_EXIT_BEFORE_MANIFEST' in (root / f'{name}-0.log').read_text(encoding='utf-8')
    run(root, name)
    assert min(step for kind, step in trace(saved, 'resume') if kind == 'before') == 3
    compare(baseline, saved, 3)
