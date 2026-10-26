"""Run the unmodified author probe with a small, explicit config variation.

Only yaml configuration is changed. Model loading, training and checkpoint code
use the current isolated checkout. All outputs are owned CPU test fixtures.
"""
import os
import runpy
import sys
from pathlib import Path

for key, value in {
    'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
    'AITK_HF_DOWNLOADER': 'default', 'PYTHONDONTWRITEBYTECODE': '1',
}.items():
    os.environ[key] = value

variant = sys.argv.pop(1)
assert variant in ('fp16-export', 'learnable-snr')
import yaml
original_load = yaml.safe_load

def varied_load(stream):
    config = original_load(stream)
    if isinstance(config, dict) and config.get('job') == 'extension':
        process = config['config']['process'][0]
        if variant == 'fp16-export':
            process['save'].pop('dtype', None)  # exercise SaveConfig's default
        elif variant == 'learnable-snr':
            process['train']['learnable_snr_gos'] = True
    return config

yaml.safe_load = varied_load
probe = Path(__file__).resolve().parents[1] / 'rawr_cpu_probe.py'
sys.argv[0] = str(probe)
runpy.run_path(str(probe), run_name='__main__')
