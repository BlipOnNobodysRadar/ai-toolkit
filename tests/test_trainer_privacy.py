"""Exercise publication and logging without contacting an external service."""
import importlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from jobs.process.BaseProcess import BaseProcess
from jobs.process.BaseTrainProcess import BaseTrainProcess
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config import get_config
from toolkit.config_modules import LoggingConfig, SaveConfig
from toolkit.logging_aitk import EmptyLogger, create_logger


def test_job_config_stays_out_of_logs_and_optional_local_backup_preserves_env_refs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('SECRET_FOR_TEST', 'private-runtime-value')
    source = tmp_path / 'job.yaml'
    source.write_text('''job: extension
config:
  name: demo
  process:
    - type: sd_trainer
      api_key: private-inline-key
      arbitrary: ${SECRET_FOR_TEST}
''')
    raw = get_config(str(source))
    process = BaseProcess(0, SimpleNamespace(name='demo', meta={}, raw_config=raw), raw['config']['process'][0])
    assert process.config['arbitrary'] == 'private-runtime-value'
    assert capsys.readouterr().out == ''
    process.save_root = tmp_path / 'output'
    BaseTrainProcess.save_training_config(process)
    backup = process.save_root / 'config.yaml'
    assert not backup.exists()
    process.config['logging'] = {'save_config': True}
    BaseTrainProcess.save_training_config(process)
    config = yaml.safe_load(backup.read_text())['config']['process'][0]
    assert config['arbitrary'] == '${SECRET_FOR_TEST}'
    assert 'api_key' not in config


def test_telemetry_disabled_before_hub_and_optional_update_imports():
    env = os.environ.copy()
    for key in ('NO_ALBUMENTATIONS_UPDATE', 'HF_HUB_DISABLE_TELEMETRY', 'DISABLE_TELEMETRY'):
        env.pop(key, None)
    subprocess.run([sys.executable, '-c', 'import os, toolkit; from huggingface_hub import constants; '
                    'assert constants.HF_HUB_DISABLE_TELEMETRY and os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"'],
                   env=env, check=True)


@pytest.mark.parametrize('samples', [False, True])
def test_wandb_metrics_keep_config_console_and_samples_private_by_default(monkeypatch, samples):
    wandb = Mock()
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    assert type(create_logger(LoggingConfig(), {})) is EmptyLogger
    logging = LoggingConfig(use_wandb=True, wandb_log_samples=samples)
    logger = create_logger(logging, {'api_key': 'private', 'prompts': ['private prompt']})
    logger.start()
    assert 'config' not in wandb.init.call_args.kwargs
    assert not wandb.init.call_args.kwargs['save_code']
    assert wandb.Settings.call_args.kwargs == dict(console='off', host='ai-toolkit', disable_git=True,
                                                   disable_code=True, _disable_meta=True, _disable_stats=True,
                                                   _disable_machine_info=True, _save_requirements=False)
    logger.log({'loss': 1})
    wandb.log.assert_called_with({'loss': 1}, commit=False)
    logger.log_image(object(), 0, 'private prompt')
    assert wandb.Image.called is samples


def hub_trainer(root):
    trainer = object.__new__(BaseSDTrainProcess)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.train_config = SimpleNamespace(steps=80)
    trainer.save_root = str(root)
    trainer.save_config = SaveConfig()
    trainer.model_config = SimpleNamespace(name_or_path='/private/model/file', is_xl=False, is_flux=False,
                                           is_lumina2=False, is_v3=False)
    trainer.job = SimpleNamespace(name='run[1]')
    trainer.sd = SimpleNamespace(arch='sdxl')
    trainer.network_config = None
    trainer.is_fine_tuning = True
    trainer.sample_config = SimpleNamespace(prompts=['sensitive prompt'])
    return trainer


def publish_final(trainer, monkeypatch, final_files):
    trainer.ema = None
    trainer.save_config.push_to_hub = True

    def save_files(self, step, pair):
        for path in final_files:
            if path.name == 'config.json':
                path.write_text('{"_name_or_path": "/private/base/checkpoint", "hidden_size": 32}')
            else:
                path.write_bytes(b'freshly saved')

    monkeypatch.setattr(BaseSDTrainProcess, '_save_checkpoint', save_files)
    trainer.save()


@pytest.mark.parametrize('samples', [False, True])
def test_hub_publishes_only_model_and_explicit_final_samples(tmp_path, monkeypatch, samples):
    trainer = hub_trainer(tmp_path)
    trainer.save_config.hf_upload_samples = samples
    files = ['.job_config.generation.json', 'log.txt', 'loss_log.db', 'optimizer.pt', 'run[1]_000000080.safetensors',
             'run[1].safetensors', 'unrelated.safetensors', '.rawr-checkpoints/slot/weights.safetensors', 'samples/123__000000080_0.jpg',
             'samples/456__000000001_0.jpg', 'run[1]/private/config.json',
             'run[1]/transformer/diffusion_pytorch_model-00001-of-00002.safetensors',
             'run[1]/transformer/config.json', 'run[1]/transformer/diffusion_pytorch_model.safetensors.index.json']
    for filename in files:
        file = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('private-data')
    publish_final(trainer, monkeypatch, [tmp_path / name for name in files[-3:]])
    hub = Mock()
    hub.repo_info.return_value.private = True
    monkeypatch.setattr(importlib.import_module(BaseSDTrainProcess.__module__), 'HfApi', lambda: hub)
    trainer.push_to_hub('user/run')
    operations = {op.path_in_repo: op for op in hub.create_commit.call_args.kwargs['operations']}
    expected = {'README.md', *files[-3:]}
    if samples:
        expected.add('samples/123__000000080_0.jpg')
    assert set(operations) == expected
    readme = operations['README.md'].path_or_fileobj.decode()
    config = operations['run[1]/transformer/config.json'].path_or_fileobj
    assert b'/private/base/checkpoint' not in config and b'hidden_size' in config
    assert ('sensitive prompt' in readme) is samples
    assert '/private/model/file' not in readme
    assert not (tmp_path / 'README.md').exists()
    hub.create_repo.assert_called_once_with('user/run', private=True, exist_ok=True)


def test_existing_public_hub_requires_explicit_public_publish(tmp_path, monkeypatch):
    trainer = hub_trainer(tmp_path)
    (tmp_path / 'run[1].safetensors').write_bytes(b'weights')
    hub = Mock()
    hub.repo_info.return_value.private = False
    monkeypatch.setattr(importlib.import_module(BaseSDTrainProcess.__module__), 'HfApi', lambda: hub)
    assert not trainer.save_config.push_to_hub
    assert trainer.save_config.hf_private
    publish_final(trainer, monkeypatch, [tmp_path / 'run[1].safetensors'])
    with pytest.raises(ValueError, match='destination repo is public'):
        trainer.push_to_hub('user/run')
    hub.create_commit.assert_not_called()
    trainer.push_to_hub('user/run', private=False)
    hub.create_commit.assert_called_once()


def test_native_writer_can_publish_a_directory_named_safetensors(tmp_path, monkeypatch):
    trainer = hub_trainer(tmp_path)
    final = tmp_path / 'run[1].safetensors' / 'transformer' / 'model.safetensors'
    final.parent.mkdir(parents=True)
    publish_final(trainer, monkeypatch, [final])
    hub = Mock()
    hub.repo_info.return_value.private = True
    monkeypatch.setattr(importlib.import_module(BaseSDTrainProcess.__module__), 'HfApi', lambda: hub)
    trainer.push_to_hub('user/run')
    assert [op.path_in_repo for op in hub.create_commit.call_args.kwargs['operations']] == [
        'run[1].safetensors/transformer/model.safetensors', 'README.md']


@pytest.mark.parametrize('pretrained', ['/private/local/text-encoder', 'Qwen/Qwen3-VL'])
def test_iris_hub_config_preserves_inference_settings_and_public_encoder_reference(tmp_path, pretrained):
    from toolkit.hub_publish import publish_payload
    file = tmp_path / 'config.yaml'
    settings = {'model': {'depth': 4}, 'text_encoder': {'pretrained': pretrained, 'dim': 8},
                'flow': {'shift': 1, 'notes': 'private'}, 'training': {'dataset': '/private'}}
    file.write_text(yaml.safe_dump(settings))
    published = yaml.safe_load(publish_payload(file, 'iris3b'))
    encoder = {'dim': 8} if pretrained.startswith('/') else {'dim': 8, 'pretrained': pretrained}
    assert published == {'model': {'depth': 4}, 'text_encoder': encoder, 'flow': {'shift': 1}}
    assert yaml.safe_load(file.read_text()) == settings


def test_quoted_false_settings_do_not_enable_uploads():
    save = SaveConfig(push_to_hub='false', hf_upload_samples='off', hf_private='false')
    logging = LoggingConfig(use_wandb='false', wandb_log_samples='no', use_ui_logger='false')
    assert not save.push_to_hub and not save.hf_upload_samples and not save.hf_private
    assert not logging.use_wandb and not logging.wandb_log_samples and not logging.use_ui_logger
    assert SaveConfig(hf_private='unknown').hf_private
