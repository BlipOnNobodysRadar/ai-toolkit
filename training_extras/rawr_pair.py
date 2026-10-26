"""Two-slot committed LoRA/optimizer pairs; export filenames are only aliases.

The active slot is never written. A manifest replacement commits the inactive
slot only after both files are complete and flushed. This bounds internal
storage to two generations without pruning user checkpoints.
"""
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import warnings

from training_extras import rawr_checkpoint as rc

MANIFEST = 'rawr-resume.json'
DIRECTORY = '.rawr-checkpoints'


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root):
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        value = json.load(f)
    if (value.get('version') != 1 or value.get('slot') not in ('a', 'b')
            or not isinstance(value.get('generation'), str) or len(value['generation']) != 32):
        raise ValueError('Invalid RAWR committed-pair manifest')
    return value


def _paths(root, slot):
    base = os.path.join(root, DIRECTORY, slot)
    return os.path.join(base, 'model.safetensors'), os.path.join(base, 'optimizer.pt')


def begin(trainer):
    if getattr(trainer, '_rawr_checkpoint_config', None) is None:
        return None
    previous = _manifest(trainer.save_root)
    # Learnable SNR state now travels in the pair either way (recorded-state path or training_state['snr']);
    # only custom SNR classes, whose state isn't captured, keep the pair off.
    snr = getattr(trainer, 'snr_gos', None)
    unsupported_snr = False
    if snr is not None:
        from toolkit.train_tools import LearnableSNRGamma
        unsupported_snr = type(snr) is not LearnableSNRGamma
    if (trainer.network is None or trainer.is_fine_tuning
            or trainer.train_config.merge_network_on_save
            or unsupported_snr
            or any(getattr(trainer, attr, None) is not None for attr in
                   ('embedding', 'adapter', 'decorator', 'ema', 'training_extras'))):
        if previous is not None:
            raise ValueError('Cannot replace a committed RAWR pair with an unsupported save mode; use a new output folder')
        return None
    slot = 'b' if previous is not None and previous['slot'] == 'a' else 'a'
    model, optimizer = _paths(trainer.save_root, slot)
    os.makedirs(os.path.dirname(model), exist_ok=True)
    return {'slot': slot, 'generation': uuid.uuid4().hex, 'model': model, 'optimizer': optimizer}


def _write_manifest(value, path):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.rawr.', suffix='.tmp',
                                         dir=os.path.dirname(path), delete=False) as f:
            temp = f.name
            json.dump(value, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            os.unlink(temp)


def _publish(source, target, writer=None):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.rawr-export.', suffix=os.path.splitext(target)[1],
                                         dir=os.path.dirname(target), delete=False) as f:
            temp = f.name
        # Close the temporary handle before serializers reopen it on Windows.
        if writer is not None:
            writer(temp)
        else:
            with open(source, 'rb') as src, open(temp, 'wb') as f:
                shutil.copyfileobj(src, f, 1024 * 1024)
        with open(temp, 'r+b') as f:
            os.fsync(f.fileno())
        os.replace(temp, target)
        temp = None
    finally:
        if temp is not None:
            os.unlink(temp)


def commit(root, pair, state, model_alias, name, step, export_model=None):
    # Model serializers close their file before returning. Explicitly flush it
    # before publishing a pointer that promises both files are complete.
    # Windows fsync requires a writable descriptor, even though no bytes change.
    with open(pair['model'], 'r+b') as f:
        os.fsync(f.fileno())
    model_hash = _hash(pair['model'])
    state[rc.KEY]['generation'] = pair['generation']
    state[rc.KEY]['model_sha256'] = model_hash
    rc.atomic_save(state, pair['optimizer'])
    manifest = {'version': 1, 'slot': pair['slot'], 'generation': pair['generation'],
                'name': name, 'step': int(step), 'model_sha256': model_hash,
                'optimizer_sha256': _hash(pair['optimizer'])}
    _write_manifest(manifest, os.path.join(root, MANIFEST))
    # The transaction is committed. Alias errors must not turn a committed save
    # into a reported failed transaction; native resume uses the manifest.
    for source, target, writer in ((pair['model'], model_alias, export_model),
                                  (pair['optimizer'], os.path.join(root, 'optimizer.pt'), None)):
        try:
            _publish(source, target, writer)
        except Exception as exc:
            warnings.warn(f'RAWR pair committed, but export alias {target} could not be updated: {exc}', RuntimeWarning)


def select(root, name):
    manifest = _manifest(root)
    if manifest is None or manifest['name'] != name:
        return None
    model, optimizer = _paths(root, manifest['slot'])
    if _hash(model) != manifest['model_sha256'] or _hash(optimizer) != manifest['optimizer_sha256']:
        raise ValueError('RAWR committed model/optimizer pair is missing or corrupt; refusing mixed resume')
    return {**manifest, 'model': model, 'optimizer': optimizer}
