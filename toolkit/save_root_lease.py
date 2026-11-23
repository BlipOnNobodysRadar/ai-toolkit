"""One writer per training save root (docs/STATE_MACHINES.md, OPUS-ONEWRITER-01).

Resume selection, pruning, the non-atomic checkpoint writes and the RAWR pair slots all assume that a single trainer
writes a save root. An open SQLite transaction in the parent directory holds the lock independently of the
checkpoint files. SQLite releases it when the trainer exits.

Training processes within one job may share the root. A second training job targeting the same canonical
directory cannot acquire the lease.
"""
import hashlib
import os
import re
import sqlite3
import sys
import threading
import unicodedata

LOCK_DIR = '.aitk_locks'

_registry_lock = threading.Lock()
_held = {}  # lock path -> _Lease


class SaveRootBusy(RuntimeError):
    pass


class _Lease:
    def __init__(self, root, path, connection, owner):
        self.root, self.path, self.connection, self.owner = root, path, connection, owner
        self.holders = set()


def _physical(path):
    """Like resources.ts physicalPath: the nearest existing ancestor resolved through junctions, symlinks and short
    names, plus the components that do not exist yet. Any failure other than a missing component raises."""
    tail = []
    current = path
    while True:
        try:
            real = os.path.realpath(current, strict=True)
            return os.path.join(real, *reversed(tail)) if tail else real
        except FileNotFoundError:
            parent = os.path.dirname(current)
            if parent == current:
                raise
            tail.append(os.path.basename(current))
            current = parent


def canonical_root(root, platform=sys.platform):
    """resources.ts canonicalFolder(dirname(root), basename(root)) for this platform."""
    absolute = _physical(os.path.abspath(root))
    if platform == 'win32':
        parts = re.split(r'[\\/]+', absolute)
        return '\\'.join(p if i == 0 else re.sub(r'[. ]+$', '', p) for i, p in enumerate(parts)).lower()
    if platform == 'darwin':
        return unicodedata.normalize('NFC', absolute).lower()
    return absolute


def lease_path(root):
    canonical = canonical_root(root)
    key = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return os.path.join(os.path.dirname(canonical), LOCK_DIR, f'writer.{key}.lock')


def _busy(root):
    return SaveRootBusy(
        f"Another training run (or a deletion) holds {root}. Two writers in one output folder corrupt each other's "
        f"checkpoints, resume point and pruning; wait for it to finish, stop it, or use another output folder.")


def acquire(root, owner, holder):
    """Lease `root` for `holder` (a training process) on behalf of `owner` (its job), before anything is written to
    it. Creates only the lock directory beside it."""
    path = lease_path(root)
    with _registry_lock:
        lease = _held.get(path)
        if lease is not None and lease.owner is owner:
            lease.holders.add(id(holder))
            return lease
        os.makedirs(os.path.dirname(path), exist_ok=True)
        connection = sqlite3.connect(path, timeout=0, isolation_level=None, check_same_thread=False)
        try:
            connection.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as e:
            connection.close()
            if 'locked' in str(e) or 'busy' in str(e).lower():
                raise _busy(root) from e
            raise
        lease = _held[path] = _Lease(root, path, connection, owner)
        lease.holders.add(id(holder))
        return lease


def release(lease, holder):
    """Drop holder's share; the last holder ends the transaction. Idempotent."""
    if lease is None:
        return
    with _registry_lock:
        lease.holders.discard(id(holder))
        if lease.holders or _held.get(lease.path) is not lease:
            return
        del _held[lease.path]
        try:
            lease.connection.execute('ROLLBACK')
        finally:
            lease.connection.close()


def release_owner(owner):
    """Release every lease `owner` (a job) holds: constructor failure, job cleanup, a failed config under --recover."""
    with _registry_lock:
        leases = [lease for lease in _held.values() if lease.owner is owner]
        for lease in leases:
            lease.holders.clear()
    for lease in leases:
        release(lease, None)


def held_roots():
    with _registry_lock:
        return [lease.root for lease in _held.values()]
