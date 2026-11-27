"""One writer per save root (docs/STATE_MACHINES.md, OPUS-ONEWRITER-01, OPUS-DIST-01). Real CPU child processes
hold the lease (stdlib only: they import toolkit/save_root_lease.py and nothing else); the trainer-side tests run the
real BaseTrainProcess/BaseJob constructors and run.py's loop with tiny stand-in processes. The cross-language half
(the UI's deletion against a Python holder) is ui/tests/writerLease.test.cjs."""
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from toolkit import save_root_lease

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDER = textwrap.dedent("""
    import importlib.util, sys, time
    spec = importlib.util.spec_from_file_location('save_root_lease', sys.argv[1])
    lease = importlib.util.module_from_spec(spec); spec.loader.exec_module(lease)
    owner = object()
    lease.acquire(sys.argv[2], owner, owner)
    print('held', flush=True)
    if sys.argv[3] == 'die':
        import os; os._exit(0)  # no release: the OS must drop the lock
    time.sleep(120)
""")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    # BaseTrainProcess asks the shared Accelerator how many processes run; don't leave a real one cached in
    # toolkit.accelerator (or in run.py's module global) for later tests in a mixed session
    import toolkit.accelerator
    monkeypatch.setattr(toolkit.accelerator, 'get_accelerator', lambda: SimpleNamespace(
        num_processes=1, is_main_process=True, is_local_main_process=True, free_memory=lambda: None))


def child_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    return env


def hold(root, mode='hold'):
    """A second OS process holding root's lease; returns once it holds it."""
    child = subprocess.Popen([sys.executable, '-B', '-I', '-c', HOLDER,
                              os.path.join(REPO, 'toolkit', 'save_root_lease.py'), str(root), mode],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_env())
    line = child.stdout.readline()
    if line.strip() != 'held':
        child.kill()
        raise AssertionError(f'holder did not start: {line!r} {child.stderr.read()}')
    return child


def stop(child):
    child.kill()
    child.wait(timeout=30)


def test_another_process_is_refused_until_it_dies(tmp_path):
    root = tmp_path / 'out' / 'job'
    child = hold(root)
    try:
        me = object()
        with pytest.raises(save_root_lease.SaveRootBusy, match='Another training run'):
            save_root_lease.acquire(str(root), me, me)
    finally:
        stop(child)
    lease = save_root_lease.acquire(str(root), me, me)  # the OS dropped the dead owner's lock
    save_root_lease.release(lease, me)


def test_an_owner_that_exits_without_releasing_frees_the_root(tmp_path):
    root = tmp_path / 'job'
    child = hold(root, 'die')
    child.wait(timeout=30)
    me = object()
    save_root_lease.release(save_root_lease.acquire(str(root), me, me), me)


def aliases(root):
    yield str(root.parent / '.' / root.name)          # another spelling
    yield str(root) + os.sep                          # trailing separator
    if os.name == 'nt':
        yield str(root).upper()                       # case
        yield str(root) + '. '                        # trailing dots and spaces, which Windows drops
        link = root.parent / 'junction'
        import _winapi
        _winapi.CreateJunction(str(root), str(link))
        yield str(link)
    else:
        link = root.parent / 'symlink'
        os.symlink(root, link)
        yield str(link)


def test_aliases_of_a_held_root_are_refused(tmp_path):
    root = tmp_path / 'job'
    root.mkdir()  # a junction or symlink needs its target
    child = hold(root)
    try:
        me = object()
        for alias in aliases(root):
            with pytest.raises(save_root_lease.SaveRootBusy):
                save_root_lease.acquire(alias, me, me)
    finally:
        stop(child)


def test_unrelated_roots_are_not_serialized(tmp_path):
    child = hold(tmp_path / 'a')
    try:
        me = object()
        lease = save_root_lease.acquire(str(tmp_path / 'b'), me, me)
        save_root_lease.release(lease, me)
    finally:
        stop(child)


def test_in_one_process_only_the_owning_job_shares_a_root(tmp_path):
    job, other_job = object(), object()
    first, second = object(), object()
    lease = save_root_lease.acquire(str(tmp_path), job, first)
    # a job's processes run in turn and may write the same folder
    assert save_root_lease.acquire(str(tmp_path), job, second) is lease
    # another job in this process (another thread's trainer) is not the owner
    with pytest.raises(save_root_lease.SaveRootBusy):
        save_root_lease.acquire(str(tmp_path / '.'), other_job, object())
    save_root_lease.release(lease, first)
    with pytest.raises(save_root_lease.SaveRootBusy):  # the second process still holds it
        save_root_lease.acquire(str(tmp_path), other_job, object())
    save_root_lease.release(lease, second)
    save_root_lease.release(lease, second)  # idempotent
    other = save_root_lease.acquire(str(tmp_path), other_job, first)
    save_root_lease.release_owner(other_job)
    assert str(tmp_path) not in save_root_lease.held_roots()
    assert other.path not in save_root_lease._held


def test_a_released_root_can_be_taken_by_another_process(tmp_path):
    me = object()
    save_root_lease.release(save_root_lease.acquire(str(tmp_path), me, me), me)
    stop(hold(tmp_path))


def test_a_held_root_is_refused_to_another_process(tmp_path):
    me = object()
    lease = save_root_lease.acquire(str(tmp_path), me, me)
    try:
        child = subprocess.run([sys.executable, '-B', '-I', '-c', HOLDER,
                                os.path.join(REPO, 'toolkit', 'save_root_lease.py'), str(tmp_path), 'die'],
                               capture_output=True, text=True, env=child_env(), timeout=60)
        assert child.returncode != 0 and 'SaveRootBusy' in child.stderr
    finally:
        save_root_lease.release(lease, me)


def test_the_lock_lives_outside_the_root_and_survives_its_deletion(tmp_path):
    # the UI deletes job folders recursively: a lock inside the root would go with it (POSIX) or half-block it
    root = tmp_path / 'job'
    child = hold(root)
    try:
        lock = save_root_lease.lease_path(str(root))
        assert os.path.dirname(os.path.dirname(lock)) == save_root_lease.canonical_root(str(tmp_path))
        assert not root.exists()  # acquiring wrote nothing into the root
        root.mkdir()
        (root / 'run_000000100.safetensors').write_bytes(b'x')
        shutil.rmtree(root)
        root.mkdir()
        me = object()
        with pytest.raises(save_root_lease.SaveRootBusy):
            save_root_lease.acquire(str(root), me, me)
    finally:
        stop(child)
    # After a crash/kill the OS drops the lock; a subsequent clean acquire + release unlinks the lock and empty dir
    me = object()
    lease = save_root_lease.acquire(str(root), me, me)
    save_root_lease.release(lease, me)
    assert not os.path.exists(lock)
    assert not os.path.exists(os.path.dirname(lock))


def test_the_lock_path_matches_the_ui_formula(tmp_path):
    # ui/cron/ownership/writerLease.ts: <dirname(canonical)>/.aitk_locks/writer.<sha256(canonical)>.lock
    root = str(tmp_path / 'Job')
    canonical = save_root_lease.canonical_root(root)
    key = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    assert save_root_lease.lease_path(root) == os.path.join(os.path.dirname(canonical), '.aitk_locks',
                                                            f'writer.{key}.lock')
    if os.name == 'nt':
        assert canonical == os.path.realpath(str(tmp_path)).lower() + '\\job'
        assert save_root_lease.canonical_root(root + '. ') == canonical
        assert save_root_lease.canonical_root(root.replace('\\', '/')) == canonical


# ---- the trainer lifecycle ----

def train_job(tmp_path, processes):
    from jobs.BaseJob import BaseJob
    job = BaseJob.__new__(BaseJob)
    job.config = {'process': processes}
    job.raw_config = {'job': 'extension', 'config': job.config}
    job.name = 'job'
    job.meta = {}
    job.training_folder = str(tmp_path)
    return job


def process_class(fail_after_base=False):
    from jobs.process.BaseTrainProcess import BaseTrainProcess

    class Tiny(BaseTrainProcess):
        def __init__(self, process_id, job, config):
            super().__init__(process_id, job, config)
            if fail_after_base:
                raise RuntimeError('subclass setup failed')
    return Tiny


def test_the_lease_is_taken_before_the_config_is_written(tmp_path, monkeypatch):
    from jobs.process.BaseTrainProcess import BaseTrainProcess
    seen = []
    monkeypatch.setattr(BaseTrainProcess, 'save_training_config',
                        lambda self: seen.append(save_root_lease.held_roots()))
    job = train_job(tmp_path, [{'type': 'tiny', 'name': 'a'}])
    job.load_processes({'tiny': process_class()})
    assert seen == [[str(tmp_path / 'a')]]
    job.cleanup()
    assert save_root_lease.held_roots() == []


def test_a_held_root_refuses_the_constructor_before_any_write(tmp_path):
    child = hold(tmp_path / 'a')
    try:
        job = train_job(tmp_path, [{'type': 'tiny', 'name': 'a'}])
        with pytest.raises(save_root_lease.SaveRootBusy):
            job.load_processes({'tiny': process_class()})
        assert not (tmp_path / 'a').exists()  # no folder, no config.yaml
    finally:
        stop(child)


def test_a_failed_constructor_releases_every_root_its_job_took(tmp_path):
    from jobs.process.BaseTrainProcess import BaseTrainProcess
    good = process_class()

    class Bad(BaseTrainProcess):
        def __init__(self, process_id, job, config):
            super().__init__(process_id, job, config)
            raise RuntimeError('subclass setup failed')
    job = train_job(tmp_path, [{'type': 'good', 'name': 'a'}, {'type': 'bad', 'name': 'b'}])
    with pytest.raises(RuntimeError, match='subclass setup failed'):
        job.load_processes({'good': good, 'bad': Bad})
    assert save_root_lease.held_roots() == []
    stop(hold(tmp_path / 'a'))
    stop(hold(tmp_path / 'b'))


def test_processes_of_one_job_may_share_a_root(tmp_path):
    job = train_job(tmp_path, [{'type': 'tiny', 'name': 'a'}, {'type': 'tiny', 'name': 'a'}])
    job.load_processes({'tiny': process_class()})
    job.process[0].release_save_root()  # the first finished its run
    assert save_root_lease.held_roots() == [str(tmp_path / 'a')]
    job.process[1].release_save_root()
    assert save_root_lease.held_roots() == []


def test_sd_trainer_run_releases_on_failure(tmp_path):
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    holder = SimpleNamespace()
    holder._save_root_lease = save_root_lease.acquire(str(tmp_path), holder, holder)

    def fail():
        raise KeyboardInterrupt
    holder._run_training = fail
    holder._remove_update_commit_observers = lambda: None
    with pytest.raises(KeyboardInterrupt):
        BaseSDTrainProcess.run(holder)
    assert save_root_lease.held_roots() == []


def test_distributed_launches_warn_by_default_and_refuse_when_strict(tmp_path, monkeypatch):
    import toolkit.accelerator
    monkeypatch.setattr(toolkit.accelerator, 'get_accelerator',
                        lambda: SimpleNamespace(num_processes=2, is_main_process=False))
    job = train_job(tmp_path, [{'type': 'tiny', 'name': 'a'}])
    with pytest.warns(UserWarning, match='2 processes'):
        job.load_processes({'tiny': process_class()})
    assert save_root_lease.held_roots() == []  # non-main rank does not deadlock on the lease
    job.cleanup()

    monkeypatch.setenv('AITK_STRICT_INVARIANTS', '1')
    job2 = train_job(tmp_path, [{'type': 'tiny', 'name': 'b'}])
    with pytest.raises(ValueError, match='2 processes'):
        job2.load_processes({'tiny': process_class()})
    assert not (tmp_path / 'b').exists() and save_root_lease.held_roots() == []


def test_run_py_releases_and_reports_safely_across_configs(tmp_path, monkeypatch, capsys):
    import run
    monkeypatch.setattr(run, 'accelerator', SimpleNamespace(is_main_process=True, free_memory=lambda: None))
    built = []

    class Job:
        def __init__(self, fail_run):
            self.fail_run = fail_run
            self.process = [SimpleNamespace(on_error=lambda e: built.append(('on_error', type(e).__name__)))]
            self.lease = save_root_lease.acquire(str(tmp_path / f'job{len(built)}'), self, self)
            built.append(('built', len(built)))

        def run(self):
            if self.fail_run:
                raise RuntimeError('run failed')

        def cleanup(self):
            self.release_save_roots()

        def release_save_roots(self):
            save_root_lease.release_owner(self)

    plan = iter([lambda: Job(True), lambda: (_ for _ in ()).throw(RuntimeError('config unreadable')), lambda: Job(False)])
    monkeypatch.setattr(run, 'get_job', lambda config, name: next(plan)())
    monkeypatch.setattr(sys, 'argv', ['run.py', '-r', 'a.yaml', 'b.yaml', 'c.yaml'])
    run.main()
    # the failed first job reported and released; the unreadable second config did not report to the first job
    assert built.count(('on_error', 'RuntimeError')) == 1
    assert save_root_lease.held_roots() == []
