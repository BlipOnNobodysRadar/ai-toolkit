"""Run-generation fencing of the Python Job writers (docs/STATE_MACHINES.md M10/SM-15, Codex UI decision item 6).

Real temporary sqlite3 databases with the Job columns these writers use plus run_id; the real UIJobStatus,
DiffusionTrainer and BaseCaptioner DB methods (the trainers built without their model setup). A fenced process
(AITK_RUN_ID set) only touches the row of its own run, stops when superseded, and refuses to start without owning its
row; without AITK_RUN_ID behaviour is unchanged.
"""
import asyncio
import concurrent.futures
import sqlite3
from types import SimpleNamespace

import pytest

from toolkit.ui_run_fence import RunFence, RunSuperseded

COLUMNS = "id TEXT PRIMARY KEY, status TEXT, info TEXT, step TEXT, stop INTEGER, return_to_queue INTEGER, " \
          "save_now INTEGER, sample_now INTEGER, run_id TEXT"


def db(tmp_path, run_id='run-b', stop=0):
    path = tmp_path / 'aitk_db.db'
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE Job ({COLUMNS})")
    conn.execute("INSERT INTO Job VALUES ('job', 'running', '', '0', ?, 0, 0, 0, ?)", (stop, run_id))
    conn.commit()
    conn.close()
    return str(path)


def row(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT status, info, step, stop, run_id FROM Job WHERE id = 'job'").fetchone()
    finally:
        conn.close()


def set_stop(path):
    conn = sqlite3.connect(path)
    conn.execute("UPDATE Job SET stop = 1 WHERE id = 'job'")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- UIJobStatus (inference engine)

def status(monkeypatch, path, run_id):
    from toolkit.ui_job_status import UIJobStatus
    monkeypatch.setenv('AITK_JOB_ID', 'job')
    if run_id is None:
        monkeypatch.delenv('AITK_RUN_ID', raising=False)
    else:
        monkeypatch.setenv('AITK_RUN_ID', run_id)
    return UIJobStatus(path)


def test_the_owning_run_reads_and_writes_its_row(tmp_path, monkeypatch):
    path = db(tmp_path)
    ui = status(monkeypatch, path, 'run-b')
    ui.update_status('running', 'Idle')
    assert row(path)[:2] == ('running', 'Idle') and not ui.should_stop()
    set_stop(path)
    assert ui.should_stop()


def test_a_superseded_run_cannot_change_the_newer_runs_row_and_stops(tmp_path, monkeypatch):
    path = db(tmp_path, run_id='run-a')
    ui = status(monkeypatch, path, 'run-a')  # started while it owned the row
    conn = sqlite3.connect(path)
    conn.execute("UPDATE Job SET run_id = 'run-b', status = 'running', info = 'new run' WHERE id = 'job'")
    conn.commit()
    conn.close()
    ui.update_status('error', 'late write from the old run')
    assert row(path)[:2] == ('running', 'new run') and row(path)[4] == 'run-b'
    assert ui.should_stop()  # the write matched nothing: superseded


def test_a_deleted_row_stops_a_fenced_run(tmp_path, monkeypatch):
    path = db(tmp_path)
    ui = status(monkeypatch, path, 'run-b')
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM Job")
    conn.commit()
    conn.close()
    assert ui.should_stop() and not ui.should_return_to_queue()


@pytest.mark.parametrize('problem', ['other run', 'no row', 'no database', 'no run_id column'])
def test_a_fenced_process_refuses_to_start_without_owning_its_row(tmp_path, monkeypatch, problem):
    path = db(tmp_path, run_id='run-a' if problem == 'other run' else 'run-b')
    if problem == 'no row':
        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM Job")
        conn.commit()
        conn.close()
    elif problem == 'no database':
        path = str(tmp_path / 'missing.db')
    elif problem == 'no run_id column':  # a database the UI migration has not reached
        conn = sqlite3.connect(path)
        conn.execute("ALTER TABLE Job RENAME TO Old")
        conn.execute("CREATE TABLE Job (id TEXT PRIMARY KEY, status TEXT, info TEXT, stop INTEGER)")
        conn.execute("INSERT INTO Job VALUES ('job', 'running', '', 0)")
        conn.commit()
        conn.close()
    with pytest.raises(RunSuperseded):
        status(monkeypatch, path, 'run-b')


def test_without_a_run_id_nothing_changes(tmp_path, monkeypatch):
    path = db(tmp_path, run_id=None)
    ui = status(monkeypatch, path, None)
    ui.update_status('running', 'legacy')
    assert row(path)[:2] == ('running', 'legacy')
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM Job")
    conn.commit()
    conn.close()
    assert not ui.should_stop()  # a missing row still reads "keep going" for an unfenced process


# ---------------------------------------------------------------- trainer and captioner writers

def trainer(cls_path, path, run_id):
    import importlib
    module, name = cls_path.rsplit('.', 1)
    cls = getattr(importlib.import_module(module), name)
    t = cls.__new__(cls)
    t.sqlite_db_path, t.job_id = path, 'job'
    t.fence = RunFence('job', run_id)
    t.is_ui_trainer = t.is_ui_captioner = True
    t.accelerator = SimpleNamespace(is_main_process=True)
    t.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t._async_tasks = []
    t.step_num = 7
    return t


WRITERS = ['extensions_built_in.sd_trainer.DiffusionTrainer.DiffusionTrainer',
           'extensions_built_in.sd_trainer.UITrainer.UITrainer',
           'extensions_built_in.captioner.BaseCaptioner.BaseCaptioner']


@pytest.mark.parametrize('cls_path', WRITERS)
def test_trainer_writes_are_fenced_and_a_stale_run_stops(tmp_path, cls_path):
    path = db(tmp_path, run_id='run-b')
    owner = trainer(cls_path, path, 'run-b')
    owner.update_status('running', 'Training')
    owner.update_step()
    assert row(path)[:3] == ('running', 'Training', '7') and not owner.should_stop()
    stale = trainer(cls_path, path, 'run-a')
    stale.update_status('error', 'late write')
    stale.update_db_key('step', 99)
    assert row(path)[:3] == ('running', 'Training', '7')
    assert stale.should_stop() and not owner.should_stop()


@pytest.mark.parametrize('cls_path', WRITERS)
def test_trainer_without_a_run_id_is_unchanged(tmp_path, cls_path):
    path = db(tmp_path, run_id=None)
    t = trainer(cls_path, path, None)
    t.update_status('running', 'legacy')
    assert row(path)[:2] == ('running', 'legacy') and not t.should_stop()
    set_stop(path)
    assert t.should_stop()
