"""UI job bookkeeping for long-running extension processes.

When a process is launched by the UI it gets AITK_JOB_ID and a sqlite db
path; status / step / info are written to the Job row and the UI's stop
button sets Job.stop. This mirrors what BaseCaptioner does inline, as a
standalone helper so other UI-driven processes (the inference engine) can
share it. Outside the UI (no job id or no db file) every call is a no-op.
"""

import os
import sqlite3
import threading
import time
from typing import Callable, Literal, Optional

from toolkit.ui_run_fence import RunFence

AITK_Status = Literal["running", "stopped", "error", "completed", "queued"]


class UIJobStatus:
    def __init__(self, sqlite_db_path: str = "./aitk_db.db"):
        self.sqlite_db_path = sqlite_db_path
        job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = job_id.strip() if job_id else None
        # a UI run id fences every Job read/write to this launch; a fenced run must own its row to start
        self.fence = RunFence(self.job_id)
        self.fence.require_owned(sqlite_db_path)
        self.is_ui_job = self.job_id is not None and os.path.exists(sqlite_db_path)
        if self.is_ui_job:
            print(f"Using SQLite database at {sqlite_db_path}")
            print(f'Job ID: "{self.job_id}"')
        self._lock = threading.Lock()
        self._watcher = None
        self._watcher_done = threading.Event()
        # held while on_stop runs (never during DB reads): stop_stop_watcher waits for a callback in flight and none
        # can start after it returns
        self._watcher_callback = threading.Lock()
        # collapse bursts of identical writes (progress ticks)
        self._last_info = None

    # ---- db plumbing ----
    def _connect(self):
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0)
        conn.isolation_level = None
        return conn

    def _retry(self, fn, max_retries=3, base_delay=2.0):
        last = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e):
                    raise
                last = e
                if attempt < max_retries:
                    time.sleep(base_delay * (2**attempt))
        raise last

    def _write(self, query: str, params: tuple):
        if not self.is_ui_job:
            return

        def _do():
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                try:
                    cur.execute(query, params)
                    self.fence.wrote(cur)
                finally:
                    cur.execute("COMMIT")

        try:
            self._retry(_do)
        except Exception as e:
            print(f"[AITK] Warning: db write failed: {e}")

    def _read_flag(self, column: str) -> bool:
        if not self.is_ui_job:
            return False

        def _do():
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT {column} FROM Job WHERE {self.fence.job_clause}", self.fence.params())
                row = cur.fetchone()
                return self.fence.flag(row, stop=column == "stop")

        return self._retry(_do)

    # ---- public api ----
    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        if info is None:
            self._write(f"UPDATE Job SET status = ? WHERE {self.fence.job_clause}", self.fence.params(status))
        else:
            self._last_info = info
            self._write(
                f"UPDATE Job SET status = ?, info = ? WHERE {self.fence.job_clause}",
                self.fence.params(status, info),
            )

    def update_info(self, info: str):
        if info == self._last_info:
            return
        self._last_info = info
        self._write(f"UPDATE Job SET info = ? WHERE {self.fence.job_clause}", self.fence.params(info))

    def update_key(self, key: str, value):
        # key is a column name from our own code, never user input
        self._write(f"UPDATE Job SET {key} = ? WHERE {self.fence.job_clause}", self.fence.params(str(value)))

    def should_stop(self) -> bool:
        if self.fence.superseded:  # a fenced write matched no row: this run no longer owns the job
            return True
        return self._read_flag("stop")

    def should_return_to_queue(self) -> bool:
        return self._read_flag("return_to_queue")

    def start_stop_watcher(self, on_stop: Callable[[str], None], interval_sec: float = 2.0):
        """Poll the stop / return_to_queue flags on a daemon thread and call
        on_stop("stopped" | "queued") once when one is set."""
        if not self.is_ui_job or self._watcher is not None:
            return

        def _dispatch(reason):
            with self._watcher_callback:
                if not self._watcher_done.is_set():  # the read may have finished after stop_stop_watcher
                    on_stop(reason)

        def _loop():
            while not self._watcher_done.is_set():
                try:
                    if self.should_stop():
                        _dispatch("stopped")
                        return
                    if self.should_return_to_queue():
                        _dispatch("queued")
                        return
                except Exception:
                    pass
                self._watcher_done.wait(interval_sec)

        self._watcher = threading.Thread(target=_loop, daemon=True)
        self._watcher.start()

    def stop_stop_watcher(self):
        """End the watcher: once this returns, on_stop has either finished or will never be called."""
        with self._watcher_callback:
            self._watcher_done.set()
