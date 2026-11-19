"""Run-generation fencing for processes the UI launches (docs/STATE_MACHINES.md M10, SM-15).

The UI gives every launch a run id (AITK_RUN_ID) next to the job id. When it is set, every Job read and write this
process makes is conditional on that run id, so a process whose run was superseded (the job was restarted, deleted or
reassigned) cannot change the newer run's row: its writes match no row, and the process stops itself. A fenced
process that cannot confirm it still owns its row at startup refuses to start rather than run unfenced. Without
AITK_RUN_ID (the CLI, or a UI that does not issue run ids) nothing changes.
"""
import os
import sqlite3


class RunSuperseded(RuntimeError):
    """This process's run no longer owns its Job row."""


class RunFence:
    def __init__(self, job_id, run_id=None):
        self.job_id = job_id
        if run_id is None:
            run_id = os.environ.get("AITK_RUN_ID", None)
        self.run_id = run_id.strip() if run_id and run_id.strip() else None
        self.superseded = False

    @property
    def fenced(self) -> bool:
        return self.run_id is not None

    @property
    def job_clause(self) -> str:
        return "id = ? AND run_id = ?" if self.fenced else "id = ?"

    def params(self, *values) -> tuple:
        return values + ((self.job_id, self.run_id) if self.fenced else (self.job_id,))

    def wrote(self, cursor):
        """After an UPDATE: a fenced write that matched no row means the run was superseded."""
        if self.fenced and cursor.rowcount == 0:
            self.superseded = True

    def flag(self, row, stop=False) -> bool:
        """A flag read's result. A missing row under a fence means superseded: report stop (only) as set. Without a
        fence a missing row reads as unset, as before."""
        if row is None:
            if self.fenced:
                self.superseded = True
                return stop
            return False
        return row[0] == 1

    def require_owned(self, sqlite_db_path):
        """Startup check for a fenced process: the DB is readable and the row still carries this run id."""
        if not self.fenced:
            return
        if self.job_id is None:
            raise RunSuperseded("AITK_RUN_ID is set but AITK_JOB_ID is not; refusing to run unfenced")
        if not os.path.exists(sqlite_db_path):
            raise RunSuperseded(f"Run {self.run_id}: job database {sqlite_db_path} not found; refusing to run unfenced")
        try:
            conn = sqlite3.connect(sqlite_db_path, timeout=30.0)
            try:
                row = conn.execute("SELECT run_id FROM Job WHERE id = ?", (self.job_id,)).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise RunSuperseded(f"Run {self.run_id}: cannot read job {self.job_id} ({e}); refusing to run unfenced") from e
        if row is None or row[0] != self.run_id:
            self.superseded = True
            raise RunSuperseded(f"Run {self.run_id} no longer owns job {self.job_id}; not starting")
