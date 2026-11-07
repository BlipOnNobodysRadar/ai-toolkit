"""Stand-in for run.py in ui/tests/launch.test.cjs: a harmless CPU-only program launched through the real relay and
supervisor. Usage: fake_run.py <config.json>. The config's process[0].test_steps is a list of steps:
  "status:<s>"         fenced UPDATE Job SET status (WHERE id AND run_id, like toolkit/ui_run_fence.py)
  "wait:<file>"        block until <file> exists
  "orphan:<file>"      start an intermediate that starts a grandchild waiting for <file>, then exits at once
  "child:<file>"       start a direct child waiting for <file>; its pid is written to <file>.started
  "sleep:<seconds>"
Every process it starts is hidden (CREATE_NO_WINDOW) and waits on a gate file, so the test owns its lifetime.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def wait_for(path):
    while not os.path.exists(path):
        time.sleep(0.05)


def spawn(code):
    return subprocess.Popen([sys.executable, "-c", code], creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    proc = cfg["config"]["process"][0]
    db = proc["sqlite_db_path"]
    job_id, run_id = os.environ["AITK_JOB_ID"], os.environ["AITK_RUN_ID"]
    print(f"fake_run start pid={os.getpid()} run={run_id} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    for step in proc.get("test_steps", []):
        kind, _, arg = step.partition(":")
        if kind == "status":
            conn = sqlite3.connect(db, timeout=30)
            try:
                cur = conn.execute("UPDATE Job SET status = ? WHERE id = ? AND run_id = ?", (arg, job_id, run_id))
                conn.commit()
                print(f"status {arg} rows={cur.rowcount}", flush=True)
            finally:
                conn.close()
        elif kind == "wait":
            wait_for(arg)
        elif kind == "orphan":
            grandchild = f"import os,time\nwhile not os.path.exists({arg!r}): time.sleep(0.05)"
            inter = f"import subprocess,sys\nsubprocess.Popen([sys.executable,'-c',{grandchild!r}],creationflags={NO_WINDOW},stdin=subprocess.DEVNULL)"
            spawn(inter).wait()
        elif kind == "child":
            c = spawn(f"import os,time\nwhile not os.path.exists({arg!r}): time.sleep(0.05)")
            with open(f"{arg}.started", "w") as f:
                f.write(str(c.pid))
        elif kind == "sleep":
            time.sleep(float(arg))
    print("fake_run done", flush=True)


if __name__ == "__main__":
    main()
