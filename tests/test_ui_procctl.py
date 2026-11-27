"""toolkit/ui_procctl.py failure classification (docs/STATE_MACHINES.md M10; Codex procctl review points 1-4).

Deterministic: Win32 calls are replaced where a failure has to be injected. Process-group tests use temporary CPU-only fixtures.
"""
import json
import os
import subprocess
import sys

import pytest

from toolkit import ui_procctl as pc

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows process APIs")


@windows
def test_wait_failed_is_unknown_not_exited(monkeypatch):
    me = pc.describe(os.getpid())
    assert me["state"] == "alive"
    monkeypatch.setattr(pc, "WaitForSingleObject", lambda h, ms: 0xFFFFFFFF)  # WAIT_FAILED
    assert pc.describe(os.getpid())["state"] == "unknown"
    assert pc.check(me["identity"]) == "unknown"


def test_reused_pid_means_the_recorded_process_exited(monkeypatch):
    monkeypatch.setattr(pc, "describe", lambda pid: {"state": "alive", "identity": f"{pid}@5000", "cmdline": "x"})
    assert pc.check("100@1000") == "exited"
    assert pc.check("100@5000") == "alive"


def test_unusable_identities_are_unknown(monkeypatch):
    monkeypatch.setattr(pc, "describe", lambda pid: {"state": "gone"})
    assert pc.check("123@None") == "unknown"
    assert pc.check("123@") == "unknown"
    assert pc.check("garbage") == "unknown"
    assert pc.check("123@77") == "exited"


def test_membership_never_comes_from_parent_links():
    # the parent-link traversal (Codex counterexamples 1 and 2) is gone; run membership is the OS boundary only
    assert not hasattr(pc, "members")
    assert not hasattr(pc, "_children_index")


@windows
def test_enumeration_error_makes_a_scan_incomplete(monkeypatch):
    real_next = pc.Process32NextW
    calls = {"n": 0}

    def next_fails_early(snap, entry):
        calls["n"] += 1
        if calls["n"] > 3:
            pc.ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED instead of ERROR_NO_MORE_FILES
            return 0
        return real_next(snap, entry)

    monkeypatch.setattr(pc, "Process32NextW", next_fails_early)
    with pytest.raises(OSError, match="stopped early"):
        pc._snapshot()
    req = {"scans": {"s": {"needle": "no-such-needle", "since_ms": 0}}}
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(req)))
    out = __import__("io").StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert pc.main(["probe"]) == 0
    assert json.loads(out.getvalue())["scans"]["s"]["complete"] is False


@windows
def test_unknown_run_has_an_absent_boundary_and_terminate_touches_nothing():
    st = pc.run_state("no-such-run-for-test", None, [])
    assert st["boundary"] == "absent" and st["supervisor"] == "unknown"
    t = pc.terminate("no-such-run-for-test")
    assert t["verified"] is False and t["boundary"] == "absent"


@windows
def test_an_existing_job_object_name_is_never_adopted(tmp_path):
    run_id = f"test-exists-{os.getpid()}"
    held = pc.CreateJobObjectW(None, pc.job_name(run_id))  # someone else already owns this name
    assert held
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-I", "-S", pc.__file__, "supervise", "--run-id", run_id, "--run-dir",
                            str(tmp_path), "--cwd", str(tmp_path), "--", sys.executable, "-c", "print('must not run')"],
                           capture_output=True, text=True, env=env, creationflags=0x08000000)
        assert r.returncode == pc.EXIT_NOT_SPAWNED
        receipt = json.loads((tmp_path / f".run.{run_id}.receipt.json").read_text())
        assert receipt["state"] == "failed" and "already exists" in receipt["error"]
        assert "must not run" not in r.stdout
    finally:
        pc.CloseHandle(held)


def test_boundary_claim_is_exclusive_and_never_replaced(tmp_path):
    files = pc.run_files(str(tmp_path), "run-x")
    won, rec = pc.claim_boundary(files["boundary"], {"run_id": "run-x", "kind": "supervisor", "supervisor": "1@1"})
    assert won and rec["kind"] == "supervisor"
    out = pc.fence("run-x", str(tmp_path), "reconciler@1")
    assert out["fenced"] is False and out["boundary"]["kind"] == "supervisor"
    other = pc.run_files(str(tmp_path), "run-y")
    assert pc.fence("run-y", str(tmp_path), "reconciler@1")["fenced"] is True
    won2, rec2 = pc.claim_boundary(other["boundary"], {"run_id": "run-y", "kind": "supervisor"})
    assert won2 is False and rec2["kind"] == "fence"  # a supervisor arriving after the fence starts nothing
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_supervisor_refuses_after_a_fence(tmp_path):
    run_id = f"fenced-{os.getpid()}"
    assert pc.fence(run_id, str(tmp_path), "test")["fenced"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([sys.executable, "-I", "-S", pc.__file__, "supervise", "--run-id", run_id, "--run-dir",
                        str(tmp_path), "--cwd", str(tmp_path), "--", sys.executable, "-c", "print('must not run')"],
                       capture_output=True, text=True, env=env, creationflags=0x08000000 if sys.platform == "win32" else 0)
    assert r.returncode == pc.EXIT_NOT_SPAWNED
    receipt = json.loads((tmp_path / f".run.{run_id}.receipt.json").read_text())
    assert receipt["state"] == "failed" and "already claimed (fence)" in receipt["error"]


@windows
def test_denied_processes_make_a_scan_incomplete(monkeypatch):
    real_open = pc._open
    monkeypatch.setattr(pc, "_open", lambda pid, *a, **k: (None, "denied") if pid != os.getpid() else real_open(pid, *a, **k))
    out = pc.scan("no-such-needle", 0)
    assert out["complete"] is False and out["denied"] > 0 and out["unreadable"] >= out["denied"]


def test_linux_kill_selection_uses_only_current_direct_children():
    # supervisor 10; 20 is its child; 30's parent 25 (an exited intermediate whose pid was reused by an unrelated
    # process 25 under init) must not be selected through ancestry; 30 is reached only once re-parented to 10
    table = [(10, 1), (20, 10), (25, 1), (30, 25), (40, 99)]
    assert pc.direct_children(10, table) == [20]
    assert pc.direct_children(10, table + [(31, 10)]) == [20, 31]


def test_boot_identity_is_read_from_the_os():
    b = pc.boot_id()
    if sys.platform == "win32":
        assert b and b.startswith("win-boot:")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_macos_process_group_waits_for_workers_and_accepts_force_stop(tmp_path, monkeypatch):
    import time

    files = pc.run_files(str(tmp_path), "group-test")
    marker = tmp_path / "worker-started"
    program = [sys.executable, "-c", (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'])\n"
        "open(sys.argv[1], 'w').close()\n"
        "time.sleep(.5)"
    ), str(marker)]
    supervisor = subprocess.Popen([
        sys.executable, "-c",
        "import json, sys; from toolkit import ui_procctl as p; "
        "sys.exit(p._supervise_process_group(*json.loads(sys.argv[1])))",
        json.dumps([files, str(tmp_path), program, {"run_id": "group-test", "supervisor": "test-supervisor"}]),
    ])
    try:
        deadline = time.monotonic() + 8
        while not (os.path.exists(files["receipt"]) and marker.exists()):
            assert supervisor.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        time.sleep(.7)
        assert supervisor.poll() is None
        assert not os.path.exists(files["exit"])
        monkeypatch.setattr(pc, "IS_MAC", True)
        assert pc.terminate("group-test", "other-supervisor", run_dir=str(tmp_path)).get("requested") is None
        assert pc.terminate("group-test", "test-supervisor", run_dir=str(tmp_path))["requested"]
        assert supervisor.wait(timeout=5) == 0
        assert json.loads(open(files["exit"]).read())["state"] == "exited"
    finally:
        open(files["stop"], "a").close()
        supervisor.wait(timeout=20)


def test_legacy_mac_force_stop_can_match_launch_when_process_environment_is_private(monkeypatch, tmp_path):
    import psutil
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(pc.__file__).resolve().parents[1] / "run.py"
    argv = ["python", str(script), str(tmp_path / "my job" / ".job_config.json")]

    def hidden():
        raise psutil.AccessDenied()

    proc = SimpleNamespace(environ=hidden, cmdline=lambda: argv)
    monkeypatch.setattr(pc, "IS_MAC", True)
    assert pc.legacy_launch_matches(proc, "old-id", "my job")
    assert not pc.legacy_launch_matches(proc, "old-id", "different")
    argv[1] = str(tmp_path / "another-install" / "run.py")
    assert not pc.legacy_launch_matches(proc, "old-id", "my job")
    argv[1], argv[2] = str(script), str(tmp_path / "my job" / ".job_config.new-id.json")
    assert not pc.legacy_launch_matches(proc, "old-id", "my job")
