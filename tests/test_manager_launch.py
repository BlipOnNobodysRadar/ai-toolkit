"""manager.launch_ui startup/cancellation ownership (docs/STATE_MACHINES.md M33; Codex manager_startup_probe).

The actual launch_ui and _open_browser_when_ready run with the process spawn, readiness request and browser call
replaced; no UI service, browser or dependency install is touched. Deterministic barriers, no timing races: the only
waits are bounded joins of the watcher thread this test owns.
"""
import threading
from unittest.mock import patch

import pytest

from manager import launch as m


def _launch_patches(popen):
    return [
        patch.object(m, "venv_python", return_value="fixture-python"),
        patch.object(m.os.path, "isfile", return_value=True),
        patch.object(m, "build_env", return_value={}),
        patch.object(m.nodejs, "find_npm", return_value="fixture-npm"),
        patch.object(m.nodejs, "have_usable_node", return_value=("fixture-node", 999)),
        patch.object(m.nodejs, "ensure_ui_deps"),
        patch.object(m, "_headless", return_value=False),
        patch.object(m.subprocess, "Popen", side_effect=popen),
    ]


def test_spawn_failure_still_stops_the_browser_watcher():
    started = []

    class RecordingThread:
        def __init__(self, *, target, args, daemon):
            self.args = args

        def start(self):
            started.append(self.args[0])

    patches = _launch_patches(OSError("injected spawn failure")) + [patch.object(m.threading, "Thread", RecordingThread)]
    for p in patches:
        p.start()
    try:
        with pytest.raises(OSError, match="injected spawn failure"):
            m.launch_ui()
    finally:
        for p in patches:
            p.stop()
    assert len(started) == 1
    assert started[0].is_set(), "the watcher's stop event is set even though Popen raised"


def test_readiness_request_blocked_when_the_ui_stops_opens_no_browser():
    stop, lock, opened = threading.Event(), threading.Lock(), []
    entered, release = threading.Event(), threading.Event()

    class Response:
        def close(self):
            pass

    def blocking_urlopen(url, timeout=None):
        entered.set()
        release.wait()
        return Response()

    with patch("urllib.request.urlopen", blocking_urlopen), patch("webbrowser.open", lambda url: opened.append(url)):
        t = threading.Thread(target=m._open_browser_when_ready, args=(stop, lock), daemon=True)
        t.start()
        assert entered.wait(10)
        with lock:  # what launch_ui's cleanup does
            stop.set()
        release.set()  # the UI answers only now
        t.join(10)
    assert not t.is_alive()
    assert opened == []


def test_ready_ui_opens_the_browser_once():
    stop, lock, opened = threading.Event(), threading.Lock(), []

    class Response:
        def close(self):
            pass

    with patch("urllib.request.urlopen", lambda url, timeout=None: Response()), patch("webbrowser.open", lambda url: opened.append(url)):
        t = threading.Thread(target=m._open_browser_when_ready, args=(stop, lock), daemon=True)
        t.start()
        t.join(10)
    assert opened == [m.UI_URL]


def test_cancellation_wakes_the_retry_wait():
    stop, lock, opened = threading.Event(), threading.Lock(), []
    failed = threading.Event()

    def refusing(url, timeout=None):
        failed.set()
        raise OSError("connection refused")

    with patch("urllib.request.urlopen", refusing), patch("webbrowser.open", lambda url: opened.append(url)):
        t = threading.Thread(target=m._open_browser_when_ready, args=(stop, lock), daemon=True)
        t.start()
        assert failed.wait(10)
        stop.set()
        t.join(10)  # without the event wait this would still be inside time.sleep(2)
    assert not t.is_alive()
    assert opened == []
