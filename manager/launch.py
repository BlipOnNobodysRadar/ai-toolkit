"""Launch the AI Toolkit web UI (ui/ -> npm run db_build_start)."""

import os
import subprocess
import threading

from . import ffmpeg, nodejs
from .util import (
    IS_LINUX,
    IS_WINDOWS,
    REPO_ROOT,
    clean_env,
    die,
    info,
    venv_dir,
    venv_python,
)

UI_PORT = 8675
UI_URL = "http://localhost:%d" % UI_PORT
BROWSER_POLL_SECONDS = 300


def build_env():
    """Scrubbed env with local node, ffmpeg, and the venv on PATH.

    Everything the UI worker spawns (training jobs) inherits this, so the
    local ffmpeg/node are visible to the whole process tree.
    """
    env = clean_env()
    path_dirs = []
    if os.path.isdir(nodejs.node_bin_dir()):
        path_dirs.append(nodejs.node_bin_dir())
    ff_paths, ff_libs = ffmpeg.env_additions()
    path_dirs += ff_paths
    vbin = os.path.join(venv_dir(), "Scripts" if IS_WINDOWS else "bin")
    if os.path.isdir(vbin):
        path_dirs.append(vbin)
    if path_dirs:
        env["PATH"] = os.pathsep.join(path_dirs) + os.pathsep + env.get("PATH", "")
    if ff_libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            ff_libs + [env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    return env


def _headless():
    return IS_LINUX and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def _open_browser_when_ready(stop_event, open_lock):
    """Poll the UI port in the background; open the browser once it responds.

    launch_ui sets stop_event while holding open_lock, and the browser is opened only under that lock after
    re-checking the event: an open begun before the UI stopped may finish, but none starts after launch_ui's
    cleanup has returned (a readiness request that was blocked when the UI stopped opens nothing)."""
    import time
    import urllib.request
    import webbrowser

    deadline = time.monotonic() + BROWSER_POLL_SECONDS
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            urllib.request.urlopen(UI_URL, timeout=2).close()
        except OSError:
            stop_event.wait(2)  # wakes at once when the UI stops
            continue
        with open_lock:
            if not stop_event.is_set():
                webbrowser.open(UI_URL)
        return


def launch_ui(open_browser=True):
    if not os.path.isfile(venv_python()):
        die("No Python environment found. Run: python3 -m manager install")

    env = build_env()
    npm = nodejs.find_npm(env)
    if not npm:
        die(
            "Node.js was not found. Run `python3 -m manager sync` to install a "
            "local copy, or install Node.js >= %d from https://nodejs.org."
            % nodejs.MIN_NODE_MAJOR
        )
    _, major = nodejs.have_usable_node()
    if major is not None and major < nodejs.MIN_NODE_MAJOR:
        die(
            "Node.js v%d found, but >= %d is required. Run `python3 -m manager sync`."
            % (major, nodejs.MIN_NODE_MAJOR)
        )

    # deps are installed here rather than by the npm script so the lockfile is
    # never rewritten (see nodejs.ensure_ui_deps); a no-op once they're in sync
    nodejs.ensure_ui_deps(env=env)

    info("Starting AI Toolkit UI (%s) ..." % UI_URL)
    stop_event = threading.Event()
    open_lock = threading.Lock()
    # the watcher is owned from its start: any failure below (including Popen raising) still stops it
    try:
        if open_browser and not _headless():
            threading.Thread(
                target=_open_browser_when_ready, args=(stop_event, open_lock), daemon=True
            ).start()

        proc = subprocess.Popen(
            [npm, "run", "db_build_start"], cwd=os.path.join(REPO_ROOT, "ui"), env=env
        )
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # terminates npm itself; whether its descendants (the UI servers) exit with it is not verified here
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 130
    finally:
        with open_lock:
            stop_event.set()
