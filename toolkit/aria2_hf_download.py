"""Route Hugging Face Hub HTTP transfers through aria2c.

Hugging Face Hub still owns repository metadata, cache layout, authentication,
and final file placement. This module only replaces the byte-transfer layer so
large model downloads get aria2's segmented transfers and visible progress.

On by default (Ryan's call, 13 Nov 2026): run.py installs the patch at startup
unless AITK_HF_DOWNLOADER says otherwise. Before that date the patch lived in a
repo-root sitecustomize.py, which Python never imports for ``python run.py``,
so jobs were in fact using Hugging Face's native downloader (Xet).

What on-by-default means in practice: Xet transfers are off for the process;
on Windows a missing aria2c is downloaded once (see AITK_ARIA2_AUTO_INSTALL);
on Linux without aria2c and without root, the first download stops with an
error that says to install aria2 or set AITK_HF_DOWNLOADER=hf.

Resuming across runs depends on the Hub version. huggingface_hub 1.23 (the
pinned version) writes each download to a uniquely named temporary file and
deletes it on failure, so an interrupted download restarts from zero there;
aria2 still retries and resumes within a run.

Environment variables:
    AITK_HF_DOWNLOADER=hf
        Turn this patch off and use Hugging Face's native downloader (Xet).
        Any value other than aria2 does the same.
    AITK_ARIA2_BIN=/path/to/aria2c
        Override aria2c executable discovery.
    AITK_ARIA2_AUTO_INSTALL=0
        Don't install aria2 when it is missing. By default Windows (x64) gets the
        official aria2 build, pinned by SHA-256, in %LOCALAPPDATA%\\ai-toolkit, and
        Linux running as root (e.g. Docker) gets it from apt-get.
    AITK_ARIA2_CONNECTIONS=16
        Number of split connections (1-16, default 16).
    AITK_ARIA2_LOG_INTERVAL=5
        Seconds between aria2 progress summaries (default 5).
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional

_ORIGINAL_HTTP_GET = None

# Official Windows build from github.com/aria2/aria2 (release-1.37.0). The SHA-256
# is the one winget's aria2.aria2 1.37.0 manifest records for the same URL.
_WIN_ARIA2_VERSION = "1.37.0"
_WIN_ARIA2_URL = (
    "https://github.com/aria2/aria2/releases/download/release-1.37.0/"
    "aria2-1.37.0-win-64bit-build1.zip"
)
_WIN_ARIA2_SHA256 = "67d015301eef0b612191212d564c5bb0a14b5b9c4796b76454276a4d28d9b288"
_WIN_ARIA2_MEMBER = "aria2-1.37.0-win-64bit-build1/aria2c.exe"


def _human_bytes(value: int) -> str:
    size = float(max(value, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    return max(minimum, min(maximum, value))


def _windows_aria2_dir() -> Path:
    """Per user, shared by every toolkit checkout."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return Path(base) / "ai-toolkit" / f"aria2-{_WIN_ARIA2_VERSION}"


def _install_aria2_windows() -> str:
    target_dir = _windows_aria2_dir()
    exe = target_dir / "aria2c.exe"
    if exe.is_file():
        return str(exe)
    print(
        f"[ai-toolkit/aria2] aria2c not found; installing the official aria2 "
        f"{_WIN_ARIA2_VERSION} build into {target_dir}",
        flush=True,
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    # staged on the same volume, so the final move is atomic
    with tempfile.TemporaryDirectory(prefix="aria2-install-", dir=target_dir.parent) as tmp:
        archive = Path(tmp) / "aria2.zip"
        with urllib.request.urlopen(_WIN_ARIA2_URL, timeout=60) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != _WIN_ARIA2_SHA256:
            raise RuntimeError(
                f"the aria2 download failed verification (sha256 {digest}, expected {_WIN_ARIA2_SHA256})"
            )
        staged = Path(tmp) / "aria2c.exe"
        with zipfile.ZipFile(archive) as zf, zf.open(_WIN_ARIA2_MEMBER) as src, open(staged, "wb") as dst:
            shutil.copyfileobj(src, dst)
        target_dir.mkdir(exist_ok=True)
        try:
            os.replace(staged, exe)
        except OSError:
            if not exe.is_file():  # otherwise a concurrent install won the race
                raise
    print(f"[ai-toolkit/aria2] Installed {exe}", flush=True)
    return str(exe)


def _install_aria2_apt() -> Optional[str]:
    """Linux as root (the Docker image): apt-get, non-interactive."""
    if getattr(os, "geteuid", lambda: -1)() != 0 or shutil.which("apt-get") is None:
        return None
    print("[ai-toolkit/aria2] aria2c not found; installing it with apt-get", flush=True)
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    subprocess.run(["apt-get", "update"], check=True, env=env)
    subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "aria2"], check=True, env=env)
    return shutil.which("aria2c")


def _install_aria2() -> Optional[str]:
    if os.name == "nt" and platform.machine().lower() in ("amd64", "x86_64"):
        return _install_aria2_windows()
    if sys.platform.startswith("linux"):
        return _install_aria2_apt()
    return None


def _aria2_binary() -> str:
    configured = os.environ.get("AITK_ARIA2_BIN")
    binary = configured or shutil.which("aria2c")
    if binary:
        return binary
    if os.environ.get("AITK_ARIA2_AUTO_INSTALL", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            binary = _install_aria2()
        except Exception as exc:
            print(f"[ai-toolkit/aria2] Automatic aria2 install failed: {exc}", flush=True)
            binary = None
        if binary:
            return binary
    raise RuntimeError(
        "AI Toolkit is configured to use aria2 for Hugging Face model downloads, "
        "but aria2c was not found and could not be installed automatically. Install "
        "aria2 (on Debian/Ubuntu: `sudo apt install aria2`), set "
        "AITK_ARIA2_BIN=/path/to/aria2c, or set AITK_HF_DOWNLOADER=hf to temporarily "
        "use Hugging Face's native downloader."
    )


def _safe_header_lines(headers: Optional[Mapping[str, Any]]) -> list[str]:
    """Build aria2 input-file header options without exposing tokens in `ps`."""
    clean: dict[str, str] = {}
    for key, value in (headers or {}).items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if not key_str:
            continue
        if "\n" in key_str or "\r" in key_str or "\n" in value_str or "\r" in value_str:
            raise RuntimeError("Refusing to pass a malformed HTTP header to aria2")
        # aria2 owns continuation/range requests. Reusing a stale Range header
        # from the caller could make a resumed file corrupt.
        if key_str.lower() in {"range", "accept-encoding"}:
            continue
        clean[key_str] = value_str

    # Hugging Face validates the final byte count. Avoid transparent gzip
    # changing the number of bytes aria2 writes.
    clean["Accept-Encoding"] = "identity"
    return [f"  header={key}: {value}" for key, value in clean.items()]


def _aria2_input(url: str, headers: Optional[Mapping[str, Any]], target: Path) -> str:
    if any(c in url for c in "\t\r\n"):
        raise RuntimeError("Refusing to pass a malformed download URL to aria2")
    if any(c in str(target) for c in "\r\n"):
        raise RuntimeError("Refusing to pass a malformed download path to aria2")

    # Per-item options override Content-Disposition. The entire item goes through
    # stdin so a terminated job cannot leave a signed URL or headers on disk.
    lines = [url, f"  dir={target.parent}", f"  out={target.name}", *_safe_header_lines(headers)]
    return "\n".join(lines) + "\n"


# aria2 reapplies custom headers on every redirect. Only pass these noncredential
# headers to it. Authenticated requests stay inside the Python HTTP client.
_PUBLIC_HEADERS = {"user-agent", "accept", "accept-encoding", "range"}
_PROGRESS = re.compile(r"^\[#[0-9a-f]+ (\d+)B/(\d+)B(?:\(\d+%\))? CN:\d+")


def _origin(url):
    import httpx
    address = httpx.URL(url)
    return address.scheme, address.host, address.port


def _authenticated_download(url, headers, temp_file, resume_size, expected_size, label, bar, retries):
    """Consume a private response or return its first off-origin URL with no credentials."""
    import httpx
    from huggingface_hub import constants
    from huggingface_hub.utils import get_session

    client = get_session()
    offset = resume_size
    if temp_file.seekable():
        temp_file.seek(offset)
        temp_file.truncate()
    for attempt in range(retries + 1):
        request_headers = httpx.Headers(headers)
        request_headers["Accept-Encoding"] = "identity"
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        request = client.build_request("GET", url, headers=request_headers, timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT)
        try:
            for _ in range(client.max_redirects + 1):
                response = client.send(request, stream=True, follow_redirects=False)
                try:
                    redirect = response.next_request
                    if redirect is not None:
                        if _origin(request.url) != _origin(redirect.url):
                            return str(redirect.url), offset
                        request = redirect
                        continue
                    response.raise_for_status()
                    if offset and response.status_code == 200:
                        temp_file.seek(0)
                        temp_file.truncate()
                        offset = 0
                    for chunk in response.iter_bytes(chunk_size=constants.DOWNLOAD_CHUNK_SIZE):
                        temp_file.write(chunk)
                        offset += len(chunk)
                    if expected_size is not None and expected_size != offset:
                        raise OSError(f"Download size mismatch for {label}: expected {expected_size} bytes, got {offset}")
                    _update_aggregate_bar(bar, max(0, offset - resume_size))
                    return None
                finally:
                    response.close()
            raise RuntimeError(f"Too many HTTP redirects while downloading {label}")
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise OSError(f"HTTP {status} while downloading {label}") from None
        except httpx.TransportError as error:
            if attempt == retries:
                raise OSError(f"{type(error).__name__} while downloading {label}") from None
        time.sleep(2)


def _run_aria2(command: list[str], item: str) -> int:
    # aria2 prints complete signed URLs even on ordinary redirects and errors.
    # Keep numeric transfer progress and the exit code, never relay raw output.
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace") as child:
        try:
            try:
                child.stdin.write(item)
                child.stdin.close()
            except BrokenPipeError:
                pass
            for line in child.stdout:
                progress = _PROGRESS.match(line.strip())
                if progress:
                    done, total = map(int, progress.groups())
                    print(f"[ai-toolkit/aria2] {_human_bytes(done)} / {_human_bytes(total)}", flush=True)
            return child.wait()
        finally:
            if child.poll() is None:
                child.kill()


def _update_aggregate_bar(bar: Any, delta: int) -> None:
    if bar is None or delta <= 0:
        return
    update_transfer = getattr(bar, "update_transfer", None)
    if callable(update_transfer):
        update_transfer(delta)
        return
    update = getattr(bar, "update", None)
    if callable(update):
        update(delta)


# Whether a failed download can be continued is huggingface_hub's decision, not this module's.
_PARTIAL_FILE_NOTE = (
    "Whether the next run continues this file depends on huggingface_hub: 1.23 deletes "
    "the partial file, so there the download starts over."
)


def _remove_orphaned_control_files(folder: Path) -> None:
    """Delete aria2 control files (<file>.aria2) whose data file is gone.

    huggingface_hub 1.23 deletes a failed download's temporary file but knows nothing about aria2's sidecar,
    and its temporary names are unique per attempt, so each failure would otherwise leave one behind for good.
    A control file without its data file is useless under any Hub version.
    """
    try:
        for control in folder.glob("*.aria2"):
            if not control.with_suffix("").exists():
                control.unlink(missing_ok=True)
    except OSError:
        pass


_LONG_PREFIX = chr(92) * 2 + "?" + chr(92)


def _plain_path(target: Path) -> str:
    s = str(target)
    return s[len(_LONG_PREFIX):] if s.startswith(_LONG_PREFIX) else s


def _too_long_for_aria2(target: Path, windows: Optional[bool] = None) -> bool:
    """True when aria2c (no long-path support on Windows) can't open target or its .aria2 control file."""
    if windows is None:
        windows = os.name == "nt"
    return windows and len(_plain_path(target)) + len(".aria2") >= 260


def aria2_http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    resume_size: int = 0,
    headers: Optional[dict[str, Any]] = None,
    expected_size: Optional[int] = None,
    displayed_filename: Optional[str] = None,
    tqdm_class: Any = None,
    _nb_retries: int = 5,
    _tqdm_bar: Any = None,
    **kwargs: Any,
) -> None:
    """Drop-in replacement for ``huggingface_hub.file_download.http_get``."""

    if expected_size is not None and resume_size == expected_size:
        target_name = getattr(temp_file, "name", None)
        if isinstance(target_name, (str, bytes, os.PathLike)):
            control_path = Path(os.fsdecode(os.fspath(target_name)) + ".aria2")
            control_path.unlink(missing_ok=True)
        return

    target_name = getattr(temp_file, "name", None)
    target = Path(os.fsdecode(target_name)).resolve() if isinstance(target_name, (str, bytes, os.PathLike)) else None
    label = displayed_filename or (target.name if target is not None else "Hub download")
    # The Hub normally supplies a model filename. Some callers pass a signed URL.
    if "://" in label:
        label = "Hub download"

    native = target is None or _too_long_for_aria2(target)
    public_headers = {k: v for k, v in (headers or {}).items() if k.lower() in _PUBLIC_HEADERS}
    if len(public_headers) != len(headers or {}):
        # Use the originating server's credentials only up to an origin change.
        # Gated models can still use segmented downloads at their signed CDN URL.
        location = _authenticated_download(url, headers, temp_file, resume_size, expected_size,
                                           label, _tqdm_bar, _nb_retries)
        if location is None:
            return
        url, resume_size = location
        headers = public_headers

    if native:
        if _ORIGINAL_HTTP_GET is None:
            raise RuntimeError("The Hugging Face downloader is unavailable for this transfer")
        return _ORIGINAL_HTTP_GET(
            url, temp_file, resume_size=resume_size, headers=headers,
            expected_size=expected_size, displayed_filename=label, tqdm_class=tqdm_class,
            _nb_retries=_nb_retries, _tqdm_bar=_tqdm_bar,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_orphaned_control_files(target.parent)

    # Hugging Face normally opens this path in append mode and derives
    # resume_size from it. Keep the path and partial bytes intact on resume.
    temp_file.flush()
    control_path = Path(str(target) + ".aria2")
    if resume_size <= 0:
        temp_file.seek(0)
        temp_file.truncate(0)
        temp_file.flush()
        # A force-download can remove Hugging Face's .incomplete file without
        # knowing about aria2's sidecar. Never let stale piece metadata apply
        # to a fresh zero-byte download.
        control_path.unlink(missing_ok=True)
        resume_size = 0

    connections = _env_int("AITK_ARIA2_CONNECTIONS", 16, 1, 16)
    log_interval = _env_int("AITK_ARIA2_LOG_INTERVAL", 5, 1, 3600)
    binary = _aria2_binary()

    if resume_size:
        print(
            f"[ai-toolkit/aria2] Resuming {label} at {_human_bytes(resume_size)}",
            flush=True,
        )
    else:
        print(f"[ai-toolkit/aria2] Downloading {label}", flush=True)

    item = _aria2_input(url, headers, target)
    command = [
        binary,
        "--continue=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--always-resume=true",
        "--auto-save-interval=5",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=1M",
        "--max-tries=10",
        "--retry-wait=2",
        "--connect-timeout=15",
        "--timeout=60",
        f"--summary-interval={log_interval}",
        "--console-log-level=notice",
        "--show-console-readout=false",
        "--human-readable=false",
        "--enable-color=false",
        f"--dir={target.parent}",
        f"--out={target.name}",
        "--input-file=-",
    ]

    try:
        returncode = _run_aria2(command, item)
        if returncode != 0:
            raise RuntimeError(
                f"aria2c exited with code {returncode} while downloading "
                f"{label}. {_PARTIAL_FILE_NOTE}"
            )
    except KeyboardInterrupt:
        print(
            f"\n[ai-toolkit/aria2] Interrupted {label}. {_PARTIAL_FILE_NOTE}",
            flush=True,
        )
        raise

    # The same file is open in this Python process while aria2 writes it.
    # Seek again so Hugging Face sees the new on-disk length.
    temp_file.seek(0, os.SEEK_END)
    actual_size = temp_file.tell()

    if expected_size is not None and actual_size != expected_size:
        raise OSError(
            f"aria2 download size mismatch for {label}: expected "
            f"{expected_size} bytes, got {actual_size} bytes. {_PARTIAL_FILE_NOTE}"
        )

    _update_aggregate_bar(_tqdm_bar, max(0, actual_size - resume_size))
    print(
        f"[ai-toolkit/aria2] Complete: {label} ({_human_bytes(actual_size)})",
        flush=True,
    )


def aria2_requested() -> bool:
    """True unless AITK_HF_DOWNLOADER is set to something other than aria2 (on by default)."""
    return os.environ.get("AITK_HF_DOWNLOADER", "aria2").strip().lower() == "aria2"


def patch_huggingface_downloads() -> bool:
    """Patch Hugging Face Hub's regular HTTP transfer function once. Returns whether the patch is active."""
    global _ORIGINAL_HTTP_GET

    if not aria2_requested():
        return False

    # Xet transfers bypass http_get, so turn them off. The environment variable covers child processes and a
    # huggingface_hub that is not imported yet; the constant covers one that already is (importing the toolkit
    # package imports it, so by the time run.py calls this the variable alone would be read too late).
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    import huggingface_hub.constants as hub_constants
    import huggingface_hub.file_download as file_download

    hub_constants.HF_HUB_DISABLE_XET = True

    current = file_download.http_get
    if getattr(current, "_aitk_aria2_patch", False):
        return True

    # The Hub can hand a signed CDN URL directly to http_get after a metadata
    # lookup. Its built-in strip covers only lowercase "authorization" and
    # compares hosts alone. Keep custom API keys, cookies and case variants at
    # the actual Hub endpoint, including an endpoint supplied by the caller.
    metadata = file_download._get_metadata_or_catch_error
    if not getattr(metadata, "_aitk_scoped_headers", False):
        @wraps(metadata)
        def scoped_metadata(**kwargs):
            result = metadata(**kwargs)
            endpoint = kwargs.get("endpoint") or hub_constants.ENDPOINT
            if result[0] and _origin(endpoint) != _origin(result[0]):
                headers = kwargs["headers"]
                for key in list(headers):
                    if key.lower() not in _PUBLIC_HEADERS:
                        del headers[key]
            return result

        scoped_metadata._aitk_scoped_headers = True
        file_download._get_metadata_or_catch_error = scoped_metadata

    _ORIGINAL_HTTP_GET = current
    aria2_http_get._aitk_aria2_patch = True
    file_download.http_get = aria2_http_get
    return True
