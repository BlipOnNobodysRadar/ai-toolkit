"""Route Hugging Face Hub HTTP transfers through aria2c.

Hugging Face Hub still owns repository metadata, cache layout, authentication,
and final file placement. This module only replaces the byte-transfer layer so
large model downloads get aria2's segmented transfers, visible progress, and
reliable continuation from the Hub's existing ``.incomplete`` files.

Environment variables:
    AITK_HF_DOWNLOADER=hf
        Disable this patch and use Hugging Face's native downloader.
    AITK_ARIA2_BIN=/path/to/aria2c
        Override aria2c executable discovery.
    AITK_ARIA2_CONNECTIONS=16
        Number of split connections (1-16, default 16).
    AITK_ARIA2_LOG_INTERVAL=5
        Seconds between aria2 progress summaries (default 5).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional

_ORIGINAL_HTTP_GET = None


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


def _aria2_binary() -> str:
    configured = os.environ.get("AITK_ARIA2_BIN")
    binary = configured or shutil.which("aria2c")
    if binary:
        return binary
    raise RuntimeError(
        "AI Toolkit is configured to use aria2 for Hugging Face model downloads, "
        "but aria2c was not found. Install aria2 (on Debian/Ubuntu: "
        "`sudo apt install aria2`), set AITK_ARIA2_BIN=/path/to/aria2c, or set "
        "AITK_HF_DOWNLOADER=hf to temporarily use Hugging Face's native downloader."
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


def _write_aria2_input(url: str, headers: Optional[Mapping[str, Any]]) -> str:
    if "\n" in url or "\r" in url:
        raise RuntimeError("Refusing to pass a malformed download URL to aria2")

    fd, path = tempfile.mkstemp(prefix="aitk-aria2-", suffix=".txt")
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(url)
            handle.write("\n")
            for line in _safe_header_lines(headers):
                handle.write(line)
                handle.write("\n")
        return path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


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

    del tqdm_class, _nb_retries, kwargs  # aria2 handles retries/progress itself.

    if expected_size is not None and resume_size == expected_size:
        target_name = getattr(temp_file, "name", None)
        if isinstance(target_name, (str, bytes, os.PathLike)):
            control_path = Path(os.fsdecode(os.fspath(target_name)) + ".aria2")
            control_path.unlink(missing_ok=True)
        return

    target_name = getattr(temp_file, "name", None)
    if not isinstance(target_name, (str, bytes, os.PathLike)):
        # This is not how hf_hub_download normally invokes http_get, but retain
        # compatibility with unusual callers rather than guessing a path.
        if _ORIGINAL_HTTP_GET is None:
            raise RuntimeError("Hugging Face provided aria2 with a non-path temporary file")
        return _ORIGINAL_HTTP_GET(
            url,
            temp_file,
            resume_size=resume_size,
            headers=headers,
            expected_size=expected_size,
            displayed_filename=displayed_filename,
            _tqdm_bar=_tqdm_bar,
        )

    target = Path(os.fsdecode(os.fspath(target_name))).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

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

    label = displayed_filename or target.name
    if resume_size:
        print(
            f"[ai-toolkit/aria2] Resuming {label} at {_human_bytes(resume_size)}",
            flush=True,
        )
    else:
        print(f"[ai-toolkit/aria2] Downloading {label}", flush=True)

    input_file = _write_aria2_input(url, headers)
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
        "--show-console-readout=true",
        f"--dir={target.parent}",
        f"--out={target.name}",
        f"--input-file={input_file}",
    ]

    try:
        # Inherit stdout/stderr so aria2's live progress is visible in the
        # ai-toolkit terminal/UI job log.
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"aria2c exited with code {completed.returncode} while downloading "
                f"{label}. The partial file was kept and will be resumed next run."
            )
    except KeyboardInterrupt:
        print(
            f"\n[ai-toolkit/aria2] Interrupted {label}; partial download kept for resume.",
            flush=True,
        )
        raise
    finally:
        try:
            os.unlink(input_file)
        except OSError:
            pass

    # The same file is open in this Python process while aria2 writes it.
    # Seek again so Hugging Face sees the new on-disk length.
    temp_file.seek(0, os.SEEK_END)
    actual_size = temp_file.tell()

    if expected_size is not None and actual_size != expected_size:
        raise OSError(
            f"aria2 download size mismatch for {label}: expected "
            f"{expected_size} bytes, got {actual_size} bytes. The partial file "
            "was kept so the next run can continue it."
        )

    _update_aggregate_bar(_tqdm_bar, max(0, actual_size - resume_size))
    print(
        f"[ai-toolkit/aria2] Complete: {label} ({_human_bytes(actual_size)})",
        flush=True,
    )


def patch_huggingface_downloads() -> bool:
    """Patch Hugging Face Hub's regular HTTP transfer function once."""
    global _ORIGINAL_HTTP_GET

    if os.environ.get("AITK_HF_DOWNLOADER", "aria2").strip().lower() != "aria2":
        return False

    # Must be set before huggingface_hub imports its constants. sitecustomize
    # calls us early enough for normal ai-toolkit startup.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    import huggingface_hub.file_download as file_download

    current = file_download.http_get
    if getattr(current, "_aitk_aria2_patch", False):
        return True

    _ORIGINAL_HTTP_GET = current
    aria2_http_get._aitk_aria2_patch = True
    file_download.http_get = aria2_http_get
    return True
