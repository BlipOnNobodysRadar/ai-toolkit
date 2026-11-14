"""aria2c on Windows can't open paths past MAX_PATH: such downloads go to Hugging Face's own downloader (14 Nov 2026)."""
import io
from pathlib import Path

from toolkit import aria2_hf_download as a


def test_too_long_for_aria2_counts_the_control_file():
    base = "C:" + chr(92) + "d" + chr(92)
    fits = Path(base + "f" * (259 - len(base) - len(".aria2")))
    over = Path(base + "f" * (260 - len(base) - len(".aria2")))
    assert not a._too_long_for_aria2(fits, windows=True)
    assert a._too_long_for_aria2(over, windows=True)
    assert a._too_long_for_aria2(Path(a._LONG_PREFIX + str(over)), windows=True)
    assert not a._too_long_for_aria2(over, windows=False)


def test_long_path_falls_back_to_the_hub_downloader(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(a, "_too_long_for_aria2", lambda target, windows=None: True)
    monkeypatch.setattr(a, "_ORIGINAL_HTTP_GET", lambda url, f, **kw: calls.append((url, kw["expected_size"])))
    monkeypatch.setattr(a, "_aria2_binary", lambda: (_ for _ in ()).throw(AssertionError("aria2 must not run")))
    f = io.BytesIO()
    f.name = str(tmp_path / "x.incomplete")
    a.aria2_http_get("http://example.invalid/x", f, headers={}, expected_size=7)
    assert calls == [("http://example.invalid/x", 7)]
