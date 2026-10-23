"""aria2c discovery and the automatic install when it is missing (no network: downloads are faked)."""

import hashlib
import io
import zipfile

import pytest

import toolkit.aria2_hf_download as aria2


def _fake_zip(exe_bytes=b"fake aria2c"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(aria2._WIN_ARIA2_MEMBER, exe_bytes)
        zf.writestr("aria2-1.37.0-win-64bit-build1/README.html", "readme")
    return buf.getvalue()


@pytest.fixture
def no_aria2(monkeypatch, tmp_path):
    monkeypatch.delenv("AITK_ARIA2_BIN", raising=False)
    monkeypatch.delenv("AITK_ARIA2_AUTO_INSTALL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(aria2.shutil, "which", lambda name: None)
    return tmp_path


def _serve(monkeypatch, payload):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(aria2.urllib.request, "urlopen", urlopen)
    return calls


def test_path_and_env_win_without_installing(monkeypatch):
    def refuse():
        raise AssertionError("installed although aria2c was available")

    monkeypatch.setattr(aria2, "_install_aria2", refuse)
    monkeypatch.delenv("AITK_ARIA2_BIN", raising=False)
    monkeypatch.setattr(aria2.shutil, "which", lambda name: "/usr/bin/aria2c")
    assert aria2._aria2_binary() == "/usr/bin/aria2c"
    monkeypatch.setenv("AITK_ARIA2_BIN", "/opt/aria2c")
    assert aria2._aria2_binary() == "/opt/aria2c"


def test_windows_install_verifies_then_reuses(monkeypatch, no_aria2):
    payload = _fake_zip()
    monkeypatch.setattr(aria2, "_WIN_ARIA2_SHA256", hashlib.sha256(payload).hexdigest())
    calls = _serve(monkeypatch, payload)
    exe = aria2._install_aria2_windows()
    assert calls == [aria2._WIN_ARIA2_URL]
    assert open(exe, "rb").read() == b"fake aria2c"
    assert exe.startswith(str(no_aria2 / "ai-toolkit" / "aria2-1.37.0"))
    # second run: already installed, nothing downloaded
    assert aria2._install_aria2_windows() == exe
    assert len(calls) == 1
    assert [p.name for p in (no_aria2 / "ai-toolkit").iterdir()] == ["aria2-1.37.0"]  # staging cleaned up


def test_windows_install_rejects_a_bad_download(monkeypatch, no_aria2):
    _serve(monkeypatch, _fake_zip(b"tampered"))  # pinned digest stays the real one
    with pytest.raises(RuntimeError, match="verification"):
        aria2._install_aria2_windows()
    assert not (no_aria2 / "ai-toolkit" / "aria2-1.37.0" / "aria2c.exe").exists()
    assert list((no_aria2 / "ai-toolkit").iterdir()) == []


def test_missing_aria2_installs_by_default(monkeypatch, no_aria2):
    monkeypatch.setattr(aria2, "_install_aria2", lambda: "C:/installed/aria2c.exe")
    assert aria2._aria2_binary() == "C:/installed/aria2c.exe"


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_auto_install_can_be_turned_off(monkeypatch, no_aria2, value):
    def refuse():
        raise AssertionError("installed although AITK_ARIA2_AUTO_INSTALL is off")

    monkeypatch.setattr(aria2, "_install_aria2", refuse)
    monkeypatch.setenv("AITK_ARIA2_AUTO_INSTALL", value)
    with pytest.raises(RuntimeError, match="aria2c was not found"):
        aria2._aria2_binary()


def test_failed_install_reports_the_usual_error(monkeypatch, no_aria2, capsys):
    def broken():
        raise OSError("network down")

    monkeypatch.setattr(aria2, "_install_aria2", broken)
    with pytest.raises(RuntimeError, match="AITK_HF_DOWNLOADER=hf"):
        aria2._aria2_binary()
    assert "network down" in capsys.readouterr().out


def test_apt_install_only_as_root(monkeypatch):
    runs = []
    monkeypatch.setattr(aria2.subprocess, "run", lambda cmd, **kw: runs.append(cmd))
    installed = {"done": False}

    def which(name):
        if name == "apt-get":
            return "/usr/bin/apt-get"
        return "/usr/bin/aria2c" if installed["done"] and name == "aria2c" else None

    monkeypatch.setattr(aria2.shutil, "which", which)
    monkeypatch.setattr(aria2.os, "geteuid", lambda: 1000, raising=False)
    assert aria2._install_aria2_apt() is None and runs == []
    monkeypatch.setattr(aria2.os, "geteuid", lambda: 0, raising=False)
    installed["done"] = True
    assert aria2._install_aria2_apt() == "/usr/bin/aria2c"
    assert runs[-1][:3] == ["apt-get", "install", "-y"] and runs[-1][-1] == "aria2"


def test_input_file_names_the_target_per_item(tmp_path):
    # aria2 ignores --dir/--out for input-file items; without per-item options it saved
    # Hugging Face blobs under their Content-Disposition name (config.json) and the
    # .incomplete file stayed empty
    target = tmp_path / "blobs" / "abc123.incomplete"
    path = aria2._write_aria2_input("https://example.invalid/f", {"authorization": "Bearer x"}, target)
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    finally:
        aria2.os.unlink(path)
    assert lines[0] == "https://example.invalid/f"
    assert f"  dir={target.parent}" in lines and "  out=abc123.incomplete" in lines
    assert "  header=authorization: Bearer x" in lines
    with pytest.raises(RuntimeError, match="malformed download path"):
        aria2._write_aria2_input("https://example.invalid/f", None, tmp_path / "bad\nname")
