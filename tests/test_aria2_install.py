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
    lines = aria2._aria2_input("https://example.invalid/f", {"user-agent": "Hub"}, target).splitlines()
    assert lines[0] == "https://example.invalid/f"
    assert f"  dir={target.parent}" in lines and "  out=abc123.incomplete" in lines
    assert "  header=user-agent: Hub" in lines
    with pytest.raises(RuntimeError, match="malformed download path"):
        aria2._aria2_input("https://example.invalid/f", None, tmp_path / "bad\nname")


def test_only_numeric_aria2_output_reaches_the_job_log(capsys):
    import sys

    # A subprocess echoing its entire signed input models aria2's redirects and errors.
    script = ("import sys; item = sys.stdin.read(); assert item.endswith('signature'); "
              "assert sys.argv[1:] == ['--input-file=-']; "
              "print(item); print('[#aa311d 10B/100B(10%) CN:1 DL:1B ETA:90s]'); "
              "print(item, file=sys.stderr); sys.exit(3)")
    rc = aria2._run_aria2([sys.executable, "-c", script, "--input-file=-"],
                          "https://cdn.invalid/blob?private-signature")
    assert rc == 3
    logs = capsys.readouterr()
    assert "10.0 B / 100.0 B" in logs.out
    assert "signature" not in logs.out + logs.err


def test_credentials_follow_same_origin_only_and_cdn_input_uses_stdin(tmp_path, monkeypatch):
    import httpx
    import huggingface_hub.utils

    def route(req):
        assert req.url.host == "hub.invalid" and req.headers["Authorization"] == "Bearer private-token"
        assert req.headers["X-Api-Key"] == "private-key"
        dest = "/renamed" if req.url.path == "/repo" else "https://storage.invalid/blob?private-signature"
        return httpx.Response(302, headers={"Location": dest})

    with httpx.Client(transport=httpx.MockTransport(route)) as client:
        monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: client)
        target = tmp_path / "model.incomplete"
        def run(command, item):
            assert "--input-file=-" in command and all("private-" not in a for a in command)
            assert "private-token" not in item and "private-key" not in item
            assert item.startswith("https://storage.invalid/blob?private-signature\n")
            target.write_bytes(b"weights")
            return 0
        monkeypatch.setattr(aria2, "_aria2_binary", lambda: "aria2c")
        monkeypatch.setattr(aria2, "_run_aria2", run)
        with target.open("w+b") as f:
            aria2.aria2_http_get("https://hub.invalid/repo", f, expected_size=7,
                                 headers={"Authorization": "Bearer private-token", "X-Api-Key": "private-key"})
            assert f.tell() == 7


def test_private_file_without_redirect_still_resumes(tmp_path, monkeypatch):
    import httpx
    import huggingface_hub.utils

    def route(req):
        assert req.headers["Authorization"] == "Bearer private-token" and req.headers["Range"] == "bytes=3-"
        return httpx.Response(206, content=b"ghts")
    with httpx.Client(transport=httpx.MockTransport(route)) as client:
        monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: client)
        with (tmp_path / "model").open("w+b") as f:
            f.write(b"wei")
            aria2.aria2_http_get("https://hub.invalid/repo", f, expected_size=7, resume_size=3,
                                 headers={"Authorization": "Bearer private-token"})
            f.seek(0)
            assert f.read() == b"weights"

def test_hub_metadata_handoff_drops_every_private_header_at_an_origin_change(monkeypatch):
    from huggingface_hub import constants, file_download

    endpoint = "https://private-hub.invalid"
    monkeypatch.setenv("AITK_HF_DOWNLOADER", "aria2")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setattr(aria2, "_ORIGINAL_HTTP_GET", None)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.setattr(file_download, "http_get", lambda *args: None)
    monkeypatch.setattr(file_download, "_get_metadata_or_catch_error", lambda **kwargs: (kwargs["location"],))
    aria2.patch_huggingface_downloads()

    private = {"Authorization": "Bearer private-token", "Cookie": "private-cookie", "X-API-Key": "private-key"}
    for location in [endpoint + "/repo", "http://private-hub.invalid/file", "https://cdn.invalid/file"]:
        headers = {**private, "User-Agent": "Hub"}
        result = file_download._get_metadata_or_catch_error(headers=headers, endpoint=endpoint, location=location)
        assert result == (location,)
        assert headers == ({**private, "User-Agent": "Hub"} if location.startswith(endpoint) else {"User-Agent": "Hub"})
