"""The two features that used to hang off a repo-root sitecustomize.py, which Python never imports for
`python run.py` (the script directory joins sys.path after the site module has run): the aria2 Hugging Face
downloader (on by default, AITK_HF_DOWNLOADER=hf turns it off) and the LTX-2 isolate_modalities dataset option."""
import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_no_repo_root_sitecustomize():
    # it would not load on its own, and with PYTHONPATH=<repo> it shadows the manager's generated one in the venv
    assert not (REPO / "sitecustomize.py").exists()


def test_script_directory_sitecustomize_is_not_imported(tmp_path):
    """The reason the hooks are wired explicitly: pin the interpreter behaviour they used to rely on."""
    (tmp_path / "sitecustomize.py").write_text("print('SITECUSTOMIZE LOADED')\n")
    (tmp_path / "run.py").write_text("print('script ran')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "run.py"], cwd=tmp_path, env=env, capture_output=True, text=True).stdout
    assert "script ran" in out and "SITECUSTOMIZE LOADED" not in out


def test_run_py_installs_the_downloader_hook_after_the_toolkit_import():
    tree = ast.parse((REPO / "run.py").read_text(encoding="utf-8"))
    calls = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "patch_huggingface_downloads"]
    torch_import = min(n.lineno for n in ast.walk(tree)
                       if isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names))
    assert len(calls) == 1 and calls[0] < torch_import  # before any model code can download


@pytest.fixture
def hub_download_state():
    import huggingface_hub.constants as constants
    import huggingface_hub.file_download as file_download
    import toolkit.aria2_hf_download as aria2

    saved = (file_download.http_get, constants.HF_HUB_DISABLE_XET, aria2._ORIGINAL_HTTP_GET,
             os.environ.get("HF_HUB_DISABLE_XET"), os.environ.get("AITK_HF_DOWNLOADER"))
    yield aria2, file_download, constants
    file_download.http_get, constants.HF_HUB_DISABLE_XET, aria2._ORIGINAL_HTTP_GET = saved[:3]
    for name, value in (("HF_HUB_DISABLE_XET", saved[3]), ("AITK_HF_DOWNLOADER", saved[4])):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.mark.parametrize("value", ["hf", "HF", "", "xet"])
def test_any_other_downloader_value_turns_aria2_off(hub_download_state, value):
    aria2, file_download, constants = hub_download_state
    native, xet_disabled = file_download.http_get, constants.HF_HUB_DISABLE_XET
    os.environ["AITK_HF_DOWNLOADER"] = value
    assert aria2.patch_huggingface_downloads() is False
    assert file_download.http_get is native and constants.HF_HUB_DISABLE_XET == xet_disabled


@pytest.mark.parametrize("value", [None, "aria2", " Aria2 "])
def test_aria2_is_on_by_default_and_turns_xet_off_after_the_hub_is_imported(hub_download_state, value):
    aria2, file_download, constants = hub_download_state
    from huggingface_hub.utils._runtime import is_xet_available

    # run.py's order: huggingface_hub is already imported (the toolkit package imports it) with Xet enabled
    constants.HF_HUB_DISABLE_XET = False
    native = file_download.http_get
    if value is None:
        os.environ.pop("AITK_HF_DOWNLOADER", None)
    else:
        os.environ["AITK_HF_DOWNLOADER"] = value
    assert aria2.patch_huggingface_downloads() is True
    assert file_download.http_get is aria2.aria2_http_get and aria2._ORIGINAL_HTTP_GET is native
    # Xet downloads bypass http_get, so the hub has to see it as unavailable from now on
    assert is_xet_available() is False and os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert aria2.patch_huggingface_downloads() is True  # idempotent
    assert aria2._ORIGINAL_HTTP_GET is native


def test_orphaned_aria2_control_files_are_removed_and_live_ones_kept(tmp_path):
    """huggingface_hub 1.23 deletes a failed download's temp file but not aria2's <file>.aria2 sidecar."""
    from toolkit.aria2_hf_download import _remove_orphaned_control_files

    (tmp_path / "abc.1234.incomplete").write_bytes(b"partial")          # a download in progress
    (tmp_path / "abc.1234.incomplete.aria2").write_bytes(b"control")
    (tmp_path / "def.5678.incomplete.aria2").write_bytes(b"control")    # its data file was deleted
    (tmp_path / "finished-blob").write_bytes(b"done")
    _remove_orphaned_control_files(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "abc.1234.incomplete", "abc.1234.incomplete.aria2", "finished-blob"]
    _remove_orphaned_control_files(tmp_path / "does-not-exist")  # never raises


def test_dataset_config_carries_isolate_modalities():
    from toolkit.config_modules import DatasetConfig

    assert DatasetConfig(folder_path="x").isolate_modalities is False
    assert DatasetConfig(folder_path="x", isolate_modalities=True).isolate_modalities is True


@pytest.mark.parametrize("isolate", [False, True])
def test_ltx_dataset_isolation_reaches_the_transformer(monkeypatch, isolate):
    from toolkit.config_modules import DatasetConfig
    import extensions_built_in.diffusion_models.ltx2 as ltx2

    seen = []

    class Transformer:
        def forward(self, **kwargs):
            seen.append(kwargs["isolate_modalities"])

    def stock_get_noise_prediction(self, *args, **kwargs):
        # the stock method always passes False (ltx2.py); the wrapper is what turns it on per dataset
        self.transformer.forward(isolate_modalities=False)
        return "pred"

    monkeypatch.setattr(ltx2, "_original_get_noise_prediction", stock_get_noise_prediction)
    model = SimpleNamespace(transformer=Transformer())
    stock_forward = model.transformer.forward
    batch = SimpleNamespace(dataset_config=DatasetConfig(folder_path="x", isolate_modalities=isolate))

    assert ltx2.LTX2Model.get_noise_prediction(model, None, None, None, batch=batch) == "pred"
    assert seen == [isolate]
    assert model.transformer.forward == stock_forward  # the per-call wrapper is removed again
