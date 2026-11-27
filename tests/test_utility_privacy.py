import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_flux_ui_upload_and_public_link_require_opt_in(tmp_path, monkeypatch, capsys):
    # Exercise the real UI wiring and callback without starting Gradio or loading a model.
    import huggingface_hub
    import toolkit.job

    gr = MagicMock()
    monkeypatch.setitem(sys.modules, "gradio", gr)
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["flux_train_ui.py"])
    for key in ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_ENABLE_HF_TRANSFER"):
        monkeypatch.delenv(key, raising=False)
    identity = Mock(return_value={"name": "example"})
    monkeypatch.setattr(huggingface_hub, "whoami", identity)
    jobs = []

    def get_job(config_path):
        path = Path(config_path)
        jobs.append((path, yaml.safe_load(path.read_text())["config"]["process"][0]))
        return Mock()

    monkeypatch.setattr(toolkit.job, "get_job", get_job)
    monkeypatch.chdir(ROOT)
    ui = runpy.run_path(str(ROOT / "flux_train_ui.py"), run_name="__main__")
    gr.Blocks.return_value.__enter__.return_value.launch.assert_called_once_with(share=False, show_error=True)
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    args = ["model", "private-caption", 1, .001, 8, "dev", False, str(tmp_path), "", "", "",
            True, "save:\n  push_to_hub: true"]
    ui["start_training"](*args)
    identity.assert_not_called()
    path, process = jobs[-1]
    assert process["save"]["push_to_hub"] is False
    assert not path.exists()
    assert "private-caption" not in capsys.readouterr().out

    ui["start_training"](*args, push_to_hub=True)
    identity.assert_called_once()
    assert jobs[-1][1]["save"]["hf_repo_id"] == "example/model"
    assert jobs[-1][1]["save"]["hf_private"] is True
    monkeypatch.setattr(sys, "argv", ["flux_train_ui.py", "--share"])
    runpy.run_path(str(ROOT / "flux_train_ui.py"), run_name="__main__")
    assert gr.Blocks.return_value.__enter__.return_value.launch.call_args.kwargs["share"] is True


def test_inference_health_exposes_job_details_only_to_the_authenticated_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from extensions_built_in.inference_engine import server
    from extensions_built_in.inference_engine.InferenceEngine import InferenceEngine

    health = {"ok": True, "current": {"prompt": "private"}, "active": {"model": "/private/model"}}
    client = TestClient(server.create_app(SimpleNamespace(health=lambda: health), token="fixture"))
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/health", headers={"x-engine-token": "incorrect"}).status_code == 401
    assert client.get("/health", headers={"authorization": "Bearer fixture"}).json() == health

    monkeypatch.setattr(server, "serve_in_thread", lambda *a: (None, None, "127.0.0.1", 1))
    endpoint_path = tmp_path / "engine.json"
    context = SimpleNamespace(
        engine_config=SimpleNamespace(host="127.0.0.1", port=0),
        ui=SimpleNamespace(job_id="fixture"), device="cpu", job_folder=str(tmp_path),
        output_folder=str(tmp_path), endpoint_path=str(endpoint_path),
    )
    InferenceEngine.start_server(context, client.app, "fixture")
    assert endpoint_path.is_file()
    if os.name == "posix":
        assert endpoint_path.stat().st_mode & 0o077 == 0


def test_pexels_api_key_stays_with_its_api(monkeypatch):
    from extensions_built_in.dataset_tools.tools import sync_tools

    response = Mock()
    response.json.side_effect = [
        {"media": [], "next_page": "https://api.pexels.com/next"},
        {"media": [], "next_page": "https://other.invalid/next"},
    ]
    get = Mock(return_value=response)
    monkeypatch.setattr(sync_tools.requests, "get", get)
    with pytest.raises(ValueError, match="outside"):
        sync_tools.get_pexels_images(SimpleNamespace(collection_id=42, api_key="fixture"))
    assert get.call_count == 2
    assert all(call.args[0].startswith("https://api.pexels.com/") for call in get.call_args_list)


def test_supporter_list_respects_github_visibility_and_counts_patrons(tmp_path, monkeypatch):
    from scripts import update_sponsors as s

    monkeypatch.setattr(s, "GITHUB_ORG", None)
    monkeypatch.setattr(s, "GITHUB_USERNAME", "fixture")
    nodes = [
        {"isActive": True, "privacyLevel": level, "sponsorEntity": {"login": name}}
        for name, level in [("shared-user", "PUBLIC"), ("unshared-user", "PRIVATE"), ("unknown-user", None)]
    ]
    response = Mock()
    response.json.return_value = {"data": {"user": {"sponsorshipsAsMaintainer": {"nodes": nodes}}}}
    post = Mock(return_value=response)
    monkeypatch.setattr(s.requests, "post", post)
    sponsors = s.fetch_github_sponsors()
    assert [x["name"] for x in sponsors] == ["shared-user"]
    assert "privacyLevel" in post.call_args.kwargs["json"]["query"]

    response.json.side_effect = [
        {"data": [{"id": 1}]},
        {"data": [{"attributes": {"patron_status": "active_patron", "full_name": "unshared-patron"}}]},
    ]
    get = Mock(return_value=response)
    monkeypatch.setattr(s.requests, "get", get)
    count = s.fetch_patreon_supporter_count()
    assert count == 1
    assert get.call_args.kwargs["params"] == {"fields[member]": "patron_status"}
    output = tmp_path / "SUPPORTERS.md"
    monkeypatch.setattr(s, "README_PATH", output)
    s.generate_readme(sponsors, count)
    rendered = output.read_text()
    assert "shared-user" in rendered and "1 active Patreon" in rendered
    assert "unshared" not in rendered and "unknown-user" not in rendered
