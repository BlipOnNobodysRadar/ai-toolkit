import requests
import pytest

from toolkit import civitai


def test_civitai_error_preserves_http_status_and_omits_signed_url(tmp_path, monkeypatch, capsys):
    url = "https://storage.invalid/weights?signature=private-signature"
    monkeypatch.setattr(civitai, "MODELS_PATH", str(tmp_path))
    monkeypatch.setattr(civitai, "get_model_download_info", lambda *args: ({"downloadUrl": url, "name": "x.pt"}, 7))
    response = requests.Response()
    response.status_code = 403
    response.url = url
    response._content_consumed = True
    monkeypatch.setattr(civitai.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(OSError, match="model 123.*HTTP 403") as error:
        civitai.get_model_path_from_url("https://civitai.com/models/123")
    logs = capsys.readouterr()
    assert "private-signature" not in logs.out + logs.err + str(error.value)
    assert not list(tmp_path.iterdir())
