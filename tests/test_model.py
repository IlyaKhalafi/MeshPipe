import io

import pytest

from meshpipe.model import download_model


def test_atomic_download(monkeypatch, tmp_path):
    payload = b"test" * (300 * 1024)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    target = tmp_path / "models" / "face.task"
    assert download_model(target) == target
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]


def test_failed_download_preserves_existing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"error"))
    target = tmp_path / "face.task"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="unexpectedly small"):
        download_model(target)
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]
