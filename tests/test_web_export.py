import json

import numpy as np
import pytest

from meshpipe.motion import EditedReader
from meshpipe.session import SessionReader
from meshpipe.web_export import export_showcase, showcase_payload

pytest_plugins = ["test_motion"]


def test_web_export_has_no_source_images(performance, tmp_path):
    reader = EditedReader(SessionReader(performance))
    directory = export_showcase(reader, tmp_path / "web", max_frames=3)
    payload = json.loads((directory / "performance.json").read_text())
    assert payload["source_kind"] == "synthetic"
    assert len(payload["frames"]) == 3
    assert payload["frames"][-1]["time"] == 400
    first = payload["frames"][0]
    reconstructed = np.asarray(first["smooth"]).reshape(468, 3) / payload["scale"]
    np.testing.assert_allclose(reconstructed, reader.frame(0).vertices, atol=0.000051)
    assert "rgb" not in first
    assert {item.name for item in directory.iterdir()} == {
        "index.html",
        "style.css",
        "viewer.js",
        "performance.json",
    }
    reader.close()


def test_invalid_frame_budget(performance):
    reader = SessionReader(performance)
    with pytest.raises(ValueError):
        showcase_payload(reader, 901)
    reader.close()
