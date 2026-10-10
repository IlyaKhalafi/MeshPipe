import pytest

from meshpipe.config import Config
from meshpipe.model import resolve_model


@pytest.mark.parametrize("arguments", [{"width": 10}, {"max_fps": 0}, {"detection_confidence": 2}])
def test_invalid_config(arguments):
    with pytest.raises(ValueError):
        Config(**arguments)


def test_missing_model_has_recovery_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-model"):
        resolve_model(tmp_path / "missing.task")
