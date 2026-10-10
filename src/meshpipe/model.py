import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def default_model_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "meshpipe" / "face_landmarker.task"


def resolve_model(path: Path | None) -> Path:
    target = path or default_model_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"Face model not found at {target}. Run 'meshpipe download-model' "
            "or provide --model /path/to/face_landmarker.task"
        )
    return target


def download_model(target: Path) -> Path:
    """Download official model atomically so interrupted transfers remain recoverable."""
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "MeshPipe/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise ValueError("Model download exceeds 100 MB safety limit")
                    output.write(chunk)
        if total < 1024 * 1024:
            raise ValueError("Model download is unexpectedly small")
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
