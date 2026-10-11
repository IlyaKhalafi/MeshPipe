import json
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np

from meshpipe.geometry import to_vertices, topology
from meshpipe.motion import MotionEdit
from meshpipe.pose import head_rotation

CHANNELS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight"]


def showcase_payload(reader, max_frames=360, signal_label=None):
    if not 2 <= max_frames <= 900:
        raise ValueError("Browser showcase frame budget must be between 2 and 900")
    if signal_label is None:
        signal_label = (
            "Edited" if getattr(reader, "edit", MotionEdit()) != MotionEdit() else "Processed"
        )
    if signal_label not in {"Edited", "Processed", "Kalman"}:
        raise ValueError("Invalid showcase signal label")
    indices = np.unique(
        np.linspace(0, len(reader.timestamps) - 1, min(max_frames, len(reader.timestamps))).astype(
            int
        )
    )
    samples = []
    for index in indices:
        frame = reader.frame(int(index))
        raw = None
        if frame.raw_landmarks is not None:
            raw = to_vertices(frame.raw_landmarks, frame.rgb.shape[1] / frame.rgb.shape[0])
        samples.append(
            {
                "time": int(frame.timestamp_ms - reader.timestamps[0]),
                "smooth": np.rint(frame.vertices * 10000).astype(int).ravel().tolist()
                if frame.vertices is not None
                else None,
                "raw": np.rint(raw * 10000).astype(int).ravel().tolist()
                if raw is not None
                else None,
                "rotation": np.round(head_rotation(frame.vertices), 5).ravel().tolist()
                if frame.vertices is not None
                else None,
                "scores": [round(frame.expressions.get(name, 0), 4) for name in CHANNELS],
            }
        )
    return {
        "version": 1,
        "scale": 10000,
        "source_kind": reader.source_kind,
        "signal_label": signal_label,
        "duration": reader.duration_ms,
        "triangles": topology().ravel().tolist(),
        "channels": CHANNELS,
        "frames": samples,
    }


def export_showcase(reader, directory: Path, max_frames=360, signal_label=None):
    text = json.dumps(showcase_payload(reader, max_frames, signal_label), separators=(",", ":"))
    if len(text) > 16 * 1024 * 1024:
        raise ValueError("Showcase motion exceeds 16 MiB. Reduce frame budget.")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "viewer.js"):
        resource = files("meshpipe").joinpath("assets/showcase", name)
        with resource.open("rb") as source, (directory / name).open("wb") as target:
            shutil.copyfileobj(source, target)
    (directory / "performance.json").write_text(text, encoding="utf8")
    return directory
