import json
from pathlib import Path

import numpy as np

from meshpipe.animation_protocol import reduce_keys, validate_animation
from meshpipe.pose import head_angles


def animation_payload(reader, fps: float = 30, tolerance: float = 0.002):
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Export FPS must be between 1 and 120")
    if len(reader.timestamps) > 36000:
        raise ValueError("Trim performance to at most 36000 frames before baking")
    times, scores, angles = [], [], []
    for index, timestamp in enumerate(reader.timestamps):
        frame = reader.frame(index)
        times.append(1 + float(timestamp - reader.timestamps[0]) * fps / 1000)
        scores.append(frame.expressions)
        angles.append(head_angles(frame.vertices) if frame.vertices is not None else (0, 0, 0))
    names = sorted({name for sample in scores for name in sample})
    channels = {
        name: reduce_keys(times, [sample.get(name, 0) for sample in scores], tolerance)
        for name in names
    }
    rotation = np.unwrap(np.asarray(angles), axis=0)
    payload = {
        "version": 1,
        "fps": fps,
        "source_kind": reader.source_kind,
        "channels": channels,
        "rotation": {
            axis: reduce_keys(times, rotation[:, index].tolist(), tolerance)
            for index, axis in enumerate("xyz")
        },
    }
    return validate_animation(payload)


def export_animation(reader, path: Path, fps=30, tolerance=0.002):
    payload = animation_payload(reader, fps, tolerance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf8")
    return path
