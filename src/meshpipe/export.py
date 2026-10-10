import json
from datetime import datetime
from pathlib import Path

import numpy as np

from meshpipe.geometry import topology
from meshpipe.pipeline import Frame


def export_frame(frame: Frame, directory: Path = Path("exports")) -> tuple[Path, Path]:
    """Write inspectable PLY geometry and normalized landmark metadata."""
    if frame.vertices is None or frame.landmarks is None:
        raise ValueError("No tracked face available to export")
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("meshpipe-%Y%m%d-%H%M%S-%f")
    ply_path = directory / f"{stem}.ply"
    json_path = directory / f"{stem}.json"
    faces = topology()
    colors = np.rint(np.clip(frame.colors, 0, 1) * 255).astype(np.uint8)
    with ply_path.open("w", encoding="ascii") as output:
        output.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(frame.vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for vertex, color in zip(frame.vertices, colors, strict=True):
            output.write(
                " ".join([*(f"{value:.7f}" for value in vertex), *(str(value) for value in color)])
                + "\n"
            )
        for face in faces:
            output.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    payload = {
        "schema_version": 1,
        "timestamp_ms": frame.timestamp_ms,
        "coordinate_system": "centered x-right y-up z-toward-viewer; outer-eye distance 2",
        "landmark_space": "MediaPipe normalized image coordinates; not metric depth",
        "landmarks": frame.landmarks.tolist(),
        "expressions": frame.expressions,
        "triangles": faces.tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ply_path, json_path
