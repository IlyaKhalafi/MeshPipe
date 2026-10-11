"""Build a small morph-only GLB example from the bundled canonical face."""

import json
import struct
from pathlib import Path

import numpy as np

from meshpipe.geometry import canonical_vertices, topology


def build(path):
    points = canonical_vertices().copy()
    points -= points.mean(axis=0)
    points *= 2 / np.linalg.norm(points[33] - points[263])
    jaw = np.zeros_like(points)
    lower = np.clip((points[1, 1] - points[:, 1]) / 1.5, 0, 1)
    jaw[:, 1] -= lower * 0.55
    smile = np.zeros_like(points)
    distance = np.minimum(
        np.linalg.norm(points - points[61], axis=1), np.linalg.norm(points - points[291], axis=1)
    )
    smile[:, 1] += np.exp(-distance * 5) * 0.22
    targets = {"jawOpen": jaw, "mouthSmileLeft": smile / 2, "mouthSmileRight": smile / 2}
    for name, upper, lower_index, region in [
        ("eyeBlinkRight", 159, 145, [33, 160, 158, 133, 153, 144, 159, 145]),
        ("eyeBlinkLeft", 386, 374, [362, 385, 387, 263, 373, 380, 386, 374]),
    ]:
        delta = np.zeros_like(points)
        middle = (points[upper, 1] + points[lower_index, 1]) / 2
        delta[region, 1] = middle - points[region, 1]
        targets[name] = delta
    binary = bytearray()
    views, accessors = [], []

    def add(array, kind, component):
        offset = len(binary)
        binary.extend(array.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes})
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component,
                "count": len(array),
                "type": kind,
            }
        )
        if kind == "VEC3":
            accessors[-1]["min"] = array.min(axis=0).tolist()
            accessors[-1]["max"] = array.max(axis=0).tolist()
        return len(accessors) - 1

    position = add(points.astype("<f4"), "VEC3", 5126)
    triangles = add(topology().ravel().astype("<u2"), "SCALAR", 5123)
    descriptors = [
        {"POSITION": add(delta.astype("<f4"), "VEC3", 5126)} for delta in targets.values()
    ]
    binary.extend(b"\0" * (-len(binary) % 4))
    document = {
        "asset": {"version": "2.0", "generator": "MeshPipe canonical morph example"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.95, 0.67, 0.28, 1]}}],
        "meshes": [
            {
                "extras": {"targetNames": list(targets)},
                "primitives": [
                    {
                        "attributes": {"POSITION": position},
                        "indices": triangles,
                        "targets": descriptors,
                        "material": 0,
                    }
                ],
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text
    body += struct.pack("<II", len(binary), 0x004E4942) + binary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body)
    return path


if __name__ == "__main__":
    print(build(Path("src/meshpipe/assets/example-head.glb")))
