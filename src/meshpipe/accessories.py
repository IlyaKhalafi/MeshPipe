import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from meshpipe.pose import head_rotation


def tube(path, radius=0.035, sides=8):
    points = np.asarray(path, dtype=float)
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    axis = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    parallel = np.abs(tangents[:, 2]) > 0.9
    axis[parallel] = [0, 1, 0]
    horizontal = np.cross(tangents, axis)
    horizontal /= np.maximum(np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-9)
    vertical = np.cross(tangents, horizontal)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    vertices = points[:, None, :] + radius * (
        np.cos(angles)[None, :, None] * horizontal[:, None, :]
        + np.sin(angles)[None, :, None] * vertical[:, None, :]
    )
    faces = []
    for index in range(len(points) - 1):
        for side in range(sides):
            first = index * sides + side
            second = index * sides + (side + 1) % sides
            faces.extend([[first, second, second + sides], [first, second + sides, first + sides]])
    return vertices.reshape(-1, 3), np.asarray(faces, dtype=np.int32)


def combine(parts):
    vertices, faces, offset = [], [], 0
    for points, triangles in parts:
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
    return np.concatenate(vertices), np.concatenate(faces)


def accessory_geometry(name):
    if name == "Glasses":
        angles = np.linspace(0, 2 * np.pi, 49)
        parts = [
            tube(
                np.column_stack(
                    [
                        0.43 * np.cos(angles) + side * 0.66,
                        0.31 * np.sin(angles),
                        np.zeros_like(angles),
                    ]
                )
            )
            for side in (-1, 1)
        ]
        parts.append(tube([[-0.23, 0.05, 0], [0, 0.12, 0.06], [0.23, 0.05, 0]]))
        for side in (-1, 1):
            parts.append(
                tube([[side * 1.09, 0, 0], [side * 1.20, 0, -0.35], [side * 1.14, -0.1, -1.2]])
            )
        return combine(parts)
    if name == "Crown":
        angles = np.linspace(0, 2 * np.pi, 49)
        points = np.column_stack(
            [1.15 * np.cos(angles), np.zeros_like(angles), 0.8 * np.sin(angles) - 0.35]
        )
        parts = [tube(points, 0.06)]
        for side in np.linspace(-0.9, 0.9, 5):
            parts.append(
                tube([[side - 0.18, 0, 0.35], [side, 0.48, 0.35], [side + 0.18, 0, 0.35]], 0.05)
            )
        return combine(parts)
    if name == "Brows":
        parts = []
        for side in (-1, 1):
            points = [[side * 0.25, 0, 0], [side * 0.65, 0.2, 0.02], [side * 1.05, 0.05, 0]]
            parts.append(tube(points, 0.085))
        return combine(parts)
    raise ValueError("Unknown accessory")


@dataclass(frozen=True)
class AccessoryAnchor:
    name: str = "Glasses"
    landmark: int = 168
    scale: float = 1
    offset: tuple[float, float, float] = (0, 0, 0.18)

    def validate(self):
        if self.name not in {"Glasses", "Crown", "Brows"}:
            raise ValueError("Unknown accessory")
        if not isinstance(self.landmark, int) or not 0 <= self.landmark < 468:
            raise ValueError("Accessory anchor must be a landmark from 0 to 467")
        if not isinstance(self.scale, (int, float)) or not np.isfinite(self.scale):
            raise ValueError("Accessory scale must be finite")
        if not 0.1 <= self.scale <= 3 or np.shape(self.offset) != (3,):
            raise ValueError("Invalid accessory scale or offset")
        if not np.isfinite(self.offset).all() or max(abs(value) for value in self.offset) > 3:
            raise ValueError("Accessory offsets must be finite and within 3 units")

    def save(self, path: Path):
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "anchor": asdict(self)}, indent=2), encoding="utf8"
        )

    @classmethod
    def load(cls, path: Path):
        if path.stat().st_size > 4096:
            raise ValueError("Accessory preset exceeds 4 KiB")
        try:
            data = json.loads(path.read_text(encoding="utf8"))
            if data["version"] != 1:
                raise ValueError("Unsupported accessory preset")
            anchor = cls(**data["anchor"])
            anchor.validate()
            return anchor
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid accessory preset") from error


class AccessoryRig:
    def __init__(self, anchor=None):
        self.configure(AccessoryAnchor() if anchor is None else anchor)

    def configure(self, anchor):
        anchor.validate()
        self.anchor = anchor
        self.vertices, self.triangles = accessory_geometry(anchor.name)

    def deform(self, vertices):
        if np.shape(vertices) != (468, 3) or not np.isfinite(vertices).all():
            raise ValueError("Accessory tracking requires 468 finite vertices")
        local = self.vertices * self.anchor.scale + self.anchor.offset
        return local @ head_rotation(vertices).T + vertices[self.anchor.landmark]
