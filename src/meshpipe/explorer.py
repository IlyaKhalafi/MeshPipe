import json
from collections import deque
from pathlib import Path

import numpy as np

REGIONS = {
    "All landmarks": tuple(range(468)),
    "Left eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "Right eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "Lips": (
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
        78,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        80,
        191,
    ),
    "Nose": (1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 326, 327),
    "Face outline": (
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ),
}


def pick_landmark(
    vertices: np.ndarray,
    projection: np.ndarray,
    view: np.ndarray,
    width: int,
    height: int,
    x: float,
    y: float,
    radius: float = 16,
) -> int | None:
    if width <= 0 or height <= 0:
        return None
    clip = np.column_stack([vertices, np.ones(len(vertices))]) @ (projection @ view).T
    front = clip[:, 3] > 1e-9
    normalized = clip[:, :3] / np.maximum(clip[:, 3:4], 1e-9)
    valid = front & (np.abs(normalized) <= 1).all(axis=1)
    pixels = (normalized[:, :2] * [0.5, -0.5] + 0.5) * [width, height]
    distance = np.linalg.norm(pixels - [x, y], axis=1)
    distance[~valid] = np.inf
    index = int(np.argmin(distance))
    return index if distance[index] <= radius else None


class LandmarkHistory:
    def __init__(self, index: int = 1, limit: int = 120):
        self.index = index
        self.values: deque = deque(maxlen=limit)

    def select(self, index: int) -> None:
        if not 0 <= index < 468:
            raise ValueError("Landmark index must be between 0 and 467")
        if self.index != index:
            self.index = index
            self.values.clear()

    def update(self, timestamp_ms: int, landmarks: np.ndarray | None) -> None:
        if landmarks is None:
            self.values.clear()
            return
        if self.values and timestamp_ms <= self.values[-1][0]:
            self.values.clear()
        self.values.append((timestamp_ms, landmarks[self.index].copy()))

    def export(self, path: Path, region: str) -> None:
        if region not in REGIONS:
            raise ValueError("Unknown landmark region")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "region": region,
                    "indices": REGIONS[region],
                    "selected": self.index,
                    "history": [
                        {"timestamp_ms": timestamp, "xyz": point.tolist()}
                        for timestamp, point in self.values
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
