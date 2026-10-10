import math

import cv2
import numpy as np
from numpy.typing import NDArray

from meshpipe.geometry import canonical_vertices


def demo_frame(seconds: float, width: int = 640) -> tuple[NDArray, NDArray, dict[str, float]]:
    """Generate deterministic geometry and expression signals without a face detector."""
    vertices = canonical_vertices().copy()
    smile = (math.sin(seconds * 1.8) + 1) / 2
    jaw = max(0.0, math.sin(seconds * 2.2)) * 0.65
    blink = max(0.0, math.cos(seconds * 3.0)) ** 18
    lower = vertices[:, 1] < -1.5
    vertices[lower, 1] -= jaw * np.minimum(-vertices[lower, 1] - 1.5, 3) * 0.3
    for corner in [61, 291, 78, 308]:
        vertices[corner, 1] += smile * 0.55
    angle = math.sin(seconds * 0.7) * 0.3
    rotation = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )
    vertices = vertices @ rotation.T
    landmarks = np.column_stack(
        [0.5 + vertices[:, 0] / 25, 0.47 - vertices[:, 1] / 19, -vertices[:, 2] / 25]
    )
    height = width * 3 // 4
    image = np.full((height, width, 3), (14, 27, 36), dtype=np.uint8)
    for column in range(0, width, 32):
        cv2.line(image, (column, 0), (column, height), (22, 42, 49), 1)
    for row in range(0, height, 32):
        cv2.line(image, (0, row), (width, row), (22, 42, 49), 1)
    cv2.putText(
        image, "SYNTHETIC DEMO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (73, 220, 170), 1
    )
    expressions = {
        "mouthSmileLeft": smile,
        "mouthSmileRight": smile,
        "jawOpen": jaw,
        "eyeBlinkLeft": blink,
        "eyeBlinkRight": blink,
    }
    return image, landmarks, expressions
