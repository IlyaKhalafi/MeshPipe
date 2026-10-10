from dataclasses import dataclass

import cv2
import numpy as np

from meshpipe.pipeline import Frame


@dataclass(frozen=True)
class Quality:
    tracked: bool
    brightness: float
    sharpness: float
    face_fraction: float
    hint: str


def assess(frame: Frame) -> Quality:
    image = frame.rgb
    fraction = 0.0
    clipped = False
    if frame.raw_landmarks is not None or frame.landmarks is not None:
        points = frame.raw_landmarks if frame.raw_landmarks is not None else frame.landmarks
        lower, upper = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
        fraction = float(np.prod(upper - lower))
        clipped = bool((lower < 0.02).any() or (upper > 0.98).any())
        height, width = image.shape[:2]
        start = np.clip(np.floor(lower * [width, height]).astype(int), 0, [width - 1, height - 1])
        stop = np.clip(np.ceil(upper * [width, height]).astype(int), start + 1, [width, height])
        image = image[start[1] : stop[1], start[0] : stop[0]]
    height, width = image.shape[:2]
    image = cv2.resize(
        image,
        (max(1, round(width * min(1, 160 / width))), max(1, round(height * min(1, 160 / width)))),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if frame.landmarks is None:
        hint = "Face lost. Face camera and check lighting."
    elif clipped:
        hint = "Keep entire face inside frame."
    elif fraction < 0.025:
        hint = "Move closer for more facial detail."
    elif brightness < 45:
        hint = "Add soft light in front of face."
    elif brightness > 235:
        hint = "Reduce glare or direct light."
    elif sharpness < 12:
        hint = "Check focus or reduce motion blur."
    else:
        hint = "Good framing. Lighting checks are heuristics."
    return Quality(frame.landmarks is not None, brightness, sharpness, fraction, hint)


class Recovery:
    def __init__(self, grace_seconds: float = 0.35):
        self.grace_seconds = grace_seconds
        self.last_frame: Frame | None = None
        self.lost_at: float | None = None
        self.reacquired = False

    def update(self, frame: Frame, clock: float) -> None:
        if frame.vertices is not None:
            self.reacquired = self.lost_at is not None
            self.last_frame = frame
            self.lost_at = None
        else:
            self.reacquired = False
            if self.lost_at is None:
                self.lost_at = clock

    def presentation(self, clock: float) -> tuple[Frame | None, float]:
        if self.lost_at is None:
            return self.last_frame, 1.0
        opacity = float(np.clip(1 - (clock - self.lost_at) / self.grace_seconds, 0, 1))
        return (self.last_frame if opacity > 0 else None), opacity
