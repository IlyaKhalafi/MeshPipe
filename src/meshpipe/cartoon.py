"""Local image puppets with three explicit feature anchors."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ANCHOR_NAMES = ("left_eye", "right_eye", "mouth")
DEFAULT_ANCHORS = {"left_eye": [0.66, 0.40], "right_eye": [0.34, 0.40], "mouth": [0.5, 0.67]}


@dataclass
class CartoonTemplate:
    rgba: np.ndarray
    anchors: dict[str, list[float]]

    def __post_init__(self) -> None:
        if self.rgba.dtype != np.uint8 or self.rgba.ndim != 3 or self.rgba.shape[2] != 4:
            raise ValueError("Cartoon image must contain uint8 RGBA pixels")
        if max(self.rgba.shape[:2]) > 1024 or min(self.rgba.shape[:2]) < 32:
            raise ValueError("Cartoon image dimensions must be between 32 and 1024")
        if not isinstance(self.anchors, dict) or set(self.anchors) != set(ANCHOR_NAMES):
            raise ValueError("Cartoon requires left eye, right eye and mouth anchors")
        for point in self.anchors.values():
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or any(not isinstance(value, (int, float)) for value in point)
                or not np.isfinite(point).all()
                or not all(0.05 <= value <= 0.95 for value in point)
            ):
                raise ValueError("Feature anchors must stay inside image with a 5 percent margin")
        if np.linalg.norm(np.array(self.anchors["left_eye"]) - self.anchors["right_eye"]) < 0.1:
            raise ValueError("Eye anchors must be at least 10 percent of image apart")

    @classmethod
    def load(cls, path: Path) -> "CartoonTemplate":
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Cartoon image exceeds 20 MB limit")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Choose a PNG cartoon image")
            if image.width * image.height > 16_000_000:
                raise ValueError("Cartoon image exceeds pixel safety limit")
            image = image.convert("RGBA")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            if min(image.size) < 32:
                raise ValueError("Cartoon image too small")
            rgba = np.array(image)
        return cls(rgba, {name: point.copy() for name, point in DEFAULT_ANCHORS.items()})

    def save_anchors(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "anchors": self.anchors}, indent=2) + "\n", encoding="utf-8"
        )

    def load_anchors(self, path: Path) -> None:
        if path.stat().st_size > 4096:
            raise ValueError("Anchor file too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported anchor format")
        validated = CartoonTemplate(self.rgba, data.get("anchors", {}))
        self.anchors = validated.anchors


def builtin_template() -> CartoonTemplate:
    image = np.zeros((384, 384, 4), dtype=np.uint8)
    cv2.rectangle(image, (46, 62), (338, 315), (80, 218, 185, 255), -1)
    cv2.rectangle(image, (54, 70), (330, 307), (28, 49, 58, 255), -1)
    cv2.line(image, (192, 62), (192, 35), (80, 218, 185, 255), 6)
    cv2.circle(image, (192, 28), 10, (248, 184, 76, 255), -1)
    for horizontal in (130, 253):
        cv2.circle(image, (horizontal, 154), 28, (233, 242, 238, 255), -1)
        cv2.circle(image, (horizontal, 154), 13, (18, 37, 44, 255), -1)
        cv2.circle(image, (horizontal - 4, 149), 4, (255, 255, 255, 255), -1)
    cv2.ellipse(image, (192, 254), (53, 13), 0, 0, 180, (248, 184, 76, 255), 7)
    for horizontal in (24, 340):
        cv2.rectangle(image, (horizontal, 141), (horizontal + 20, 228), (248, 184, 76, 255), -1)
    return CartoonTemplate(image, {name: point.copy() for name, point in DEFAULT_ANCHORS.items()})


class CartoonRig:
    def __init__(self, template: CartoonTemplate):
        self.template = template
        height, width = template.rgba.shape[:2]
        self.horizontal, self.vertical = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        self.fields = {}
        for name, point in template.anchors.items():
            center_x, center_y = np.array(point) * [width, height]
            offset_x, offset_y = self.horizontal - center_x, self.vertical - center_y
            spread_x = width * (0.18 if name == "mouth" else 0.12)
            spread_y = height * (0.13 if name == "mouth" else 0.10)
            weight = np.exp(-0.5 * ((offset_x / spread_x) ** 2 + (offset_y / spread_y) ** 2))
            self.fields[name] = (offset_x, offset_y, weight)

    def render(self, scores: dict[str, float], roll: float = 0, opacity: float = 1) -> np.ndarray:
        height, width = self.horizontal.shape
        map_x, map_y = self.horizontal.copy(), self.vertical.copy()
        for name, channel in [("left_eye", "eyeBlinkLeft"), ("right_eye", "eyeBlinkRight")]:
            _, offset_y, weight = self.fields[name]
            blink = float(np.clip(scores.get(channel, 0), 0, 1))
            map_y += offset_y * weight * blink * 3.5
        offset_x, offset_y, weight = self.fields["mouth"]
        smile = float(
            np.clip((scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2, 0, 1)
        )
        jaw = float(np.clip(scores.get("jawOpen", 0), 0, 1))
        map_y -= offset_y * weight * jaw * 0.65
        map_y += weight * smile * height * 0.055 * np.minimum((offset_x / (width * 0.12)) ** 2, 2)
        map_x -= offset_x * weight * smile * 0.15
        warped = cv2.remap(
            self.template.rgba, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        transform = cv2.getRotationMatrix2D(
            (width / 2, height / 2), float(np.clip(np.degrees(roll), -35, 35)), 0.88
        )
        warped = cv2.warpAffine(warped, transform, (width, height), flags=cv2.INTER_LINEAR)
        alpha = warped[:, :, 3:4].astype(np.float32) / 255 * np.clip(opacity, 0, 1)
        return (warped[:, :, :3] * alpha + np.array([9, 17, 22]) * (1 - alpha)).astype(np.uint8)
