"""Personal expression calibration with no emotion classification."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHANNELS = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight")


@dataclass
class Calibration:
    neutral: dict[str, float] = field(default_factory=dict)
    maximum: dict[str, float] = field(default_factory=dict)
    sensitivity: float = 1.0
    dead_zone: float = 0.04

    def apply(self, scores: dict[str, float]) -> dict[str, float]:
        result = scores.copy()
        for name in CHANNELS:
            if name not in scores:
                continue
            baseline = self.neutral.get(name, 0.0)
            ceiling = self.maximum.get(name, 1.0)
            value = (scores[name] - baseline) / max(ceiling - baseline, 0.1)
            value = (value - self.dead_zone) / (1 - self.dead_zone)
            result[name] = float(np.clip(value * self.sensitivity, 0, 1))
        return result

    def fit(self, samples: list[dict[str, float]], neutral: bool) -> None:
        if len(samples) < 10:
            raise ValueError("Calibration needs at least ten tracked frames")
        if any(any(name not in sample for name in CHANNELS) for sample in samples):
            raise ValueError("Calibration requires blink, jaw and smile channels")
        target = self.neutral if neutral else self.maximum
        for name in CHANNELS:
            values = [sample[name] for sample in samples]
            if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
                raise ValueError("Invalid calibration expression scores")
        for name in CHANNELS:
            target[name] = float(
                np.percentile([sample[name] for sample in samples], 50 if neutral else 95)
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "neutral": self.neutral,
                    "maximum": self.maximum,
                    "sensitivity": self.sensitivity,
                    "dead_zone": self.dead_zone,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        if path.stat().st_size > 16_384:
            raise ValueError("Calibration file too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported calibration format")
        neutral, maximum = data.get("neutral", {}), data.get("maximum", {})
        for scores in [neutral, maximum]:
            if not isinstance(scores, dict) or any(
                name not in CHANNELS
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 1
                for name, value in scores.items()
            ):
                raise ValueError("Invalid calibration channels")
        sensitivity, dead_zone = data.get("sensitivity", 1), data.get("dead_zone", 0.04)
        if not isinstance(sensitivity, (int, float)) or not 0.25 <= sensitivity <= 3:
            raise ValueError("Sensitivity must be between 0.25 and 3")
        if not isinstance(dead_zone, (int, float)) or not 0 <= dead_zone <= 0.4:
            raise ValueError("Dead zone must be between 0 and 0.4")
        return cls(neutral, maximum, float(sensitivity), float(dead_zone))
