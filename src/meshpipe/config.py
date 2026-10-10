from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    source: str = "0"
    model: Path | None = None
    width: int = 640
    max_fps: float = 30.0
    detection_confidence: float = 0.5
    smoothing: bool = True
    demo: bool = False
    loop: bool = False

    def __post_init__(self) -> None:
        if self.width < 160:
            raise ValueError("Processing width must be at least 160 pixels")
        if not 1 <= self.max_fps <= 120:
            raise ValueError("Maximum FPS must be between 1 and 120")
        if not 0 < self.detection_confidence <= 1:
            raise ValueError("Detection confidence must be in (0, 1]")
