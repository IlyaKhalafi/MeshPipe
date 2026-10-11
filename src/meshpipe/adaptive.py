import math


class AdaptiveBudget:
    """Slow, hysteretic controller for inference size and live-camera cadence."""

    def __init__(self, width: int, fps: float, budget_ms: float = 33):
        if width < 160 or not 1 <= fps <= 120:
            raise ValueError("Invalid adaptive width or rate")
        self.maximum_width = width
        self.maximum_fps = fps
        self.width = width
        self.fps = fps
        self.mean_ms = 0.0
        self.samples = 0
        self.configure(False, budget_ms)

    def configure(self, enabled: bool, budget_ms: float) -> None:
        if not math.isfinite(budget_ms) or not 8 <= budget_ms <= 200:
            raise ValueError("CPU budget must be between 8 and 200 ms")
        self.enabled = enabled
        self.budget_ms = budget_ms
        self.samples = 0
        if not enabled:
            self.width, self.fps = self.maximum_width, self.maximum_fps

    def observe(self, duration_ms: float) -> None:
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            return
        self.mean_ms = duration_ms if not self.mean_ms else self.mean_ms * 0.9 + duration_ms * 0.1
        self.samples += 1
        if not self.enabled or self.samples < 30:
            return
        self.samples = 0
        if self.mean_ms > self.budget_ms * 1.15:
            self.width = max(160, int(self.width * 0.85) // 8 * 8)
            self.fps = max(5, min(self.maximum_fps, 800 / self.mean_ms))
        elif self.mean_ms < self.budget_ms * 0.7:
            self.width = min(self.maximum_width, int(self.width * 1.1 + 8) // 8 * 8)
            self.fps = min(self.maximum_fps, self.fps * 1.1)
