import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from meshpipe.filter import LandmarkFilter


@dataclass(frozen=True)
class LabResult:
    raw: np.ndarray
    kalman: np.ndarray
    ema: np.ndarray
    costs_ms: tuple[float, float]
    motion: tuple[float, float, float]
    lag_ms: tuple[float | None, float | None]


class SmoothingLab:
    def __init__(self, alpha: float = 0.35):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.filter = LandmarkFilter()
        self.ema: np.ndarray | None = None
        self.previous_ms: int | None = None
        self.history: deque = deque(maxlen=90)

    def reset(self) -> None:
        self.filter.reset()
        self.ema = None
        self.previous_ms = None
        self.history.clear()

    def update(self, raw: np.ndarray | None, timestamp_ms: int) -> LabResult | None:
        if raw is None:
            self.reset()
            return None
        if self.previous_ms is not None and not 0 < timestamp_ms - self.previous_ms <= 500:
            self.reset()
        dt = (timestamp_ms - self.previous_ms) / 1000 if self.previous_ms is not None else 1 / 30
        started = time.perf_counter()
        kalman = self.filter.update(raw, dt)
        kalman_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        effective_alpha = 1 - (1 - self.alpha) ** (dt * 30)
        self.ema = raw.copy() if self.ema is None else self.ema + effective_alpha * (raw - self.ema)
        ema_ms = (time.perf_counter() - started) * 1000
        self.previous_ms = timestamp_ms
        traces = np.array([raw[1, :2], kalman[1, :2], self.ema[1, :2]])
        self.history.append((timestamp_ms, traces))
        history = np.array([sample[1] for sample in self.history])
        motion = (0.0, 0.0, 0.0)
        lag = (None, None)
        if len(history) >= 3:
            motion = tuple(
                float(value)
                for value in np.sqrt(np.mean(np.diff(history, n=2, axis=0) ** 2, axis=(0, 2)))
            )
        if len(history) >= 20:
            timestamps = np.array([sample[0] for sample in self.history])
            step = float(np.median(np.diff(timestamps)))
            velocity = np.diff(history, axis=0)
            axis = int(np.argmax(np.var(velocity[:, 0], axis=0)))
            reference = velocity[:, 0, axis]
            if np.std(reference) > 1e-5:
                estimates = []
                for filter_index in (1, 2):
                    correlations = []
                    for delay in range(7):
                        start = reference[: len(reference) - delay] if delay else reference
                        end = velocity[delay:, filter_index, axis]
                        correlation = (
                            np.corrcoef(start, end)[0, 1]
                            if np.std(end) > 1e-9 and np.std(start) > 1e-9
                            else -1
                        )
                        correlations.append(float(correlation) if np.isfinite(correlation) else -1)
                    estimates.append(float(np.argmax(correlations)) * step)
                lag = tuple(estimates)
        return LabResult(raw, kalman, self.ema.copy(), (kalman_ms, ema_ms), motion, lag)


def trace_image(values: np.ndarray, width: int = 300, height: int = 100) -> np.ndarray:
    image = np.full((height, width, 3), [15, 27, 33], dtype=np.uint8)
    for vertical in range(20, height, 20):
        cv2.line(image, (0, vertical), (width - 1, vertical), (29, 47, 55), 1)
    values = np.asarray(values)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        return image
    lower, upper = float(values.min()), float(values.max())
    normalized = (values - lower) / max(upper - lower, 1e-6)
    horizontal = np.linspace(4, width - 5, len(values))
    for index, color in enumerate([(91, 214, 184), (245, 185, 74), (130, 170, 230)]):
        if index >= values.shape[1]:
            break
        points = np.column_stack(
            [horizontal, height - 5 - normalized[:, index] * (height - 10)]
        ).astype(np.int32)
        cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
    return image
