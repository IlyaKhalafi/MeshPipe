import threading
import time
from dataclasses import dataclass

from numpy.typing import NDArray

from meshpipe.capture import VideoSource
from meshpipe.config import Config
from meshpipe.demo import demo_frame
from meshpipe.detector import FaceDetector
from meshpipe.filter import LandmarkFilter
from meshpipe.geometry import normals, sample_colors, to_vertices, topology


@dataclass(frozen=True)
class Frame:
    rgb: NDArray
    timestamp_ms: int
    landmarks: NDArray | None
    vertices: NDArray | None
    colors: NDArray | None
    normals: NDArray | None
    expressions: dict[str, float]
    processing_ms: float
    fps: float = 0.0


class Processor:
    def __init__(self, config: Config):
        self.config = config
        self.filter = LandmarkFilter()
        self.detector = None if config.demo else FaceDetector(config)
        self.previous_ms: int | None = None
        self.was_smoothing = config.smoothing
        self.faces = topology()

    def process(
        self,
        rgb: NDArray,
        timestamp_ms: int,
        smoothing: bool | None = None,
        demo_landmarks: NDArray | None = None,
        demo_expressions: dict[str, float] | None = None,
    ) -> Frame:
        started = time.perf_counter()
        smoothing = self.config.smoothing if smoothing is None else smoothing
        if self.detector is not None:
            landmarks, expressions = self.detector.detect(rgb, timestamp_ms)
        else:
            landmarks, expressions = demo_landmarks, demo_expressions or {}
        if self.was_smoothing != smoothing:
            self.filter.reset()
        self.was_smoothing = smoothing
        vertices = colors = vertex_normals = None
        if landmarks is None:
            self.filter.reset()
            self.previous_ms = None
        else:
            if self.previous_ms is not None and timestamp_ms - self.previous_ms > 500:
                self.filter.reset()
            dt = (
                max((timestamp_ms - self.previous_ms) / 1000, 0.001)
                if self.previous_ms is not None
                else 1 / 30
            )
            self.previous_ms = timestamp_ms
            raw_landmarks = landmarks
            if smoothing:
                landmarks = self.filter.update(landmarks, dt)
            vertices = to_vertices(landmarks, rgb.shape[1] / rgb.shape[0])
            colors = sample_colors(rgb, raw_landmarks)
            vertex_normals = normals(vertices, self.faces)
        return Frame(
            rgb,
            timestamp_ms,
            landmarks,
            vertices,
            colors,
            vertex_normals,
            expressions,
            (time.perf_counter() - started) * 1000,
        )

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()


class Worker:
    """Own all inference state off the GUI thread and publish through one mailbox."""

    def __init__(self, config: Config):
        self.config = config
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest: Frame | None = None
        self.error: str | None = None
        self.finished = False
        self.paused = False
        self.smoothing = config.smoothing
        self.thread = threading.Thread(target=self._run, name="meshpipe-inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_paused(self, enabled: bool) -> None:
        with self.lock:
            self.paused = enabled

    def set_smoothing(self, enabled: bool) -> None:
        with self.lock:
            self.smoothing = enabled

    def take(self) -> tuple[Frame | None, str | None, bool]:
        with self.lock:
            frame, self.latest = self.latest, None
            return frame, self.error, self.finished

    def _run(self) -> None:
        processor = source = None
        try:
            processor = Processor(self.config)
            if not self.config.demo:
                source = VideoSource(self.config.source, self.config.width, self.config.loop)
            previous = time.perf_counter()
            count = 0
            ema_fps = 0.0
            while not self.stop.is_set():
                with self.lock:
                    paused, smoothing = self.paused, self.smoothing
                if paused:
                    self.stop.wait(0.05)
                    previous = time.perf_counter()
                    continue
                started = time.perf_counter()
                if self.config.demo:
                    seconds = count / self.config.max_fps
                    rgb, landmarks, expressions = demo_frame(seconds, self.config.width)
                    frame = processor.process(
                        rgb, int(seconds * 1000), smoothing, landmarks, expressions
                    )
                else:
                    captured = source.read()
                    if captured is None:
                        break
                    frame = processor.process(*captured, smoothing=smoothing)
                interval = started - previous
                previous = started
                instant_fps = 1 / max(interval, 1e-6) if count else 0
                ema_fps = instant_fps if not ema_fps else ema_fps * 0.9 + instant_fps * 0.1
                frame = Frame(**{**vars(frame), "fps": ema_fps})
                with self.lock:
                    self.latest = frame
                count += 1
                rate = self.config.max_fps
                if source is not None and not source.camera:
                    rate = min(rate, source.fps)
                self.stop.wait(max(0, 1 / rate - (time.perf_counter() - started)))
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                if source is not None:
                    source.close()
                if processor is not None:
                    processor.close()
            finally:
                with self.lock:
                    self.finished = True

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
