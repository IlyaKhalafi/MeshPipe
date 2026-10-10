import threading
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoSource:
    """Sequential file decoding or a single-slot, latest-frame webcam reader."""

    def __init__(self, source: str, width: int, loop: bool = False):
        self.camera = source.isdecimal()
        if not self.camera and not Path(source).is_file():
            raise FileNotFoundError(f"Video input does not exist: {source}")
        self.capture = cv2.VideoCapture(int(source) if self.camera else source)
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Cannot open {'webcam' if self.camera else 'video'} {source}")
        self.width = width
        self.loop = loop
        if self.camera:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 3 // 4)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        reported_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if np.isfinite(reported_fps) and 1 <= reported_fps <= 240 else 30.0
        self.index = 0
        self.offset_ms = 0.0
        self.last_timestamp_ms = -1.0
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.latest: tuple[NDArray, int] | None = None
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        if self.camera:
            self.thread = threading.Thread(
                target=self._read_camera, name="meshpipe-capture", daemon=True
            )
            self.thread.start()

    def _read_camera(self) -> None:
        try:
            while not self.stop.is_set():
                success, frame = self.capture.read()
                if not success:
                    raise OSError("Webcam disconnected or returned an empty frame")
                with self.condition:
                    self.latest = (frame, int((time.monotonic() - self.started) * 1000))
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.capture.release()

    def read(self) -> tuple[NDArray, int] | None:
        if self.camera:
            with self.condition:
                ready = self.condition.wait_for(
                    lambda: self.latest is not None or self.error is not None or self.stop.is_set(),
                    timeout=5,
                )
                if self.error:
                    raise self.error
                if not ready:
                    raise TimeoutError("No webcam frame received within 5 seconds")
                if self.stop.is_set():
                    return None
                frame, timestamp = self.latest
                self.latest = None
        else:
            success, frame = self.capture.read()
            if not success and self.loop and self.index:
                self.offset_ms = self.last_timestamp_ms + 1000 / self.fps
                self.index = 0
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.capture.read()
            if not success:
                return None
            presentation_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            candidate = self.offset_ms + presentation_ms
            if (
                not np.isfinite(presentation_ms)
                or presentation_ms < 0
                or candidate <= self.last_timestamp_ms
            ):
                candidate = max(
                    self.offset_ms + self.index * 1000 / self.fps,
                    self.last_timestamp_ms + 1000 / self.fps,
                )
            self.last_timestamp_ms = candidate
            timestamp = round(candidate)
            self.index += 1
        height, width = frame.shape[:2]
        if width > self.width:
            frame = cv2.resize(
                frame,
                (self.width, round(height * self.width / width)),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp

    def close(self) -> None:
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=2)
        else:
            self.capture.release()
