"""Bounded background recording of the actual rendered studio view."""

import queue
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def compose_capture(
    source: np.ndarray, rendered: np.ndarray, title: str = "MeshPipe"
) -> np.ndarray:
    canvas = np.full((540, 960, 3), [9, 17, 22], dtype=np.uint8)
    cv2.putText(
        canvas, "MESHPIPE", (24, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (85, 221, 187), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        title[:60],
        (175, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (205, 219, 222),
        1,
        cv2.LINE_AA,
    )
    for image, rect in [(source, (20, 65, 280, 450)), (rendered, (316, 65, 624, 450))]:
        x, y, width, height = rect
        if image.ndim != 3 or image.shape[2] < 3 or not image.size:
            raise ValueError("Capture requires valid RGB images")
        factor = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image[:, :, :3],
            (max(1, round(image.shape[1] * factor)), max(1, round(image.shape[0] * factor))),
        )
        left, top = x + (width - resized.shape[1]) // 2, y + (height - resized.shape[0]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


class ViewRecorder:
    def __init__(self, path: Path, fps: int = 15, gif_seconds: int = 10):
        self.path = Path(path)
        if self.path.suffix.lower() not in {".gif", ".mp4"}:
            raise ValueError("Recording format must be GIF or MP4")
        if self.path.exists():
            raise ValueError("Recording destination already exists")
        if not 1 <= fps <= 30 or not 1 <= gif_seconds <= 20:
            raise ValueError("Invalid recording rate or GIF duration")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.limit = fps * (gif_seconds if self.path.suffix.lower() == ".gif" else 600)
        self.frames: queue.Queue = queue.Queue(maxsize=4)
        self.stop = threading.Event()
        self.error: str | None = None
        self.written = 0
        self.dropped = 0
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._run, name="meshpipe-video-writer", daemon=True)
        self.thread.start()

    def submit(self, source: np.ndarray, rendered: np.ndarray, seconds: float, title: str) -> bool:
        if self.stop.is_set() or self.finished.is_set() or self.error:
            return False
        try:
            self.frames.put_nowait((source, rendered, seconds, title))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        writer = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="meshpipe-record-", dir=self.path.parent
            ) as directory:
                root = Path(directory)
                partial = root / self.path.name
                gif = self.path.suffix.lower() == ".gif"
                if not gif:
                    writer = cv2.VideoWriter(
                        str(partial), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (960, 540)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("MP4 encoder unavailable. Try GIF recording instead.")
                previous = None
                origin = None
                while (
                    not self.stop.is_set() or not self.frames.empty()
                ) and self.written < self.limit:
                    try:
                        source, rendered, seconds, title = self.frames.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if not np.isfinite(seconds) or seconds < 0:
                        raise ValueError("Invalid capture timestamp")
                    origin = seconds if origin is None else origin
                    target = min(round((seconds - origin) * self.fps), self.limit - 1)
                    if target < self.written:
                        continue
                    canvas = compose_capture(source, rendered, title)
                    while self.written <= target:
                        image = canvas if self.written == target or previous is None else previous
                        if gif:
                            Image.fromarray(image).resize(
                                (800, 450), Image.Resampling.LANCZOS
                            ).save(root / f"{self.written:04d}.png")
                        else:
                            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        self.written += 1
                    previous = canvas
                if writer is not None:
                    writer.release()
                    writer = None
                if not self.written:
                    raise ValueError("No rendered frames captured")
                if gif:
                    images = []
                    try:
                        palette = Image.open(root / "0000.png").convert("RGB").quantize(colors=128)
                        for index in range(self.written):
                            with Image.open(root / f"{index:04d}.png") as image:
                                images.append(image.convert("RGB").quantize(palette=palette))
                        durations = [
                            (round((index + 1) * 100 / self.fps) - round(index * 100 / self.fps))
                            * 10
                            for index in range(self.written)
                        ]
                        images[0].save(
                            partial,
                            save_all=True,
                            append_images=images[1:],
                            duration=durations,
                            loop=0,
                            optimize=True,
                            disposal=1,
                        )
                        palette.close()
                    finally:
                        for image in images:
                            image.close()
                if self.path.exists():
                    raise ValueError("Recording destination appeared while recording")
                partial.replace(self.path)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if writer is not None:
                writer.release()
            self.finished.set()

    def finish(self) -> None:
        self.stop.set()

    def close(self) -> Path:
        self.finish()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError("Recording encoder is still finishing")
        if self.error:
            raise RuntimeError(self.error)
        return self.path
