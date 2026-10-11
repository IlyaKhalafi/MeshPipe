"""Versioned, bounded performance recording and inference-free replay."""

import json
import os
import queue
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meshpipe.geometry import normals, sample_colors, to_vertices, topology
from meshpipe.pipeline import Frame

VERSION = 1
MAX_FRAMES = 100_000
MAX_BLOB_BYTES = 16 * 1024 * 1024


def _encode_points(points: np.ndarray | None) -> bytes | None:
    return None if points is None else np.asarray(points, dtype="<f4").tobytes()


def _decode_points(data: bytes | None) -> np.ndarray | None:
    if data is None:
        return None
    if not isinstance(data, bytes) or len(data) != 468 * 3 * 4:
        raise ValueError("Session landmark array has invalid dimensions")
    points = np.frombuffer(data, dtype="<f4").reshape(468, 3).astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Session contains nonfinite landmarks")
    return points


def _inflate(data: bytes) -> bytes:
    inflater = zlib.decompressobj()
    try:
        decoded = inflater.decompress(data, MAX_BLOB_BYTES + 1)
    except zlib.error as error:
        raise ValueError("Invalid compressed session image") from error
    if len(decoded) > MAX_BLOB_BYTES or not inflater.eof or inflater.unused_data:
        raise ValueError("Invalid or oversized session frame")
    return decoded


class SessionRecorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError("Session destination already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        descriptor = os.open(self.partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=8)
        self.stopping = threading.Event()
        self.error: str | None = None
        self.written = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, name="meshpipe-session", daemon=True)
        self.thread.start()

    def submit(self, frame: Frame) -> bool:
        if self.stopping.is_set() or self.error:
            return False
        try:
            self.frames.put_nowait(frame)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        connection = None
        try:
            connection = sqlite3.connect(self.partial)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute(f"PRAGMA user_version={VERSION}")
            connection.execute(
                "CREATE TABLE frames (timestamp INTEGER PRIMARY KEY, width INTEGER, "
                "height INTEGER, rgb BLOB, landmarks BLOB, raw BLOB, expressions TEXT, "
                "processing REAL)"
            )
            previous = -1
            while not self.stopping.is_set() or not self.frames.empty():
                try:
                    frame = self.frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.written >= MAX_FRAMES:
                    raise ValueError("Session reached its 100,000 frame safety limit")
                timestamp = max(previous + 1, frame.timestamp_ms)
                previous = timestamp
                height, width = frame.rgb.shape[:2]
                if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (height, width, 3):
                    raise ValueError("Session requires uint8 RGB frames")
                if frame.rgb.nbytes > MAX_BLOB_BYTES:
                    raise ValueError("Session image exceeds safety limit")
                connection.execute(
                    "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        width,
                        height,
                        zlib.compress(frame.rgb.tobytes(), 1),
                        _encode_points(frame.landmarks),
                        _encode_points(frame.raw_landmarks),
                        json.dumps(frame.expressions, allow_nan=False),
                        frame.processing_ms,
                    ),
                )
                self.written += 1
                if self.written % 30 == 0:
                    connection.commit()
            connection.commit()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if connection is not None:
                connection.close()

    def close(self) -> Path:
        if getattr(self, "finalized", False):
            return self.path
        self.stopping.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Session writer is still finishing")
        if self.error:
            raise RuntimeError(self.error)
        if not self.written:
            self.partial.unlink(missing_ok=True)
            raise ValueError("No frames recorded")
        if self.path.exists():
            raise ValueError("Session destination appeared while recording")
        self.partial.replace(self.path)
        self.finalized = True
        return self.path


class SessionReader:
    def __init__(self, path: Path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("Session file does not exist")
        try:
            self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error as error:
            raise ValueError("Cannot open MeshPipe session database") from error
        try:
            self.connection.execute("PRAGMA query_only=ON")
            if self.connection.execute("PRAGMA user_version").fetchone()[0] != VERSION:
                raise ValueError("Unsupported MeshPipe session version")
            count = self.connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            if not 1 <= count <= MAX_FRAMES:
                raise ValueError("Session has no frames or exceeds safety limit")
            self.timestamps = np.array(
                [
                    row[0]
                    for row in self.connection.execute(
                        "SELECT timestamp FROM frames ORDER BY timestamp"
                    )
                ],
                dtype=np.int64,
            )
            if self.timestamps[0] < 0:
                raise ValueError("Session timestamps must be nonnegative")
        except sqlite3.Error as error:
            self.connection.close()
            raise ValueError("Invalid MeshPipe session database") from error
        except ValueError:
            self.connection.close()
            raise

    @property
    def duration_ms(self) -> int:
        return int(self.timestamps[-1] - self.timestamps[0])

    def frame(self, index: int) -> Frame:
        if not 0 <= index < len(self.timestamps):
            raise ValueError("Replay frame index out of range")
        row = self.connection.execute(
            "SELECT width, height, rgb, landmarks, raw, expressions, processing "
            "FROM frames WHERE timestamp=?",
            (int(self.timestamps[index]),),
        ).fetchone()
        width, height, data, points, raw, expressions, processing = row
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or not 1 <= width <= 4096
            or not 1 <= height <= 4096
        ):
            raise ValueError("Invalid session image dimensions")
        if not isinstance(data, bytes) or len(data) > MAX_BLOB_BYTES:
            raise ValueError("Invalid session image")
        decoded = _inflate(data)
        if len(decoded) != width * height * 3:
            raise ValueError("Session image dimensions do not match payload")
        rgb = np.frombuffer(decoded, dtype=np.uint8).reshape(height, width, 3).copy()
        landmarks = _decode_points(points)
        raw_landmarks = _decode_points(raw)
        if not isinstance(expressions, str) or len(expressions) > 32_768:
            raise ValueError("Invalid session expression payload")
        scores = json.loads(expressions)
        if (
            not isinstance(scores, dict)
            or len(scores) > 100
            or any(
                not isinstance(name, str)
                or len(name) > 80
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 1
                for name, value in scores.items()
            )
        ):
            raise ValueError("Invalid session expressions")
        if (
            not isinstance(processing, (int, float))
            or not np.isfinite(processing)
            or processing < 0
        ):
            raise ValueError("Invalid session timing")
        vertices = colors = vertex_normals = None
        if landmarks is not None:
            vertices = to_vertices(landmarks, width / height)
            colors = sample_colors(rgb, raw_landmarks if raw_landmarks is not None else landmarks)
            vertex_normals = normals(vertices, topology())
        return Frame(
            rgb,
            int(self.timestamps[index]),
            landmarks,
            vertices,
            colors,
            vertex_normals,
            scores,
            processing,
            raw_landmarks=raw_landmarks,
        )

    def close(self) -> None:
        self.connection.close()


@dataclass
class Playback:
    reader: SessionReader
    speed: float = 1.0
    paused: bool = False
    position_ms: float = 0
    last_clock: float | None = None
    last_index: int = -1
    loop: bool = False

    def seek(self, milliseconds: float) -> None:
        if not np.isfinite(milliseconds):
            raise ValueError("Replay position must be finite")
        self.position_ms = float(np.clip(milliseconds, 0, self.reader.duration_ms))
        self.last_index = -1
        self.last_clock = None

    def step(self, clock: float | None = None) -> Frame | None:
        clock = time.monotonic() if clock is None else clock
        if self.last_clock is not None and not self.paused:
            self.position_ms += max(0, clock - self.last_clock) * self.speed * 1000
        self.last_clock = clock
        if self.position_ms > self.reader.duration_ms:
            if self.loop and self.reader.duration_ms:
                self.position_ms %= self.reader.duration_ms
            else:
                self.position_ms = self.reader.duration_ms
                self.paused = True
        target = self.reader.timestamps[0] + self.position_ms
        index = max(0, int(np.searchsorted(self.reader.timestamps, target, side="right")) - 1)
        if index == self.last_index:
            return None
        self.last_index = index
        return self.reader.frame(index)
