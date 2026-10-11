import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from meshpipe.config import Config
from meshpipe.demo import demo_frame
from meshpipe.pipeline import Processor
from meshpipe.session import Playback, SessionReader, SessionRecorder, _inflate


def make_frame(timestamp=0):
    image, landmarks, scores = demo_frame(timestamp / 1000, 160)
    processor = Processor(Config(demo=True, width=160))
    return processor.process(image, timestamp, demo_landmarks=landmarks, demo_expressions=scores)


def record(path, frames):
    recorder = SessionRecorder(path)
    for frame in frames:
        assert recorder.submit(frame)
    recorder.close()
    return recorder


def test_lossless_rgb_and_timestamps_roundtrip(tmp_path):
    original = make_frame(300)
    path = tmp_path / "take.meshpipe"
    record(path, [original, replace(original, timestamp_ms=400)])
    reader = SessionReader(path)
    replay = reader.frame(0)
    np.testing.assert_array_equal(replay.rgb, original.rgb)
    np.testing.assert_allclose(replay.landmarks, original.landmarks, atol=1e-7)
    np.testing.assert_allclose(replay.raw_landmarks, original.raw_landmarks, atol=1e-7)
    assert replay.expressions == original.expressions
    assert reader.duration_ms == 100
    reader.close()


def test_face_loss_roundtrip(tmp_path):
    path = tmp_path / "lost.meshpipe"
    record(
        path,
        [replace(make_frame(), landmarks=None, raw_landmarks=None, vertices=None, expressions={})],
    )
    reader = SessionReader(path)
    assert reader.frame(0).vertices is None
    reader.close()


def test_playback_seek_pause_speed_loop(tmp_path):
    path = tmp_path / "take.meshpipe"
    record(path, [make_frame(index * 100) for index in range(4)])
    reader = SessionReader(path)
    player = Playback(reader)
    assert player.step(0).timestamp_ms == 0
    assert player.step(0.15).timestamp_ms == 100
    player.paused = True
    assert player.step(2) is None
    player.seek(300)
    assert player.step(3).timestamp_ms == 300
    player.paused = False
    player.loop = True
    player.speed = 2
    assert player.step(3.1).timestamp_ms == 200
    reader.close()


def test_empty_recording_and_overwrite_protection(tmp_path):
    path = tmp_path / "empty.meshpipe"
    recorder = SessionRecorder(path)
    with pytest.raises(ValueError, match="No frames"):
        recorder.close()
    assert not path.exists()
    path.write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        SessionRecorder(path)


def test_bad_session_rejected(tmp_path):
    path = tmp_path / "bad.meshpipe"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=42")
    with pytest.raises(ValueError, match="Unsupported"):
        SessionReader(path)
    with pytest.raises(ValueError):
        _inflate(b"broken")


def test_writer_failure_surfaces(tmp_path):
    recorder = SessionRecorder(tmp_path / "bad.meshpipe")
    recorder.submit(replace(make_frame(), rgb=np.zeros((10, 10, 3), dtype=np.float64)))
    with pytest.raises(RuntimeError, match="uint8"):
        recorder.close()
    assert not recorder.path.exists()


def test_reader_open_failure_becomes_actionable_error(tmp_path, monkeypatch):
    path = tmp_path / "inaccessible.meshpipe"
    path.write_bytes(b"placeholder")

    def unavailable(*arguments, **keywords):
        raise sqlite3.OperationalError("permission denied")

    monkeypatch.setattr(sqlite3, "connect", unavailable)
    with pytest.raises(ValueError, match="Cannot open"):
        SessionReader(path)


@pytest.mark.parametrize(
    "column,payload",
    [
        ("expressions", None),
        ("raw", "bad"),
        ("expressions", "x" * 40_000),
        ("rgb", b"not compressed"),
    ],
)
def test_corrupt_frame_payloads_rejected(tmp_path, column, payload):
    path = tmp_path / "corrupt.meshpipe"
    record(path, [make_frame()])
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE frames SET {column}=?", (payload,))
    reader = SessionReader(path)
    try:
        with pytest.raises(ValueError):
            reader.frame(0)
    finally:
        reader.close()
