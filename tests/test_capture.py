import time

import cv2
import numpy as np
import pytest

from meshpipe.capture import VideoSource


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((240, 320, 3), index * 80, dtype=np.uint8))
    writer.release()
    return path


def test_sequential_video_timestamps_and_eof(video):
    source = VideoSource(str(video), 160)
    try:
        for index in range(3):
            image, timestamp = source.read()
            assert image.shape == (120, 160, 3)
            assert timestamp == index * 40
        assert source.read() is None
    finally:
        source.close()


def test_loop_keeps_monotonic_timestamps(video):
    source = VideoSource(str(video), 640, loop=True)
    try:
        timestamps = [source.read()[1] for _ in range(7)]
        assert timestamps == list(range(0, 280, 40))
    finally:
        source.close()


def test_missing_video_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video input does not exist"):
        VideoSource(str(tmp_path / "missing.mp4"), 640)


class FakeCamera:
    def __init__(self, disconnected=False):
        self.disconnected = disconnected
        self.released = False
        self.count = 0

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, *args):
        return 30

    def read(self):
        time.sleep(0.01)
        self.count += 1
        return not self.disconnected, np.full((16, 16, 3), self.count % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_mailbox_and_shutdown(monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: camera)
    source = VideoSource("0", 640)
    try:
        first, first_time = source.read()
        second, second_time = source.read()
        assert second[0, 0, 0] > first[0, 0, 0]
        assert second_time > first_time
    finally:
        source.close()
    assert camera.released
    assert not source.thread.is_alive()


def test_camera_disconnection_surfaces_error(monkeypatch):
    camera = FakeCamera(disconnected=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: camera)
    source = VideoSource("0", 640)
    try:
        with pytest.raises(OSError, match="disconnected"):
            source.read()
    finally:
        source.close()


def test_camera_coarse_clock_keeps_timestamps_strictly_increasing(monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: camera)
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    source = VideoSource("0", 640)
    try:
        timestamps = [source.read()[1] for _ in range(3)]
        assert timestamps[0] >= 0
        assert all(
            current > previous
            for previous, current in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
    finally:
        source.close()
    assert camera.released
    assert not source.thread.is_alive()


def test_variable_frame_rate_uses_decoder_timestamps(monkeypatch, tmp_path):
    path = tmp_path / "variable.mp4"
    path.write_bytes(b"placeholder")

    class VariableRateCapture(FakeCamera):
        def get(self, property_id):
            if property_id == cv2.CAP_PROP_POS_MSEC:
                return [0, 50, 120][self.count - 1]
            return 30

    monkeypatch.setattr(cv2, "VideoCapture", lambda source: VariableRateCapture())
    source = VideoSource(str(path), 640)
    try:
        assert [source.read()[1] for _ in range(3)] == [0, 50, 120]
    finally:
        source.close()
