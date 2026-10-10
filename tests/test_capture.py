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
