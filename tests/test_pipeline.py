import json
import time

import numpy as np

from meshpipe.config import Config
from meshpipe.demo import demo_frame
from meshpipe.export import export_frame
from meshpipe.pipeline import Processor, Worker


def test_demo_processing_and_export(tmp_path):
    processor = Processor(Config(demo=True))
    image, landmarks, expressions = demo_frame(0.3)
    frame = processor.process(image, 100, demo_landmarks=landmarks, demo_expressions=expressions)
    assert frame.vertices.shape == (468, 3)
    assert np.isfinite(frame.vertices).all()
    paths = export_frame(frame, tmp_path)
    assert "element vertex 468" in paths[0].read_text()
    assert json.loads(paths[1].read_text())["landmarks"] == frame.landmarks.tolist()
    assert len(list(tmp_path.glob("*"))) == 2
    processor.close()


def test_face_loss_clears_filter_and_geometry():
    processor = Processor(Config(demo=True))
    image, landmarks, expressions = demo_frame(0)
    processor.process(image, 0, demo_landmarks=landmarks)
    lost = processor.process(image, 33)
    assert lost.vertices is None
    assert processor.filter.position is None
    recovered = processor.process(image, 66, demo_landmarks=landmarks)
    np.testing.assert_array_equal(recovered.landmarks, landmarks)


def test_smoothing_toggle_reinitializes_filter():
    processor = Processor(Config(demo=True))
    image, landmarks, _ = demo_frame(0)
    processor.process(image, 0, demo_landmarks=landmarks)
    shifted = landmarks + 0.1
    raw = processor.process(image, 33, smoothing=False, demo_landmarks=shifted)
    np.testing.assert_array_equal(raw.landmarks, shifted)
    resumed = processor.process(image, 66, smoothing=True, demo_landmarks=landmarks)
    np.testing.assert_array_equal(resumed.landmarks, landmarks)


def test_worker_stops_without_unbounded_queue():
    worker = Worker(Config(demo=True, max_fps=60))
    worker.start()
    deadline = time.monotonic() + 3
    frame = None
    while frame is None and time.monotonic() < deadline:
        time.sleep(0.01)
        frame, error, _ = worker.take()
        assert error is None
    worker.close()
    assert frame is not None
    assert not worker.thread.is_alive()
    assert worker.finished
