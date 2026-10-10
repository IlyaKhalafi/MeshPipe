import json
import socket
import struct
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from test_session import make_frame

from meshpipe.cartoon import CartoonRig, CartoonTemplate, builtin_template
from meshpipe.explorer import REGIONS, LandmarkHistory, pick_landmark
from meshpipe.expressions import CHANNELS, Calibration
from meshpipe.lab import SmoothingLab, trace_image
from meshpipe.media import ViewRecorder, compose_capture
from meshpipe.playground import ExpressionGame, GestureController
from meshpipe.pose import head_rotation
from meshpipe.quality import Recovery, assess
from meshpipe.stream import LocalStream, osc_message, packet


def test_calibration_fit_and_persistence(tmp_path):
    calibration = Calibration()
    calibration.fit([{name: 0.1 for name in CHANNELS}] * 12, True)
    calibration.fit([{name: 0.8 for name in CHANNELS}] * 12, False)
    assert calibration.apply({name: 0.1 for name in CHANNELS})["jawOpen"] == 0
    assert calibration.apply({name: 0.8 for name in CHANNELS})["jawOpen"] == pytest.approx(1)
    path = tmp_path / "personal.json"
    calibration.save(path)
    assert Calibration.load(path) == calibration
    path.write_text(json.dumps({"version": 1, "dead_zone": 1}))
    with pytest.raises(ValueError):
        Calibration.load(path)


def test_calibration_short_or_missing_channels_rejected():
    with pytest.raises(ValueError):
        Calibration().fit([{}], True)
    with pytest.raises(ValueError):
        Calibration().fit([{}] * 20, True)


def test_template_rgba_animation_and_anchor_validation(tmp_path):
    template = builtin_template()
    rig = CartoonRig(template)
    neutral = rig.render({})
    expressive = rig.render({"jawOpen": 1, "eyeBlinkLeft": 1}, roll=0.2)
    assert neutral.shape == (384, 384, 3)
    assert np.mean(np.abs(neutral.astype(float) - expressive)) > 1
    path = tmp_path / "cartoon.png"
    Image.fromarray(template.rgba).save(path)
    loaded = CartoonTemplate.load(path)
    anchor_path = tmp_path / "anchors.json"
    loaded.save_anchors(anchor_path)
    loaded.load_anchors(anchor_path)
    assert loaded.anchors == template.anchors
    with pytest.raises(ValueError):
        CartoonTemplate(template.rgba, {"left_eye": [0, 0]})


def test_osc_encoding_and_loopback_delivery():
    message = osc_message("/test", (0.25,))
    assert message.startswith(b"/test\0\0\0,f\0\0")
    assert struct.unpack(">f", message[-4:])[0] == 0.25
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        stream = LocalStream(receiver.getsockname()[1])
        frame = make_frame()
        assert stream.send(frame, frame.expressions, 0)
        data = receiver.recv(8192)
        assert data.startswith(b"#bundle\0")
        assert b"/meshpipe/rotation\0" in data
        assert not stream.send(frame, frame.expressions, 0.001)
        stream.close()
    lost = replace(frame, vertices=None, landmarks=None)
    assert b"/meshpipe/rotation" not in packet(lost, {}, 2)


def test_pose_orthonormal():
    rotation = head_rotation(make_frame().vertices)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_gesture_edges_cooldowns_and_loss():
    controller = GestureController()
    scores = {"jawOpen": 0.9}
    assert [event.name for event in controller.update(scores, True, 0)] == ["Open mouth"]
    assert controller.update(scores, True, 1) == []
    controller.update({}, True, 1.1)
    assert len(controller.update(scores, True, 2)) == 1
    assert controller.update(scores, False, 3) == []
    assert len(controller.update(scores, True, 4)) == 1
    assert controller.update({"eyeBlinkLeft": 1, "eyeBlinkRight": 1}, True, 5) == []
    game = ExpressionGame()
    assert not game.hit("Open mouth", 0)
    assert game.hit("Smile", 1)
    assert game.target == "Open mouth"


def test_recovery_fade_and_quality():
    frame = make_frame()
    recovery = Recovery(0.4)
    recovery.update(frame, 0)
    lost = replace(frame, vertices=None, landmarks=None, raw_landmarks=None)
    recovery.update(lost, 1)
    assert recovery.presentation(1.2)[1] == pytest.approx(0.5)
    assert recovery.presentation(2)[0] is None
    recovery.update(frame, 3)
    assert recovery.reacquired
    assert "Face lost" in assess(lost).hint
    assert assess(frame).tracked


def test_filter_lab_and_history():
    lab = SmoothingLab()
    points = make_frame().raw_landmarks
    result = None
    for index in range(40):
        result = lab.update(points + np.sin(index * 0.2) * 0.01, index * 33)
    assert result.raw.shape == result.kalman.shape == result.ema.shape == (468, 3)
    assert all(cost >= 0 for cost in result.costs_ms)
    assert all(lag is not None for lag in result.lag_ms)
    assert trace_image(np.zeros((20, 3))).shape == (100, 300, 3)
    assert lab.update(None, 2000) is None
    assert len(lab.history) == 0


def test_landmark_projection_history_and_export(tmp_path):
    vertices = np.array([[0, 0, 0], [0.5, 0, 0]])
    assert pick_landmark(vertices, np.eye(4), np.eye(4), 100, 100, 50, 50) == 0
    assert pick_landmark(vertices, np.eye(4), np.eye(4), 100, 100, 0, 0) is None
    history = LandmarkHistory()
    frame = make_frame()
    history.update(100, frame.landmarks)
    history.update(50, frame.landmarks)
    assert len(history.values) == 1
    history.select(33)
    assert len(history.values) == 0
    path = tmp_path / "region.json"
    history.export(path, "Right eye")
    assert json.loads(path.read_text())["indices"] == list(REGIONS["Right eye"])


@pytest.mark.parametrize("extension", ["gif", "mp4"])
def test_actual_view_recording(tmp_path, extension):
    source = np.full((120, 160, 3), 100, dtype=np.uint8)
    rendered = np.full((200, 300, 3), [20, 50, 80], dtype=np.uint8)
    assert compose_capture(source, rendered).shape == (540, 960, 3)
    path = tmp_path / f"view.{extension}"
    recorder = ViewRecorder(path)
    assert recorder.submit(source, rendered, 0, "Test")
    assert recorder.submit(source, rendered + 20, 0.2, "Test")
    recorder.close()
    assert recorder.written == 4
    assert path.stat().st_size > 0
    if extension == "gif":
        with Image.open(path) as image:
            assert image.size == (800, 450)


def test_view_recording_failure_surfaces(tmp_path):
    recorder = ViewRecorder(tmp_path / "empty.gif")
    with pytest.raises(RuntimeError, match="No rendered frames"):
        recorder.close()
