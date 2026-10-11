import importlib.util
import struct
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_session import make_frame

from meshpipe import osc_protocol
from meshpipe.cartoon import CartoonRig, CartoonTemplate, builtin_template
from meshpipe.cli import main, parser
from meshpipe.config import Config
from meshpipe.osc_protocol import decode_bundle, osc_message
from meshpipe.playground import GestureController
from meshpipe.session import SessionReader, SessionRecorder
from meshpipe.stream import packet


def bundle(message):
    return b"#bundle\0" + struct.pack(">Q", 1) + struct.pack(">I", len(message)) + message


def test_osc_roundtrip_and_face_loss():
    frame = make_frame()
    decoded = decode_bundle(packet(frame, {"jawOpen": 0.7}, 12))
    assert decoded["/meshpipe/frame"][0] == 12
    assert decoded["/meshpipe/blendshape/jawOpen"][0] == pytest.approx(0.7)
    frame = replace(frame, vertices=None)
    decoded = decode_bundle(packet(frame, {}, 13))
    assert decoded["/meshpipe/tracked"] == (0.0,)
    assert "/meshpipe/rotation" not in decoded


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"#bundle\0" * 1100,
        bundle(b"broken!"),
        bundle(osc_message("/other", (1,))),
        bundle(osc_message("/meshpipe/test", tuple(range(17)))),
        bundle(osc_message("/meshpipe/test", (1,)))[:-1],
    ],
)
def test_malformed_osc_rejected(payload):
    with pytest.raises(ValueError):
        decode_bundle(payload)


def test_nonfinite_osc_rejected():
    payload = bundle(osc_message("/meshpipe/test", (1,)))
    with pytest.raises(ValueError):
        decode_bundle(payload[:-4] + struct.pack(">f", float("nan")))


@pytest.mark.parametrize(
    "anchors",
    [
        None,
        [],
        {"left_eye": None, "right_eye": [0.3, 0.4], "mouth": [0.5, 0.7]},
        {"left_eye": ["bad", 0.4], "right_eye": [0.3, 0.4], "mouth": [0.5, 0.7]},
    ],
)
def test_malformed_cartoon_anchors_rejected(anchors):
    with pytest.raises(ValueError):
        CartoonTemplate(builtin_template().rgba, anchors)


def test_turn_gestures_require_release():
    controller = GestureController()
    assert controller.update({}, True, 0, -0.4)[0].name == "Turn left"
    assert controller.update({}, True, 1, -0.4) == []
    controller.update({}, True, 2, 0)
    assert controller.update({}, True, 3, 0.4)[0].name == "Turn right"


def test_cartoon_transparency_and_fade():
    rig = CartoonRig(builtin_template())
    image = rig.render({}, rgba=True)
    assert image.shape == (384, 384, 4)
    assert image[:, :, 3].min() == 0 and image[:, :, 3].max() == 255
    assert rig.render({}, opacity=0, rgba=True)[:, :, 3].max() == 0


def test_workbench_cli_and_invalid_config(capsys, tmp_path):
    options = parser().parse_args(["run", "--replay", "take.meshpipe", "--osc-port", "9000"])
    assert options.replay == Path("take.meshpipe") and options.osc_port == 9000
    for arguments in [
        {"osc_port": 1000},
        {"osc_port": 65536},
        {"replay": Path("take"), "demo": True},
    ]:
        with pytest.raises(ValueError):
            Config(**arguments)
    assert main(["run", "--replay", str(tmp_path / "missing.meshpipe"), "--demo"]) == 1
    assert "Replay and synthetic demo cannot be used together" in capsys.readouterr().err


def test_session_close_idempotent_and_non_database_rejected(tmp_path):
    recorder = SessionRecorder(tmp_path / "take.meshpipe")
    recorder.submit(make_frame())
    assert recorder.close() == recorder.close()
    bad = tmp_path / "bad.meshpipe"
    bad.write_bytes(b"not sqlite")
    with pytest.raises(ValueError, match="database"):
        SessionReader(bad)


def test_blender_preserves_unrelated_keys_and_restores_driven_keys(monkeypatch):
    plugin_name = "meshpipe_receiver_test"
    monkeypatch.setitem(
        sys.modules,
        "bpy",
        SimpleNamespace(
            types=SimpleNamespace(Operator=object, Panel=object), data=SimpleNamespace(objects={})
        ),
    )
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=object))
    monkeypatch.setitem(sys.modules, plugin_name + ".osc_protocol", osc_protocol)
    path = Path(__file__).resolve().parents[1] / "integrations/blender/__init__.py"
    spec = importlib.util.spec_from_file_location(plugin_name, path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    class Keys(dict):
        def __iter__(self):
            return iter(self.values())

    keys = Keys(
        {
            name: SimpleNamespace(name=name, value=value)
            for name, value in [("Basis", 0), ("jawOpen", 0.2), ("CustomPose", 0.8)]
        }
    )
    receiver = plugin.MESHPIPE_OT_receive()
    receiver._object = SimpleNamespace(
        data=SimpleNamespace(shape_keys=SimpleNamespace(key_blocks=keys))
    )
    receiver._driven = {}
    receiver._scene = SimpleNamespace(meshpipe_running=True)
    receiver._apply({"/meshpipe/blendshape/jawOpen": (0.9,)})
    assert keys["jawOpen"].value == 0.9
    assert keys["CustomPose"].value == 0.8
    receiver._apply({})
    assert keys["jawOpen"].value == 0
    assert keys["CustomPose"].value == 0.8
    receiver.cancel(None)
    assert keys["jawOpen"].value == 0.2
    assert not receiver._scene.meshpipe_running


def test_blender_archive_contains_compilable_shared_codec(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_addon", root / "tools/build_blender_addon.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = builder.build(tmp_path / "receiver.zip")
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "meshpipe_receiver/__init__.py",
            "meshpipe_receiver/osc_protocol.py",
        }
        for name in archive.namelist():
            compile(archive.read(name), name, "exec")
        assert (
            archive.read("meshpipe_receiver/osc_protocol.py")
            == (root / "src/meshpipe/osc_protocol.py").read_bytes()
        )
