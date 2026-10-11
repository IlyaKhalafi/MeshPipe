import json
import struct

import numpy as np
import pytest

from meshpipe.avatar import load_glb


def write_avatar(path, mutate=lambda doc: None):
    points = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    delta = np.zeros_like(points)
    delta[2, 1] = 1
    binary = points.tobytes() + delta.tobytes()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36},
        ],
        "accessors": [
            {"bufferView": index, "componentType": 5126, "count": 3, "type": "VEC3"}
            for index in range(2)
        ],
        "meshes": [
            {
                "extras": {"targetNames": ["jawOpen"]},
                "primitives": [{"attributes": {"POSITION": 0}, "targets": [{"POSITION": 1}]}],
            }
        ],
        "nodes": [{"mesh": 0, "translation": [0, 2, 0]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    mutate(document)
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    contents = struct.pack("<II", len(text), 0x4E4F534A) + text
    contents += struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, len(contents) + 12) + contents)


def test_glb_morph_mapping_and_presets(tmp_path):
    path = tmp_path / "head.glb"
    write_avatar(path)
    avatar = load_glb(path)
    assert avatar.names == ["jawOpen"]
    part = avatar.parts[0]
    shifted = avatar.deform(part, {"jawOpen": 0.5})
    assert shifted[2, 1] > part.vertices[2, 1]
    avatar.set_mapping({"jawOpen": {"channel": "smile", "strength": 2}})
    np.testing.assert_allclose(avatar.deform(part, {"smile": 0.25}), shifted)
    preset = tmp_path / "rig.json"
    avatar.save_preset(preset)
    avatar.mapping = {}
    avatar.load_preset(preset)
    assert avatar.mapping["jawOpen"]["channel"] == "smile"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["bufferViews"][0].update(byteLength=9999),
        lambda doc: doc["nodes"][0].update(children=[0]),
        lambda doc: doc["nodes"][0].update(skin=0),
        lambda doc: doc.update(extensionsRequired=["KHR_draco_mesh_compression"]),
        lambda doc: doc["accessors"][0].update(count=99999),
        lambda doc: doc["meshes"][0].update(extras={}),
    ],
    ids=["bounds", "cycle", "skin", "compression", "count", "names"],
)
def test_invalid_glb_rejected(tmp_path, mutate):
    path = tmp_path / "head.glb"
    write_avatar(path, mutate)
    with pytest.raises(ValueError):
        load_glb(path)


def test_truncated_glb(tmp_path):
    path = tmp_path / "head.glb"
    path.write_bytes(b"glTF")
    with pytest.raises(ValueError):
        load_glb(path)


def test_bundled_morph_head():
    from importlib.resources import as_file, files

    with as_file(files("meshpipe").joinpath("assets/example-head.glb")) as path:
        avatar = load_glb(path)
    assert len(avatar.names) == 5
    for part in avatar.parts:
        assert np.isfinite(avatar.deform(part, {"jawOpen": 1})).all()
