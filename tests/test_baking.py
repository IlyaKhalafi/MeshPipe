import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meshpipe.animation_protocol import load_animation, reduce_keys, validate_animation
from meshpipe.baking import export_animation
from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.session import SessionReader

pytest_plugins = ["test_motion"]


def test_reduction_bounds():
    times = np.arange(100).tolist()
    values = np.sin(np.arange(100) / 10).tolist()
    reduced = reduce_keys(times, values, 0.01)
    assert len(reduced) < len(times)
    error = np.abs(np.interp(times, *np.asarray(reduced).T) - values)
    assert error.max() <= 0.01


def test_bake_edited_take(performance, tmp_path):
    reader = EditedReader(SessionReader(performance))
    reader.set_edit(MotionEdit(start_ms=100, end_ms=300, strength=0.5))
    path = export_animation(reader, tmp_path / "motion.json", fps=60, tolerance=0)
    payload = load_animation(path)
    assert payload["source_kind"] == "synthetic"
    assert payload["channels"]["jawOpen"][0] == [1, 0.1]
    assert payload["channels"]["jawOpen"][-1][0] == 13
    assert set(payload["rotation"]) == {"x", "y", "z"}
    reader.close()


def test_animation_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_animation({"version": 1, "fps": 30, "channels": {"jawOpen": [[1, float("nan")]]}})
    with pytest.raises(ValueError):
        reduce_keys([0, 0], [0, 1])


def test_blender_bake_restores_values_and_preserves_old_action(monkeypatch):
    from meshpipe import animation_protocol

    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(types=SimpleNamespace(Operator=object)))
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=object))
    monkeypatch.setitem(sys.modules, "bake_test.animation_protocol", animation_protocol)
    path = Path(__file__).resolve().parents[1] / "integrations/blender/bake.py"
    spec = importlib.util.spec_from_file_location("bake_test.bake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_action = object()
    owner = SimpleNamespace(
        animation_data=SimpleNamespace(action=old_action), animation_data_create=lambda: None
    )
    inserted = []

    def insert(**kwargs):
        owner.animation_data.action = SimpleNamespace(name="", fcurves=[])
        inserted.append((kwargs["frame"], key.value))

    key = SimpleNamespace(value=0.3, slider_min=0, slider_max=1, keyframe_insert=insert)
    owner.key_blocks = {"MouthOpen": key}
    target = SimpleNamespace(type="MESH", data=SimpleNamespace(shape_keys=owner))
    context = SimpleNamespace(
        scene=SimpleNamespace(
            meshpipe_target=target, meshpipe_head=None, render=SimpleNamespace(fps=60, fps_base=1)
        )
    )
    payload = {"fps": 30, "channels": {"jawOpen": [[1, 0.2], [31, 0.8]]}, "rotation": {}}
    assert module.bake_animation(context, payload, {"jawOpen": "MouthOpen"}, 0.5, 10) == 1
    assert inserted == [(10, 0.1), (70, 0.4)]
    assert key.value == 0.3
    assert owner.animation_data.action is not old_action
