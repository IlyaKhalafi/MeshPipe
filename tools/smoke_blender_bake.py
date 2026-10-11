"""Validate offline baking with an actual standalone bpy installation."""

import importlib
import json
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]


def run_case(mode, plugin):
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new("MeshPipe bake test")
    mesh.from_pydata([(-1, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    face = bpy.data.objects.new("Expression test", mesh)
    scene.collection.objects.link(face)
    face.shape_key_add(name="Basis")
    mouth = face.shape_key_add(name="MouthOpen")
    mouth.data[2].co.y += 0.3
    mouth.value = 0.3
    mouth.keyframe_insert(data_path="value", frame=1)
    custom = face.shape_key_add(name="CustomPose")
    custom.value = 0.7
    custom.keyframe_insert(data_path="value", frame=1)
    owner = mesh.shape_keys
    original_action = owner.animation_data.action
    root = bpy.data.objects.new("Head root test", None)
    scene.collection.objects.link(root)
    root.rotation_mode = mode
    root.rotation_euler.z = 0.2
    path = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(
        mode, "rotation_euler"
    )
    root.keyframe_insert(data_path=path, frame=1)
    original_root_action = root.animation_data.action
    scene.meshpipe_target, scene.meshpipe_head = face, root
    scene.render.fps, scene.render.fps_base = 60, 1
    payload = {
        "version": 1,
        "fps": 30,
        "channels": {"jawOpen": [[1, 0.2], [16, 0.9], [31, 0.1]]},
        "rotation": {
            "x": [[1, 0], [16, 0.2], [31, 0]],
            "y": [[1, 0], [16, 0.1], [31, 0]],
            "z": [[1, 0], [16, 0.05], [31, 0]],
        },
    }
    count = plugin.bake_animation(bpy.context, payload, {"jawOpen": "MouthOpen"}, 0.8, 10)
    assert count == 1
    assert abs(mouth.value - 0.3) < 1e-6 and abs(custom.value - 0.7) < 1e-6
    assert root.rotation_mode == mode
    assert owner.animation_data.action != original_action
    assert root.animation_data.action != original_root_action
    curves = owner.animation_data.action.fcurves
    assert len(curves) == 1 and "MouthOpen" in curves[0].data_path
    assert [round(point.co.x) for point in curves[0].keyframe_points] == [10, 40, 70]
    assert all(point.interpolation == "LINEAR" for point in curves[0].keyframe_points)
    assert all(curve.data_path == path for curve in root.animation_data.action.fcurves)
    scene.frame_set(10)
    assert abs(mouth.value - 0.16) < 1e-6
    scene.frame_set(40)
    assert abs(mouth.value - 0.72) < 1e-6
    assert abs(custom.value - 0.7) < 1e-6
    print(f"Blender {bpy.app.version_string} / {mode} baking passed")


def main():
    builder = runpy.run_path(str(ROOT / "tools/build_blender_addon.py"))["build"]
    with tempfile.TemporaryDirectory(prefix="meshpipe-blender-") as directory:
        root = Path(directory)
        archive = builder(root / "addon.zip")
        with zipfile.ZipFile(archive) as package:
            package.extractall(root)
        sys.path.insert(0, str(root))
        addon = importlib.import_module("meshpipe_receiver")
        addon.register()
        try:
            plugin = importlib.import_module("meshpipe_receiver.bake")
            for mode in ("ZYX", "QUATERNION", "AXIS_ANGLE"):
                run_case(mode, plugin)
            path = root / "motion.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "fps": 30,
                        "channels": {"jawOpen": [[1, 0.2], [31, 0.8]]},
                        "rotation": {},
                    }
                ),
                encoding="utf8",
            )
            scene = bpy.context.scene
            scene.meshpipe_bake_file = str(path)
            scene.meshpipe_bake_mapping = '{"jawOpen":"MouthOpen"}'
            assert bpy.ops.meshpipe.bake() == {"FINISHED"}
            print("Packaged add-on registration and bake operator passed")
        finally:
            addon.unregister()


if __name__ == "__main__":
    main()
