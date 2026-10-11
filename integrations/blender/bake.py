import json

import bpy
from mathutils import Euler

from .animation_protocol import load_animation


def linear_keys(owner):
    animation = owner.animation_data
    if not animation or not animation.action:
        return
    action = animation.action
    if hasattr(action, "fcurves"):
        curves = action.fcurves
    else:
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(animation.action_slot)
                if bag:
                    curves.extend(bag.fcurves)
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def bake_animation(context, payload, mapping, strength, start_frame):
    target = context.scene.meshpipe_target
    root = context.scene.meshpipe_head
    if target is None or target.type != "MESH" or target.data.shape_keys is None:
        raise ValueError("Choose an expression mesh with shape keys")
    keys = target.data.shape_keys
    resolved = {}
    for channel, track in payload["channels"].items():
        name = mapping.get(channel, channel)
        key = keys.key_blocks.get(name)
        if key is not None and name != "Basis":
            if name in resolved:
                raise ValueError("Multiple channels map to the same shape key")
            resolved[name] = (key, track)
    if not resolved:
        raise ValueError("No matching expression keys. Check remapping JSON.")
    owners = [keys] + ([root] if root and payload["rotation"] else [])
    previous = []
    original_values = {name: key.value for name, (key, _) in resolved.items()}
    root_state = None
    success = False
    try:
        for owner in owners:
            owner.animation_data_create()
            previous.append(
                (
                    owner,
                    owner.animation_data.action,
                    getattr(owner.animation_data, "action_slot", None),
                )
            )
            owner.animation_data.action = None
        scene_fps = context.scene.render.fps / context.scene.render.fps_base
        factor = scene_fps / payload["fps"]
        for key, track in resolved.values():
            for frame, value in track:
                key.value = max(key.slider_min, min(key.slider_max, value * strength))
                key.keyframe_insert(data_path="value", frame=start_frame + (frame - 1) * factor)
        if root and payload["rotation"]:
            root_state = (
                root.rotation_mode,
                root.rotation_euler.copy(),
                root.rotation_quaternion.copy(),
                tuple(root.rotation_axis_angle),
            )
            base = root.matrix_basis.to_quaternion()
            tracks = payload["rotation"]
            times = sorted({pair[0] for track in tracks.values() for pair in track})

            def value_at(track, time):
                if time <= track[0][0]:
                    return track[0][1]
                for left, right in zip(track, track[1:], strict=False):
                    if time <= right[0]:
                        fraction = (time - left[0]) / (right[0] - left[0])
                        return left[1] + (right[1] - left[1]) * fraction
                return track[-1][1]

            root.rotation_mode = "XYZ"
            last_euler = root.rotation_euler.copy()
            for frame in times:
                angles = [value_at(tracks.get(axis, [[1, 0]]), frame) for axis in "xyz"]
                root.rotation_euler = (base @ Euler(angles, "XYZ").to_quaternion()).to_euler(
                    "XYZ", last_euler
                )
                last_euler = root.rotation_euler.copy()
                root.keyframe_insert(
                    data_path="rotation_euler", frame=start_frame + (frame - 1) * factor
                )
        for owner in owners:
            if owner.animation_data.action:
                owner.animation_data.action.name = "MeshPipe baked performance"
            linear_keys(owner)
        success = True
    finally:
        for name, value in original_values.items():
            resolved[name][0].value = value
        if root_state:
            (
                root.rotation_mode,
                root.rotation_euler,
                root.rotation_quaternion,
                root.rotation_axis_angle,
            ) = root_state
        if not success:
            for owner, action, slot in previous:
                failed = owner.animation_data.action
                owner.animation_data.action = action
                if slot is not None:
                    owner.animation_data.action_slot = slot
                if failed and failed.users == 0:
                    bpy.data.actions.remove(failed)
    return len(resolved)


class MESHPIPE_OT_bake(bpy.types.Operator):
    bl_idname = "meshpipe.bake"
    bl_label = "Bake Saved Performance"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.scene.meshpipe_running:
            self.report({"ERROR"}, "Stop live receiver before baking")
            return {"CANCELLED"}
        try:
            payload = load_animation(bpy.path.abspath(context.scene.meshpipe_bake_file))
            mapping = json.loads(context.scene.meshpipe_bake_mapping)
            if not isinstance(mapping, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
            ):
                raise ValueError("Remapping must be a JSON object of channel and shape key names")
            count = bake_animation(
                context,
                payload,
                mapping,
                context.scene.meshpipe_bake_strength,
                context.scene.meshpipe_bake_start,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked {count} expression channels into new actions")
        return {"FINISHED"}
