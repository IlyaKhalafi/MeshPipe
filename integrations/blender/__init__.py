"""Install the ZIP built by tools/build_blender_addon.py in Blender."""

import socket
import time

import bpy
from mathutils import Euler

from .osc_protocol import decode_bundle

bl_info = {
    "name": "MeshPipe Local Receiver",
    "author": "MeshPipe contributors",
    "version": (0, 3, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > MeshPipe",
    "description": "Receive local facial motion and bake saved expression performances",
    "category": "Animation",
}

ACTIVE_RECEIVERS = []


class MESHPIPE_OT_receive(bpy.types.Operator):
    bl_idname = "meshpipe.receive"
    bl_label = "Start Local Receiver"
    bl_options = {"REGISTER"}
    _socket = None
    _timer = None
    _object = None
    _root = None
    _base_rotation = None
    _last_packet = 0.0
    _sequence = -1.0

    def execute(self, context):
        if context.scene.meshpipe_running:
            self.report({"WARNING"}, "Receiver already running")
            return {"CANCELLED"}
        target = context.scene.meshpipe_target
        if target is None or target.type != "MESH" or target.data.shape_keys is None:
            self.report({"ERROR"}, "Choose a mesh with named expression shape keys")
            return {"CANCELLED"}
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(("127.0.0.1", context.scene.meshpipe_port))
            self._socket.setblocking(False)
        except OSError as error:
            if self._socket:
                self._socket.close()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self._object = target
        self._driven = {}
        self._scene = context.scene
        self._manager = context.window_manager
        self._root = context.scene.meshpipe_head
        if self._root:
            self._base_rotation = self._root.matrix_basis.to_quaternion()
            self._original_rotation = (
                self._root.rotation_mode,
                self._root.rotation_euler.copy(),
                self._root.rotation_quaternion.copy(),
                tuple(self._root.rotation_axis_angle),
            )
        self._last_packet = time.monotonic()
        self._sequence = -1.0
        self._timer = context.window_manager.event_timer_add(1 / 30, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.scene.meshpipe_running = True
        ACTIVE_RECEIVERS.append(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if self._socket is None or not self._scene.meshpipe_running or event.type == "ESC":
            self.cancel(context)
            return {"CANCELLED"}
        if event.type != "TIMER" or event.timer != self._timer:
            return {"PASS_THROUGH"}
        try:
            present = self._object.name in bpy.data.objects
        except ReferenceError:
            present = False
        if not present:
            self.cancel(context)
            return {"CANCELLED"}
        latest = None
        try:
            for _ in range(64):
                try:
                    data, _ = self._socket.recvfrom(8193)
                except BlockingIOError:
                    break
                try:
                    candidate = decode_bundle(data)
                except ValueError:
                    continue
                frame = candidate.get("/meshpipe/frame", ())
                if len(frame) == 2 and frame[0] > self._sequence:
                    latest = candidate
                    self._sequence = frame[0]
        except OSError:
            self.cancel(context)
            return {"CANCELLED"}
        if latest:
            self._last_packet = time.monotonic()
            tracked = latest.get("/meshpipe/tracked", (0,)) == (1.0,)
            self._apply(latest if tracked else {})
        elif time.monotonic() - self._last_packet > 0.5:
            self._apply({})
            self._sequence = -1.0
        return {"PASS_THROUGH"}

    def _apply(self, messages):
        keys = self._object.data.shape_keys
        if keys is None:
            return
        for key in keys.key_blocks:
            if key.name == "Basis":
                continue
            value = messages.get("/meshpipe/blendshape/" + key.name)
            if value and len(value) == 1:
                self._driven.setdefault(key.name, key.value)
                key.value = max(0.0, min(1.0, value[0]))
            elif key.name in self._driven:
                key.value = 0.0
        rotation = messages.get("/meshpipe/rotation")
        try:
            root_present = self._root is not None and self._root.name in bpy.data.objects
        except ReferenceError:
            root_present = False
        if root_present and self._base_rotation:
            quaternion = self._base_rotation.copy()
            if rotation and len(rotation) == 3:
                quaternion = quaternion @ Euler(rotation, "XYZ").to_quaternion()
            self._root.rotation_mode = "QUATERNION"
            self._root.rotation_quaternion = quaternion

    def cancel(self, context):
        if self in ACTIVE_RECEIVERS:
            ACTIVE_RECEIVERS.remove(self)
        if hasattr(self, "_scene"):
            self._scene.meshpipe_running = False
        if self._timer:
            self._manager.event_timer_remove(self._timer)
            self._timer = None
        if self._socket:
            self._socket.close()
            self._socket = None
        try:
            if self._object and self._object.data.shape_keys:
                for name, original in self._driven.items():
                    key = self._object.data.shape_keys.key_blocks.get(name)
                    if key is not None:
                        key.value = original
            if self._root and hasattr(self, "_original_rotation"):
                mode, euler, quaternion, axis_angle = self._original_rotation
                self._root.rotation_euler = euler
                self._root.rotation_quaternion = quaternion
                self._root.rotation_axis_angle = axis_angle
                self._root.rotation_mode = mode
        except ReferenceError:
            pass


class MESHPIPE_OT_stop(bpy.types.Operator):
    bl_idname = "meshpipe.stop"
    bl_label = "Stop Receiver"

    def execute(self, context):
        for receiver in list(ACTIVE_RECEIVERS):
            if receiver._scene == context.scene:
                receiver.cancel(context)
        return {"FINISHED"}


class MESHPIPE_PT_panel(bpy.types.Panel):
    bl_label = "MeshPipe"
    bl_idname = "MESHPIPE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MeshPipe"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        column = layout.column()
        column.enabled = not scene.meshpipe_running
        column.prop(scene, "meshpipe_target")
        column.prop(scene, "meshpipe_head")
        column.prop(scene, "meshpipe_port")
        layout.label(text="127.0.0.1 only. No video transmitted.")
        layout.operator("meshpipe.stop" if scene.meshpipe_running else "meshpipe.receive")
        layout.separator()
        layout.label(text="Offline performance baking")
        for name in (
            "meshpipe_bake_file",
            "meshpipe_bake_mapping",
            "meshpipe_bake_strength",
            "meshpipe_bake_start",
        ):
            layout.prop(scene, name)
        layout.operator("meshpipe.bake")


CLASSES = (MESHPIPE_OT_receive, MESHPIPE_OT_stop, MESHPIPE_PT_panel)


def register():
    from .bake import MESHPIPE_OT_bake

    bpy.utils.register_class(MESHPIPE_OT_bake)
    bpy.types.Scene.meshpipe_bake_file = bpy.props.StringProperty(
        name="Animation JSON", subtype="FILE_PATH"
    )
    bpy.types.Scene.meshpipe_bake_mapping = bpy.props.StringProperty(
        name="Remap JSON", default="{}"
    )
    bpy.types.Scene.meshpipe_bake_strength = bpy.props.FloatProperty(
        name="Strength", default=1, min=0, max=3
    )
    bpy.types.Scene.meshpipe_bake_start = bpy.props.IntProperty(
        name="Start frame", default=1, min=1
    )
    for plugin in CLASSES:
        bpy.utils.register_class(plugin)
    bpy.types.Scene.meshpipe_target = bpy.props.PointerProperty(
        name="Expression mesh", type=bpy.types.Object
    )
    bpy.types.Scene.meshpipe_head = bpy.props.PointerProperty(
        name="Head root", type=bpy.types.Object
    )
    bpy.types.Scene.meshpipe_port = bpy.props.IntProperty(
        name="OSC port", default=9000, min=1024, max=65535
    )
    bpy.types.Scene.meshpipe_running = bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})


def unregister():
    from .bake import MESHPIPE_OT_bake

    bpy.utils.unregister_class(MESHPIPE_OT_bake)
    for name in (
        "meshpipe_bake_file",
        "meshpipe_bake_mapping",
        "meshpipe_bake_strength",
        "meshpipe_bake_start",
    ):
        delattr(bpy.types.Scene, name)
    for receiver in list(ACTIVE_RECEIVERS):
        receiver.cancel(bpy.context)
    for scene in bpy.data.scenes:
        scene.meshpipe_running = False
    for plugin in reversed(CLASSES):
        bpy.utils.unregister_class(plugin)
    for name in ("meshpipe_target", "meshpipe_head", "meshpipe_port", "meshpipe_running"):
        delattr(bpy.types.Scene, name)
