from open3d.visualization import gui

from meshpipe.accessories import AccessoryAnchor
from meshpipe.avatar import load_glb
from meshpipe.jobs import ExportJob, MotionSource
from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.quality import Recovery
from meshpipe.widgets import button, combo, row, section, slider


class AdvancedTools:
    def __init__(self, tools):
        self.tools = tools
        self.job = None
        panel = gui.ScrollableVert(8, gui.Margins(0, 8, 0, 4))
        panel.background_color = gui.Color(0.055, 0.095, 0.115)
        self.trim_start = slider(0, 600000, 0, lambda _: None)
        self.trim_end = slider(0, 600000, 600000, lambda _: None)
        self.strength = slider(0, 3, 1, lambda _: None)
        self.fade = slider(0, 5000, 0, lambda _: None)
        self.despike = gui.Checkbox("Remove single-frame spikes")
        self.blend = slider(0, 1, 0, lambda _: None)
        self.channel = gui.TextEdit()
        self.channel.text_value = "jawOpen"
        self.curve_start = slider(0, 3, 1, lambda _: None)
        self.curve_mid = slider(0, 3, 1, lambda _: None)
        self.curve_end = slider(0, 3, 1, lambda _: None)
        safe = tools._safe
        panel.add_child(
            section(
                "01  MOTION EDITOR",
                gui.Label("Open a performance in Create first."),
                gui.Label("Trim start / end in ms"),
                self.trim_start,
                self.trim_end,
                gui.Label("Expression strength"),
                self.strength,
                gui.Label("Fade in / out in ms"),
                self.fade,
                self.despike,
                button(
                    "Load blend performance",
                    lambda: tools._file(
                        "Blend performance", ".meshpipe", lambda path: self.reader().set_blend(path)
                    ),
                ),
                gui.Label("Blend weight"),
                self.blend,
                gui.Label("Expression curve channel"),
                self.channel,
                gui.Label("Gain at start / midpoint / end"),
                self.curve_start,
                self.curve_mid,
                self.curve_end,
                button("Apply edit", safe(self.apply)),
                row(
                    button("Undo", safe(lambda: self.history(False))),
                    button("Redo", safe(lambda: self.history(True))),
                ),
                row(
                    button("Save edit", safe(self.save)),
                    button(
                        "Load edit", lambda: tools._file("Load motion edit", ".json", self.load)
                    ),
                ),
            )
        )
        tools.tabs.add_tab("Motion", panel)
        self.panel = panel
        if tools.studio.playback:
            self.sync()
        panel.add_child(
            section(
                "BROWSER SHOWCASE",
                gui.Label("Share motion without source images."),
                button("Export browser studio", safe(self.export_web)),
            )
        )
        style_panel = gui.ScrollableVert(8, gui.Margins(0, 8, 0, 4))
        style_panel.background_color = gui.Color(0.055, 0.095, 0.115)
        tools.tabs.add_tab("Style", style_panel)
        self.accessory_name = combo(["Glasses", "Crown", "Brows"], lambda *_: None)
        self.accessory_index = gui.NumberEdit(gui.NumberEdit.INT)
        self.accessory_index.set_limits(0, 467)
        self.accessory_index.int_value = 168
        self.accessory_scale = slider(0.1, 3, 1, lambda _: None)
        self.accessory_offsets = [slider(-3, 3, value, lambda _: None) for value in (0, 0, 0.18)]
        style_panel.add_child(
            section(
                "FACE ACCESSORIES",
                self.accessory_name,
                gui.Label("Anchor landmark"),
                self.accessory_index,
                gui.Label("Scale"),
                self.accessory_scale,
                gui.Label("Local offsets x / y / depth"),
                *self.accessory_offsets,
                button("Apply accessory", safe(self.apply_accessory)),
                row(
                    button("Save anchor", safe(self.save_accessory)),
                    button(
                        "Load anchor",
                        lambda: tools._file("Load accessory anchor", ".json", self.load_accessory),
                    ),
                ),
                gui.Label("Face surface provides real depth occlusion."),
            )
        )
        self.heat_limit = slider(
            0.005, 0.2, 0.08, lambda value: setattr(tools.studio.content.heatmap, "limit", value)
        )
        style_panel.add_child(
            section(
                "EXPRESSION DISPLACEMENT",
                gui.Label("Cyan quiet. Amber active. Coral high."),
                button("Capture neutral and show heatmap", safe(self.neutral)),
                gui.Label("Color limit / eye-distance fraction"),
                self.heat_limit,
                gui.Label("Rigid head motion removed. Not an emotion score."),
            )
        )
        self.bake_fps = slider(1, 120, 30, lambda _: None)
        self.bake_tolerance = slider(0, 0.05, 0.002, lambda _: None)
        panel.add_child(
            section(
                "OFFLINE BLENDER BAKE",
                gui.Label("Expression and head motion JSON"),
                gui.Label("Animation FPS"),
                self.bake_fps,
                gui.Label("Linear key reduction error"),
                self.bake_tolerance,
                button("Export Blender motion", safe(self.export_bake)),
            )
        )
        self.target = gui.Combobox()
        self.rig_channel = gui.TextEdit()
        self.rig_channel.text_value = "jawOpen"
        self.rig_strength = slider(0, 3, 1, lambda _: None)
        style_panel.add_child(
            section(
                "02  CUSTOM GLB AVATAR",
                gui.Label("Named morph-target head. No skeleton required."),
                button("Load GLB", lambda: tools._file("Load avatar", ".glb", self.load_avatar)),
                button("Try canonical morph head", safe(self.builtin_avatar)),
                gui.Label("Morph target"),
                self.target,
                gui.Label("Expression channel"),
                self.rig_channel,
                gui.Label("Mapping strength"),
                self.rig_strength,
                button("Map target", safe(self.map_target)),
                row(
                    button("Save rig", safe(self.save_rig)),
                    button(
                        "Load rig",
                        lambda: tools._file(
                            "Load rig preset", ".json", lambda path: self.avatar().load_preset(path)
                        ),
                    ),
                ),
            )
        )

    def avatar(self):
        avatar = self.tools.studio.content.custom_avatar
        if avatar is None:
            raise ValueError("Load a GLB avatar first")
        return avatar

    def apply_accessory(self):
        anchor = AccessoryAnchor(
            self.accessory_name.selected_text,
            self.accessory_index.int_value,
            self.accessory_scale.double_value,
            tuple(control.double_value for control in self.accessory_offsets),
        )
        self.tools.studio.content.accessory.configure(anchor)
        studio = self.tools.studio
        studio.mode.selected_index = studio.modes.index("Accessories")
        studio._mode_changed("Accessories", studio.mode.selected_index)
        self.tools.notice.text = "Accessory attached. Rotate face to inspect depth."

    def save_accessory(self):
        from meshpipe.studio_tools import export_path

        path = export_path("accessory-anchor", ".json")
        self.tools.studio.content.accessory.anchor.save(path)
        self.tools.notice.text = f"Saved {path.name}"

    def load_accessory(self, path):
        anchor = AccessoryAnchor.load(path)
        self.accessory_name.selected_index = ["Glasses", "Crown", "Brows"].index(anchor.name)
        self.accessory_index.int_value = anchor.landmark
        self.accessory_scale.double_value = anchor.scale
        for control, value in zip(self.accessory_offsets, anchor.offset, strict=True):
            control.double_value = value
        self.apply_accessory()

    def neutral(self):
        studio = self.tools.studio
        studio.content.heatmap.capture(studio.frame.vertices if studio.frame else None)
        studio.mode.selected_index = studio.modes.index("Heatmap")
        studio._mode_changed("Heatmap", studio.mode.selected_index)
        self.tools.notice.text = "Neutral geometry captured. Move jaw or brows."

    def export_bake(self):
        from meshpipe.baking import export_animation
        from meshpipe.studio_tools import export_path

        path = export_path("blender-motion", ".json")
        fps, tolerance = self.bake_fps.double_value, self.bake_tolerance.double_value
        self.start_export(lambda reader: export_animation(reader, path, fps, tolerance))

    def export_web(self):
        from meshpipe.studio_tools import export_path
        from meshpipe.web_export import export_showcase

        path = export_path("browser-studio", "")
        self.start_export(lambda reader: export_showcase(reader, path))

    def start_export(self, action):
        if self.job is not None:
            raise ValueError("Wait for the current motion export to finish")
        self.job = ExportJob(MotionSource.snapshot(self.reader()), action)
        self.tools.notice.text = "Exporting motion in background. Replay stays responsive."

    def tick(self):
        if self.job is not None and self.job.done.is_set():
            job, self.job = self.job, None
            self.tools.notice.text = job.error or f"Saved {job.result.name}"
            return True
        return False

    def close(self):
        if self.job:
            self.job.close()

    def load_avatar(self, path):
        avatar = load_glb(path)
        self.tools.studio.content.custom_avatar = avatar
        self.target.clear_items()
        for name in avatar.names:
            self.target.add_item(name)
        studio = self.tools.studio
        studio.mode.selected_index = studio.modes.index("Custom Avatar")
        studio._mode_changed("Custom Avatar", studio.mode.selected_index)
        self.tools.notice.text = f"Avatar loaded. {len(avatar.names)} morph targets."

    def builtin_avatar(self):
        from importlib.resources import as_file, files

        with as_file(files("meshpipe").joinpath("assets/example-head.glb")) as path:
            self.load_avatar(path)

    def map_target(self):
        avatar = self.avatar()
        if not avatar.names:
            raise ValueError("GLB has no named expression morph targets")
        mapping = dict(avatar.mapping)
        mapping[self.target.selected_text] = {
            "channel": self.rig_channel.text_value,
            "strength": self.rig_strength.double_value,
        }
        avatar.set_mapping(mapping)
        self.tools.notice.text = "Rig mapping updated"

    def save_rig(self):
        from meshpipe.studio_tools import export_path

        path = export_path("avatar-rig", ".json")
        self.avatar().save_preset(path)
        self.tools.notice.text = f"Saved {path.name}"

    def reader(self):
        playback = self.tools.studio.playback
        if playback is None:
            raise ValueError("Open a performance before editing motion")
        if not isinstance(playback.reader, EditedReader):
            playback.reader = EditedReader(playback.reader)
        return playback.reader

    def refresh(self):
        playback = self.tools.studio.playback
        playback.seek(0)
        self.tools.timeline.set_limits(0, max(1, playback.reader.duration_ms))
        self.tools.recovery = Recovery()
        self.tools.notice.text = "Edit applied. Original recording unchanged."

    def apply(self):
        reader = self.reader()
        duration = reader.base.duration_ms
        start = min(self.trim_start.double_value, duration)
        end = min(self.trim_end.double_value, duration)
        curves = dict(reader.edit.curves)
        if self.channel.text_value.strip() and duration:
            curves[self.channel.text_value.strip()] = [
                [0, self.curve_start.double_value],
                [duration / 2, self.curve_mid.double_value],
                [duration, self.curve_end.double_value],
            ]
        edit = MotionEdit(
            start,
            end,
            self.strength.double_value,
            self.fade.double_value,
            self.despike.checked,
            self.blend.double_value,
            curves,
        )
        reader.set_edit(edit)
        self.refresh()

    def history(self, redo):
        self.reader().undo(redo)
        self.sync()
        self.refresh()

    def sync(self):
        reader = self.reader()
        edit = reader.edit
        duration = max(1, reader.base.duration_ms)
        self.trim_start.set_limits(0, duration)
        self.trim_end.set_limits(0, duration)
        self.trim_start.double_value = edit.start_ms
        self.trim_end.double_value = reader.base.duration_ms if edit.end_ms is None else edit.end_ms
        self.strength.double_value = edit.strength
        self.fade.double_value = edit.fade_ms
        self.despike.checked = edit.despike
        self.blend.double_value = edit.blend_weight
        curve = edit.curves.get(self.channel.text_value)
        if curve:
            self.curve_start.double_value = curve[0][1]
            self.curve_mid.double_value = curve[len(curve) // 2][1]
            self.curve_end.double_value = curve[-1][1]

    def save(self):
        from meshpipe.studio_tools import export_path

        reader = self.reader()
        path = export_path("motion-edit", ".json")
        reader.edit.save(path, reader.base.duration_ms)
        self.tools.notice.text = f"Saved {path.name}"

    def load(self, path):
        reader = self.reader()
        reader.set_edit(MotionEdit.load(path, reader.base.duration_ms))
        self.sync()
        self.refresh()
