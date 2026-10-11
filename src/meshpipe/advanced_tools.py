from open3d.visualization import gui

from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.quality import Recovery
from meshpipe.widgets import button, row, section, slider


class AdvancedTools:
    def __init__(self, tools):
        self.tools = tools
        panel = gui.ScrollableVert(8, gui.Margins(0, 8, 0, 4))
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
