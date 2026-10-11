import math
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from meshpipe.config import Config
from meshpipe.export import export_frame
from meshpipe.pipeline import Frame, Worker
from meshpipe.scene import FaceScene
from meshpipe.session import Playback, SessionReader, private_preview
from meshpipe.studio_tools import StudioTools
from meshpipe.widgets import button, combo


class Studio:
    def __init__(self, config: Config):
        self.config = config
        self.playback = (
            Playback(SessionReader(config.replay), loop=config.loop) if config.replay else None
        )
        self.worker = None if self.playback else Worker(config)
        self.frame: Frame | None = None
        self.paused = False
        self.spin = False
        self.angle = 0.0
        self.failed = False
        self.closed = False
        self.show_tools = True
        self.last_tick = time.monotonic()
        title_font = gui.Application.instance.add_font(
            gui.FontDescription("monospace", gui.FontStyle.BOLD, 28)
        )
        self.window = gui.Application.instance.create_window(
            "MeshPipe | Face Mesh Studio", 1480, 900
        )
        self.window.set_on_layout(self._layout)
        self.window.set_on_tick_event(self._tick)
        self.window.set_on_close(self._close)
        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.window.renderer)
        self.view.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.content = FaceScene(self.view.scene)
        self._reset_camera()
        self.sidebar = gui.ScrollableVert(8, gui.Margins(18, 16, 18, 16))
        self.sidebar.background_color = gui.Color(0.07, 0.12, 0.15)
        title = gui.Label("MESHPIPE")
        title.font_id = title_font
        title.text_color = gui.Color(0.1, 0.65, 0.48)
        self.sidebar.add_child(title)
        self.sidebar.add_child(gui.Label("FACE MESH STUDIO  /  CPU"))
        self.sidebar.add_child(gui.Label("01  INPUT"))
        self.preview = gui.ImageWidget()
        self.sidebar.add_child(self.preview)
        source = (
            "Recorded performance"
            if config.replay
            else "Synthetic demo"
            if config.demo
            else (
                f"Webcam {config.source}" if config.source.isdecimal() else Path(config.source).name
            )
        )
        self.sidebar.add_child(gui.Label(source[:38]))
        self.status = gui.Label("Starting pipeline...")
        self.sidebar.add_child(self.status)
        self.metrics = gui.Label("468 landmarks  /  single-face tracking")
        self.sidebar.add_child(self.metrics)
        self.sidebar.add_child(gui.Label("02  VIEW"))
        self.mode = gui.Combobox()
        self.modes = [
            "Surface",
            "Wireframe",
            "Points",
            "Source Color",
            "Textured",
            "Emoji",
            "Cartoon",
            "Custom Avatar",
            "Heatmap",
            "Accessories",
        ]
        for item in self.modes:
            self.mode.add_item(item)
        self.mode.set_on_selection_changed(self._mode_changed)
        self.sidebar.add_child(self.mode)
        palette = gui.Combobox()
        self.palette = palette
        for item in ["Mint", "Amber", "Cloud"]:
            palette.add_item(item)
        palette.set_on_selection_changed(self._palette_changed)
        self.sidebar.add_child(palette)
        self.sidebar.add_child(combo(["Soft studio", "Flat", "Dramatic"], self._lighting_changed))
        self.sidebar.add_child(gui.Label("03  MOTION"))
        self.smooth = gui.Checkbox("Kalman smoothing")
        self.smooth.checked = config.smoothing
        self.smooth.enabled = self.worker is not None
        self.smooth.set_on_checked(self._smoothing_changed)
        self.sidebar.add_child(self.smooth)
        spin = gui.Checkbox("Auto orbit")
        spin.set_on_checked(self._spin_changed)
        self.sidebar.add_child(spin)
        self.pause_button = gui.Button("Pause")
        self.pause_button.set_on_clicked(self._pause)
        self.sidebar.add_child(self.pause_button)
        controls = gui.Horiz(8)
        reset = gui.Button("Reset view")
        reset.set_on_clicked(self._reset_camera)
        save = gui.Button("Export PLY + JSON")
        save.set_on_clicked(self._export)
        controls.add_child(reset)
        controls.add_child(save)
        self.sidebar.add_child(controls)
        self.sidebar.add_child(button("Show / hide workbench", self._toggle_tools))
        self.sidebar.add_child(gui.Label("Drag to orbit  /  Scroll to zoom"))
        self.sidebar.add_child(gui.Label("Ctrl + drag to pan"))
        self.sidebar.add_child(gui.Label("Local processing. No frame uploads."))
        self.window.add_child(self.sidebar)
        self.window.add_child(self.view)
        self.tools = StudioTools(self)
        self.window.add_child(self.tools.panel)
        if self.worker is not None:
            self.worker.start()

    def _layout(self, context) -> None:
        rect = self.window.content_rect
        sidebar_width = min(320, max(250, int(rect.width * 0.24)))
        tools_width = (
            min(350, max(280, int(rect.width * 0.24)))
            if self.show_tools and rect.width >= 960
            else 0
        )
        self.sidebar.frame = gui.Rect(rect.x, rect.y, sidebar_width, rect.height)
        self.view.frame = gui.Rect(
            rect.x + sidebar_width,
            rect.y,
            max(1, rect.width - sidebar_width - tools_width),
            rect.height,
        )
        self.tools.panel.visible = tools_width > 0
        self.tools.panel.frame = gui.Rect(
            rect.get_right() - tools_width, rect.y, tools_width, rect.height
        )

    def _toggle_tools(self) -> None:
        self.show_tools = not self.show_tools
        self.window.set_needs_layout()

    def _reset_camera(self) -> None:
        self.angle = 0
        bounds = o3d.geometry.AxisAlignedBoundingBox([-2.5, -2.5, -2.5], [2.5, 2.5, 2.5])
        self.view.setup_camera(45, bounds, [0, 0, 0])
        distance = 13.5 if hasattr(self, "tools") and self.tools.lab_enabled else 7.5
        self.view.look_at([0, 0, 0], [0, 0.1, distance], [0, 1, 0])

    def _mode_changed(self, text: str, index: int) -> None:
        if text == "Custom Avatar" and self.content.custom_avatar is None:
            self.mode.selected_index = self.modes.index("Surface")
            self._message("Avatar needed", "Load a GLB or try the bundled head in Style first.")
            text = "Surface"
        if text == "Heatmap" and self.content.heatmap.reference is None:
            self.mode.selected_index = self.modes.index("Surface")
            self._message("Neutral reference needed", "Capture neutral geometry in Motion first.")
            text = "Surface"
        if (
            hasattr(self, "tools")
            and text in {"Textured", "Source Color"}
            and (
                not self.tools.record_images
                or (self.frame is not None and not self.frame.source_available)
            )
        ):
            self.mode.selected_index = self.modes.index("Surface")
            self._message(
                "Source images unavailable", "Landmark-only mode cannot show source textures."
            )
            text = "Surface"
        self.content.mode = text
        if hasattr(self, "tools"):
            self.tools.present(time.monotonic())
        self.view.force_redraw()
        self.window.post_redraw()

    def _palette_changed(self, text: str, index: int) -> None:
        self.content.palette = text
        self.tools.present(time.monotonic())

    def _lighting_changed(self, text: str, index: int) -> None:
        self.content.set_lighting(text)
        self.tools.present(time.monotonic())

    def _smoothing_changed(self, enabled: bool) -> None:
        if self.worker is not None:
            self.worker.set_smoothing(enabled)

    def _spin_changed(self, enabled: bool) -> None:
        self.spin = enabled

    def _pause(self) -> None:
        self.paused = not self.paused
        if self.worker is not None:
            self.worker.set_paused(self.paused)
        if self.playback:
            if (
                self.paused is False
                and self.playback.position_ms >= self.playback.reader.duration_ms
            ):
                self.playback.seek(0)
            self.playback.paused = self.paused
            self.playback.last_clock = None
        self.pause_button.text = "Resume" if self.paused else "Pause"

    def _export(self) -> None:
        try:
            if self.frame is None:
                raise ValueError("Wait for a tracked face before exporting")
            paths = export_frame(self.frame)
            self._message("Export saved", f"Saved {paths[0].name}\nand matching JSON in exports/")
        except (OSError, ValueError) as error:
            self._message("Export unavailable", str(error))

    def _message(self, title: str, text: str) -> None:
        dialog = gui.Dialog(title)
        panel = gui.Vert(12, gui.Margins(20, 20, 20, 20))
        panel.add_child(gui.Label(text))
        dismiss = gui.Button("Close")
        dismiss.set_on_clicked(self.window.close_dialog)
        panel.add_child(dismiss)
        dialog.add_child(panel)
        self.window.show_dialog(dialog)

    def _tick(self) -> bool:
        previous_status = self.status.text
        now = time.monotonic()
        elapsed = min(now - self.last_tick, 0.1)
        self.last_tick = now
        if self.playback is not None:
            error, finished = None, False
            try:
                frame = self.playback.step(now)
            except (OSError, ValueError, sqlite3.Error) as failure:
                frame = None
                error = str(failure)
                self.playback.paused = True
        else:
            frame, error, finished = self.worker.take()
        if error:
            self.failed = True
            self.status.text = "Pipeline stopped"
            self._message("Input error", error)
            if self.worker is not None:
                with self.worker.lock:
                    self.worker.error = None
        if frame is not None:
            self.frame = frame
            self._update_preview(frame)
            self.tools.on_frame(frame, now)
            self.status.text = "Tracking face" if frame.vertices is not None else "No face detected"
            self.metrics.text = (
                "Recorded motion / no inference"
                if self.playback
                else f"{frame.fps:.1f} FPS  /  {frame.processing_ms:.1f} ms processing"
            )
        elif finished and not self.failed:
            self.status.text = "Video ended. Last frame available for export."
        if self.failed:
            self.status.text = "Pipeline stopped"
        tools_redraw = self.tools.tick(now)
        if self.spin and not self.paused:
            self.angle += elapsed * 0.3
            distance = 13.5 if self.tools.lab_enabled else 7.5
            self.view.look_at(
                [0, 0, 0],
                [distance * math.sin(self.angle), 0.3, distance * math.cos(self.angle)],
                [0, 1, 0],
            )
        return (
            frame is not None
            or tools_redraw
            or (self.spin and not self.paused)
            or self.status.text != previous_status
        )

    def _update_preview(self, frame: Frame) -> None:
        visible = frame.source_available and self.tools.record_images
        preview = (
            frame.rgb.copy() if visible else private_preview(frame.rgb.shape[1], frame.rgb.shape[0])
        )
        if not visible:
            cv2.putText(
                preview,
                "NO SOURCE IMAGES",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (85, 210, 180),
                1,
                cv2.LINE_AA,
            )
        if frame.landmarks is not None:
            height, width = preview.shape[:2]
            for point in frame.landmarks:
                cv2.circle(
                    preview,
                    (round(point[0] * width), round(point[1] * height)),
                    1,
                    (70, 240, 181),
                    -1,
                )
        preview = cv2.resize(preview, (300, round(preview.shape[0] * 300 / preview.shape[1])))
        self.preview.update_image(o3d.geometry.Image(np.ascontiguousarray(preview)))

    def _close(self) -> bool:
        if self.closed:
            return True
        self.closed = True
        if self.worker is not None:
            self.worker.close()
        for error in self.tools.close():
            print(f"MeshPipe shutdown warning: {error}", file=sys.stderr)
        if self.playback:
            self.playback.reader.close()
            self.playback = None
        return True


def run_studio(config: Config) -> None:
    application = gui.Application.instance
    application.initialize()
    studio = Studio(config)
    try:
        application.run()
    finally:
        studio._close()
