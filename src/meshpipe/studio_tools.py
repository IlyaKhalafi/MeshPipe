"""Optional studio tools. Inference remains independent of UI and integrations."""

import sqlite3
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui

from meshpipe.cartoon import CartoonRig, CartoonTemplate, builtin_template
from meshpipe.cartoon_editor import CartoonEditor
from meshpipe.explorer import REGIONS, LandmarkHistory, pick_landmark
from meshpipe.expressions import Calibration
from meshpipe.lab import SmoothingLab, trace_image
from meshpipe.media import ViewRecorder
from meshpipe.playground import ExpressionGame, GestureController
from meshpipe.pose import head_angles
from meshpipe.quality import Recovery, assess
from meshpipe.session import Playback, SessionReader, SessionRecorder, private_preview
from meshpipe.stream import LocalStream
from meshpipe.studio_panels import build_create, build_inspect, build_play


def export_path(prefix: str, suffix: str) -> Path:
    return Path("exports") / (prefix + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + suffix)


class StudioTools:
    def __init__(self, studio):
        self.studio = studio
        self.calibration = Calibration()
        self.calibration_enabled = False
        self.replay_speed = studio.playback.speed if studio.playback else 1.0
        self.replay_loop = studio.playback.loop if studio.playback else False
        self.yaw = 0.0
        self.calibrating: tuple[bool, float] | None = None
        self.samples: list[dict[str, float]] = []
        self.session: SessionRecorder | None = None
        self.finalizing_session: SessionRecorder | None = None
        self.capture: ViewRecorder | None = None
        self.capture_started = 0.0
        self.capture_last = 0.0
        self.capture_pending = False
        self.capture_token = 0
        self.stream: LocalStream | None = None
        self.lab = SmoothingLab()
        self.lab_result = None
        self.lab_enabled = False
        self.history = LandmarkHistory()
        self.inspect_enabled = False
        self.region = "Nose"
        self.game = ExpressionGame()
        self.gestures = GestureController()
        self.play_enabled = False
        self.recovery = Recovery()
        self.last_quality = 0.0
        self.last_graph = 0.0
        self.last_play_graph = 0.0
        self.render_frame = None
        self.editor: CartoonEditor | None = None
        self.closed = False
        self.record_images = not studio.config.landmark_only
        self.panel = gui.Vert(10, gui.Margins(14, 16, 14, 10))
        self.panel.background_color = gui.Color(0.055, 0.095, 0.115)
        heading = gui.Label("WORKBENCH")
        heading.text_color = gui.Color(0.96, 0.72, 0.29)
        self.panel.add_child(heading)
        self.notice = gui.Label("Create. Inspect. Play.")
        self.panel.add_child(self.notice)
        self.tabs = gui.TabControl()
        self.panel.add_child(self.tabs)
        build_create(self)
        build_inspect(self)
        build_play(self)
        from meshpipe.advanced_tools import AdvancedTools

        self.advanced = AdvancedTools(self)
        self.panel.add_child(gui.Label("Tools stay local and opt-in."))
        self.studio.view.set_on_mouse(self._pick)
        self.studio.view.set_on_key(self._key)
        if studio.config.cartoon:
            self.load_cartoon(studio.config.cartoon)
        if studio.config.osc_port:
            self.port.int_value = studio.config.osc_port
            self._toggle_stream()

    def _safe(self, action):
        def wrapped(*arguments):
            try:
                return action(*arguments)
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                self.notice.text = "Action unavailable"
                self.studio._message("Tool unavailable", str(error))
                return None

        return wrapped

    def _file(self, title: str, extension: str, callback) -> None:
        dialog = gui.FileDialog(gui.FileDialog.OPEN, title, self.studio.window.theme)
        dialog.add_filter(extension, extension + " files")
        dialog.set_on_cancel(self.studio.window.close_dialog)

        def chosen(path):
            self.studio.window.close_dialog()
            self._safe(callback)(Path(path))

        dialog.set_on_done(chosen)
        self.studio.window.show_dialog(dialog)

    def _toggle_session(self) -> None:
        if self.finalizing_session:
            raise ValueError("Previous performance is still saving")
        if self.session is None:
            source_kind = (
                self.studio.playback.reader.source_kind
                if self.studio.playback
                and self.studio.playback.reader.source_kind in {"tracked", "synthetic"}
                else (
                    "replay"
                    if self.studio.playback
                    else "synthetic"
                    if self.studio.config.demo
                    else "tracked"
                )
            )
            self.session = SessionRecorder(
                export_path("performance", ".meshpipe"),
                save_images=self.record_images
                and (self.studio.frame is None or self.studio.frame.source_available),
                source_kind=source_kind,
            )
            self.session_button.text = "Stop performance"
            self.notice.text = (
                "Recording source images and motion"
                if self.session.save_images
                else "Recording landmarks only. No source images saved."
            )
        else:
            self.session.stopping.set()
            self.finalizing_session, self.session = self.session, None
            self.session_button.text = "Saving performance..."
            self.session_button.enabled = False

    def _privacy(self, enabled: bool) -> None:
        if self.session or self.finalizing_session or self.capture:
            self.privacy_checkbox.checked = not self.record_images
            raise ValueError("Finish recording before changing privacy mode")
        self.record_images = not enabled
        if enabled and self.studio.content.mode in {"Textured", "Source Color"}:
            self.studio.mode.selected_index = self.studio.modes.index("Surface")
            self.studio._mode_changed("Surface", self.studio.modes.index("Surface"))
        if self.studio.frame is not None:
            self.studio._update_preview(self.studio.frame)
            self.present(time.monotonic())
        self.notice.text = "Landmark-only data is still sensitive facial motion."

    def _toggle_capture(self) -> None:
        if self.capture is not None:
            self.capture.finish()
            self.capture_button.text = "Finishing..."
            self.capture_button.enabled = False
            return
        extension = ".gif" if self.capture_format.selected_text == "GIF" else ".mp4"
        self.capture = ViewRecorder(export_path("studio", extension))
        self.capture_started = time.monotonic()
        self.capture_last = 0
        self.capture_token += 1
        self.capture_button.text = "Stop recording"
        self.capture_format.enabled = False

    def _toggle_stream(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
            self.stream_button.text = "Start OSC stream"
            self.port.enabled = True
        else:
            self.stream = LocalStream(self.port.int_value)
            self.stream_button.text = "Stop OSC stream"
            self.port.enabled = False
            self.notice.text = f"OSC to 127.0.0.1 port {self.port.int_value}"

    def open_replay(self, path: Path) -> None:
        if self.session or self.finalizing_session:
            raise ValueError("Finish performance recording before loading replay")
        reader = SessionReader(path)
        try:
            initial = reader.frame(0)
        except Exception:
            reader.close()
            raise
        if self.studio.worker is not None:
            self.studio.worker.close()
            self.studio.worker = None
        if self.studio.playback:
            self.studio.playback.reader.close()
        self.studio.playback = Playback(reader, speed=self.replay_speed, loop=self.replay_loop)
        self.studio.paused = False
        self.studio.failed = False
        self.studio.pause_button.text = "Pause"
        self.studio.smooth.enabled = False
        self.timeline.enabled = True
        self.timeline.set_limits(0, max(1, reader.duration_ms))
        self.lab.reset()
        self.history.values.clear()
        self.recovery = Recovery()
        self.studio.frame = initial
        self.notice.text = f"Replay loaded. {len(reader.timestamps)} frames."
        self.advanced.sync()

    def _seek(self, value: float) -> None:
        if self.studio.playback:
            self.studio.playback.seek(value)
            self.lab.reset()
            self.history.values.clear()
            self.recovery = Recovery()

    def _speed(self, text: str, index: int) -> None:
        self.replay_speed = [0.25, 0.5, 1, 2][index]
        if self.studio.playback:
            self.studio.playback.speed = self.replay_speed

    def _replay_loop(self, enabled: bool) -> None:
        self.replay_loop = enabled
        if self.studio.playback:
            self.studio.playback.loop = enabled

    def _calibrate(self, neutral: bool) -> None:
        self.calibrating = (neutral, time.monotonic() + (3 if neutral else 6))
        self.samples.clear()
        self.calibration_status.text = (
            "Hold neutral face" if neutral else "Smile, blink both eyes and open jaw"
        )

    def _save_calibration(self) -> None:
        path = export_path("calibration", ".json")
        self.calibration.save(path)
        self.notice.text = f"Saved {path.name}"

    def _load_calibration(self, path: Path) -> None:
        self.calibration = Calibration.load(path)
        self.sensitivity.double_value = self.calibration.sensitivity
        self.dead_zone.double_value = self.calibration.dead_zone
        self.calibration_status.text = "Personal calibration loaded"

    def _builtin_cartoon(self) -> None:
        self._use_cartoon(builtin_template())

    def load_cartoon(self, path: Path) -> None:
        if self.editor is not None and not self.editor.closed:
            raise ValueError("Finish current cartoon editor first")
        self.editor = CartoonEditor(CartoonTemplate.load(path), self._use_cartoon, self._safe)

    def _use_cartoon(self, template: CartoonTemplate) -> None:
        self.studio.content.cartoon = CartoonRig(template)
        index = self.studio.modes.index("Cartoon")
        self.studio.mode.selected_index = index
        self.studio._mode_changed("Cartoon", index)
        self.notice.text = "Cartoon ready. Image warping, not 3D reconstruction."

    def _save_anchors(self) -> None:
        if self.studio.content.cartoon is None:
            raise ValueError("Load a cartoon first")
        path = export_path("anchors", ".json")
        self.studio.content.cartoon.template.save_anchors(path)
        self.notice.text = f"Saved {path.name}"

    def _load_anchors(self, path: Path) -> None:
        if self.studio.content.cartoon is None:
            raise ValueError("Load a cartoon first")
        template = self.studio.content.cartoon.template
        template.load_anchors(path)
        self._use_cartoon(template)

    def _inspect(self, enabled: bool) -> None:
        self.inspect_enabled = enabled
        self.inspect_checkbox.checked = enabled
        self.studio.content.highlight = REGIONS[self.region] if enabled else ()
        self.studio.content.selected = self.history.index if enabled else None
        if self.render_frame is not None:
            self.present(time.monotonic())

    def _region(self, text: str, index: int) -> None:
        self.region = text
        self._inspect(self.inspect_enabled)

    def _select_index(self, value: float) -> None:
        self.history.select(int(value))
        self.studio.content.selected = int(value) if self.inspect_enabled else None
        self._inspect(self.inspect_enabled)

    def _pick(self, event):
        if (
            self.inspect_enabled
            and not self.lab_enabled
            and self.studio.content.mode not in {"Cartoon", "Emoji"}
            and event.type == gui.MouseEvent.BUTTON_DOWN
            and event.is_modifier_down(gui.KeyModifier.SHIFT)
        ):
            frame = self.studio.frame
            if frame is not None and frame.vertices is not None:
                rect = self.studio.view.frame
                camera = self.studio.view.scene.camera
                index = pick_landmark(
                    frame.vertices,
                    camera.get_projection_matrix(),
                    camera.get_view_matrix(),
                    rect.width,
                    rect.height,
                    event.x - rect.x,
                    event.y - rect.y,
                )
                if index is not None:
                    self.landmark_index.int_value = index
                    self._select_index(index)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _lab(self, enabled: bool) -> None:
        self.lab_enabled = enabled
        self.lab_checkbox.checked = enabled
        self.lab.reset()
        self.lab_result = None
        self.studio._reset_camera()
        if self.studio.frame:
            self.on_frame(self.studio.frame, time.monotonic(), record=False)

    def _alpha(self, value: float) -> None:
        self.lab.alpha = value
        self.lab.reset()

    def _export_region(self) -> None:
        path = export_path("landmarks", ".json")
        self.history.export(path, self.region)
        self.notice.text = f"Saved {path.name}"

    def _export_lab(self) -> None:
        import json

        if self.lab_result is None:
            raise ValueError("Enable lab with a tracked face first")
        path = export_path("smoothing-lab", ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        result = self.lab_result
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "ema_alpha_at_30_fps": self.lab.alpha,
                    "filter_cost_ms": dict(zip(["kalman", "ema"], result.costs_ms, strict=True)),
                    "estimated_motion_lag_ms": dict(
                        zip(["kalman", "ema"], result.lag_ms, strict=True)
                    ),
                    "second_difference_rms": dict(
                        zip(["raw", "kalman", "ema"], result.motion, strict=True)
                    ),
                    "scope": "Nose x/y motion. Roughness includes real acceleration, "
                    "not just noise.",
                    "samples": [
                        {"timestamp_ms": sample[0], "nose_xy": sample[1].tolist()}
                        for sample in self.lab.history
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.notice.text = f"Saved {path.name}"

    def _game_hit(self, name: str) -> None:
        if self.game.hit(name, time.monotonic()):
            self.game_target.text = "NEXT  /  " + self.game.target
            self.game_score.text = f"{self.game.score} gestures matched"
            self._draw_game(time.monotonic(), {})

    def _reset_game(self) -> None:
        self.game.reset()
        self.gestures = GestureController()
        self.game_target.text = "NEXT  /  Smile"
        self.game_score.text = "0 gestures matched"
        self._draw_game(0, {})

    def _key(self, event):
        if self.play_enabled and event.type == gui.KeyEvent.DOWN and not event.is_repeat:
            names = {
                ord("s"): "Smile",
                ord("j"): "Open mouth",
                ord("l"): "Left wink",
                ord("r"): "Right wink",
                ord("a"): "Turn left",
                ord("d"): "Turn right",
            }
            if int(event.key) in names:
                self._game_hit(names[int(event.key)])
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _draw_game(self, clock: float, scores: dict[str, float]) -> None:
        image = np.full((220, 300, 3), [15, 27, 33], dtype=np.uint8)
        for radius in (74, 84, 96):
            cv2.circle(image, (150, 103), radius, (28, 49, 58), 1, cv2.LINE_AA)
        strength = {
            "Smile": (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2,
            "Open mouth": scores.get("jawOpen", 0),
            "Left wink": scores.get("eyeBlinkLeft", 0),
            "Right wink": scores.get("eyeBlinkRight", 0),
            "Turn left": max(0, -self.yaw / 0.45),
            "Turn right": max(0, self.yaw / 0.45),
        }[self.game.target]
        cv2.ellipse(
            image,
            (150, 103),
            (84, 84),
            -90,
            0,
            max(1, 360 * min(1, strength)),
            (85, 221, 187),
            5,
            cv2.LINE_AA,
        )
        text = str(self.game.score)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.7, 2)[0]
        cv2.putText(
            image,
            text,
            (150 - size[0] // 2, 119),
            cv2.FONT_HERSHEY_DUPLEX,
            1.7,
            (245, 185, 74),
            2,
            cv2.LINE_AA,
        )
        self.game_image.update_image(o3d.geometry.Image(image))

    def on_frame(self, frame, clock: float, record: bool = True) -> None:
        if record and self.session:
            self.session.submit(frame)
        scores = (
            self.calibration.apply(frame.expressions)
            if self.calibration_enabled
            else frame.expressions
        )
        self.render_frame = replace(
            frame,
            expressions=scores,
            source_available=frame.source_available and self.record_images,
        )
        self.recovery.update(self.render_frame, clock)
        if self.calibrating and frame.vertices is not None:
            self.samples.append(frame.expressions.copy())
        if self.stream and record:
            try:
                self.stream.send(frame, scores, clock)
            except (OSError, ValueError) as error:
                self._toggle_stream()
                self.notice.text = f"OSC stopped. {error}"
        if clock - self.last_quality >= 0.5:
            if frame.source_available:
                quality = assess(frame)
                self.quality.text = quality.hint
                self.quality_values.text = (
                    f"Light {quality.brightness:.0f}/255  /  Face {quality.face_fraction:.0%}"
                )
            else:
                self.quality.text = "Landmark-only replay. Source images unavailable."
                self.quality_values.text = "Lighting and focus cannot be measured."
            self.last_quality = clock
        if self.inspect_enabled:
            self.history.update(frame.timestamp_ms, frame.landmarks)
            if frame.landmarks is not None:
                point = frame.landmarks[self.history.index]
                self.coordinate.text = (
                    f"#{self.history.index}  x {point[0]:.3f}  y {point[1]:.3f}  z {point[2]:.3f}"
                )
        if self.lab_enabled:
            raw = frame.raw_landmarks
            if raw is None and frame.landmarks is not None:
                self.lab_metrics.text = "Raw landmarks unavailable in this performance"
            self.lab_result = self.lab.update(raw, frame.timestamp_ms)
        if self.play_enabled:
            self.yaw = head_angles(frame.vertices)[1] if frame.vertices is not None else 0
            for event in self.gestures.update(scores, frame.vertices is not None, clock, self.yaw):
                self._game_hit(event.name)
            if clock - self.last_play_graph > 0.1:
                self._draw_game(clock, scores)
                self.last_play_graph = clock
        self.present(clock)

    def present(self, clock: float) -> None:
        if self.lab_enabled:
            if self.studio.frame is not None:
                self.studio.content.update_lab(
                    self.lab_result, self.studio.frame.rgb.shape[1] / self.studio.frame.rgb.shape[0]
                )
        else:
            frame, opacity = self.recovery.presentation(clock)
            self.studio.content.opacity = opacity
            if frame is not None:
                self.studio.content.update(frame)
            else:
                self.studio.content.clear()

    def tick(self, clock: float) -> bool:
        redraw = self.advanced.tick()
        if self.calibrating and clock >= self.calibrating[1]:
            neutral = self.calibrating[0]
            self.calibrating = None
            try:
                self.calibration.fit(self.samples, neutral)
                self.calibration_status.text = (
                    "Neutral captured" if neutral else "Expression range captured"
                )
            except ValueError as error:
                self.calibration_status.text = str(error)
            self.samples.clear()
            redraw = True
        if self.session and self.session.error:
            self.notice.text = f"Performance stopped. {self.session.error}"
            self.session.stopping.set()
            self.finalizing_session, self.session = self.session, None
        if self.finalizing_session and not self.finalizing_session.thread.is_alive():
            recorder, self.finalizing_session = self.finalizing_session, None
            try:
                path = recorder.close()
                self.notice.text = f"Saved {path.name}. Dropped {recorder.dropped}."
            except (OSError, ValueError, RuntimeError) as error:
                self.notice.text = str(error)
            self.session_button.text = "Record performance"
            self.session_button.enabled = True
            redraw = True
        if self.capture and self.capture.finished.is_set():
            recorder, self.capture = self.capture, None
            self.capture_token += 1
            try:
                path = recorder.close()
                self.notice.text = f"Saved {path.name}"
            except (OSError, RuntimeError) as error:
                self.notice.text = str(error)
            self.capture_status.text = f"{recorder.written} frames. Dropped {recorder.dropped}."
            self.capture_button.text = "Record view"
            self.capture_button.enabled = True
            self.capture_format.enabled = True
            redraw = True
        if (
            self.capture
            and not self.capture.stop.is_set()
            and not self.capture_pending
            and self.studio.frame is not None
            and clock - self.capture_last >= 1 / self.capture.fps
        ):
            self._capture(clock)
        if self.studio.playback:
            playback = self.studio.playback
            self.timeline.set_limits(0, max(1, playback.reader.duration_ms))
            self.timeline.double_value = playback.position_ms
            self.replay_time.text = (
                f"{playback.position_ms / 1000:.1f} / {playback.reader.duration_ms / 1000:.1f} s"
            )
            self.studio.paused = playback.paused
            self.studio.pause_button.text = "Resume" if playback.paused else "Pause"
        if self.recovery.lost_at is not None and self.recovery.last_frame is not None:
            self.present(clock)
            if clock - self.recovery.lost_at > self.recovery.grace_seconds:
                self.recovery.last_frame = None
            redraw = True
        if clock - self.last_graph >= 0.2:
            if self.inspect_enabled:
                self.landmark_graph.update_image(
                    o3d.geometry.Image(
                        trace_image(np.array([point for _, point in self.history.values]))
                    )
                )
                redraw = True
            if self.lab_enabled and self.lab_result is not None:
                result = self.lab_result
                costs = result.costs_ms
                lag = ["n/a" if value is None else f"{value:.0f}" for value in result.lag_ms]
                self.lab_metrics.text = (
                    f"Kalman {costs[0]:.2f} ms / EMA {costs[1]:.2f} ms\n"
                    f"Lag estimate {lag[0]} / {lag[1]} ms"
                )
                self.lab_motion.text = "Roughness " + " / ".join(
                    f"{value:.5f}" for value in result.motion
                )
                self.lab_graph.update_image(
                    o3d.geometry.Image(
                        trace_image(np.array([sample[1][:, 0] for sample in self.lab.history]))
                    )
                )
                redraw = True
            self.last_graph = clock
        return redraw

    def _capture(self, clock: float) -> None:
        recorder = self.capture
        token = self.capture_token
        frame = self.studio.frame
        source = (
            frame.rgb
            if frame.source_available and self.record_images
            else private_preview(frame.rgb.shape[1], frame.rgb.shape[0])
        )
        timestamp = clock - self.capture_started
        self.capture_last = clock
        self.capture_pending = True
        title = (
            "Synthetic demo"
            if self.studio.config.demo and not self.studio.playback
            else "Recorded performance"
            if self.studio.playback
            else "Live tracking"
        )
        title += " / " + self.studio.content.mode
        if not frame.source_available or not self.record_images:
            title += " / No source images"

        def rendered(image):
            if not self.closed and token == self.capture_token:
                recorder.submit(
                    source,
                    np.asarray(image).copy(),
                    timestamp,
                    title,
                )
            self.capture_pending = False

        self.studio.view.scene.scene.render_to_image(rendered)

    def close(self) -> list[str]:
        if self.closed:
            return []
        self.closed = True
        self.advanced.close()
        self.capture_token += 1
        errors = []
        for resource in [self.stream, self.session, self.finalizing_session, self.capture]:
            if resource is not None:
                try:
                    resource.close()
                except (OSError, ValueError, RuntimeError) as error:
                    errors.append(str(error))
        if self.editor is not None and not self.editor.closed:
            gui.Application.instance.post_to_main_thread(
                self.editor.window, self.editor.window.close
            )
        return errors
