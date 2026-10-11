"""View builders for the studio workbench, separate from tool lifecycle."""

import numpy as np
import open3d as o3d
from open3d.visualization import gui

from meshpipe.explorer import REGIONS
from meshpipe.lab import trace_image
from meshpipe.widgets import button, combo, row, section, slider


def build_create(tools) -> None:
    panel = gui.ScrollableVert(8, gui.Margins(0, 8, 0, 4))
    panel.background_color = gui.Color(0.055, 0.095, 0.115)
    tools.session_button = button("Record performance", tools._safe(tools._toggle_session))
    tools.capture_button = button("Record view", tools._safe(tools._toggle_capture))
    tools.capture_format = combo(["MP4", "GIF"], lambda *_: None)
    tools.capture_status = gui.Label("GIF up to 10 s. MP4 up to 10 min.")
    tools.privacy_checkbox = gui.Checkbox("Landmark-only recording / hide source")
    tools.privacy_checkbox.checked = not tools.record_images
    tools.privacy_checkbox.set_on_checked(tools._safe(tools._privacy))
    panel.add_child(
        section(
            "01  RECORD",
            tools.privacy_checkbox,
            tools.session_button,
            row(tools.capture_format, tools.capture_button),
            tools.capture_status,
        )
    )
    tools.timeline = slider(0, 1, 0, tools._safe(tools._seek))
    tools.timeline.enabled = tools.studio.playback is not None
    tools.replay_time = gui.Label("No performance loaded")
    speed = combo(["0.25x", "0.5x", "1x", "2x"], tools._speed)
    speed.selected_index = 2
    replay_loop = gui.Checkbox("Loop replay")
    replay_loop.checked = tools.replay_loop
    replay_loop.set_on_checked(tools._replay_loop)
    panel.add_child(
        section(
            "02  REPLAY",
            button(
                "Open performance",
                lambda: tools._file("Open performance", ".meshpipe", tools.open_replay),
            ),
            tools.timeline,
            tools.replay_time,
            row(gui.Label("Speed"), speed),
            replay_loop,
        )
    )
    tools.calibration_status = gui.Label("Neutral first. Then blink, smile, open jaw.")
    calibrated = gui.Checkbox("Apply personal calibration")
    calibrated.set_on_checked(lambda value: setattr(tools, "calibration_enabled", value))
    tools.sensitivity = slider(
        0.25, 3, 1, lambda value: setattr(tools.calibration, "sensitivity", value)
    )
    tools.dead_zone = slider(
        0, 0.4, 0.04, lambda value: setattr(tools.calibration, "dead_zone", value)
    )
    panel.add_child(
        section(
            "03  EXPRESSIONS",
            calibrated,
            tools.calibration_status,
            row(
                button("Neutral 3 s", lambda: tools._calibrate(True)),
                button("Range 6 s", lambda: tools._calibrate(False)),
            ),
            gui.Label("Sensitivity"),
            tools.sensitivity,
            gui.Label("Dead zone"),
            tools.dead_zone,
            row(
                button("Save", tools._safe(tools._save_calibration)),
                button(
                    "Load",
                    lambda: tools._file("Load calibration", ".json", tools._load_calibration),
                ),
            ),
            opened=False,
        )
    )
    panel.add_child(
        section(
            "04  YOUR CARTOON",
            gui.Label("PNG puppet with explicit eye and mouth anchors."),
            button(
                "Load PNG and mark anchors",
                lambda: tools._file("Choose cartoon PNG", ".png", tools.load_cartoon),
            ),
            button("Try paper robot", tools._builtin_cartoon),
            row(
                button("Save anchors", tools._safe(tools._save_anchors)),
                button(
                    "Load anchors",
                    lambda: tools._file("Load anchors", ".json", tools._load_anchors),
                ),
            ),
            opened=False,
        )
    )
    tools.port = gui.NumberEdit(gui.NumberEdit.INT)
    tools.port.set_limits(1024, 65535)
    tools.port.int_value = 9000
    tools.stream_button = button("Start OSC stream", tools._safe(tools._toggle_stream))
    panel.add_child(
        section(
            "05  LOCAL STREAM",
            gui.Label("Loopback OSC. No images transmitted."),
            row(gui.Label("Port"), tools.port),
            tools.stream_button,
            opened=False,
        )
    )
    tools.tabs.add_tab("Create", panel)


def build_inspect(tools) -> None:
    panel = gui.ScrollableVert(8, gui.Margins(0, 8, 0, 4))
    panel.background_color = gui.Color(0.055, 0.095, 0.115)
    tools.quality = gui.Label("Waiting for input quality checks")
    tools.quality_values = gui.Label("Framing and lighting, not confidence scores")
    panel.add_child(section("01  TRACKING HEALTH", tools.quality, tools.quality_values))
    inspect = gui.Checkbox("Enable landmark explorer")
    tools.inspect_checkbox = inspect
    inspect.set_on_checked(tools._inspect)
    tools.landmark_index = gui.NumberEdit(gui.NumberEdit.INT)
    tools.landmark_index.set_limits(0, 467)
    tools.landmark_index.int_value = 1
    tools.landmark_index.set_on_value_changed(tools._select_index)
    region = combo(REGIONS, tools._region)
    region.selected_index = list(REGIONS).index(tools.region)
    tools.coordinate = gui.Label("Shift-click mesh or choose index")
    tools.landmark_graph = gui.ImageWidget(o3d.geometry.Image(trace_image(np.empty((0, 3)))))
    panel.add_child(
        section(
            "02  LANDMARK EXPLORER",
            inspect,
            row(gui.Label("Index"), tools.landmark_index),
            region,
            tools.coordinate,
            tools.landmark_graph,
            gui.Label("History colors follow x, y and z."),
            button("Export region and history", tools._safe(tools._export_region)),
        )
    )
    lab = gui.Checkbox("Compare raw / Kalman / EMA")
    tools.lab_checkbox = lab
    lab.set_on_checked(tools._lab)
    alpha = slider(0.05, 1, 0.35, tools._alpha)
    tools.lab_metrics = gui.Label("Compare filtering cost and motion lag")
    tools.lab_motion = gui.Label("Second differences are motion roughness")
    tools.lab_graph = gui.ImageWidget(o3d.geometry.Image(trace_image(np.empty((0, 3)))))
    panel.add_child(
        section(
            "03  SMOOTHING LAB",
            lab,
            gui.Label("EMA response at 30 FPS"),
            alpha,
            tools.lab_metrics,
            tools.lab_motion,
            tools.lab_graph,
            gui.Label("Mint raw. Amber Kalman. Blue EMA."),
            gui.Label("Left raw. Center Kalman. Right EMA."),
            button("Export lab measurements", tools._safe(tools._export_lab)),
        )
    )
    tools.tabs.add_tab("Inspect", panel)


def build_play(tools) -> None:
    panel = gui.ScrollableVert(10, gui.Margins(0, 8, 0, 4))
    panel.background_color = gui.Color(0.055, 0.095, 0.115)
    enabled = gui.Checkbox("Enable expression playground")
    enabled.set_on_checked(lambda value: setattr(tools, "play_enabled", value))
    panel.add_child(enabled)
    tools.game_target = gui.Label("NEXT  /  Smile")
    tools.game_score = gui.Label("0 gestures matched")
    tools.game_image = gui.ImageWidget()
    panel.add_child(tools.game_target)
    panel.add_child(tools.game_score)
    panel.add_child(tools.game_image)
    panel.add_child(gui.Label("Match target with expression or keyboard."))
    panel.add_child(gui.Label("S smile. J jaw. L/R wink. A/D turn."))
    for name in tools.game.targets:
        panel.add_child(button(name, lambda target=name: tools._game_hit(target)))
    panel.add_child(button("Reset round", tools._reset_game))
    panel.add_child(gui.Label("Calibrate expressions for comfortable control."))
    panel.add_child(gui.Label("Nothing controls your OS or other apps."))
    tools._draw_game(0, {})
    tools.tabs.add_tab("Play", panel)
