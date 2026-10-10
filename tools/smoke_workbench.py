"""Exercise optional desktop workflows using generated or genuine video input."""

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import mss
from open3d.visualization import gui
from PIL import Image

from meshpipe import studio_tools
from meshpipe.app import Studio
from meshpipe.cartoon import builtin_template
from meshpipe.config import Config
from meshpipe.session import SessionReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/meshpipe-workbench.png"))
    options = parser.parse_args()
    application = gui.Application.instance
    application.initialize()
    with tempfile.TemporaryDirectory(prefix="meshpipe-smoke-") as directory:
        root = Path(directory)
        studio_tools.export_path = lambda prefix, suffix: root / (prefix + suffix)
        studio = Studio(
            Config(
                demo=options.source is None,
                source=options.source or "0",
                model=options.model,
                loop=True,
            )
        )
        tools = studio.tools
        started = time.monotonic()
        phase = 0
        changed = started
        failure = None
        finished = False

        def advance() -> None:
            nonlocal phase, changed
            phase += 1
            changed = time.monotonic()

        def tick() -> bool:
            nonlocal failure, finished
            try:
                redraw = studio._tick()
                clock = time.monotonic()
                if clock - started > 40:
                    raise TimeoutError(f"Workbench stalled in phase {phase}")
                if studio.failed:
                    raise RuntimeError("Tracking pipeline failed")
                if studio.frame is None or studio.frame.vertices is None:
                    return redraw
                elapsed = clock - changed
                if phase == 0:
                    tools._toggle_session()
                    tools._toggle_stream()
                    tools._inspect(True)
                    tools.play_enabled = True
                    tools.calibration_enabled = True
                    tools.calibration.fit([studio.frame.expressions.copy()] * 12, True)
                    tools._save_calibration()
                    tools._load_calibration(root / "calibration.json")
                    tools._lab(True)
                    advance()
                elif phase == 1 and elapsed > 1.4:
                    tools._export_lab()
                    tools._export_region()
                    tools._lab(False)
                    studio._mode_changed("Textured", studio.modes.index("Textured"))
                    studio.mode.selected_index = studio.modes.index("Textured")
                    tools.tabs.selected_tab_index = 1
                    advance()
                elif phase == 2 and elapsed > 0.7:
                    rect = studio.window.os_frame
                    with mss.MSS() as capture:
                        shot = capture.grab(
                            {
                                "left": rect.x,
                                "top": rect.y,
                                "width": rect.width,
                                "height": rect.height,
                            }
                        )
                        Image.frombytes("RGB", shot.size, shot.rgb).save(options.output)
                    tools._toggle_session()
                    tools._toggle_stream()
                    advance()
                elif phase == 3 and tools.finalizing_session is None:
                    tools._speed("2x", 3)
                    tools._replay_loop(True)
                    tools.open_replay(root / "performance.meshpipe")
                    assert studio.worker is None
                    assert studio.playback.speed == 2 and studio.playback.loop
                    tools._seek(studio.playback.reader.duration_ms / 2)
                    studio._pause()
                    assert studio.playback.paused
                    studio._pause()
                    tools.capture_format.selected_index = 1
                    tools._toggle_capture()
                    advance()
                elif phase == 4 and elapsed > 1.3:
                    tools._toggle_capture()
                    advance()
                elif phase == 5 and tools.capture is None:
                    with Image.open(root / "studio.gif") as image:
                        assert image.n_frames > 1
                    tools.capture_format.selected_index = 0
                    tools._toggle_capture()
                    advance()
                elif phase == 6 and elapsed > 1.3:
                    tools._toggle_capture()
                    advance()
                elif phase == 7 and tools.capture is None:
                    video = cv2.VideoCapture(str(root / "studio.mp4"))
                    try:
                        assert video.isOpened() and video.read()[0]
                    finally:
                        video.release()
                    template = builtin_template()
                    Image.fromarray(template.rgba).save(root / "robot.png")
                    tools.load_cartoon(root / "robot.png")
                    tools.editor._accept()
                    tools._save_anchors()
                    tools._load_anchors(root / "anchors.json")
                    for target in tools.game.targets:
                        tools._game_hit(target)
                    assert tools.game.score >= len(tools.game.targets)
                    reader = SessionReader(root / "performance.meshpipe")
                    try:
                        assert len(reader.timestamps) > 1
                        assert reader.frame(0).vertices is not None
                    finally:
                        reader.close()
                    advance()
                elif phase == 8 and elapsed > 0.4:
                    finished = True
                    application.post_to_main_thread(studio.window, studio.window.close)
                return True
            except Exception as error:
                failure = error
                application.post_to_main_thread(studio.window, studio.window.close)
                return True

        studio.window.set_on_tick_event(tick)
        try:
            application.run()
            if failure:
                raise failure
            if not finished:
                raise RuntimeError("Workbench closed before validation finished")
            print(
                "Session, replay, GIF, MP4, calibration, cartoon, OSC, "
                "lab, explorer and game passed"
            )
        finally:
            studio._close()


if __name__ == "__main__":
    main()
