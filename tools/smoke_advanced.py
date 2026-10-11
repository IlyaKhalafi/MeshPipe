"""Exercise new native workflows against a genuine video source."""

import argparse
import sqlite3
import tempfile
import time
from pathlib import Path

import mss
from open3d.visualization import gui
from PIL import Image

from meshpipe import studio_tools
from meshpipe.animation_protocol import load_animation
from meshpipe.app import Studio
from meshpipe.config import Config
from meshpipe.motion import MotionEdit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/meshpipe-advanced.png"))
    options = parser.parse_args()
    application = gui.Application.instance
    application.initialize()
    with tempfile.TemporaryDirectory(prefix="meshpipe-advanced-") as directory:
        root = Path(directory)
        studio_tools.export_path = lambda prefix, suffix: root / (prefix + suffix)
        studio = Studio(
            Config(
                source=options.source,
                model=options.model,
                loop=True,
                landmark_only=True,
                adaptive_cpu=True,
            )
        )
        tools, advanced = studio.tools, studio.tools.advanced
        phase = 0
        started = changed = time.monotonic()
        failure = None
        finished = False

        def advance():
            nonlocal phase, changed
            phase += 1
            changed = time.monotonic()

        def tick():
            nonlocal failure, finished
            try:
                redraw = studio._tick()
                clock = time.monotonic()
                if clock - started > 45:
                    raise TimeoutError(f"Advanced studio stalled in phase {phase}")
                if studio.failed:
                    raise RuntimeError("Pipeline failed")
                if studio.frame is None or studio.frame.vertices is None:
                    return redraw
                elapsed = clock - changed
                if phase == 0:
                    tools._toggle_session()
                    advance()
                elif phase == 1 and elapsed > 1.5:
                    tools._toggle_session()
                    advance()
                elif phase == 2 and tools.finalizing_session is None:
                    path = root / "performance.meshpipe"
                    with sqlite3.connect(path) as connection:
                        assert (
                            connection.execute(
                                "SELECT count(*) FROM frames WHERE rgb IS NOT NULL"
                            ).fetchone()[0]
                            == 0
                        )
                    tools.open_replay(path)
                    reader = advanced.reader()
                    assert not reader.has_images and reader.source_kind == "tracked"
                    reader.set_blend(path)
                    advanced.strength.double_value = 0.8
                    advanced.despike.checked = True
                    advanced.blend.double_value = 0.3
                    advanced.apply()
                    advanced.save()
                    advanced.history(False)
                    advanced.history(True)
                    advanced.load(root / "motion-edit.json")
                    assert (
                        MotionEdit.load(root / "motion-edit.json", reader.base.duration_ms).strength
                        == 0.8
                    )
                    advanced.export_bake()
                    advance()
                elif phase == 3 and advanced.job is None:
                    assert (root / "blender-motion.json").is_file(), tools.notice.text
                    load_animation(root / "blender-motion.json")
                    advanced.export_web()
                    advance()
                elif phase == 4 and advanced.job is None:
                    assert (root / "browser-studio/index.html").is_file(), tools.notice.text
                    advanced.builtin_avatar()
                    advanced.save_rig()
                    advanced.avatar().load_preset(root / "avatar-rig.json")
                    advance()
                elif phase == 5 and elapsed > 0.7:
                    advanced.neutral()
                    advance()
                elif phase == 6 and elapsed > 0.7:
                    advanced.apply_accessory()
                    advanced.save_accessory()
                    advanced.load_accessory(root / "accessory-anchor.json")
                    tools.tabs.selected_tab_index = 4
                    advance()
                elif phase == 7 and elapsed > 0.7:
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
                    studio._mode_changed("Textured", studio.modes.index("Textured"))
                    assert studio.content.mode == "Surface"
                    studio.window.close_dialog()
                    finished = True
                    application.post_to_main_thread(studio.window, studio.window.close)
                return True
            except Exception as error:
                failure = error
                application.post_to_main_thread(studio.window, studio.window.close)
                return False

        studio.window.set_on_tick_event(tick)
        application.run()
        if failure:
            raise failure
        if not finished:
            raise RuntimeError("Native workflow ended before validation finished")
    print("Privacy, adaptive CPU, editing, blending, GLB, heatmap, accessories and exports passed")


if __name__ == "__main__":
    main()
