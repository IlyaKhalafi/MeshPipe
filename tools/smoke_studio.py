"""Exercise all desktop modes and save a screenshot on a real display."""

import argparse
import tempfile
import time
from pathlib import Path

import mss
from open3d.visualization import gui
from PIL import Image

from meshpipe.app import Studio
from meshpipe.config import Config
from meshpipe.export import export_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/meshpipe-studio.png"))
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--video", action="store_true")
    options = parser.parse_args()
    application = gui.Application.instance
    application.initialize()
    studio = Studio(Config(demo=not options.video, source=options.source, model=options.model))
    started = time.monotonic()
    modes = studio.modes
    previous_mode = -1
    screenshot = False
    prepared = False
    error = None

    def tick() -> bool:
        nonlocal previous_mode, screenshot, prepared, error
        redraw = studio._tick()
        try:
            elapsed = time.monotonic() - started
            if elapsed > 30:
                raise TimeoutError("Studio smoke test did not finish within 30 seconds")
            if studio.failed:
                raise RuntimeError("Studio pipeline failed")
            if studio.frame is None:
                return redraw
            index = min(int(elapsed / 0.9), len(modes) - 1)
            if index != previous_mode:
                studio._mode_changed(modes[index], index)
                studio.mode.selected_index = index
                studio._palette_changed("Amber", 1)
                studio.palette.selected_index = 1
                studio._spin_changed(True)
                previous_mode = index
                studio.window.post_redraw()
            if elapsed > 8 and not prepared:
                studio._pause()
                studio._spin_changed(False)
                studio._reset_camera()
                prepared = True
                return True
            if elapsed > 8.3:
                with tempfile.TemporaryDirectory() as directory:
                    export_frame(studio.frame, Path(directory))
                rect = studio.window.os_frame
                with mss.MSS() as capture:
                    shot = capture.grab(
                        {"left": rect.x, "top": rect.y, "width": rect.width, "height": rect.height}
                    )
                    Image.frombytes("RGB", shot.size, shot.rgb).save(options.output)
                screenshot = True
                application.post_to_main_thread(studio.window, studio.window.close)
                return True
            return redraw
        except Exception as failure:
            error = failure
            application.post_to_main_thread(studio.window, studio.window.close)
            return True

    studio.window.set_on_tick_event(tick)
    try:
        application.run()
        if error is not None:
            raise error
        if not screenshot:
            raise RuntimeError("Studio closed before smoke test completed")
        print(f"Studio modes, orbit, pause and export passed. Screenshot at {options.output}")
    finally:
        studio._close()


if __name__ == "__main__":
    main()
