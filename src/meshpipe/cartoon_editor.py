from collections.abc import Callable

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui

from meshpipe.cartoon import ANCHOR_NAMES, CartoonTemplate
from meshpipe.widgets import button, row


class CartoonEditor:
    def __init__(self, template: CartoonTemplate, accept: Callable, report: Callable):
        self.template = template
        self.accept = accept
        self.report = report
        self.index = 0
        self.closed = False
        self.window = gui.Application.instance.create_window("Cartoon anchor studio", 640, 560)
        self.window.set_on_layout(self._layout)
        self.window.set_on_close(self._close)
        self.panel = gui.Vert(10, gui.Margins(20, 16, 20, 16))
        self.panel.add_child(gui.Label("MAKE IT YOURS"))
        self.instruction = gui.Label("")
        self.panel.add_child(self.instruction)
        self.image = gui.ImageWidget()
        self.image.set_on_mouse(self._mouse)
        self.panel.add_child(self.image)
        self.panel.add_child(gui.Label("Left and right mean subject's anatomical sides."))
        self.panel.add_child(gui.Label("Click left eye, right eye, then mouth center."))
        self.panel.add_child(
            row(button("Start again", self._reset), button("Use this cartoon", self._accept))
        )
        self.window.add_child(self.panel)
        height, width = template.rgba.shape[:2]
        self.scale = min(600 / width, 340 / height)
        self.offset_x = (600 - round(width * self.scale)) // 2
        self.offset_y = (340 - round(height * self.scale)) // 2
        self._refresh()

    def _layout(self, context) -> None:
        self.panel.frame = self.window.content_rect

    def _refresh(self) -> None:
        image = np.full((340, 600, 3), [15, 27, 33], dtype=np.uint8)
        rgba = self.template.rgba
        alpha = rgba[:, :, 3:4].astype(float) / 255
        rgb = (rgba[:, :, :3] * alpha + np.array([15, 27, 33]) * (1 - alpha)).astype(np.uint8)
        height, width = rgba.shape[:2]
        resized = cv2.resize(rgb, (round(width * self.scale), round(height * self.scale)))
        image[
            self.offset_y : self.offset_y + resized.shape[0],
            self.offset_x : self.offset_x + resized.shape[1],
        ] = resized
        for anchor_index, name in enumerate(ANCHOR_NAMES):
            point = self.template.anchors[name]
            horizontal = round(point[0] * width * self.scale + self.offset_x)
            vertical = round(point[1] * height * self.scale + self.offset_y)
            cv2.circle(image, (horizontal, vertical), 7, (245, 185, 74), 2)
            cv2.putText(
                image,
                str(anchor_index + 1),
                (horizontal + 10, vertical),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 185, 74),
                1,
            )
        self.image.update_image(o3d.geometry.Image(np.ascontiguousarray(image)))
        self.instruction.text = (
            "Anchors ready. Adjust again or use cartoon."
            if self.index == 3
            else f"Click {ANCHOR_NAMES[self.index].replace('_', ' ')} center"
        )
        self.window.post_redraw()

    def _mouse(self, event):
        if event.type != gui.MouseEvent.BUTTON_DOWN or not event.is_button_down(
            gui.MouseButton.LEFT
        ):
            return gui.Widget.EventCallbackResult.IGNORED
        rect = self.image.frame
        height, width = self.template.rgba.shape[:2]
        factor = min(rect.width / 600, rect.height / 340)
        origin_x = rect.x + (rect.width - 600 * factor) / 2
        origin_y = rect.y + (rect.height - 340 * factor) / 2
        horizontal = ((event.x - origin_x) / factor - self.offset_x) / (width * self.scale)
        vertical = ((event.y - origin_y) / factor - self.offset_y) / (height * self.scale)
        if self.index < 3 and 0.05 <= horizontal <= 0.95 and 0.05 <= vertical <= 0.95:
            self.template.anchors[ANCHOR_NAMES[self.index]] = [horizontal, vertical]
            self.index += 1
            self._refresh()
        return gui.Widget.EventCallbackResult.HANDLED

    def _reset(self) -> None:
        self.index = 0
        self._refresh()

    def _accept(self) -> None:
        try:
            validated = CartoonTemplate(self.template.rgba, self.template.anchors)
            self.accept(validated)
            gui.Application.instance.post_to_main_thread(self.window, self.window.close)
        except (OSError, ValueError) as error:
            self.instruction.text = str(error)
            self.window.post_redraw()

    def _close(self) -> bool:
        self.closed = True
        return True
