from collections.abc import Callable

from open3d.visualization import gui


def button(text: str, action: Callable) -> gui.Button:
    widget = gui.Button(text)
    widget.set_on_clicked(action)
    return widget


def row(*widgets) -> gui.Horiz:
    container = gui.Horiz(6)
    for widget in widgets:
        container.add_child(widget)
    return container


def section(title: str, *widgets, opened: bool = True) -> gui.CollapsableVert:
    container = gui.CollapsableVert(title, 6, gui.Margins(4, 5, 4, 10))
    container.background_color = gui.Color(0.055, 0.095, 0.115)
    container.set_is_open(opened)
    for widget in widgets:
        container.add_child(widget)
    return container


def slider(lower: float, upper: float, value: float, action: Callable) -> gui.Slider:
    widget = gui.Slider(gui.Slider.DOUBLE)
    widget.set_limits(lower, upper)
    widget.double_value = value
    widget.set_on_value_changed(action)
    return widget


def combo(items, action: Callable) -> gui.Combobox:
    widget = gui.Combobox()
    for item in items:
        widget.add_item(item)
    widget.set_on_selection_changed(action)
    return widget
