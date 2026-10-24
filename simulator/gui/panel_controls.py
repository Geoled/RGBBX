"""The control deck: throttle, starter, ignition, limiter, brake and volume.

The deck is one row of fat, keyboard-mirrored controls -- the things you touch
while listening -- a full-width throttle rail, then the secondary sliders and
switches. Every button shows its shortcut on its face, because an interface
that has to be discovered is an interface that fails.

Layout is explicit: each row is placed with SetCursorScreenPos at a known
offset from the panel's origin, so nothing can silently slide out of the
window (the previous flow layout overflowed the 176 px card and the sliders
ended up below the clip rect -- unclickable).
"""

from __future__ import annotations

from imgui_bundle import imgui

from . import theme, widgets

# Row geometry, in pixels from the panel's content origin (after 12 px pad).
ROW_TOP = 18.0     # engine picker + action buttons
ROW_THROTTLE = 66.0   # full-width throttle rail
ROW_SLIDERS = 108.0   # brake | volume
ROW_SWITCHES = 150.0  # ignition | limiter | ghost | re-frame
SLIDER_H = 22.0


def _button(label: str, shortcut: str, *, active: bool = False, w: float = -1.0, accent=theme.ORANGE) -> bool:
    """A big deck button with the shortcut printed on its face."""
    imgui.push_style_color(imgui.Col_.button, theme.PANEL_RAISED if not active else theme.with_alpha(accent, 0.30))
    imgui.push_style_color(imgui.Col_.button_hovered, theme.mix(theme.PANEL_RAISED, theme.INK, 0.12))
    imgui.push_style_color(imgui.Col_.button_active, theme.with_alpha(accent, 0.45))
    imgui.push_style_color(imgui.Col_.text, theme.INK if not active else accent)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.0)
    size = imgui.ImVec2(w, 40.0) if w > 0 else imgui.ImVec2(0, 40.0)
    clicked = imgui.button(label, size)
    imgui.pop_style_var()
    imgui.pop_style_color(4)
    if shortcut:
        dl = imgui.get_window_draw_list()
        p = imgui.get_item_rect_min()
        sz = imgui.get_item_rect_size()
        text = shortcut.upper()
        ts = imgui.calc_text_size(text)
        dl.add_text(
            imgui.ImVec2(p.x + sz.x - ts.x - 6.0, p.y + sz.y - ts.y - 4.0),
            widgets._color_u32(theme.INK_FAINT, 0.9),
            text,
        )
    return clicked


def _rail(label: str, value: float, origin, w: float, *, accent, fmt: str = "%.0f%%") -> tuple[bool, float]:
    """A full-width slider rail: caption above the grab area, value on the right.

    The stock slider is 20 px tall and eats horizontal padding for its label;
    drawing the caption as text keeps the grab area the full width.
    """
    dl = imgui.get_window_draw_list()
    line_h = imgui.get_text_line_height()
    # caption + live value
    widgets.label_at(imgui.ImVec2(origin[0], origin[1]), label, theme.INK_FAINT)
    value_text = fmt % (value * 100.0)
    vw = imgui.calc_text_size(value_text).x
    dl.add_text(
        imgui.ImVec2(origin[0] + w - vw, origin[1]),
        widgets._color_u32(accent),
        value_text,
    )
    # the slider itself, in its own row below the caption
    imgui.set_cursor_screen_pos(imgui.ImVec2(origin[0], origin[1] + line_h + 3.0))
    imgui.push_item_width(w)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 3.0)
    changed, new = imgui.slider_float(
        "##" + label, value, 0.0, 1.0, "",
        flags=imgui.SliderFlags_.always_clamp | imgui.SliderFlags_.no_input,
    )
    imgui.pop_style_var()
    imgui.pop_item_width()
    return changed, new


def draw_controls(app, state, w: float, h: float, pos=None) -> None:
    widgets.panel("controls", w, h, accent=theme.BLUE, pos=pos)
    try:
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        inner_w = avail.x

        # --- row 1: engine picker + action buttons ------------------------ #
        imgui.set_cursor_screen_pos(imgui.ImVec2(origin.x, origin.y + ROW_TOP))
        current = state._preset_names[state.preset_index] if state._preset_names else "-"
        if len(current) > 34:
            current = current[:32] + "..."

        # The catalogue button: opens the modal browser with previews.
        imgui.push_style_color(imgui.Col_.button, theme.PANEL_RAISED)
        imgui.push_style_color(imgui.Col_.button_hovered, theme.mix(theme.PANEL_RAISED, theme.BLUE, 0.18))
        imgui.push_style_color(imgui.Col_.button_active, theme.with_alpha(theme.BLUE, 0.35))
        imgui.push_style_color(imgui.Col_.text, theme.INK)
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.0)
        if imgui.button("ENGINE CATALOGUE  ##open_catalog", imgui.ImVec2(210.0, 40.0)):
            app.catalogue.show(state)
        imgui.pop_style_var()
        imgui.pop_style_color(4)
        # current engine name beside it
        widgets.label_at(
            imgui.ImVec2(origin.x + 222.0, origin.y + ROW_TOP + 12.0), current, theme.INK_DIM
        )
        dl = imgui.get_window_draw_list()
        _ = dl  # (label_at already draws; kept for symmetry with other panels)

        imgui.same_line()
        imgui.set_cursor_screen_pos(imgui.ImVec2(origin.x + 460.0, origin.y + ROW_TOP))
        if _button("RUN PULL", "D", active=state.dyno_state is not None, w=140.0):
            app.start_dyno()
        imgui.same_line()
        if _button("BLIP", "space", w=100.0):
            app.blip()
        imgui.same_line()
        if _button("CRANK" if not state.cranking else "CRANKING", "S", active=state.cranking, w=130.0):
            state.cranking = not state.cranking
            if state.cranking:
                state.throttle = 0.0

        # --- row 2: throttle rail ------------------------------------------- #
        changed, value = _rail("throttle", state.throttle, (origin.x, origin.y + ROW_THROTTLE), inner_w, accent=theme.ORANGE)
        if changed:
            if state.dyno_state is None:
                state.throttle = value
            else:
                # a pull owns the pedal; ignore edits rather than fight the dyno
                pass

        # --- row 3: brake | volume ------------------------------------------ #
        half = (inner_w - 12.0) * 0.5
        changed, brake = _rail(
            "brake load", state.brake, (origin.x, origin.y + ROW_SLIDERS), half, accent=theme.YELLOW
        )
        if changed:
            state.brake = brake
        changed, vol = _rail(
            "volume", state.volume, (origin.x + half + 12.0, origin.y + ROW_SLIDERS), half, accent=theme.BLUE
        )
        if changed:
            state.volume = vol
            if state.audio is not None:
                state.audio.set_volume(vol)

        # --- row 4: switches -------------------------------------------------- #
        imgui.set_cursor_screen_pos(imgui.ImVec2(origin.x, origin.y + ROW_SWITCHES))
        if _switch("IGNITION", state.ignition, theme.GREEN):
            state.ignition = not state.ignition
            state.status = "ignition " + ("on" if state.ignition else "off")
        imgui.same_line()
        if _switch("LIMITER", state.limiter, theme.RED):
            state.limiter = not state.limiter
            state.status = "limiter " + ("on" if state.limiter else "off")
        imgui.same_line()
        if _switch("GHOST SHELL", app.viewport.show_shell, theme.BLUE):
            app.viewport.show_shell = not app.viewport.show_shell
        imgui.same_line()
        if _switch("RE-FRAME", False, theme.STEEL):
            app.viewport.frame_camera()
    finally:
        widgets.end_panel()


def _switch(label: str, on: bool, color) -> bool:
    """A toggle pill: filled with its signal colour while on."""
    imgui.push_style_color(imgui.Col_.button, theme.with_alpha(color, 0.10) if on else theme.PANEL_SUNK)
    imgui.push_style_color(imgui.Col_.button_hovered, theme.mix(theme.PANEL_RAISED, color, 0.25 if on else 0.08))
    imgui.push_style_color(imgui.Col_.button_active, theme.with_alpha(color, 0.40))
    imgui.push_style_color(imgui.Col_.text, color if on else theme.INK_DIM)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 12.0)
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(12, 5))
    clicked = imgui.button(label)
    imgui.pop_style_var(2)
    imgui.pop_style_color(4)
    return clicked
