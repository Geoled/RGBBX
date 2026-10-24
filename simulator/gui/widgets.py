"""Custom dashboard widgets drawn straight onto the imgui DrawList.

None of these are stock imgui widgets: a dyno-cell interface needs a dial whose
sweep matches the engine's actual redline, bars with peak-hold markers and state
chips that light up only while a condition is live. All of them are immediate
mode -- they take the current values and draw, no retained state -- so there is
nothing to invalidate and nothing to sync.

Everything here is drawn in the current window's local coordinates; callers use
the layout helpers to lay the panels out.
"""

from __future__ import annotations

import math

from imgui_bundle import imgui

from . import theme

# The dial sweeps clockwise over this arc (screen angles: 0 = +x, y is down).
_DIAL_START = 135.0
_DIAL_SPAN = 270.0


def _color_u32(color, alpha: float = 1.0):
    """imgui's packed ABG R for the DrawList calls, with multiplied alpha."""
    base_a = color[3] if len(color) > 3 else 1.0
    a = int(max(0.0, min(1.0, base_a * alpha)) * 255.0)
    r = int(max(0.0, min(1.0, color[0])) * 255.0)
    g = int(max(0.0, min(1.0, color[1])) * 255.0)
    b = int(max(0.0, min(1.0, color[2])) * 255.0)
    return (a << 24) | (b << 16) | (g << 8) | r


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #
def rect(w: float, h: float):
    """Reserve an item box and return its (top-left, bottom-right) corners."""
    size = imgui.ImVec2(w, h)
    cursor = imgui.get_cursor_screen_pos()
    imgui.dummy(size)
    return cursor, imgui.ImVec2(cursor.x + w, cursor.y + h)


def panel(title: str, w: float, h: float, accent=theme.ORANGE, *, pos=None) -> None:
    """A framed card at an explicit position (or the flowing cursor).

    These are real top-level windows, not child windows: a child window clips
    its DrawList commands to its own rect *after* they are emitted, which is
    exactly what made the header and panel frames disappear in the first
    screenshots. A plain window with NoDecoration + NoSavedSettings draws
    everything and stays out of the .ini file.

    ``Always`` position/size means the layout is owned by the code, so the
    grid cannot be dragged apart by a stray mouse gesture.
    """
    import imgui_bundle  # noqa: F401  (keep the import local and cheap)

    from imgui_bundle import imgui as _imgui

    flags = (
        _imgui.WindowFlags_.no_title_bar
        | _imgui.WindowFlags_.no_resize
        | _imgui.WindowFlags_.no_move
        | _imgui.WindowFlags_.no_collapse
        | _imgui.WindowFlags_.no_scrollbar
        | _imgui.WindowFlags_.no_scroll_with_mouse
        | _imgui.WindowFlags_.no_saved_settings
        | _imgui.WindowFlags_.no_nav
        | _imgui.WindowFlags_.no_bring_to_front_on_focus
    )
    if pos is not None:
        _imgui.set_next_window_pos(_imgui.ImVec2(pos[0], pos[1]), _imgui.Cond_.always)
        _imgui.set_next_window_size(_imgui.ImVec2(w, h), _imgui.Cond_.always)
    _imgui.push_style_var(_imgui.StyleVar_.window_rounding, 0.0)
    _imgui.push_style_var(_imgui.StyleVar_.window_border_size, 0.0)
    _imgui.push_style_var(_imgui.StyleVar_.window_padding, _imgui.ImVec2(12, 12))
    _imgui.push_style_color(_imgui.Col_.window_bg, theme.PANEL_B)
    _imgui.begin(title, None, flags)
    _imgui.pop_style_color()
    _imgui.pop_style_var(3)
    _push_window()

    dl = _imgui.get_window_draw_list()
    p = _imgui.get_cursor_screen_pos()
    p = _imgui.ImVec2(p.x - 12, p.y - 12)
    avail = _imgui.get_content_region_avail()
    full = _imgui.ImVec2(p.x + avail.x + 12, p.y + avail.y + 12)

    dl.add_rect_filled(p, full, _color_u32(theme.PANEL_B), 0.0)
    dl.add_rect(p, full, _color_u32(theme.EDGE), 0.0)
    dl.add_line(
        _imgui.ImVec2(p.x + 1, p.y + 1),
        _imgui.ImVec2(full.x - 1, p.y + 1),
        _color_u32(theme.EDGE_HOT),
    )
    dl.add_rect_filled(
        _imgui.ImVec2(p.x, p.y + 1), _imgui.ImVec2(p.x + 3, p.y + 18), _color_u32(accent)
    )


def end_panel() -> None:
    from imgui_bundle import imgui as _imgui

    _pop_window()
    _imgui.end()


def overlay_layer():
    """A draw list that is valid with *no* window open."""
    return imgui.get_foreground_draw_list()


# How many of our panels are currently open. Tracked here rather than asked of
# imgui: GetCurrentWindow() keeps pointing at the *last* window after End(), so
# a "is a window open?" test based on it says yes during the header bar and the
# following GetWindowDrawList() then opens imgui's fallback window
# ("Debug##Default") on top of the UI. Our own stack is exact.
_DEPTH = 0


def window_layer():
    """The current panel's draw list -- only call between begin and end."""
    return imgui.get_window_draw_list()


def in_window() -> bool:
    """True while one of our panels (or the catalogue) is open."""
    return _DEPTH > 0


def _push_window() -> None:
    global _DEPTH
    _DEPTH += 1


def _pop_window() -> None:
    global _DEPTH
    if _DEPTH > 0:
        _DEPTH -= 1


def label_at(pos, text: str, color=theme.INK_FAINT, dl=None) -> None:
    """Spaced capitals at an absolute position (panel headers etc.).

    Inside a panel the label goes to that panel's list, so it is clipped and
    ordered with the panel (and the catalogue modal covers it). Outside every
    window -- the header bar is drawn before the panels open -- it goes to the
    foreground list, which needs no window at all.
    """
    if dl is not None:
        target = dl
    elif in_window():
        target = imgui.get_window_draw_list()
    else:
        target = overlay_layer()
    target.add_text(pos, _color_u32(color), text.upper())


def caption(text: str, value_text: str = "", *, color=theme.INK_FAINT, vcolor=theme.INK, w: float = 240.0) -> None:
    """A row: caption left, value right. Returns nothing."""
    dl = imgui.get_window_draw_list()
    p = imgui.get_cursor_screen_pos()
    h = imgui.get_text_line_height()
    imgui.dummy(imgui.ImVec2(w, h))
    dl.add_text(p, _color_u32(color), text.upper())
    if value_text:
        tw = imgui.calc_text_size(value_text).x
        dl.add_text(imgui.ImVec2(p.x + w - tw, p.y), _color_u32(vcolor), value_text)


# --------------------------------------------------------------------------- #
# dial gauge
# --------------------------------------------------------------------------- #
def dial(
    centre,
    radius: float,
    value: float,
    vmin: float,
    vmax: float,
    *,
    unit: str = "",
    title: str = "",
    redline: float | None = None,
    accent=theme.ORANGE,
    thickness: float = 9.0,
    ticks: int = 5,
    peak: float | None = None,
) -> None:
    """A ring gauge: thin arc over a hairline track, value in the middle.

    The redline zone is drawn as its own red arc segment, so an engine's whole
    personality (idle placement, powerband width, how close the limiter sits)
    reads from across the room.

    Note: this binding's ``path_stroke`` argument order is
    ``(col, thickness, flags)``, unlike C++ ``(col, flags, thickness)``.
    """
    dl = imgui.get_window_draw_list()
    span = _DIAL_SPAN
    scale = (vmax - vmin) or 1.0

    def ang(v: float) -> float:
        return math.radians(_DIAL_START + span * max(0.0, min(1.0, (v - vmin) / scale)))

    r = radius - thickness * 0.5

    # track
    dl.path_clear()
    dl.path_arc_to(centre, r, math.radians(_DIAL_START), math.radians(_DIAL_START + span), 0)
    dl.path_stroke(_color_u32(theme.TRACK), thickness, imgui.ImDrawFlags_.none)

    if redline is not None and redline < vmax:
        dl.path_clear()
        dl.path_arc_to(centre, r, ang(redline), math.radians(_DIAL_START + span), 0)
        dl.path_stroke(_color_u32(theme.RED_DIM, 0.95), thickness, imgui.ImDrawFlags_.none)
        dl.path_clear()
        dl.path_arc_to(centre, r, ang(redline), math.radians(_DIAL_START + span), 0)
        dl.path_stroke(_color_u32(theme.RED, 0.6), thickness * 0.45, imgui.ImDrawFlags_.none)

    # value arc
    frac = max(0.0, min(1.0, (value - vmin) / scale))
    if frac > 0.003:
        dl.path_clear()
        dl.path_arc_to(centre, r, math.radians(_DIAL_START), ang(value), 0)
        dl.path_stroke(_color_u32(accent), thickness, imgui.ImDrawFlags_.none)

    # peak-hold ghost marker
    if peak is not None and peak > vmin + (vmax - vmin) * 0.01:
        pp = ang(peak)
        cos_a, sin_a = math.cos(pp), math.sin(pp)
        dl.add_line(
            imgui.ImVec2(centre.x + cos_a * (r - thickness * 0.72), centre.y + sin_a * (r - thickness * 0.72)),
            imgui.ImVec2(centre.x + cos_a * (r + thickness * 0.72), centre.y + sin_a * (r + thickness * 0.72)),
            _color_u32(theme.mix(accent, theme.INK, 0.4), 0.85),
            1.5,
        )

    # ticks
    step = max(1, ticks // 4)
    for i in range(ticks + 1):
        a = math.radians(_DIAL_START + span * i / ticks)
        major = i % step == 0 or i == ticks
        r_in = r - thickness * (0.85 if major else 0.6)
        r_out = r - thickness * (1.5 if major else 1.2)
        dl.add_line(
            imgui.ImVec2(centre.x + math.cos(a) * r_in, centre.y + math.sin(a) * r_in),
            imgui.ImVec2(centre.x + math.cos(a) * r_out, centre.y + math.sin(a) * r_out),
            _color_u32(theme.INK_DIM if major else theme.INK_OFF),
            1.0,
        )

    # number in the middle: big tabular figures
    text = f"{value:,.0f}"
    ts = imgui.calc_text_size(text)
    dl.add_text(imgui.ImVec2(centre.x - ts.x * 0.5, centre.y + radius * 0.20), _color_u32(theme.INK), text)
    if unit:
        us = imgui.calc_text_size(unit)
        dl.add_text(imgui.ImVec2(centre.x - us.x * 0.5, centre.y + radius * 0.20 + ts.y), _color_u32(theme.INK_FAINT), unit)
    if title:
        t = title.upper()
        ts2 = imgui.calc_text_size(t)
        dl.add_text(imgui.ImVec2(centre.x - ts2.x * 0.5, centre.y - radius * 0.58), _color_u32(theme.INK_FAINT), t)


# --------------------------------------------------------------------------- #
# horizontal meter bar
# --------------------------------------------------------------------------- #
def meter(
    caption_text: str,
    fraction: float,
    text: str,
    *,
    accent=theme.ORANGE,
    warn_from: float | None = None,
    peak: float | None = None,
    w: float = 240.0,
    h: float = 24.0,
) -> None:
    """A labelled bar with the numeric readout on the right, peak tick optional."""
    dl = imgui.get_window_draw_list()
    p = imgui.get_cursor_screen_pos()
    imgui.dummy(imgui.ImVec2(w, h))

    line_h = imgui.get_text_line_height()
    label_y = p.y
    dl.add_text(imgui.ImVec2(p.x, label_y), _color_u32(theme.INK_FAINT), caption_text.upper())
    tw = imgui.calc_text_size(text).x
    dl.add_text(imgui.ImVec2(p.x + w - tw, label_y), _color_u32(theme.INK), text)

    bar_y = p.y + line_h + 6.0
    bar_h = 5.0
    dl.add_rect_filled(imgui.ImVec2(p.x, bar_y), imgui.ImVec2(p.x + w, bar_y + bar_h), _color_u32(theme.TRACK), 2.0)

    frac = max(0.0, min(1.0, fraction))
    if frac > 0.0:
        color = accent
        if warn_from is not None and frac >= warn_from:
            color = theme.YELLOW if frac < 0.97 else theme.RED
        dl.add_rect_filled(imgui.ImVec2(p.x, bar_y), imgui.ImVec2(p.x + w * frac, bar_y + bar_h), _color_u32(color), 2.0)

    if peak is not None and 0.005 < peak < 0.995:
        px = p.x + w * max(0.0, min(1.0, peak))
        dl.add_line(
            imgui.ImVec2(px, bar_y - 1.5), imgui.ImVec2(px, bar_y + bar_h + 1.5),
            _color_u32(theme.INK, 0.5), 1.0,
        )


# --------------------------------------------------------------------------- #
# state chip
# --------------------------------------------------------------------------- #
def chip(text: str, *, color=theme.INK_DIM, lit: bool = False) -> None:
    """A small status pill. Lit = filled with the signal colour."""
    dl = imgui.get_window_draw_list()
    pad_x, pad_y = 7.0, 2.5
    size = imgui.calc_text_size(text)
    w = size.x + pad_x * 2
    h = size.y + pad_y * 2
    p = imgui.get_cursor_screen_pos()
    imgui.dummy(imgui.ImVec2(w + 6.0, h))
    b = imgui.ImVec2(p.x + w, p.y + h)
    if lit:
        dl.add_rect_filled(p, b, _color_u32(color, 0.20), 3.0)
        dl.add_rect(p, b, _color_u32(color), 3.0)
        dl.add_text(imgui.ImVec2(p.x + pad_x, p.y + pad_y), _color_u32(color), text)
    else:
        dl.add_rect(p, b, _color_u32(theme.EDGE_HOT), 3.0)
        dl.add_text(imgui.ImVec2(p.x + pad_x, p.y + pad_y), _color_u32(theme.INK_OFF), text)
