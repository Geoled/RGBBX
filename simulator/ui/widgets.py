"""Pillow-drawn dashboard widgets: dials, meters, sliders, buttons, switches.

Everything here draws onto a plain RGB image. Widgets that need a soft surface
(a groove, a button skin, a knob) paste a cached image from :mod:`theme`.
"""

from __future__ import annotations

import math
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

from . import theme

DIAL_START = 142.0
DIAL_END = 398.0


def _arc_cap(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, degrees: float, color, width: int) -> None:
    a = math.radians(degrees)
    x = cx + r * math.cos(a)
    y = cy + r * math.sin(a)
    draw.ellipse([x - width / 2.0, y - width / 2.0, x + width / 2.0, y + width / 2.0], fill=color)


# --------------------------------------------------------------------------- #
# gauges
# --------------------------------------------------------------------------- #
def draw_dial(
    draw: ImageDraw.ImageDraw,
    box,
    value: float,
    vmin: float,
    vmax: float,
    unit: str,
    title: str,
    *,
    redline: float | None = None,
    accent=theme.ACCENT,
    fmt: str = "{:.0f}",
    ticks: int = 9,
) -> None:
    """A Fluent ring gauge: a thin arc over a light track, number in the middle."""
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    radius = 0.5 * min(x1 - x0, y1 - y0) - 6.0
    if radius <= 14.0:
        return
    width = max(7, int(radius * 0.115))
    r = radius - width * 0.5
    span = DIAL_END - DIAL_START
    scale = (vmax - vmin) or 1.0

    def angle(v: float) -> float:
        return DIAL_START + span * theme.clamp01((v - vmin) / scale)

    circle = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(circle, DIAL_START, DIAL_END, fill=theme.TRACK, width=width)
    _arc_cap(draw, cx, cy, r, DIAL_START, theme.TRACK, width)
    _arc_cap(draw, cx, cy, r, DIAL_END, theme.TRACK, width)

    if redline is not None and redline < vmax:
        draw.arc(circle, angle(redline), DIAL_END, fill=theme.DANGER, width=width)
        _arc_cap(draw, cx, cy, r, DIAL_END, theme.DANGER, width)

    now = angle(value)
    if now > DIAL_START + 0.6:
        draw.arc(circle, DIAL_START, now, fill=accent, width=width)
        # a start cap on a near-zero reading is just a floating dot, so it is
        # only drawn once the value arc has some length to round off
        if now > DIAL_START + span * 0.02:
            _arc_cap(draw, cx, cy, r, DIAL_START, accent, width)
        _arc_cap(draw, cx, cy, r, now, accent, width)

    for i in range(ticks):
        a = math.radians(DIAL_START + span * i / (ticks - 1))
        r_in = r - width * 0.70
        r_out = r - width * 1.62
        over = redline is not None and (vmin + (vmax - vmin) * i / (ticks - 1)) >= redline
        draw.line(
            [
                cx + r_in * math.cos(a),
                cy + r_in * math.sin(a),
                cx + r_out * math.cos(a),
                cy + r_out * math.sin(a),
            ],
            fill=theme.DANGER if over else theme.INK_FAINT,
            width=1,
        )

    draw.text(
        (cx, cy - radius * 0.02),
        fmt.format(value),
        font=theme.font(int(radius * 0.56), bold=True),
        fill=theme.INK,
        anchor="mm",
    )
    if unit:
        draw.text(
            (cx, cy + radius * 0.36),
            unit,
            font=theme.font(int(radius * 0.20)),
            fill=theme.INK_DIM,
            anchor="mm",
        )
    if title:
        draw.text(
            (cx, cy - radius * 0.46),
            title,
            font=theme.font(int(radius * 0.19), bold=True),
            fill=theme.INK_FAINT,
            anchor="mm",
        )


def draw_bar(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    fraction: float,
    title: str,
    value: str,
    *,
    color=theme.ACCENT,
    warn_from: float | None = None,
    warn_color=theme.WARM,
) -> None:
    """A labelled horizontal meter with the number on the right."""
    x0, y0, x1, y1 = box
    draw.text((x0, y0), title, font=theme.font(11, bold=True), fill=theme.INK_FAINT, anchor="la")
    draw.text((x1, y0), value, font=theme.font(13, bold=True), fill=theme.INK, anchor="ra")

    top = int(y0 + 17)
    height = 9
    track = theme.groove((int(x1 - x0), height), radius=height // 2)
    img.paste(track, (int(x0), top), track)

    f = theme.clamp01(fraction)
    if f > 0.004:
        fill_color = color
        if warn_from is not None and f >= warn_from:
            fill_color = theme.mix(color, warn_color, (f - warn_from) / max(1e-6, 1.0 - warn_from))
        span = max(int((x1 - x0 - 6) * f), height - 4)
        draw.rounded_rectangle(
            [x0 + 3, top + 3, x0 + 3 + span, top + height - 3],
            radius=(height - 6) // 2,
            fill=fill_color,
        )


def draw_stat(draw: ImageDraw.ImageDraw, box, label: str, value: str, color=theme.INK) -> None:
    """A label/value pair, label faint on the left, value strong on the right."""
    x0, y0, x1, y1 = box
    cy = 0.5 * (y0 + y1)
    draw.text((x0, cy), label, font=theme.font(12), fill=theme.INK_DIM, anchor="lm")
    draw.text((x1, cy), value, font=theme.font(14, bold=True), fill=color, anchor="rm")


def draw_chip(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: str,
    *,
    color=theme.ACCENT_2,
    face=None,
) -> float:
    """A small tinted status pill. Returns the x to continue from."""
    face = face or theme.font(11, bold=True)
    pad = 9
    w, h = theme.text_size(draw, value, face)
    box = [x, y, x + w + pad * 2, y + h + pad]
    draw.rounded_rectangle(box, radius=(h + pad) * 0.5, fill=theme.mix(theme.CARD_TOP, color, 0.17))
    draw.text((x + pad + w * 0.5, y + (h + pad) * 0.5 - 1), value, font=face, fill=color, anchor="mm")
    return box[2] + 6


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _knob(diameter: int = 22, color=theme.ACCENT) -> Image.Image:
    d = int(diameter)
    size = d + 8
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([3, 5, size - 4, size - 2], fill=120)
    mask = mask.filter(ImageFilter.GaussianBlur(2.6))
    img = Image.composite(Image.new("RGBA", (size, size), theme.SHADOW + (255,)), img, mask)
    d2 = ImageDraw.Draw(img)
    d2.ellipse([4, 4, size - 5, size - 5], fill=(255, 255, 255, 255))
    d2.ellipse([4, 4, size - 5, size - 5], outline=color, width=3)
    return img


def draw_slider(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    fraction: float,
    *,
    accent=theme.ACCENT,
    enabled: bool = True,
    hover: bool = False,
) -> None:
    """A neumorphic slider: an inset groove, an accent fill and a round knob."""
    x0, y0, x1, y1 = box
    cy = int(0.5 * (y0 + y1))
    height = 10
    gx0, gx1 = int(x0) + 2, int(x1) - 2
    groove = theme.groove((max(gx1 - gx0, 8), height), radius=height // 2)
    img.paste(groove, (gx0, cy - height // 2), groove)

    knob_r = 11
    travel = max(gx1 - gx0 - 2 * knob_r, 1)
    kx = gx0 + knob_r + travel * theme.clamp01(fraction)
    fill = accent if enabled else theme.INK_FAINT
    if kx > gx0 + knob_r + 2:
        draw.rounded_rectangle(
            [gx0 + 3, cy - height // 2 + 3, kx, cy + height // 2 - 3],
            radius=(height - 6) // 2,
            fill=fill,
        )
    knob = _knob(22, fill)
    img.paste(knob, (int(kx - knob.size[0] * 0.5), int(cy - knob.size[1] * 0.5)), knob)
    if hover and enabled:
        draw.ellipse(
            [kx - knob_r - 3, cy - knob_r - 3, kx + knob_r + 3, cy + knob_r + 3],
            outline=theme.mix(theme.CARD_TOP, accent, 0.45),
            width=1,
        )


def draw_button(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    text: str,
    *,
    primary: bool = False,
    state: str = "normal",
    enabled: bool = True,
) -> None:
    """A raised pill button. ``state`` is normal | hover | pressed."""
    if not enabled:
        state = "disabled"
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return
    skin = theme.skin((w, h), radius=min(12, h // 2), kind=state, primary=primary)
    img.paste(skin, (x0, y0))
    if primary and state != "disabled":
        colour = theme.INK_ON_ACCENT
    elif not enabled:
        colour = theme.INK_FAINT
    else:
        colour = theme.INK
    cy = 0.5 * (y0 + y1) + (-1 if state == "pressed" else 0)
    draw.text((0.5 * (x0 + x1), cy), text, font=theme.font(12, bold=True), fill=colour, anchor="mm")


def draw_toggle(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    on: bool,
    label: str,
    *,
    color=theme.ACCENT,
    enabled: bool = True,
) -> None:
    """A Windows 11 style switch with a label on the left."""
    x0, y0, x1, y1 = (int(v) for v in box)
    cy = 0.5 * (y0 + y1)
    draw.text((x0, cy), label, font=theme.font(12, bold=True), fill=theme.INK if on else theme.INK_DIM, anchor="lm")

    sw, sh = 44, 24
    sx1 = x1
    sx0 = sx1 - sw
    sy0 = int(cy - sh * 0.5)
    track_color = color if (on and enabled) else theme.TRACK
    draw.rounded_rectangle([sx0, sy0, sx1, sy0 + sh], radius=sh // 2, fill=track_color)
    draw.rounded_rectangle(
        [sx0, sy0, sx1, sy0 + sh], radius=sh // 2, outline=theme.mix(track_color, (255, 255, 255), 0.35), width=1
    )
    kr = 9
    kx = sx1 - kr - 4 if on else sx0 + kr + 4
    draw.ellipse(
        [kx - kr, cy - kr, kx + kr, cy + kr],
        fill=(255, 255, 255),
    )


def draw_combo(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    text: str,
    *,
    state: str = "normal",
    sub: str = "",
    enabled: bool = True,
) -> None:
    """A raised select field with a chevron."""
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    skin = theme.skin((w, h), radius=10, kind=state if enabled else "disabled")
    img.paste(skin, (x0, y0))
    cy = 0.5 * (y0 + y1)
    draw.text((x0 + 16, cy if not sub else cy - 7), text, font=theme.font(13, bold=True), fill=theme.INK, anchor="lm")
    if sub:
        draw.text((x0 + 17, cy + 9), sub, font=theme.font(11), fill=theme.INK_FAINT, anchor="lm")

    ax = x1 - 22
    draw.line([ax - 5, cy - 2, ax, cy + 2, ax + 5, cy - 2], fill=theme.INK_DIM, width=2, joint="curve")


def draw_caption(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    title: str,
    value: str = "",
    *,
    right: float | None = None,
    title_color=theme.INK_DIM,
    value_color=theme.INK,
) -> None:
    """A small caption above a control, with an optional right-aligned value."""
    draw.text((x, y), title, font=theme.font(11, bold=True), fill=title_color, anchor="la")
    if value and right is not None:
        draw.text((right, y), value, font=theme.font(12, bold=True), fill=value_color, anchor="ra")
