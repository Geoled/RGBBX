"""The dyno chart: a live torque and power plot on the simulator's own screen."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from ..physics import TWO_PI
from . import theme


def summarise(points: list[tuple[float, float]]) -> dict:
    """Peak figures for a curve of ``(rpm, brake torque)`` points."""
    if not points:
        return {}

    def power_kw(point: tuple[float, float]) -> float:
        return point[0] * point[1] * TWO_PI / 60.0 / 1000.0

    peak_t_rpm, peak_t = max(points, key=lambda p: p[1])
    peak_p_rpm, peak_p = max(points, key=power_kw)
    return {
        "peak_torque": peak_t,
        "peak_torque_rpm": peak_t_rpm,
        "peak_power_kw": power_kw((peak_p_rpm, peak_p)),
        "peak_power_rpm": peak_p_rpm,
        "peak_power_hp": power_kw((peak_p_rpm, peak_p)) * 1000.0 / 745.7,
    }


def _nice(value: float) -> float:
    """Round a scale maximum up to a readable 1-2-5 step."""
    if value <= 0:
        return 1.0
    step = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= step * factor:
            return step * factor
    return step * 10.0


def _legend(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color, anchor: str = "ra") -> float:
    face = theme.font(11, bold=True)
    w, _ = theme.text_size(draw, text, face)
    if anchor == "ra":
        draw.text((x, y), text, font=face, fill=color, anchor="ra")
        draw.rounded_rectangle([x - w - 16, y - 4, x - w - 8, y + 4], radius=4, fill=color)
        return x - w - 24
    draw.rounded_rectangle([x, y - 4, x + 8, y + 4], radius=4, fill=color)
    draw.text((x + 14, y), text, font=face, fill=color, anchor="la")
    return x + w + 28


def draw_chart(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    points: list[tuple[float, float]],
    *,
    redline_rpm: float,
    torque_label: str = "torque, Nm",
    power_label: str = "power, kW",
    running: bool = False,
    progress: float = 0.0,
    message: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    pad_l, pad_r, pad_t, pad_b = 58, 64, 28, 24
    left, right = x0 + pad_l, x1 - pad_r
    top, bottom = y0 + pad_t, y1 - pad_b
    if right - left < 60 or bottom - top < 40:
        return

    draw.text((x0, y0 + 2), "DYNO PULL", font=theme.font(13, bold=True), fill=theme.INK, anchor="la")
    _legend(draw, x0 + 118, y0 + 8, torque_label, theme.ACCENT, anchor="la")
    _legend(draw, x0 + 236, y0 + 8, power_label, theme.WARM, anchor="la")

    x_max = _nice(max(redline_rpm * 1.04, 1200.0))
    summary = summarise(points)
    t_max = _nice(max(50.0, (summary.get("peak_torque", 0.0) or 0.0) * 1.18))
    p_max = _nice(max(20.0, (summary.get("peak_power_kw", 0.0) or 0.0) * 1.18))
    k_t = (bottom - top) / t_max
    k_p = (bottom - top) / p_max

    def sx(rpm: float) -> float:
        return left + (right - left) * theme.clamp01(rpm / x_max)

    # inset plot well
    well = [left - 10, top - 8, right + 10, bottom + 6]
    draw.rounded_rectangle(well, radius=12, fill=theme.mix(theme.CARD_TOP, (255, 255, 255), 0.4))
    draw.rounded_rectangle(well, radius=12, outline=theme.EDGE, width=1)

    grid = theme.mix(theme.CARD_TOP, theme.EDGE, 0.75)
    face = theme.font(11)
    for i in range(6):
        f = i / 5.0
        gx = left + (right - left) * f
        draw.line([gx, top, gx, bottom], fill=grid, width=1)
        draw.text((gx, bottom + 8), f"{int(x_max * f)}", font=face, fill=theme.INK_FAINT, anchor="ma")
    for i in range(5):
        f = i / 4.0
        gy = bottom - (bottom - top) * f
        draw.line([left, gy, right, gy], fill=grid, width=1)
        draw.text((left - 8, gy), f"{int(t_max * f)}", font=face, fill=theme.ACCENT, anchor="rm")
        draw.text((right + 8, gy), f"{int(p_max * f)}", font=face, fill=theme.WARM, anchor="lm")

    draw.text((x0 + 8, bottom + 8), "rpm", font=face, fill=theme.INK_DIM, anchor="la")

    rx = sx(redline_rpm)
    if rx < right:
        draw.line([rx, top, rx, bottom], fill=theme.mix(theme.CARD_TOP, theme.DANGER, 0.55), width=1)
        draw.text((rx - 4, top - 16), "redline", font=face, fill=theme.DANGER, anchor="ra")

    if len(points) >= 2:
        torque_pts = [(sx(rpm), bottom - t * k_t) for rpm, t in points]
        power_pts = [(sx(rpm), bottom - (rpm * t * TWO_PI / 60.0 / 1000.0) * k_p) for rpm, t in points]
        draw.polygon(
            torque_pts + [(torque_pts[-1][0], bottom), (torque_pts[0][0], bottom)],
            fill=theme.mix(theme.CARD_TOP, theme.ACCENT, 0.13),
        )
        draw.line(power_pts, fill=theme.WARM, width=2, joint="curve")
        draw.line(torque_pts, fill=theme.ACCENT, width=3, joint="curve")

        if summary:
            # the peak figures sit on their own card so they never fight the
            # curves for legibility at the low-rpm end of the plot
            lines = [
                (
                    f"{summary['peak_torque']:.0f} Nm @ {summary['peak_torque_rpm']:.0f}",
                    theme.ACCENT,
                ),
                (
                    f"{summary['peak_power_kw']:.1f} kW / {summary['peak_power_hp']:.0f} hp"
                    f" @ {summary['peak_power_rpm']:.0f}",
                    theme.WARM,
                ),
            ]
            face = theme.font(15, bold=True)
            widths = [theme.text_size(draw, text, face)[0] for text, _ in lines]
            cx0, cy0 = left + 12, top + 8
            cx1 = cx0 + max(widths) + 20
            cy1 = cy0 + len(lines) * 22 + 8
            draw.rounded_rectangle(
                [cx0, cy0, cx1, cy1],
                radius=8,
                fill=theme.mix(theme.CARD_TOP, (255, 255, 255), 0.72),
                outline=theme.EDGE,
            )
            for i, (text, color) in enumerate(lines):
                draw.text((cx0 + 10, cy0 + 6 + i * 22), text, font=face, fill=color, anchor="la")

    if running:
        track = [left, bottom + 16, right, bottom + 20]
        draw.rounded_rectangle(track, radius=2, fill=theme.TRACK)
        draw.rounded_rectangle(
            [left, bottom + 16, left + (right - left) * theme.clamp01(progress), bottom + 20],
            radius=2,
            fill=theme.ACCENT_2,
        )
    elif message and not points:
        draw.text(
            ((left + right) * 0.5, (top + bottom) * 0.5),
            message,
            font=theme.font(14),
            fill=theme.INK_FAINT,
            anchor="mm",
        )
