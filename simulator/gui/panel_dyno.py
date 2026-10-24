"""The dyno chart: a live torque and power plot with a progress bar."""

from __future__ import annotations

import math

from imgui_bundle import imgui

from ..physics import TWO_PI
from . import theme, widgets
from .state import summarise


def _nice(value: float) -> float:
    """Round a scale maximum up to a readable 1-2-5 step."""
    if value <= 0:
        return 1.0
    step = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= step * factor:
            return step * factor
    return step * 10.0


def draw_dyno(app, state, w: float, h: float, pos=None) -> None:
    spec = state.spec
    if spec is None:
        return

    widgets.panel("dyno", w, h, accent=theme.ORANGE, pos=pos)
    try:
        dl = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        inner_w = avail.x

        widgets.label_at(pos, "dyno pull")
        legend_x = pos.x + 90.0
        for text, color in (("torque, Nm", theme.ORANGE), ("power, kW", theme.YELLOW)):
            widgets.label_at(imgui.ImVec2(legend_x, pos.y), text, color)
            legend_x += imgui.calc_text_size(text.upper()).x + 34.0

        # summary card on the right
        summary = state.dyno_summary or summarise(state.dyno_points)
        if summary:
            t_line = f"{summary['peak_torque']:.0f} Nm @ {summary['peak_torque_rpm']:.0f} rpm"
            p_line = (
                f"{summary['peak_power_kw']:.1f} kW · {summary['peak_power_hp']:.0f} hp"
                f" @ {summary['peak_power_rpm']:.0f} rpm"
            )
            tw = max(imgui.calc_text_size(t_line).x, imgui.calc_text_size(p_line).x)
            y = pos.y + avail.y - 2.0 * imgui.get_text_line_height() - 4.0
            dl.add_text(imgui.ImVec2(pos.x + inner_w - tw, y), widgets._color_u32(theme.ORANGE), t_line)
            dl.add_text(
                imgui.ImVec2(pos.x + inner_w - tw, y + imgui.get_text_line_height() + 2.0),
                widgets._color_u32(theme.YELLOW),
                p_line,
            )

        # plot area
        top = pos.y + 26.0
        left = pos.x + 44.0
        right = pos.x + inner_w - 40.0
        bottom = pos.y + avail.y - 30.0
        if right - left < 60 or bottom - top < 30:
            return

        x_max = _nice(max(spec.redline_rpm * 1.04, 1200.0))
        t_max = _nice(max(50.0, summary.get("peak_torque", 0.0) * 1.18 if summary else 50.0))
        p_max = _nice(max(20.0, summary.get("peak_power_kw", 0.0) * 1.18 if summary else 20.0))
        k_t = (bottom - top) / t_max
        k_p = (bottom - top) / p_max

        def sx(rpm: float) -> float:
            return left + (right - left) * max(0.0, min(1.0, rpm / x_max))

        # grid
        for i in range(6):
            f = i / 5.0
            gx = left + (right - left) * f
            dl.add_line(
                imgui.ImVec2(gx, top), imgui.ImVec2(gx, bottom),
                widgets._color_u32(theme.EDGE, 0.7), 1.0,
            )
            text = f"{int(x_max * f)}"
            ts = imgui.calc_text_size(text)
            dl.add_text(
                imgui.ImVec2(gx - ts.x * 0.5, bottom + 6.0),
                widgets._color_u32(theme.INK_FAINT), text,
            )
        for i in range(5):
            f = i / 4.0
            gy = bottom - (bottom - top) * f
            dl.add_line(
                imgui.ImVec2(left, gy), imgui.ImVec2(right, gy),
                widgets._color_u32(theme.EDGE, 0.7), 1.0,
            )
            tt = f"{int(t_max * f)}"
            ts = imgui.calc_text_size(tt)
            dl.add_text(
                imgui.ImVec2(left - ts.x - 6.0, gy - ts.y * 0.5),
                widgets._color_u32(theme.ORANGE, 0.8), tt,
            )
            pt = f"{int(p_max * f)}"
            ps = imgui.calc_text_size(pt)
            dl.add_text(
                imgui.ImVec2(right + 6.0, gy - ps.y * 0.5),
                widgets._color_u32(theme.YELLOW, 0.8), pt,
            )

        # redline marker
        rx = sx(spec.redline_rpm)
        if rx < right:
            dl.add_line(
                imgui.ImVec2(rx, top), imgui.ImVec2(rx, bottom),
                widgets._color_u32(theme.RED, 0.5), 1.0,
            )
            widgets.label_at(imgui.ImVec2(rx - 56.0, top - 2.0), "redline", theme.RED)

        # curves
        points = state.dyno_points
        if len(points) >= 2:
            torque_pts = [(sx(rpm), bottom - t * k_t) for rpm, t in points]
            power_pts = [
                (sx(rpm), bottom - (rpm * t * TWO_PI / 60.0 / 1000.0) * k_p)
                for rpm, t in points
            ]
            # torque fill under the curve
            dl.path_clear()
            for pnt in torque_pts:
                dl.path_line_to(imgui.ImVec2(*pnt))
            dl.path_line_to(imgui.ImVec2(torque_pts[-1][0], bottom))
            dl.path_line_to(imgui.ImVec2(torque_pts[0][0], bottom))
            dl.path_fill_convex(widgets._color_u32(theme.ORANGE, 0.10))
            for pts, color, width in ((power_pts, theme.YELLOW, 2.0), (torque_pts, theme.ORANGE, 2.5)):
                dl.path_clear()
                for pnt in pts:
                    dl.path_line_to(imgui.ImVec2(*pnt))
                dl.path_stroke(widgets._color_u32(color), width, imgui.ImDrawFlags_.none)

        elif state.dyno_message and not points:
            msg = state.dyno_message
            ms = imgui.calc_text_size(msg)
            dl.add_text(
                imgui.ImVec2((left + right - ms.x) * 0.5, (top + bottom - ms.y) * 0.5),
                widgets._color_u32(theme.INK_DIM),
                msg,
            )

        # progress bar while the pull runs
        if state.dyno_state:
            bar_y = bottom + imgui.get_text_line_height() + 10.0
            dl.add_rect_filled(
                imgui.ImVec2(left, bar_y), imgui.ImVec2(right, bar_y + 4.0),
                widgets._color_u32(theme.TRACK), 2.0,
            )
            dl.add_rect_filled(
                imgui.ImVec2(left, bar_y),
                imgui.ImVec2(left + (right - left) * max(0.0, min(1.0, state.dyno_progress)), bar_y + 4.0),
                widgets._color_u32(theme.ORANGE), 2.0,
            )
    finally:
        widgets.end_panel()
