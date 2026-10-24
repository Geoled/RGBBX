"""The instrument cluster: dials, meters and the live state chips."""

from __future__ import annotations

from imgui_bundle import imgui

from ..physics import TWO_PI
from . import theme, widgets


def draw_telemetry(app, state, w: float, h: float, pos=None) -> None:
    model = state.model
    spec = state.spec
    if model is None or spec is None:
        return

    widgets.panel("cluster", w, h, accent=theme.ORANGE, pos=pos)
    try:
        avail = imgui.get_content_region_avail()
        inner_w = avail.x

        dyno_active = model.dyno.active
        shown_torque = model.brake_torque if dyno_active else model.net_torque
        shown_power_kw = max(0.0, shown_torque * model.rpm * TWO_PI / 60.0 / 1000.0)

        turbine = spec.kind == "turbine"
        if turbine:
            t_max, p_max = 2400.0, 6400.0
        else:
            t_max = max(model.full_load_torque * 1.18, 40.0)
            p_max = max(t_max * spec.redline_rpm * 0.62 * TWO_PI / 60.0 / 1000.0, 20.0)

        # --- two dials side by side -------------------------------------- #
        dial_r = max(46.0, min(inner_w * 0.25, 86.0))
        cy = imgui.get_cursor_screen_pos().y + dial_r + 14.0
        base_x = imgui.get_cursor_screen_pos().x
        widgets.dial(
            imgui.ImVec2(base_x + dial_r + 6.0, cy),
            dial_r,
            model.rpm,
            0.0,
            spec.redline_rpm * 1.05,
            unit="rpm",
            title="engine speed",
            redline=spec.redline_rpm,
            accent=theme.ORANGE,
            ticks=7,
            peak=state.rpm_peak,
        )
        widgets.dial(
            imgui.ImVec2(base_x + dial_r * 2 + 30.0 + dial_r + 6.0, cy),
            dial_r,
            shown_power_kw,
            0.0,
            p_max,
            unit="kW",
            title="power",
            accent=theme.YELLOW,
            ticks=5,
            peak=state.kw_peak,
        )
        imgui.dummy(imgui.ImVec2(inner_w, dial_r * 2 + 24.0))

        # --- meters ------------------------------------------------------- #
        bar_w = inner_w
        widgets.meter(
            "throttle", model.throttle, f"{model.throttle * 100:.0f} %",
            accent=theme.ORANGE, w=bar_w,
        )
        widgets.meter(
            "manifold",
            (model.map_bar - 0.10) / max(1.05 + spec.boost_bar - 0.10, 0.2),
            f"{model.map_bar:.2f} bar",
            accent=theme.BLUE, w=bar_w,
        )
        widgets.meter(
            "load", model.sound_load, f"{model.sound_load * 100:.0f} %",
            accent=theme.ORANGE, w=bar_w,
        )
        widgets.meter(
            "torque",
            shown_torque / t_max,
            f"{shown_torque:.0f} Nm",
            accent=theme.YELLOW, warn_from=0.9, w=bar_w,
        )
        if spec.boost_bar > 0.0 or model.boost_bar > 0.0:
            widgets.meter(
                "boost", model.boost_bar / max(spec.boost_bar, 0.05),
                f"{model.boost_bar:.2f} bar",
                accent=theme.BLUE, w=bar_w,
            )

        # --- readout rows ------------------------------------------------- #
        imgui.dummy(imgui.ImVec2(0, 4))
        rows = [
            ("peak torque", f"{model.full_load_torque:.0f} Nm", theme.ORANGE),
            (
                "power",
                f"{shown_power_kw:.1f} kW · {shown_power_kw * 1000 / 745.7:.0f} hp",
                theme.YELLOW,
            ),
        ]
        if spec.kind == "petrol":
            rows.append(
                (
                    "knock",
                    f"{model.knock_level * 100:.0f} % · retard {model.knock_retard * 100:.0f} %",
                    theme.RED if model.knock_level > 0.04 else theme.INK_DIM,
                )
            )
        if spec.kind == "turbine":
            rows.append(("shaft power", f"{model.turbine_power_kw:.0f} kW", theme.BLUE))
        for label_text, value, color in rows:
            widgets.caption(label_text, value, color=theme.INK_FAINT, vcolor=color, w=bar_w)

        # --- state chips --------------------------------------------------- #
        # A flowing row: each chip reserves its own box, then SameLine packs
        # the next one right after it, wrapping naturally at the panel edge.
        # The last SameLine of a row is harmless: imgui clips the empty gap.
        imgui.dummy(imgui.ImVec2(0, 6))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(6, 4))
        chips = []
        if not state.ignition:
            chips.append(("IGNITION OFF", theme.RED, True))
        elif model.running:
            chips.append(("RUNNING", theme.GREEN, True))
        else:
            chips.append(("STOPPED", theme.INK_DIM, False))
        if model.limiter_cut:
            chips.append(("LIMITER", theme.RED, True))
        if model.overrun_cut:
            chips.append(("OVERRUN CUT", theme.YELLOW, True))
        if state.backfire_flash > 0.0:
            chips.append(("BACKFIRE", theme.YELLOW, True))
        if state.knock_flash > 0.0:
            chips.append(("KNOCK", theme.RED, True))
        if state.dyno_state:
            chips.append(("DYNO PULL", theme.ORANGE, True))
        if state.audio is None and state.audio_wanted:
            chips.append(("NO AUDIO", theme.RED, True))
        for i, (text, color, lit) in enumerate(chips):
            widgets.chip(text, color=color, lit=lit)
            if i < len(chips) - 1:
                imgui.same_line()
        imgui.pop_style_var()
    finally:
        widgets.end_panel()