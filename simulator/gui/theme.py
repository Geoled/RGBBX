"""Visual language of the new simulator interface.

A near-black carbon room in which the bright 3D engine is the only loud thing:
everything else is thin hairlines, spaced capitals and tabular figures, the way
a real dyno-cell dashboard reads at a glance. Signal colours have strict
meanings and are never reused decoratively:

* orange -- power and work (torque, power, the tachometer needle)
* blue   -- air and boost (manifold, turbo)
* green  -- healthy running state
* yellow -- caution (knock, high load)
* red    -- danger (redline, limiter, ignition off while cranking)

The palette module also owns the application fonts: a condensed display face
(Bahnschrift) for headlines, Segoe UI for prose and Cascadia Mono for numbers,
because a live readout must not change width as it counts.
"""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import imgui

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
BG = (0.0, 0.0, 0.0, 0.0)  # clear colour: hello_imgui fills the backdrop

INK = (0.914, 0.937, 0.969, 1.0)  # primary text
INK_DIM = (0.541, 0.592, 0.659, 1.0)  # secondary text
INK_FAINT = (0.333, 0.380, 0.435, 1.0)  # labels
INK_OFF = (0.227, 0.267, 0.310, 1.0)  # disabled

ORANGE = (1.00, 0.416, 0.102, 1.0)
ORANGE_TOP = (1.00, 0.541, 0.239, 1.0)
ORANGE_DIM = (0.478, 0.204, 0.063, 1.0)
BLUE = (0.180, 0.608, 1.00, 1.0)
BLUE_DIM = (0.071, 0.251, 0.431, 1.0)
RED = (1.00, 0.176, 0.239, 1.0)
RED_DIM = (0.42, 0.067, 0.098, 1.0)
GREEN = (0.169, 0.886, 0.49, 1.0)
GREEN_DIM = (0.059, 0.29, 0.173, 1.0)
YELLOW = (1.0, 0.82, 0.18, 1.0)
YELLOW_DIM = (0.36, 0.29, 0.05, 1.0)
STEEL = (0.56, 0.627, 0.706, 1.0)

PANEL_A = (0.063, 0.082, 0.110, 1.0)  # card top
PANEL_B = (0.043, 0.059, 0.082, 1.0)  # card bottom
PANEL_RAISED = (0.086, 0.110, 0.145, 1.0)
PANEL_SUNK = (0.035, 0.047, 0.067, 1.0)

EDGE = (0.137, 0.173, 0.220, 1.0)
EDGE_HOT = (0.20, 0.251, 0.310, 1.0)
TRACK = (0.10, 0.125, 0.16, 1.0)

GLOW = (0.063, 0.082, 0.110, 1.0)

FONT_DIR = Path(__file__).resolve().parents[2] / "simulator" / "gui" / "fonts"


def mix(a, b, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else float(t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def with_alpha(color, a: float):
    return (color[0], color[1], color[2], max(0.0, min(1.0, float(a))))


def fade(color, t: float):
    """Dim towards the panel, for unfilled gauge tracks."""
    return mix(color, PANEL_B, t)


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
# hello_imgui loads fonts by asset name; the system faces below live in
# C:/Windows/Fonts on every Windows 10+ box, so the interface gets real
# typography (condensed display, tabular mono numbers) with zero packaging.

FONT_DISPLAY = "C:/Windows/Fonts/Bahnschrift.ttf"  # headlines
FONT_UI = "C:/Windows/Fonts/segoeui.ttf"  # prose
FONT_UI_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_MONO = "C:/Windows/Fonts/CascadiaMono.ttf"  # tabular figures

_fallbacks = {
    FONT_DISPLAY: "DroidSans.ttf",
    FONT_UI: "DroidSans.ttf",
    FONT_UI_BOLD: "DroidSans.ttf",
    FONT_MONO: "Inconsolata-Medium.ttf",
}


def _first_existing(preferred: str, fallback: str) -> str:
    from pathlib import Path as _P

    if _P(preferred).is_file():
        return preferred
    return fallback  # a bundled asset name for hello_imgui.load_font_ttf


def load_fonts() -> None:
    """Load the four faces into the atlas (called once at startup).

    The base size is in ems so the whole interface scales with the monitor DPI
    the way hello_imgui expects; 16 px at 1x is the Windows text baseline.
    """
    from imgui_bundle import hello_imgui

    size = 16.0 * hello_imgui.dpi_window_size_factor()
    for path in (FONT_UI, FONT_UI_BOLD, FONT_MONO, FONT_DISPLAY):
        actual = _first_existing(path, _fallbacks[path])
        cfg = imgui.ImFontConfig()
        cfg.merge_mode = False
        hello_imgui.load_font_ttf(actual, size, cfg)


# --------------------------------------------------------------------------- #
# imgui style
# --------------------------------------------------------------------------- #
def install_style() -> None:
    """Push the dark carbon style onto the imgui context (after its creation)."""
    style = imgui.get_style()
    for i in range(int(imgui.Col_.count)):
        style.set_color_(i, imgui.ImVec4(*PANEL_A))
    style.set_color_(imgui.Col_.text, imgui.ImVec4(*INK))
    style.set_color_(imgui.Col_.text_disabled, imgui.ImVec4(*INK_OFF))
    style.set_color_(imgui.Col_.window_bg, imgui.ImVec4(*(0.027, 0.031, 0.043, 1.0)))
    style.set_color_(imgui.Col_.child_bg, imgui.ImVec4(*(0.0, 0.0, 0.0, 0.0)))
    style.set_color_(imgui.Col_.popup_bg, imgui.ImVec4(*(0.043, 0.055, 0.075, 0.985)))
    style.set_color_(imgui.Col_.border, imgui.ImVec4(*EDGE))
    style.set_color_(imgui.Col_.border_shadow, imgui.ImVec4(*(0.0, 0.0, 0.0, 0.0)))
    style.set_color_(imgui.Col_.frame_bg, imgui.ImVec4(*PANEL_SUNK))
    style.set_color_(imgui.Col_.frame_bg_hovered, imgui.ImVec4(*PANEL_RAISED))
    style.set_color_(imgui.Col_.frame_bg_active, imgui.ImVec4(*(0.102, 0.129, 0.169, 1.0)))
    style.set_color_(imgui.Col_.title_bg, imgui.ImVec4(*(0.02, 0.025, 0.035, 1.0)))
    style.set_color_(imgui.Col_.title_bg_active, imgui.ImVec4(*(0.03, 0.04, 0.055, 1.0)))
    style.set_color_(imgui.Col_.title_bg_collapsed, imgui.ImVec4(*(0.02, 0.025, 0.035, 0.5)))
    style.set_color_(imgui.Col_.menu_bar_bg, imgui.ImVec4(*(0.03, 0.039, 0.055, 1.0)))
    style.set_color_(imgui.Col_.scrollbar_bg, imgui.ImVec4(*(0.0, 0.0, 0.0, 0.0)))
    style.set_color_(imgui.Col_.scrollbar_grab, imgui.ImVec4(*(0.11, 0.14, 0.18, 1.0)))
    style.set_color_(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(*EDGE_HOT))
    style.set_color_(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(*ORANGE_DIM))
    style.set_color_(imgui.Col_.check_mark, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.slider_grab, imgui.ImVec4(*(0.28, 0.33, 0.40, 1.0)))
    style.set_color_(imgui.Col_.slider_grab_active, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.button, imgui.ImVec4(*(0.09, 0.11, 0.145, 1.0)))
    style.set_color_(imgui.Col_.button_hovered, imgui.ImVec4(*(0.13, 0.16, 0.20, 1.0)))
    style.set_color_(imgui.Col_.button_active, imgui.ImVec4(*ORANGE_DIM))
    style.set_color_(imgui.Col_.header, imgui.ImVec4(*(0.075, 0.098, 0.13, 1.0)))
    style.set_color_(imgui.Col_.header_hovered, imgui.ImVec4(*(0.11, 0.14, 0.185, 1.0)))
    style.set_color_(imgui.Col_.header_active, imgui.ImVec4(*ORANGE_DIM))
    style.set_color_(imgui.Col_.separator, imgui.ImVec4(*EDGE))
    style.set_color_(imgui.Col_.separator_hovered, imgui.ImVec4(*EDGE_HOT))
    style.set_color_(imgui.Col_.separator_active, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.resize_grip, imgui.ImVec4(*(0.08, 0.10, 0.13, 0.25)))
    style.set_color_(imgui.Col_.resize_grip_hovered, imgui.ImVec4(*ORANGE_DIM))
    style.set_color_(imgui.Col_.resize_grip_active, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.tab, imgui.ImVec4(*(0.03, 0.039, 0.055, 1.0)))
    style.set_color_(imgui.Col_.tab_hovered, imgui.ImVec4(*(0.10, 0.13, 0.17, 1.0)))
    style.set_color_(imgui.Col_.tab_selected, imgui.ImVec4(*(0.12, 0.15, 0.19, 1.0)))
    style.set_color_(imgui.Col_.tab_dimmed, imgui.ImVec4(*(0.02, 0.025, 0.035, 1.0)))
    style.set_color_(imgui.Col_.tab_dimmed_selected, imgui.ImVec4(*(0.06, 0.078, 0.10, 1.0)))
    style.set_color_(imgui.Col_.docking_preview, imgui.ImVec4(*with_alpha(ORANGE, 0.35)))
    style.set_color_(imgui.Col_.docking_empty_bg, imgui.ImVec4(*(0.012, 0.016, 0.023, 1.0)))
    style.set_color_(imgui.Col_.plot_lines, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.plot_lines_hovered, imgui.ImVec4(*ORANGE_TOP))
    style.set_color_(imgui.Col_.plot_histogram, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.plot_histogram_hovered, imgui.ImVec4(*ORANGE_TOP))
    style.set_color_(imgui.Col_.table_header_bg, imgui.ImVec4(*(0.05, 0.065, 0.09, 1.0)))
    style.set_color_(imgui.Col_.table_border_strong, imgui.ImVec4(*EDGE))
    style.set_color_(imgui.Col_.table_border_light, imgui.ImVec4(*(0.09, 0.11, 0.145, 1.0)))
    style.set_color_(imgui.Col_.table_row_bg, imgui.ImVec4(*(0.0, 0.0, 0.0, 0.0)))
    style.set_color_(imgui.Col_.table_row_bg_alt, imgui.ImVec4(*(0.02, 0.025, 0.035, 0.4)))
    style.set_color_(imgui.Col_.text_selected_bg, imgui.ImVec4(*with_alpha(BLUE, 0.35)))
    style.set_color_(imgui.Col_.drag_drop_target, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.nav_cursor, imgui.ImVec4(*ORANGE))
    style.set_color_(imgui.Col_.nav_windowing_highlight, imgui.ImVec4(*with_alpha(INK, 0.7)))
    style.set_color_(imgui.Col_.nav_windowing_dim_bg, imgui.ImVec4(*(0.05, 0.05, 0.05, 0.55)))
    style.set_color_(imgui.Col_.modal_window_dim_bg, imgui.ImVec4(*(0.02, 0.02, 0.03, 0.6)))

    style.window_rounding = 0.0
    style.child_rounding = 0.0
    style.popup_rounding = 0.0
    style.frame_rounding = 3.0
    style.grab_rounding = 3.0
    style.tab_rounding = 3.0
    style.window_border_size = 0.0
    style.child_border_size = 0.0
    style.popup_border_size = 1.0
    style.frame_border_size = 0.0
    style.window_padding = imgui.ImVec2(8, 8)
    style.frame_padding = imgui.ImVec2(10, 5)
    style.item_spacing = imgui.ImVec2(8, 6)
    style.item_inner_spacing = imgui.ImVec2(6, 4)
    style.indent_spacing = 18.0
    style.scrollbar_size = 11.0
    style.grab_min_size = 9.0
    style.window_title_align = imgui.ImVec2(0.0, 0.5)
    style.window_min_size = imgui.ImVec2(160, 60)
    style.alpha = 1.0
