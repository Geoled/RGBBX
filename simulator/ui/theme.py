"""Racing / automotive visual language for the simulator.

Everything the new interface shows is drawn from this module: a near-black
carbon background, thin machined edges, and a small set of signal colours with
strict meanings (orange = power, blue = air/boost, red = danger, green = ok).
Numbers are set in a tabular face so a live readout does not jitter, and the
labels are spaced capitals, which is how a real dashboard reads at a glance.

The palette is deliberately dark and low-chroma except for the signals: on a
screen the 3D engine is the brightest thing in the room, and the chrome is not
allowed to compete with it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
BG = "#080A0E"
BG_DEEP = "#05070A"
CARBON_A = "#12161D"
CARBON_B = "#0C1015"

PANEL = "#10151C"
PANEL_RAISED = "#161C25"
PANEL_SUNK = "#0A0E13"

EDGE = "#232C38"
EDGE_HOT = "#33404F"
EDGE_LIT = "#3E4C5E"

INK = "#E9EFF7"
INK_DIM = "#8A97A8"
INK_FAINT = "#55616F"
INK_OFF = "#3A444F"

ORANGE = "#FF6A1A"
ORANGE_TOP = "#FF8A3D"
ORANGE_DIM = "#7A3410"
BLUE = "#2E9BFF"
BLUE_DIM = "#12406E"
RED = "#FF2D3D"
RED_DIM = "#6B1119"
GREEN = "#2BE27D"
GREEN_DIM = "#0F4A2C"
YELLOW = "#FFD12E"
YELLOW_DIM = "#5C4A0C"
STEEL = "#8FA0B4"

# alpha helpers for translucent overlays
OVERLAY = QtGui.QColor(10, 14, 19, 200)

# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
_MONO_FAMILIES = ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "Courier New")
_UI_FAMILIES = ("Segoe UI Variable", "Segoe UI", "Inter", "Arial", "DejaVu Sans")
_CONDENSED = ("Bahnschrift SemiCondensed", "Bahnschrift", "Segoe UI Semibold", "Segoe UI")


def _pick(families) -> str:
    available = set(QtGui.QFontDatabase.families())
    for name in families:
        if name in available:
            return name
    return families[-1]


_ui_family: str | None = None
_mono_family: str | None = None
_cond_family: str | None = None


def _ensure_families() -> None:
    global _ui_family, _mono_family, _cond_family
    if _ui_family is None:
        _ui_family = _pick(_UI_FAMILIES)
        _mono_family = _pick(_MONO_FAMILIES)
        _cond_family = _pick(_CONDENSED)


def font(size: int, *, bold: bool = False, mono: bool = False, condensed: bool = False,
         spacing: float = 0.0) -> QtGui.QFont:
    """A font from the palette's families, with optional letter spacing."""
    _ensure_families()
    family = _mono_family if mono else (_cond_family if condensed else _ui_family)
    f = QtGui.QFont(family, size)
    if bold:
        f.setWeight(QtGui.QFont.Weight.DemiBold)
    if spacing:
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spacing)
    # tabular figures where the face supports them: a live number must not
    # change width as it counts, or the whole panel twitches
    f.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    return f


# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #
def qc(value) -> QtGui.QColor:
    """QColor from a hex string or an existing colour/tuple."""
    if isinstance(value, QtGui.QColor):
        return QtGui.QColor(value)
    if isinstance(value, (tuple, list)):
        return QtGui.QColor(*value)
    return QtGui.QColor(value)


def mix(a, b, t: float) -> QtGui.QColor:
    """Linear mix of two colours, ``t`` = 0 gives ``a``."""
    ca, cb = qc(a), qc(b)
    t = max(0.0, min(1.0, float(t)))
    return QtGui.QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
        round(ca.alpha() + (cb.alpha() - ca.alpha()) * t),
    )


def alpha(color, a: float) -> QtGui.QColor:
    c = qc(color)
    c.setAlphaF(max(0.0, min(1.0, float(a))))
    return c


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp01(t)


# --------------------------------------------------------------------------- #
# painting helpers
# --------------------------------------------------------------------------- #
def rounded(rect: QtCore.QRectF, radius: float) -> QtCore.QRectF:
    return rect


def panel_path(rect: QtCore.QRectF, radius: float = 10.0) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def draw_panel(
    p: QtGui.QPainter,
    rect: QtCore.QRectF,
    *,
    fill=PANEL,
    border=EDGE,
    radius: float = 10.0,
    top=None,
    glow=None,
) -> None:
    """A card: subtle vertical gradient, one hairline edge, optional glow."""
    path = panel_path(rect, radius)
    grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
    grad.setColorAt(0.0, qc(top if top is not None else mix(fill, "#FFFFFF", 0.045)))
    grad.setColorAt(1.0, qc(fill))
    p.save()
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.fillPath(path, QtGui.QBrush(grad))
    if glow is not None:
        p.setPen(QtGui.QPen(alpha(glow, 0.55), 1.0))
        p.drawPath(path)
    p.setPen(QtGui.QPen(qc(border), 1.0))
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.restore()


def draw_label(
    p: QtGui.QPainter,
    x: float,
    y: float,
    text: str,
    *,
    color=INK_FAINT,
    size: int = 9,
    anchor: str = "left",
    spacing: float = 1.4,
) -> None:
    """Spaced small capitals, the standard label of this interface."""
    p.save()
    p.setFont(font(size, bold=True, spacing=spacing))
    p.setPen(qc(color))
    p.drawText(QtCore.QPointF(x, y), text.upper())
    p.restore()


def draw_value(
    p: QtGui.QPainter,
    rect: QtCore.QRectF,
    text: str,
    *,
    color=INK,
    size: int = 20,
    mono: bool = True,
    align="right",
) -> None:
    """A numeric readout in the tabular face."""
    p.save()
    p.setFont(font(size, bold=True, mono=mono))
    p.setPen(qc(color))
    flag = {
        "right": QtCore.Qt.AlignmentFlag.AlignRight,
        "left": QtCore.Qt.AlignmentFlag.AlignLeft,
        "center": QtCore.Qt.AlignmentFlag.AlignHCenter,
    }[align]
    p.drawText(rect, flag | QtCore.Qt.AlignmentFlag.AlignVCenter, text)
    p.restore()


def draw_glow(
    p: QtGui.QPainter,
    rect: QtCore.QRectF,
    color,
    *,
    radius: float = 10.0,
    strength: float = 0.6,
) -> None:
    """A soft outer glow, faked by stroking the same path several times."""
    p.save()
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    for i in range(4):
        spread = (i + 1) * 1.6
        p.setPen(QtGui.QPen(alpha(color, strength * (0.34 ** i)), 1.0))
        p.drawPath(panel_path(rect.adjusted(-spread, -spread, spread, spread), radius + spread))
    p.restore()


def tick_marks(
    p: QtGui.QPainter,
    centre: QtCore.QPointF,
    radius: float,
    start_deg: float,
    span_deg: float,
    count: int,
    *,
    major_every: int = 5,
    color=INK_FAINT,
    major_color=INK_DIM,
    length: float = 6.0,
    major_length: float = 12.0,
    width: float = 1.0,
) -> None:
    """Radiating ticks for a dial, drawn along the arc."""
    import math

    p.save()
    for i in range(count + 1):
        frac = i / max(count, 1)
        ang = math.radians(start_deg + span_deg * frac)
        major = (i % major_every) == 0
        r0 = radius - (major_length if major else length)
        c, ln, w = (major_color, major_length, width * 1.8) if major else (color, length, width)
        p.setPen(QtGui.QPen(qc(c), w, cap=QtCore.Qt.PenCapStyle.FlatCap))
        p.drawLine(
            QtCore.QPointF(centre.x() + math.cos(ang) * r0, centre.y() + math.sin(ang) * r0),
            QtCore.QPointF(centre.x() + math.cos(ang) * radius, centre.y() + math.sin(ang) * radius),
        )
    p.restore()


# --------------------------------------------------------------------------- #
# application-wide stylesheet
# --------------------------------------------------------------------------- #
def stylesheet() -> str:
    _ensure_families()
    return f"""
    QWidget {{
        background: {BG};
        color: {INK};
    }}
    QToolTip {{
        background: {PANEL_RAISED};
        color: {INK};
        border: 1px solid {EDGE_HOT};
        padding: 5px 8px;
    }}
    QLineEdit {{
        background: {PANEL_SUNK};
        border: 1px solid {EDGE};
        border-radius: 7px;
        padding: 6px 10px;
        color: {INK};
        selection-background-color: {ORANGE_DIM};
    }}
    QLineEdit:focus {{
        border: 1px solid {ORANGE};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {EDGE_HOT};
        border-radius: 4px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ORANGE_DIM};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    """
