"""Live charts: the dyno plot and the engine scope, drawn with QPainter only.

Both widgets are pull-only: they never own a clock, they never build a
``QPixmap`` per frame, and they keep their samples in a preallocated numpy ring
so that a push costs a few float writes and a repaint costs one small array
copy plus a polyline. That is the budget that lets the 3D view keep the frame.

The simulation feeds these widgets asynchronously -- possibly before the first
show, possibly with a solver that has not settled yet -- so every incoming
number is screened for NaN/inf at the door instead of poisoning the axes.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

__all__ = ["LiveDynoChart", "LiveOscilloscope"]

# ~15 s of history at 60 Hz: enough for a full pull to stay visible on screen.
DYNO_CAPACITY = 900
# ~10 s window for the scope; a scrolling window longer than this reads as a
# static line rather than a pulse.
SCOPE_CAPACITY = 600

# Paint-time colours are resolved once: QColor construction inside a 60 Hz
# paintEvent is pure waste, and theme.qc() copies on every call.
_BG = theme.qc(theme.BG)
_PANEL_SUNK = theme.qc(theme.PANEL_SUNK)
_EDGE = theme.qc(theme.EDGE)
_EDGE_HOT = theme.qc(theme.EDGE_HOT)
_INK = theme.qc(theme.INK)
_INK_DIM = theme.qc(theme.INK_DIM)
_INK_FAINT = theme.qc(theme.INK_FAINT)
_ORANGE = theme.qc(theme.ORANGE)
_BLUE = theme.qc(theme.BLUE)
_RED = theme.qc(theme.RED)
_GREEN = theme.qc(theme.GREEN)
_YELLOW = theme.qc(theme.YELLOW)

_GRID = theme.mix(theme.PANEL_SUNK, theme.EDGE, 0.85)
_GRID_FAINT = theme.mix(theme.PANEL_SUNK, theme.EDGE, 0.45)
_AXIS_T = theme.mix(theme.ORANGE, theme.INK_DIM, 0.62)
_AXIS_P = theme.mix(theme.BLUE, theme.INK_DIM, 0.58)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=48)
def _font(size: int, bold: bool = False, mono: bool = False, spacing: float = 0.0) -> QtGui.QFont:
    """Cached face from the palette: QFont objects are immutable enough here."""
    return theme.font(size, bold=bold, mono=mono, spacing=spacing)


def _caps(
    p: QtGui.QPainter,
    x: float,
    y: float,
    text: str,
    *,
    color=_INK_FAINT,
    size: int = 9,
    align: str = "left",
    spacing: float = 1.2,
) -> None:
    """Spaced small capitals with an anchor other than the left edge.

    ``theme.draw_label`` draws from the baseline leftward-unaware, so this is
    the same type at the two other anchors the charts need.
    """
    face = _font(size, True, False, spacing)
    label = text.upper()
    width = QtGui.QFontMetricsF(face).horizontalAdvance(label)
    if align == "right":
        x -= width
    elif align == "center":
        x -= width * 0.5
    p.save()
    p.setFont(face)
    p.setPen(theme.qc(color))
    p.drawText(QtCore.QPointF(x, y), label)
    p.restore()


def _text_width(face: QtGui.QFont, text: str) -> float:
    return QtGui.QFontMetricsF(face).horizontalAdvance(text)


def _nice(value: float) -> float:
    """Round a scale maximum up to a 1-2-5 step.

    Axis maxima that are not round numbers force the reader to decode
    "382" mid-corner; a 1-2-5 step keeps every tick label readable.
    """
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    step = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if value <= step * factor + 1e-9:
            return step * factor
    return step * 10.0


def _finite(*values: float) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True


class _Ring:
    """A fixed-size sample store shared by both charts.

    A deque of tuples would allocate per sample and then force a Python-level
    conversion on every repaint; a preallocated array keeps push O(1) and makes
    the ordered read a slice (or one concatenation once the buffer wrapped).

    ``retain`` switches the store from "oldest sample falls off the back" to
    "grow instead of dropping": a dyno pull is a measurement record, so the
    beginning of the pull must survive even when it runs past the window.
    Growth is capped, because a stuck pull must not turn into a memory leak.
    """

    __slots__ = ("capacity", "base", "channels", "buf", "head", "count", "retain", "ceiling")

    MAX_GROWTH = 6

    def __init__(self, capacity: int, channels: int) -> None:
        self.capacity = int(capacity)
        self.base = self.capacity
        self.ceiling = self.capacity * self.MAX_GROWTH
        self.channels = int(channels)
        self.buf = np.empty((self.capacity, self.channels), dtype=np.float32)
        self.head = 0  # next write slot
        self.count = 0
        self.retain = False

    def _grow(self) -> None:
        """Re-linearise into a larger buffer, keeping the oldest rows."""
        ordered = self.ordered()
        new_cap = min(self.capacity * 2, max(self.ceiling, self.count + 1))
        grown = np.empty((new_cap, self.channels), dtype=np.float32)
        grown[: self.count] = ordered
        self.buf = grown
        self.capacity = new_cap
        self.head = self.count

    def push(self, *values: float) -> None:
        if self.count >= self.capacity:
            if self.retain and self.capacity < self.ceiling:
                self._grow()
        row = self.buf[self.head]
        for i, value in enumerate(values):
            row[i] = value
        self.head += 1
        if self.head >= self.capacity:
            self.head = 0
        if self.count < self.capacity:
            self.count += 1

    def clear(self) -> None:
        self.head = 0
        self.count = 0
        if self.capacity > self.base:
            # a finished pull leaves an oversized buffer behind; give that ram
            # back now rather than at some arbitrary later point
            self.capacity = self.base
            self.buf = np.empty((self.capacity, self.channels), dtype=np.float32)

    def ordered(self) -> np.ndarray:
        """Rows oldest -> newest, as a view when possible."""
        n = self.count
        if n == 0:
            return self.buf[:0]
        if n < self.capacity:
            # head only wraps once the buffer is full, so a plain prefix is
            # already in time order and needs no copy
            return self.buf[:n]
        if self.head == 0:
            return self.buf
        return np.concatenate((self.buf[self.head:], self.buf[: self.head]))


def _axis_ticks(span: float, count: int) -> list[float]:
    """Round tick values covering ``0..span``, roughly ``count`` of them.

    A live axis whose labels read 1484 / 2968 / 4452 forces the operator to do
    arithmetic mid-pull; ticks on a 1-2-5 step can be read at a glance.
    """
    if not math.isfinite(span) or span <= 0.0 or count < 2:
        return [0.0]
    raw = span / max(count - 1, 1)
    step = _nice(raw)
    # _nice() rounds up, which can leave the last tick well short of the span;
    # halving once keeps the labels on a readable step and covers the axis
    if step * (count - 1) < span * 0.72:
        step = _nice(raw * 0.5)
    values: list[float] = []
    value = 0.0
    while value <= span * 1.0001 and len(values) < count:
        values.append(value)
        value += step
    return values or [0.0]


def _num(p: QtGui.QPainter, rect: QtCore.QRectF, value: float, *, color=_INK_FAINT, size: int = 8, align="right") -> None:
    """A scale number: thousands separators only above 9999 to save width."""
    text = f"{value:,.0f}" if abs(value) >= 9999.0 else f"{value:.0f}"
    theme.draw_value(p, rect, text, color=color, size=size, align=align)


def _stroke(p: QtGui.QPainter, poly: QtGui.QPolygonF, color, width: float, glow: float = 0.0) -> None:
    """One curve, optionally with a wide translucent underlay for the glow.

    Stroking the same geometry twice is far cheaper than a real blur, and at
    60 Hz the eye reads the underlay as luminance spill, not as a second line.
    """
    if glow > 0.0:
        p.setPen(
            QtGui.QPen(
                theme.alpha(color, glow),
                width * 2.8,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.RoundCap,
                QtCore.Qt.PenJoinStyle.RoundJoin,
            )
        )
        p.drawPolyline(poly)
    p.setPen(
        QtGui.QPen(
            qt_color(color),
            width,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
    )
    p.drawPolyline(poly)


def qt_color(value) -> QtGui.QColor:
    """Coerce a palette entry to a QColor without copying when possible."""
    if isinstance(value, QtGui.QColor):
        return value
    return theme.qc(value)


# --------------------------------------------------------------------------- #
# dyno chart
# --------------------------------------------------------------------------- #
class LiveDynoChart(QtWidgets.QWidget):
    """Brake torque (left axis) and power (right axis) plotted against rpm.

    The curve is drawn in arrival order, not sorted by rpm: this is a live
    trace of the engine, so a pull that is interrupted and restarted must show
    the hysteresis rather than a fabricated smooth curve.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # the widget paints every pixel itself, so Qt must not pre-erase it
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(360, 150)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._ring = _Ring(DYNO_CAPACITY, 3)
        self._redline = 7000.0
        self._peak_torque_hint = 260.0
        # seeded from the redline so the power axis starts at a plausible scale
        # instead of collapsing onto the first few samples
        self._t_max = 320.0
        self._p_max = 160.0
        self._pulling = False
        self._pull_progress = 0.0
        self._message = "AWAITING THE FIRST PULL"

    # -- public API -------------------------------------------------------- #
    def set_limits(self, redline_rpm: float, peak_torque_nm: float) -> None:
        """Anchor the axes to the engine's own numbers."""
        redline = float(redline_rpm) if _finite(redline_rpm) else 0.0
        torque = float(peak_torque_nm) if _finite(peak_torque_nm) else 0.0
        self._redline = max(1000.0, redline)
        self._peak_torque_hint = max(10.0, torque)
        self._t_max = _nice(max(self._peak_torque_hint * 1.15, 20.0))
        # ~0.62 of the theoretical peak is what a real torque curve reaches at
        # the rpm where it makes its best power; good enough to size the axis
        estimate = self._peak_torque_hint * self._redline * 0.62 * math.tau / 60.0 / 1000.0
        self._p_max = _nice(max(estimate, 20.0))
        self.update()

    def clear(self) -> None:
        """Drop the history and re-seed the axes from the last ``set_limits``."""
        self._ring.clear()
        self._pull_progress = 0.0
        self.set_limits(self._redline, self._peak_torque_hint)
        self.update()

    def push(self, rpm: float, torque_nm: float, power_kw: float) -> None:
        """Append one live sample; non-finite input is ignored."""
        if not _finite(rpm, torque_nm, power_kw):
            return
        self._ring.push(rpm, torque_nm, power_kw)
        # feeding data is what makes this widget live; the update request is
        # coalesced by Qt, so a 60 Hz feed still costs at most one paint
        self.update()

    def set_dyno_pull(self, active: bool, progress: float = 0.0) -> None:
        """Show the pull progress bar and stop trimming the history.

        While a pull is running every sample is part of the measurement, so the
        ring grows instead of evicting the start of the pull.
        """
        active = bool(active)
        progress = theme.clamp01(progress) if _finite(progress) else 0.0
        if active == self._pulling and progress == self._pull_progress:
            return
        self._pulling = active
        self._pull_progress = progress
        self._ring.retain = active
        self.update()

    def set_message(self, text: str) -> None:
        """Text shown while there is no data yet."""
        self._message = str(text)
        self.update()

    # -- geometry ---------------------------------------------------------- #
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt naming
        return QtCore.QSize(900, 260)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt naming
        return QtCore.QSize(360, 150)

    def _plot_rect(self, card: QtCore.QRectF) -> QtCore.QRectF:
        # below the plot sit the rpm tick row and its unit caption, and in pull
        # mode the progress bar under both
        bottom_pad = 52.0 if self._pulling else 36.0
        return QtCore.QRectF(
            card.left() + 58.0,
            card.top() + 40.0,
            max(card.width() - 58.0 - 66.0, 1.0),
            max(card.height() - 40.0 - bottom_pad, 1.0),
        )

    # -- painting ---------------------------------------------------------- #
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, p: QtGui.QPainter) -> None:
        card = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if card.width() < 120.0 or card.height() < 90.0:
            return

        if self._pulling:
            # a soft orange halo is the only chrome that changes while pulling,
            # which is what tells the operator "this is being measured, wait"
            theme.draw_glow(p, card, theme.ORANGE, radius=10.0, strength=0.35)
        theme.draw_panel(p, card, fill=theme.PANEL, border=theme.EDGE, radius=10.0)

        self._paint_header(p, card)
        plot = self._plot_rect(card)
        if plot.width() < 60.0 or plot.height() < 40.0:
            return

        theme.draw_panel(p, plot.adjusted(-8.0, -6.0, 8.0, 6.0), fill=theme.PANEL_SUNK, border=theme.EDGE, radius=8.0)

        ordered = self._ring.ordered()
        n = self._ring.count
        if n:
            self._grow_limits(ordered)

        x_max = self._redline * 1.06
        self._paint_grid(p, plot, x_max)
        self._paint_redline(p, plot, x_max)

        if n:
            self._paint_data(p, plot, ordered, x_max)
        else:
            theme.draw_value(p, plot, self._message, color=_INK_FAINT, size=10, align="center")

        if self._pulling:
            self._paint_progress(p, plot)

    def _paint_header(self, p: QtGui.QPainter, card: QtCore.QRectF) -> None:
        x = card.left() + 14.0
        baseline = card.top() + 20.0
        theme.draw_label(p, x, baseline, "live dyno", color=theme.INK, size=10)
        self._legend(p, x + 98.0, baseline - 4.0, "torque  Nm", _ORANGE)
        self._legend(p, x + 214.0, baseline - 4.0, "power  kW", _BLUE)

        if self._pulling:
            text = f"PULL {self._pull_progress * 100.0:.0f}%"
            color = _YELLOW
        else:
            text = "LIVE"
            color = _GREEN
        dot_x = card.right() - 14.0 - _text_width(_font(9, True, False, 1.2), text.upper())
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QtCore.QPointF(dot_x - 9.0, baseline - 3.5), 2.6, 2.6)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        _caps(p, card.right() - 14.0, baseline, text, color=color, size=9, align="right")

    def _legend(self, p: QtGui.QPainter, x: float, y: float, text: str, color) -> None:
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QtCore.QRectF(x, y - 4.0, 9.0, 9.0), 2.0, 2.0)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        _caps(p, x + 15.0, y + 4.0, text, color=theme.mix(color, theme.INK_DIM, 0.55), size=8, spacing=0.9)

    def _grow_limits(self, ordered: np.ndarray) -> None:
        """Widen an axis only when a sample leaves it.

        Re-deriving the axis every frame makes the whole plot breathe and the
        labels flicker; a dyno operator also wants the scale they started with.
        """
        t_peak = float(ordered[:, 1].max())
        p_peak = float(ordered[:, 2].max())
        if t_peak > self._t_max:
            self._t_max = _nice(t_peak * 1.12)
        if p_peak > self._p_max:
            self._p_max = _nice(p_peak * 1.12)

    def _paint_grid(self, p: QtGui.QPainter, plot: QtCore.QRectF, x_max: float) -> None:
        left, right = plot.left(), plot.right()
        top, bottom = plot.top(), plot.bottom()

        # ticks are computed first and the grid drawn from them, so a line and
        # its label can never disagree
        x_ticks = _axis_ticks(x_max, 6)
        p.setPen(QtGui.QPen(_GRID, 1.0))
        for value in x_ticks:
            gx = left + plot.width() * theme.clamp01(value / max(x_max, 1e-6))
            p.drawLine(QtCore.QPointF(gx, top), QtCore.QPointF(gx, bottom))
        for i in range(1, 4):
            gy = top + plot.height() * i / 4.0
            p.drawLine(QtCore.QPointF(left, gy), QtCore.QPointF(right, gy))

        # frame of the plot well
        p.setPen(QtGui.QPen(_EDGE_HOT, 1.0))
        p.drawLine(QtCore.QPointF(left, bottom), QtCore.QPointF(right, bottom))

        label_rect = QtCore.QRectF(plot.left() - 54.0, 0.0, 46.0, 14.0)
        for i in range(5):
            frac = i / 4.0
            gy = bottom - plot.height() * frac
            label_rect.moveTop(gy - 7.0)
            _num(p, label_rect, self._t_max * frac, color=_AXIS_T, align="right")
            value_rect = QtCore.QRectF(right + 8.0, gy - 7.0, 56.0, 14.0)
            _num(p, value_rect, self._p_max * frac, color=_AXIS_P, align="left")

        # fixed-width boxes that are moved, never resized: QRectF.setLeft()
        # keeps the right edge and would collapse the box to a negative width
        # at the end of the axis, silently dropping the label
        x_label = QtCore.QRectF(0.0, bottom + 5.0, 58.0, 12.0)
        half = 29.0
        for value in x_ticks:
            gx = left + plot.width() * theme.clamp01(value / max(x_max, 1e-6))
            # the end labels are re-anchored rather than clamped so they stay
            # optically attached to their tick's column
            if gx - left < half:
                x_label.moveLeft(left)
                _num(p, x_label, value, color=_INK_FAINT, align="left")
            elif right - gx < half:
                x_label.moveLeft(right - 58.0)
                _num(p, x_label, value, color=_INK_FAINT, align="right")
            else:
                x_label.moveLeft(gx - half)
                _num(p, x_label, value, color=_INK_FAINT, align="center")

        _caps(p, right, bottom + 26.0, "rpm", color=_INK_DIM, size=8, align="right", spacing=1.0)

    def _paint_redline(self, p: QtGui.QPainter, plot: QtCore.QRectF, x_max: float) -> None:
        x = plot.left() + plot.width() * theme.clamp01(self._redline / x_max)
        if x >= plot.right() - 1.0:
            return
        p.setPen(QtGui.QPen(theme.alpha(theme.RED, 0.30), 3.0))
        p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
        p.setPen(QtGui.QPen(_RED, 1.2))
        p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
        _caps(p, x - 4.0, plot.top() + 11.0, "redline", color=_RED, size=8, align="right", spacing=1.0)

    def _paint_data(
        self,
        p: QtGui.QPainter,
        plot: QtCore.QRectF,
        ordered: np.ndarray,
        x_max: float,
    ) -> None:
        rpm = ordered[:, 0].tolist()
        torque = ordered[:, 1].tolist()
        power = ordered[:, 2].tolist()
        n = len(rpm)

        left, right = plot.left(), plot.right()
        top, bottom = plot.top(), plot.bottom()
        inv_x = plot.width() / max(x_max, 1.0)
        k_t = plot.height() / max(self._t_max, 1e-6)
        k_p = plot.height() / max(self._p_max, 1e-6)

        # more samples than device pixels cannot show more detail; skipping them
        # keeps the polyline inside the pixel budget on a wide monitor
        stride = 1
        if n > int(plot.width()):
            stride = max(1, n // max(1, int(plot.width())))
        indices = list(range(0, n, stride))
        if indices[-1] != n - 1:
            indices.append(n - 1)

        torque_poly = QtGui.QPolygonF()
        power_poly = QtGui.QPolygonF()
        first_x = 0.0
        last_x = 0.0
        last_torque_y = bottom
        for step, i in enumerate(indices):
            x = left + rpm[i] * inv_x
            if x < left:
                x = left
            elif x > right:
                x = right
            y_t = bottom - torque[i] * k_t
            if y_t < top:
                y_t = top
            elif y_t > bottom:
                y_t = bottom
            y_p = bottom - power[i] * k_p
            if y_p < top:
                y_p = top
            elif y_p > bottom:
                y_p = bottom
            if step == 0:
                first_x = x
            last_x = x
            last_torque_y = y_t
            torque_poly.append(QtCore.QPointF(x, y_t))
            power_poly.append(QtCore.QPointF(x, y_p))

        if n >= 2:
            # the fill is what makes the torque curve read as the "area of
            # work" under a pull, so it is drawn before either line
            fill = QtGui.QPainterPath()
            fill.addPolygon(torque_poly)
            fill.lineTo(last_x, bottom)
            fill.lineTo(first_x, bottom)
            fill.closeSubpath()
            gradient = QtGui.QLinearGradient(QtCore.QPointF(0.0, top), QtCore.QPointF(0.0, bottom))
            gradient.setColorAt(0.0, theme.alpha(theme.ORANGE, 0.34))
            gradient.setColorAt(1.0, theme.alpha(theme.ORANGE, 0.02))
            p.save()
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.fillPath(fill, QtGui.QBrush(gradient))
            p.restore()

            p.save()
            p.setClipRect(plot.adjusted(-1.0, -3.0, 1.0, 1.0))
            _stroke(p, power_poly, _BLUE, 1.9, glow=0.16)
            _stroke(p, torque_poly, _ORANGE, 2.6, glow=0.20)
            p.restore()

            self._paint_peak_dot(p, last_x, last_torque_y)

        self._paint_peak_badge(p, plot, max(torque), max(power))

    def _paint_peak_dot(self, p: QtGui.QPainter, x: float, y: float) -> None:
        p.save()
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(theme.alpha(theme.ORANGE, 0.22))
        p.drawEllipse(QtCore.QPointF(x, y), 6.0, 6.0)
        p.setBrush(_ORANGE)
        p.drawEllipse(QtCore.QPointF(x, y), 2.4, 2.4)
        p.restore()

    def _paint_peak_badge(self, p: QtGui.QPainter, plot: QtCore.QRectF, peak_t: float, peak_p: float) -> None:
        rows = ((f"MAX {peak_t:,.0f} Nm", _ORANGE), (f"MAX {peak_p:,.1f} kW", _BLUE))
        face = _font(9, True, True)
        width = max(_text_width(face, text) for text, _ in rows) + 30.0
        height = 2 * 15.0 + 8.0
        box = QtCore.QRectF(plot.left() + 10.0, plot.top() + 10.0, width, height)
        theme.draw_panel(
            p,
            box,
            fill=theme.mix(theme.PANEL_SUNK, theme.PANEL, 0.45),
            border=theme.EDGE,
            radius=6.0,
        )
        for i, (text, color) in enumerate(rows):
            y = box.top() + 4.0 + i * 15.0
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QtCore.QRectF(box.left() + 8.0, y + 4.0, 6.0, 6.0), 1.5, 1.5)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            theme.draw_value(
                p,
                QtCore.QRectF(box.left() + 20.0, y, width - 26.0, 14.0),
                text,
                color=color,
                size=9,
                align="left",
            )

    def _paint_progress(self, p: QtGui.QPainter, plot: QtCore.QRectF) -> None:
        # sits below the rpm tick row so the axis labels stay legible mid-pull
        track = QtCore.QRectF(plot.left(), plot.bottom() + 32.0, plot.width(), 5.0)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_PANEL_SUNK)
        p.drawRoundedRect(track, 2.5, 2.5)
        p.setPen(QtGui.QPen(_EDGE, 1.0))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(track, 2.5, 2.5)
        filled = QtCore.QRectF(track.left(), track.top(), track.width() * self._pull_progress, track.height())
        if filled.width() > 1.0:
            gradient = QtGui.QLinearGradient(filled.topLeft(), filled.topRight())
            gradient.setColorAt(0.0, _ORANGE)
            gradient.setColorAt(1.0, _YELLOW)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QBrush(gradient))
            p.drawRoundedRect(filled, 2.5, 2.5)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)


# --------------------------------------------------------------------------- #
# oscilloscope
# --------------------------------------------------------------------------- #
class LiveOscilloscope(QtWidgets.QWidget):
    """Three stacked traces of the engine's instantaneous state.

    Rpm, throttle and load share one scrolling time base so that a throttle
    stab and the rpm that follows it line up vertically -- that correlation is
    the whole point of the widget.
    """

    # lane share of the plot height: rpm is the headline trace and gets the room
    _LANES = (
        ("RPM", _INK, 0.42),
        ("THR", _GREEN, 0.29),
        ("LOAD", _ORANGE, 0.29),
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(360, 130)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._ring = _Ring(SCOPE_CAPACITY, 3)
        # rpm is normalised against a slowly moving maximum, otherwise the
        # trace would rescale on every throttle stab and look like noise
        self._rpm_peak = 1200.0

    # -- public API -------------------------------------------------------- #
    def push(self, rpm: float, throttle: float, load: float) -> None:
        """Append one frame of engine state; non-finite input is ignored."""
        if not _finite(rpm, throttle, load):
            return
        rpm = max(0.0, float(rpm))
        self._ring.push(rpm, theme.clamp01(throttle), theme.clamp01(load))
        self._track_peak(rpm)
        self.update()

    def clear(self) -> None:
        """Forget the trace and reset the rpm normalisation."""
        self._ring.clear()
        self._rpm_peak = 1200.0
        self.update()

    def _track_peak(self, rpm: float) -> None:
        if rpm > self._rpm_peak:
            # headroom so a new maximum does not pin the trace to the ceiling
            self._rpm_peak = rpm * 1.08
        else:
            # exponential relax back down, per sample: slow enough that the
            # trace keeps its shape while the engine idles
            self._rpm_peak += (max(rpm, 1200.0) - self._rpm_peak) * 0.012
        if self._rpm_peak < 1200.0:
            self._rpm_peak = 1200.0

    # -- geometry ---------------------------------------------------------- #
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt naming
        return QtCore.QSize(900, 180)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt naming
        return QtCore.QSize(360, 130)

    def _plot_rect(self, card: QtCore.QRectF) -> QtCore.QRectF:
        return QtCore.QRectF(
            card.left() + 54.0,
            card.top() + 28.0,
            max(card.width() - 54.0 - 66.0, 1.0),
            max(card.height() - 28.0 - 9.0, 1.0),
        )

    def _lane_rects(self, plot: QtCore.QRectF) -> list[QtCore.QRectF]:
        gap = 6.0
        usable = plot.height() - gap * (len(self._LANES) - 1)
        rects: list[QtCore.QRectF] = []
        y = plot.top()
        for _, _, weight in self._LANES:
            height = usable * weight
            rects.append(QtCore.QRectF(plot.left(), y, plot.width(), height))
            y += height + gap
        return rects

    # -- painting ---------------------------------------------------------- #
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, p: QtGui.QPainter) -> None:
        card = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if card.width() < 120.0 or card.height() < 80.0:
            return
        theme.draw_panel(p, card, fill=theme.PANEL, border=theme.EDGE, radius=10.0)

        baseline = card.top() + 19.0
        theme.draw_label(p, card.left() + 14.0, baseline, "engine scope", color=theme.INK, size=10)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_GREEN)
        p.drawEllipse(QtCore.QPointF(card.right() - 58.0, baseline - 3.5), 2.6, 2.6)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        _caps(p, card.right() - 14.0, baseline, "live", color=_GREEN, size=9, align="right")

        plot = self._plot_rect(card)
        if plot.width() < 60.0 or plot.height() < 40.0:
            return

        ordered = self._ring.ordered()
        n = self._ring.count
        columns = [ordered[:, i].tolist() for i in range(3)] if n else ([], [], [])
        scales = (max(self._rpm_peak, 1.0), 1.0, 1.0)

        for index, rect in enumerate(self._lane_rects(plot)):
            caption, color, _ = self._LANES[index]
            values = columns[index]
            scale = scales[index]
            self._paint_lane(p, rect, caption, color, values, scale, n)

        if n == 0:
            _caps(
                p,
                plot.center().x(),
                plot.center().y(),
                "waiting for data",
                color=_INK_FAINT,
                size=9,
                align="center",
            )

    def _paint_lane(
        self,
        p: QtGui.QPainter,
        rect: QtCore.QRectF,
        caption: str,
        color,
        values: list[float],
        scale: float,
        n: int,
    ) -> None:
        theme.draw_panel(p, rect, fill=theme.PANEL_SUNK, border=theme.EDGE, radius=5.0)

        # time grid: fixed pixel divisions, so the lines scroll with the data
        p.setPen(QtGui.QPen(_GRID_FAINT, 1.0))
        for k in range(1, 6):
            gx = rect.left() + rect.width() * k / 6.0
            p.drawLine(QtCore.QPointF(gx, rect.top() + 1.0), QtCore.QPointF(gx, rect.bottom() - 1.0))
        p.setPen(QtGui.QPen(_GRID, 1.0))
        p.drawLine(QtCore.QPointF(rect.left() + 1.0, rect.center().y()), QtCore.QPointF(rect.right() - 1.0, rect.center().y()))

        _caps(p, rect.left() - 10.0, rect.center().y() + 3.5, caption, color=theme.mix(color, theme.INK_DIM, 0.35), size=8, align="right", spacing=1.0)

        if n == 0:
            theme.draw_value(p, QtCore.QRectF(rect.right() + 8.0, rect.center().y() - 8.0, 52.0, 16.0), "--", color=_INK_FAINT, size=10, align="right")
            return

        latest = values[-1]
        if caption == "RPM":
            readout = f"{latest:.0f}"
            unit = "rpm"
        else:
            readout = f"{latest * 100.0:.0f}"
            unit = "%"
        # one compact line in the right gutter: the number leads, a small unit
        # trails it, so three lanes of readouts stay optically aligned
        gutter = rect.right() + 60.0
        unit_face = _font(7, True, False, 0.6)
        unit_width = _text_width(unit_face, unit.upper()) + 4.0
        p.save()
        p.setFont(unit_face)
        p.setPen(_INK_FAINT)
        p.drawText(
            QtCore.QRectF(gutter - unit_width, rect.center().y() - 8.0, unit_width, 14.0),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            unit.upper(),
        )
        p.setFont(_font(10, True, True))
        p.setPen(color)
        p.drawText(
            QtCore.QRectF(rect.right() + 6.0, rect.center().y() - 8.0, gutter - unit_width - rect.right() - 6.0, 14.0),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            readout,
        )
        p.restore()

        p.save()
        p.setClipRect(rect.adjusted(1.0, 0.0, -1.0, 0.0))
        self._paint_trace(p, rect, values, scale, color, fill=caption == "RPM")
        p.restore()

    def _paint_trace(
        self,
        p: QtGui.QPainter,
        rect: QtCore.QRectF,
        values: list[float],
        scale: float,
        color,
        *,
        fill: bool,
    ) -> None:
        n = len(values)
        if n == 0:
            return
        # the newest sample is pinned to the right edge and history moves left,
        # so a freshly cleared scope fills in from the right like a real scope
        dx = rect.width() / max(SCOPE_CAPACITY - 1, 1)
        right = rect.right() - 1.0
        inner_h = rect.height() - 4.0
        bottom = rect.bottom() - 2.0
        inv_scale = 1.0 / max(scale, 1e-6)

        poly = QtGui.QPolygonF()
        first_x = 0.0
        last_x = right
        last_y = bottom
        for index, value in enumerate(values):
            x = right - (n - 1 - index) * dx
            frac = value * inv_scale
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            y = bottom - frac * inner_h
            if index == 0:
                first_x = x
            last_x = x
            last_y = y
            poly.append(QtCore.QPointF(x, y))

        if fill and n >= 2:
            path = QtGui.QPainterPath()
            path.addPolygon(poly)
            path.lineTo(last_x, bottom)
            path.lineTo(first_x, bottom)
            path.closeSubpath()
            gradient = QtGui.QLinearGradient(QtCore.QPointF(0.0, rect.top()), QtCore.QPointF(0.0, bottom))
            gradient.setColorAt(0.0, theme.alpha(color, 0.16))
            gradient.setColorAt(1.0, theme.alpha(color, 0.01))
            p.save()
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.fillPath(path, QtGui.QBrush(gradient))
            p.restore()

        if n >= 2:
            _stroke(p, poly, color, 1.7, glow=0.18)

        # the head of the trace: a dot with a short afterglow, which reads as
        # "this is the value right now" against the scrolling body
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(theme.alpha(color, 0.20))
        p.drawEllipse(QtCore.QPointF(last_x, last_y), 5.0, 5.0)
        p.setBrush(color)
        p.drawEllipse(QtCore.QPointF(last_x, last_y), 2.2, 2.2)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
