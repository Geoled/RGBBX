"""Shared helpers for the GUI panels (kept import-cycle free)."""

from __future__ import annotations

from ..physics import TWO_PI


def summarise(points: list[tuple[float, float]]) -> dict:
    """Peak figures for a curve of ``(rpm, brake torque)`` points."""
    if not points:
        return {}

    def power_kw(point):
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
