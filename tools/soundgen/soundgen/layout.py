"""Engine geometry, firing order and bank assignment.

Everything here is pure geometry/timing: it converts a block specification into
firing frequencies, per-cylinder phase offsets (in units of one full cycle) and
which exhaust bank each cylinder feeds.
"""

from __future__ import annotations

import math

from .config import Block


def cycle_revolutions(block: Block) -> float:
    """Crankshaft revolutions per full cycle.

    A 4-stroke cycle takes two revolutions, a 2-stroke and a Wankel one.
    """
    if block.layout == "rotary":
        return 1.0
    if block.layout == "radial":
        return 2.0
    return 2.0 if int(block.strokes) == 4 else 1.0


def events_per_cycle(block: Block) -> int:
    """Combustion events per full cycle (one per cylinder / rotor)."""
    return max(int(block.cylinders), 1)


def events_per_revolution(block: Block) -> float:
    return events_per_cycle(block) / cycle_revolutions(block)


def firing_frequency(block: Block, rpm: float) -> float:
    """Combustion events per second. This is what sets the perceived pitch."""
    cyc = cycle_revolutions(block) * 60.0 / max(rpm, 1.0)
    return events_per_cycle(block) / cyc


def crankshaft_frequency(rpm: float) -> float:
    return max(rpm, 1.0) / 60.0


def bank_assignment(block: Block) -> list[int]:
    """Map cylinder index -> exhaust bank index."""
    n = max(int(block.cylinders), 1)
    if block.layout in ("v", "radial") and n > 1:
        return [i % 2 for i in range(n)]
    if block.layout == "boxer" and n > 1:
        return [i % 2 for i in range(n)]
    return [0] * n


def firing_offsets(block: Block) -> list[float]:
    """Phase offset of each cylinder's firing, in fractions of one full cycle.

    Even firing is the default (offsets spread uniformly). Explicit
    ``firing_offsets`` in the preset override it, which allows uneven-firing
    character (big-bang, V-twin, cross-plane burble).
    """
    n = max(int(block.cylinders), 1)
    if block.firing_offsets:
        offs = [float(v) % 1.0 for v in block.firing_offsets]
        while len(offs) < n:
            offs.append(0.0)
        return offs[:n]
    return [i / n for i in range(n)]


def firing_events(block: Block) -> list[tuple[int, float]]:
    """Return ``(bank, offset)`` for every combustion event in the cycle."""
    banks = bank_assignment(block)
    offs = firing_offsets(block)
    return [(banks[i], offs[i]) for i in range(len(offs))]


def cylinder_cc(block: Block) -> float:
    """Swept volume of one cylinder in cubic centimetres."""
    r_cm = (block.bore_mm / 10.0) / 2.0
    h_cm = block.stroke_mm / 10.0
    return math.pi * r_cm * r_cm * h_cm


def displacement_cc(block: Block) -> float:
    return cylinder_cc(block) * max(int(block.cylinders), 1)


def displacement_litres(block: Block) -> float:
    return displacement_cc(block) / 1000.0


def size_factor(block: Block, reference_cc: float = 500.0) -> float:
    """Frequency scaling from engine size.

    Bigger cylinders -> heavier, lower-pitched combustion and resonance.
    Returns a multiplier applied to reference frequencies (about 1.0 for a
    500 cc/cylinder reference such as a 2.0 L four).
    """
    cc = max(cylinder_cc(block), 1.0)
    return float((cc / reference_cc) ** (-0.30))


def bore_stroke_ratio(block: Block) -> float:
    return block.bore_mm / max(block.stroke_mm, 1e-6)


def describe(block: Block) -> str:
    layout = block.layout
    if layout == "inline":
        name = f"I{block.cylinders}"
    elif layout == "v":
        name = f"V{block.cylinders}"
    elif layout == "boxer":
        name = f"F{block.cylinders}"
    elif layout == "rotary":
        return f"{block.cylinders}-rotor Wankel {displacement_litres(block):.1f} L"
    elif layout == "radial":
        name = f"{block.cylinders}-cyl radial"
    elif layout == "turbine":
        return "gas turbine"
    else:
        name = f"{block.cylinders}-cyl"
    return f"{name} {block.strokes}-stroke {displacement_litres(block):.1f} L"
