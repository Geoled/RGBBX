"""Gas bank: render the loop layers from the GAS core (roadmap B3).

Same five operating points and the same LoopBank/LoopPlayer machinery as
the procedural asset mode -- only the renderer behind ``render_layer``
changes. This is the acceptance bridge of phase B: the gas voice's quality
becomes available both offline (WAV assets, unchanged pipeline) and live
(asset playback, unchanged player), so the two paradigms can be A/B'd on
identical infrastructure.
"""

from __future__ import annotations

import numpy as np

from . import dsp, gasvoice, layers
from .config import Preset


def render_gas_layer(
    preset: Preset,
    spec: layers.LayerSpec,
    sr: int | None = None,
    normalize_peak: float = 0.79,
) -> tuple[np.ndarray, float]:
    """Render one loop layer with the gas-dynamic voice.

    Returns ``(buffer, ref_rpm)`` like the bank entries. The gas voice is
    not exactly periodic, so the loop contract (whole cycles, seam-free)
    cannot be copied from the procedural renderer; instead the layer is
    rendered with its natural period snap: the buffer length is rounded to
    a whole number of firing periods at the layer's rpm, and the seam is
    equal-power crossfaded over 15 ms (uncorrelated at the gas voice's
    per-event turbulence, so the equal-power assumption holds here).
    """
    sr = int(sr or preset.render.sample_rate)
    rpm = layers.quantized_rpm(preset, layers.layer_rpm(preset, spec), sr)
    voice = gasvoice.GasEngineVoice(preset, sr=sr, block=256, output_gain=1.0)
    voice.set_operating_point(rpm, spec.load, spec.throttle)

    # warm the network to steady flow (1 s), then render the loop
    voice.process(sr)
    period = int(round(layers.cycle_period(preset, rpm) * sr))
    cycles = max(1, int(round(preset.render.loop_seconds * sr / period)))
    total = cycles * period
    x = voice.process(total)

    if spec.limiter:
        x = layers.apply_limiter(x, sr, seed=preset.seed)

    # seam: equal-power crossfade of the tail into the head
    fade = min(int(0.015 * sr), x.size // 4)
    if fade > 1:
        w = np.linspace(0.0, np.pi / 2.0, fade)
        head = x[:fade] * np.sin(w) + x[-fade:] * np.cos(w)
        x = np.concatenate([head, x[fade:-fade]])

    pk = float(np.abs(x).max())
    if pk > 1e-9:
        x = x * (normalize_peak / pk)
    return x, rpm


def gas_bank_entries(preset: Preset, sr: int | None = None):
    """All five layers as ``{key: (buffer, ref_rpm)}`` -- bank-compatible."""
    out = {}
    for spec in layers.DEFAULT_LAYERS:
        out[spec.key] = render_gas_layer(preset, spec, sr)
    return out
