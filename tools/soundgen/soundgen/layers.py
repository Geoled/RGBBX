"""Loop layers: the steady-state sounds that Roblox crossfades at runtime.

For every preset we render a small set of loops at fixed operating points. The
game selects/blends them by RPM and load and shifts playback speed, which is how
a continuous, latency-free engine note is produced from a handful of assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp, layout
from .config import Preset
from .voice import EngineVoice


@dataclass(frozen=True)
class LayerSpec:
    key: str
    label: str
    rpm_frac: float = 0.0
    load: float = 0.0
    throttle: float = 0.0
    limiter: bool = False


DEFAULT_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("idle", "Idle / closed throttle", rpm_frac=0.00, load=0.03, throttle=0.00),
    LayerSpec("mid", "Part throttle", rpm_frac=0.42, load=0.45, throttle=0.42),
    LayerSpec("wot", "Wide open throttle", rpm_frac=0.72, load=1.00, throttle=1.00),
    # Coast is rendered with a little load on purpose. A truly loadless
    # render is almost pure turbulence: 54 % of its energy sat at 1-3 kHz and
    # lifting off at speed was a hiss with no body. The pumping loss that
    # dominates real engine braking rides on the same pulse train as
    # part-load combustion, so a small load restores the low end without
    # making it sound like a throttle that is still open.
    LayerSpec("coast", "Overrun / engine braking", rpm_frac=0.58, load=0.18, throttle=0.08),
    LayerSpec("limiter", "Rev limiter bounce", rpm_frac=1.00, load=1.00, throttle=1.00, limiter=True),
)


def layer_rpm(preset: Preset, spec: LayerSpec) -> float:
    if spec.rpm_frac <= 0.0:
        return preset.engine.idle_rpm
    span = preset.engine.redline_rpm - preset.engine.idle_rpm
    return preset.engine.idle_rpm + spec.rpm_frac * span


def cycle_period(preset: Preset, rpm: float) -> float:
    return layout.cycle_revolutions(preset.block) * 60.0 / max(rpm, 1.0)


def period_samples(preset: Preset, rpm: float, sr: int) -> int:
    """Length of one engine cycle, rounded to a whole number of samples.

    Always even. Some excitations (injector impulses) live on a 2-sample grid, so
    an odd loop length would flip their parity after one loop and break the seam.
    """
    p = max(int(round(cycle_period(preset, rpm) * sr)), 2)
    return p + (p % 2)


def quantized_rpm(preset: Preset, rpm: float, sr: int) -> float:
    """The rpm that actually corresponds to the quantised sample period.

    Combustion events are placed on integer samples, so unless the cycle period
    is a whole number of samples the event grid never repeats and the loop has a
    seam. Snapping the period shifts rpm by a fraction of a percent -- inaudible
    -- and this is the rpm the rendered loop really is.
    """
    return layout.cycle_revolutions(preset.block) * 60.0 * sr / period_samples(preset, rpm, sr)


def loop_length_samples(preset: Preset, rpm: float, sr: int) -> int:
    """Length of one loop: a whole number of quantised engine cycles."""
    p = period_samples(preset, rpm, sr)
    cycles = max(1, int(round(preset.render.loop_seconds * sr / p)))
    return cycles * p


def warmup_samples(preset: Preset, rpm: float, sr: int) -> int:
    """Samples to discard so every filter has forgotten the silent start.

    Must exceed the longest filter memory, otherwise the state at the start and
    the end of the loop differ and the seam is not clean. The reverb tail counts
    as filter memory: an LTI reverb is periodic in its steady state, but only
    once its impulse response has fully played in.
    """
    core = max(preset.render.resonator_length_s, 0.25)
    memory_s = core
    sp = preset.space
    if sp.enabled and sp.mix > 0.0:
        # The reverb needs its whole impulse response to play in *and* the dry
        # chain to have settled before that response was fed a steady signal.
        memory_s = sp.predelay_ms / 1000.0 + sp.decay_s + core
    return int(max(preset.render.warmup_cycles * cycle_period(preset, rpm), memory_s) * sr)


def apply_limiter(x: np.ndarray, sr: int, rate: float = 26.0, depth: float = 0.82, seed: int = 0) -> np.ndarray:
    """Chop the note the way a hard rev limiter cuts fuel/spark.

    Uses several incommensurate modulators so the bounce is irregular, which is
    what a real limiter sounds like. Every modulator is snapped to a whole
    number of cycles across the buffer and the gate is smoothed circularly, so
    the result still loops seamlessly.
    """
    n = x.size
    if n == 0:
        return x
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    ph = float(rng.uniform(0.0, dsp.TWO_PI))

    def snap(freq: float) -> float:
        cycles = max(1, int(round(freq * n / sr)))
        return cycles * sr / n

    s = (
        np.sin(dsp.TWO_PI * snap(rate) * t + ph)
        + 0.55 * np.sin(dsp.TWO_PI * snap(rate * 0.37) * t + ph * 1.7)
        + 0.70 * np.sin(dsp.TWO_PI * snap(rate * 0.11) * t + ph * 2.3)
    )
    gate = (s > 0.15).astype(np.float64)
    k = max(int(0.0015 * sr), 1)
    ker = np.ones(k) / k
    # circular smoothing: wrap the signal instead of zero-padding, otherwise the
    # start and end of the buffer get different treatment and the loop clicks
    ext = np.concatenate([gate[-k:], gate, gate[:k]])
    gate = np.convolve(ext, ker, mode="same")[k : k + n]
    return x * (1.0 - depth + depth * gate)


def render_layer(
    preset: Preset,
    spec: LayerSpec,
    output_gain: float,
    sr: int | None = None,
    block: int | None = None,
    crossfade_ms: float = 0.0,
) -> np.ndarray:
    """Render one steady-state loop.

    The buffer is an exact whole number of engine cycles rendered from steady
    state, so it is *exactly* periodic and loops with no seam. Deliberately no
    crossfade is applied: an equal-power crossfade is only correct for
    uncorrelated material and would add up to +3 dB on this correlated signal.
    """
    sr = int(sr or preset.render.sample_rate)
    block = int(block or preset.render.block_size)
    rpm = quantized_rpm(preset, layer_rpm(preset, spec), sr)
    total = loop_length_samples(preset, rpm, sr)

    voice = EngineVoice(
        preset,
        rpm=rpm,
        load=spec.load,
        throttle=spec.throttle,
        sr=sr,
        block=block,
        output_gain=output_gain,
        quantize_period=True,
        loop_samples=total,
    )

    voice.process(warmup_samples(preset, rpm, sr))  # steady state, then discarded
    x = voice.process(total)

    if spec.limiter:
        x = apply_limiter(x, sr, seed=preset.seed)

    if crossfade_ms > 0.0:
        x = dsp.equal_power_crossfade_loop(x, int(crossfade_ms / 1000.0 * sr))
    return x


def layer_descriptors(preset: Preset, sr: int | None = None) -> list[tuple[LayerSpec, float, int]]:
    """Return ``(spec, rpm, length_samples)`` for every layer.

    ``rpm`` is the quantised reference speed the loop was actually rendered at,
    which is what the runtime must use to compute playback speed.
    """
    sr = int(sr or preset.render.sample_rate)
    out = []
    for spec in DEFAULT_LAYERS:
        rpm = quantized_rpm(preset, layer_rpm(preset, spec), sr)
        out.append((spec, rpm, loop_length_samples(preset, rpm, sr)))
    return out
