"""One-shot and looping sound events.

These are triggered discretely by the simulator: starter, catch, shutdown,
backfire, blow-off, turbo spool and detonation. Unlike the steady layers, some
randomness here is desirable (no two backfires should be identical), so these
renderers deliberately use fresh random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .config import Preset
from .voice import EngineVoice


@dataclass(frozen=True)
class EventSpec:
    key: str
    label: str
    duration_s: float
    looping: bool = False
    peak_db: float = -3.0


EVENTS: tuple[EventSpec, ...] = (
    EventSpec("start_crank", "Starter cranking", 2.2, peak_db=-6.0),
    EventSpec("start_catch", "Engine catches", 1.5, peak_db=-2.0),
    EventSpec("shutdown", "Shutdown", 2.0, peak_db=-3.0),
    EventSpec("backfire", "Backfire pop", 0.55, peak_db=-1.5),
    EventSpec("blow_off", "Blow-off valve", 0.65, peak_db=-5.0),
    EventSpec("turbo_spool", "Turbo spool", 2.0, looping=True, peak_db=-9.0),
    EventSpec("knock", "Detonation ping", 1.6, looping=True, peak_db=-14.0),
)


def _conv(kernel: np.ndarray, x: np.ndarray, block: int) -> np.ndarray:
    """Convolve a finite kernel with a signal, block-aligned."""
    conv = dsp.FFTConvolver(kernel, block)
    pad = (-x.size) % block
    y = conv.process(np.concatenate([x, np.zeros(pad, dtype=np.float64)]))
    return y[: x.size]


def _render_ramped(
    preset: Preset,
    rpm_at,
    duration_s: float,
    sr: int,
    block: int,
    gain: float,
    load_at=None,
    thr_at=None,
    warm_s: float = 0.25,
) -> np.ndarray:
    """Render while the operating point changes over time (start / shutdown)."""
    voice = EngineVoice(
        preset,
        rpm=rpm_at(0.0),
        load=load_at(0.0) if load_at else 0.0,
        throttle=thr_at(0.0) if thr_at else 0.0,
        sr=sr,
        block=block,
        output_gain=gain,
    )
    voice.process(int(warm_s * sr))
    n = int(duration_s * sr)
    out = np.empty(n, dtype=np.float64)
    pos = 0
    while pos < n:
        tsec = pos / sr
        voice.set_params(
            rpm=rpm_at(tsec),
            load=load_at(tsec) if load_at else None,
            throttle=thr_at(tsec) if thr_at else None,
        )
        take = min(block, n - pos)
        out[pos : pos + take] = voice.process(take)
        pos += take
    return out


# --------------------------------------------------------------------------- #
# individual events
# --------------------------------------------------------------------------- #
def _start_crank(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    n = int(EVENTS[0].duration_s * sr)
    starter = preset.engine.starter_rpm
    x = _render_ramped(
        preset,
        rpm_at=lambda t: starter * (0.94 + 0.06 * np.sin(dsp.TWO_PI * 1.7 * t)),
        load_at=lambda t: 0.22,
        duration_s=EVENTS[0].duration_s,
        sr=sr,
        block=block,
        gain=gain,
        warm_s=0.1,
    )
    t = np.arange(n, dtype=np.float64) / sr
    # starter motor whine with a compression-synced wobble
    osc1 = dsp.PhaseOsc(sr)
    osc2 = dsp.PhaseOsc(sr)
    whine = osc1.render(n, 1180.0) + 0.45 * osc2.render(n, 1180.0 * 2.03)
    wobble = 0.55 + 0.45 * np.abs(np.sin(dsp.TWO_PI * 11.0 * t))
    x = x + whine * wobble * 0.09
    # a faint gear mesh rattle
    rattle = _conv(dsp.bandpass_kernel(sr, 1800.0, 5200.0, 2, 257), rng.standard_normal(n), block)
    x = x + rattle * (0.05 * wobble)
    return dsp.fade_edges(x, int(0.01 * sr), int(0.05 * sr))


def _start_catch(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    dur = EVENTS[1].duration_s
    starter = preset.engine.starter_rpm
    idle = preset.engine.idle_rpm

    def rpm_at(t: float) -> float:
        if t < 0.45:
            k = t / 0.45
            return starter + (idle * 1.9 - starter) * (k**0.6)
        k = min((t - 0.45) / 0.85, 1.0)
        return idle * (1.0 + 0.9 * (1.0 - k) ** 2)

    thr_at = lambda t: 0.55 * max(0.0, 1.0 - t / 0.7)  # throttle blip to catch
    x = _render_ramped(preset, rpm_at, dur, sr, block, gain, thr_at=thr_at, warm_s=0.08)
    return dsp.fade_edges(x, int(0.005 * sr), int(0.08 * sr))


def _shutdown(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    dur = EVENTS[2].duration_s
    idle = preset.engine.idle_rpm

    def rpm_at(t: float) -> float:
        k = dsp.clamp(t / 1.35, 0.0, 1.0)
        return max(idle * (1.0 - k) ** 1.6 + 12.0 * (1.0 - k), 6.0)

    def load_at(t: float) -> float:
        return 0.25 * max(0.0, 1.0 - t / 1.1)

    x = _render_ramped(preset, rpm_at, dur, sr, block, gain, load_at=load_at, warm_s=0.1)
    n = x.size
    t = np.arange(n, dtype=np.float64) / sr
    # final compression "clunk"
    clunk_env = np.exp(-np.maximum(t - 1.32, 0.0) / 0.06) * (t >= 1.32)
    clunk = 0.35 * clunk_env * np.sin(dsp.TWO_PI * 58.0 * t)
    x = x + clunk
    return dsp.fade_edges(x, int(0.004 * sr), int(0.12 * sr))


def _backfire(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    n = int(EVENTS[3].duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    noise = rng.standard_normal(n)
    # low thump
    body = _conv(dsp.bandpass_kernel(sr, 70.0, 900.0, 2, 257), noise, block) * np.exp(-t / 0.11)
    # sharp crack
    crack = _conv(dsp.highpass_kernel(sr, 1400.0, 2, 193), noise, block) * np.exp(-t / 0.018)
    # a second, smaller pop
    t2 = np.maximum(t - 0.16, 0.0)
    pop2 = _conv(dsp.bandpass_kernel(sr, 120.0, 1400.0, 2, 193), rng.standard_normal(n), block)
    pop2 *= np.exp(-t2 / 0.07) * (t >= 0.16)
    x = body * 1.0 + crack * 0.55 + pop2 * 0.45
    return dsp.fade_edges(x, int(0.001 * sr), int(0.06 * sr))


def _blow_off(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    n = int(EVENTS[4].duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    noise = rng.standard_normal(n)
    bp = _conv(dsp.bandpass_kernel(sr, 2200.0, 9000.0, 2, 257), noise, block)
    env = (1.0 - np.exp(-t / 0.004)) * np.exp(-t / 0.22)
    flutter = 0.72 + 0.28 * np.abs(np.sin(dsp.TWO_PI * 22.0 * t))
    x = bp * env * flutter
    return dsp.fade_edges(x, int(0.001 * sr), int(0.08 * sr))


def _turbo_spool(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    n = int(EVENTS[5].duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    k = t / max(t[-1], 1e-6)
    f = 1500.0 + 4200.0 * k
    osc1 = dsp.PhaseOsc(sr)
    osc2 = dsp.PhaseOsc(sr)
    whistle = osc1.render(n, f) + 0.4 * osc2.render(n, f * 1.51)
    whistle *= 0.16 * (0.15 + 0.85 * k)
    whoosh = _conv(dsp.bandpass_kernel(sr, 900.0, 6500.0, 2, 257), rng.standard_normal(n), block)
    whoosh *= 0.13 * (0.2 + 0.8 * k)
    x = whistle + whoosh
    x = dsp.equal_power_crossfade_loop(x, int(0.03 * sr))
    return x


def _knock(preset: Preset, sr: int, block: int, gain: float, rng) -> np.ndarray:
    n = int(EVENTS[6].duration_s * sr)
    x = np.zeros(n, dtype=np.float64)
    ping = dsp.resonator_kernel(
        sr,
        [(6300.0, 22.0, 1.0), (8100.0, 18.0, 0.5)],
        length_s=0.09,
    )
    count = int(rng.integers(5, 10))
    for _ in range(count):
        i = int(rng.integers(0, max(n - ping.size, 1)))
        amp = float(rng.uniform(0.35, 1.0))
        m = min(ping.size, n - i)
        x[i : i + m] += amp * ping[:m]
    x = dsp.equal_power_crossfade_loop(x, int(0.02 * sr))
    return x


_RENDERERS = {
    "start_crank": _start_crank,
    "start_catch": _start_catch,
    "shutdown": _shutdown,
    "backfire": _backfire,
    "blow_off": _blow_off,
    "turbo_spool": _turbo_spool,
    "knock": _knock,
}


def render_event(
    preset: Preset,
    spec: EventSpec,
    output_gain: float,
    sr: int | None = None,
    block: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Render one event, normalised to its own target peak.

    One-shots are discrete sounds; the game's mixer sets how loud they sit
    against the engine layers, so each is peak-normalised to a designed target
    rather than sharing the layer calibration gain.
    """
    sr = int(sr or preset.render.sample_rate)
    block = int(block or preset.render.block_size)
    stable = sum(ord(ch) * (i + 1) for i, ch in enumerate(spec.key))
    rng = np.random.default_rng(seed if seed is not None else preset.seed + stable)
    fn = _RENDERERS[spec.key]
    x = fn(preset, sr, block, output_gain, rng)
    return dsp.normalize(x, spec.peak_db)


def events_for(preset: Preset) -> list[EventSpec]:
    """Events appropriate for a preset.

    Boost events need a boost device, a blow-off valve has to actually be
    fitted, and detonation ("ping") only means anything on a spark-ignition
    engine: a diesel burns by compression and a turbine has no discrete
    combustion event at all.
    """
    out = []
    for spec in EVENTS:
        if spec.key in ("blow_off", "turbo_spool") and not preset.turbo.enabled:
            continue
        if spec.key == "blow_off" and not preset.turbo.blow_off:
            continue
        if spec.key == "knock" and preset.block.ignition != "spark":
            continue
        out.append(spec)
    return out
