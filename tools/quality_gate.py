"""Quality gate: the numbers the roadmap is measured against.

Every roadmap stage should move at least one of these in the right direction,
so this is the acceptance harness for "did the simulation get better".

    python tools/quality_gate.py            # full report
    python tools/quality_gate.py --quick    # audio metrics only, one preset

Metrics
    rtf           synthesiser speed (must stay >= 1.2x for a live audio thread)
    sr_dev        worst 1/3-octave deviation between 44.1 / 48 / 96 kHz
                  (a physical timbre must not depend on the output rate)
    lufs_spread   loudness range across presets at the same operating point
    latency       worst-case queued audio latency (stage 6)
    crank_deg     crank degrees per physics substep at the redline
    phase_lock    whether the audio firing grid follows the physics crank
    core          native C++ DSP core in use, or the numpy reference (stage 5)
    paths         per-cylinder exhaust chains, not per-bank (stage 4)
    wiebe         per-cylinder burn: pulse depth and energy conservation (stage 7)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(_ROOT, ".pycache"))

import numpy as np  # noqa: E402

import simulator.bootstrap  # noqa: F401,E402
from soundgen.config import discover_presets  # noqa: E402
from simulator.bootstrap import preset_dir  # noqa: E402
from simulator.view.repro import load_spec  # noqa: E402

DEFAULT_SR = 48000


def _presets():
    return discover_presets(preset_dir())


def load(stem: str):
    for p in _presets():
        if p.stem == stem:
            return load_spec(p)
    raise SystemExit(f"no preset named {stem}")


def render(preset, sr=DEFAULT_SR, rpm=None, seconds=4.0, block=1024):
    from soundgen.voice import EngineVoice, calibrate_gain

    rpm = float(rpm if rpm is not None else preset.engine.redline_rpm * 0.6)
    gain = calibrate_gain(preset, sr=sr, block=block)
    voice = EngineVoice(preset, rpm=rpm, sr=sr, block=block, output_gain=gain)
    voice.set_params(rpm=rpm, load=0.9, throttle=0.9)
    for _ in range(16):
        voice.process(block)
    n = int(seconds * sr)
    chunks, done = [], 0
    while done < n:
        chunks.append(voice.process(block))
        done += block
    return np.concatenate(chunks)


def rtf(preset, block=1024, seconds=3.0):
    from soundgen.voice import EngineVoice, calibrate_gain

    gain = calibrate_gain(preset, sr=DEFAULT_SR, block=block)
    voice = EngineVoice(preset, rpm=3000.0, sr=DEFAULT_SR, block=block, output_gain=gain)
    voice.set_params(rpm=preset.engine.redline_rpm * 0.6, load=0.9, throttle=0.9)
    for _ in range(6):
        voice.process(block)
    n = int(seconds * DEFAULT_SR)
    t0 = time.perf_counter()
    done = 0
    while done < n:
        voice.process(block)
        done += block
    elapsed = time.perf_counter() - t0
    return (n / DEFAULT_SR) / elapsed


def third_octave_bands(x, sr, fmin=39.0, fmax=16000.0):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    centres = 1000.0 * 2.0 ** (np.arange(-16, 5) / 3.0)
    centres = centres[(centres >= fmin) & (centres <= fmax)]
    out = []
    for fc in centres:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        out.append(spec[(freqs >= lo) & (freqs < hi)].sum())
    out = np.asarray(out)
    return out / (out.sum() + 1e-30)


def sr_deviation(preset):
    """Worst octave-band difference between sample rates, in dB.

    Octave bands, not 1/3-octave: an engine is a line spectrum, so a narrow band
    holds two or three harmonics and a few degrees of phase shift reads as 3 dB
    of "change" that is not audible. Octave bands hold 8-30 lines and measure
    the timbre instead of the harmonic alignment. Several speeds are averaged
    for the same reason.
    """
    speeds = (0.20, 0.35, 0.50, 0.65, 0.80)
    rows = {}
    for sr in (44100, 48000, 96000):
        acc = np.zeros(len(dsp_octave_edges()))
        for frac in speeds:
            rpm = preset.engine.idle_rpm + frac * (
                preset.engine.redline_rpm - preset.engine.idle_rpm
            )
            x = render(preset, sr=sr, rpm=rpm, seconds=2.5)
            acc += dsp_octave_bands(x, sr)
        rows[sr] = acc / len(speeds)
    worst, corr = 0.0, 1.0
    for a, b in ((44100, 48000), (48000, 96000), (44100, 96000)):
        da = 10 * np.log10(rows[a] + 1e-30)
        db = 10 * np.log10(rows[b] + 1e-30)
        worst = max(worst, float(np.abs(da - db).max()))
        corr = min(corr, float(np.corrcoef(np.log(rows[a] + 1e-30), np.log(rows[b] + 1e-30))[0, 1]))
    return worst, corr


def dsp_octave_edges():
    from soundgen import dsp

    return dsp._OCTAVE_EDGES


def dsp_octave_bands(x, sr):
    from soundgen import dsp

    return dsp.octave_bands(x, sr)


def sweep_loudness(preset, sr=DEFAULT_SR, block=1024):
    """Loudness integrated over the operating sweep, the same basis calibrate_gain uses.

    Measuring at one rpm would not test the contract: the calibration promises
    equal loudness *across the rev range*, so a preset may legitimately be a bit
    louder at one point and quieter at another as long as the integral matches.
    """
    from soundgen import dsp
    from soundgen.voice import EngineVoice, calibrate_gain

    gain = calibrate_gain(preset, sr=sr, block=block)
    idle, red = preset.engine.idle_rpm, preset.engine.redline_rpm
    v = EngineVoice(preset, rpm=idle, sr=sr, block=block, output_gain=gain, clip=True)
    v.process(int(0.25 * sr))
    num = den = 0.0
    peak = 0.0
    for i in range(9):
        f = i / 8.0
        rpm = idle + f * (red - idle)
        for load, thr in ((1.0, 1.0), (0.45, 0.42)):
            v.set_params(rpm=rpm, load=load, throttle=thr)
            v.process(int(0.05 * sr))
            frame = v.process(int(0.20 * sr))
            num += dsp.loudness_lufs(frame, sr) * frame.size
            den += frame.size
            peak = max(peak, dsp.peak(frame))
    return (num / den if den else -120.0), peak


def loudness_spread(stems):
    rows = []
    for stem in stems:
        preset, _ = load(stem)
        lufs, peak = sweep_loudness(preset)
        rows.append((stem, lufs, peak))
    values = np.array([r[1] for r in rows])
    return rows, float(values.max() - values.min())


def latency_table(block, buffers):
    per_block_ms = block / DEFAULT_SR * 1000.0
    return (buffers - 1) * per_block_ms, per_block_ms


def crank_degrees(preset):
    from simulator.physics import EngineModel

    deg = 360.0 * preset.engine.redline_rpm / 60.0 * EngineModel.FIXED_DT
    return deg, 720.0 / deg


def phase_lock(preset):
    """Does the synthesiser follow the physics crank, or free-run on its own?

    A locked synthesiser accepts ``crank_phase`` and reports ``crank_locked``.
    Without it two engines at the same rpm are audible copies of each other and
    no firing can be aligned to a real piston position.
    """
    from soundgen.voice import EngineVoice

    voice = EngineVoice(preset, rpm=3000.0, sr=DEFAULT_SR, block=1024)
    if not hasattr(voice, "crank_locked"):
        return False
    voice.set_params(rpm=3000.0, crank_cycles=1.25)
    voice.process(1024)
    return bool(voice.crank_locked)


def exhaust_paths(preset):
    """Stage 4: per-cylinder exhaust chains, or the legacy per-bank pair.

    A V8 with two chains filters four cylinders through one transfer function
    each; only their timing differs. Eight chains give every pulse its own
    pipe. Returns ``(chains, cylinders)``.
    """
    from soundgen.voice import EngineVoice

    voice = EngineVoice(preset, rpm=3000.0, sr=DEFAULT_SR, block=1024)
    return len(voice._bank_chains), voice.n_cyl


def wiebe_report(preset):
    """Stage 7: per-cylinder burn depth and energy conservation.

    Returns ``(pulse_depth, conservation)``:

    * ``pulse_depth`` -- peak of the Wiebe torque envelope over one cycle
      divided by its own mean. 1.0 means no pulses (mean-value); 3-8 is a
      healthy four-stroke pulse train resolved by the substep grid.
    * ``conservation`` -- the angle-integral of the torque multiplier over one
      cycle. The mean-value contract requires 1.0; the dyno curve, BMEP and
      every power figure are this integral.
    """
    import math

    from simulator.physics import TWO_PI, EngineModel, EngineSpec

    spec = EngineSpec.from_preset(preset)
    model = EngineModel(spec)
    rpm = 0.5 * (spec.idle_rpm + spec.redline_rpm)
    model.rpm = rpm
    omega = rpm * TWO_PI / 60.0
    cycle_rad = TWO_PI * spec.cycle_revs
    dt = EngineModel.FIXED_DT
    dtheta = omega * dt
    steps = max(int(round(cycle_rad / dtheta)), 8)
    total = 0.0
    peak = 0.0
    for _ in range(steps):
        burn = model._step_cylinders(dt, omega)
        total += burn * dtheta
        peak = max(peak, burn)
        model.theta = (model.theta + dtheta) % cycle_rad
    mean = total / cycle_rad
    depth = peak / mean if mean > 1e-9 else 1.0
    return depth, mean


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="audio metrics on one preset only")
    ap.add_argument("--preset", default="v8_flatplane")
    args = ap.parse_args(argv)

    preset, spec = load(args.preset)
    print(f"preset: {preset.name}\n")

    r = rtf(preset)
    print(f"rtf            {r:6.2f}x realtime  ({100/r:.1f}% of one core)")

    dev, corr = sr_deviation(preset)
    print(f"sr_dev         {dev:6.1f} dB worst octave band, envelope corr {corr:.3f}")

    if not args.quick:
        stems = ["i4_gasoline", "v8_flatplane", "v12_na", "lambo_huracan",
                 "single_2stroke", "rotary_2rotor", "crazy_screamer_14500"]
        rows, spread = loudness_spread(stems)
        print(f"lufs_spread    {spread:6.1f} dB across {len(stems)} presets over the operating sweep")
        for stem, value, peak in sorted(rows, key=lambda r: r[1]):
            print(f"                 {stem:24} {value:7.1f} LUFS  peak {peak:.3f}")

    queued, one = latency_table(128, 3)
    print(f"latency        {queued + one:6.1f} ms queued (block 128, 3 buffers -- stage 6)")

    deg, steps = crank_degrees(preset)
    print(f"crank_deg      {deg:6.1f} deg per substep at {preset.engine.redline_rpm:.0f} rpm "
          f"({steps:.0f} steps/cycle)")

    print(f"phase_lock     {'yes' if phase_lock(preset) else 'NO -- audio free-runs'}")

    from soundgen import native

    print(f"core           {native.describe()}")

    chains, cylinders = exhaust_paths(preset)
    per_cyl = "per-cylinder" if chains == cylinders else f"PER-BANK only ({chains} chains)"
    print(f"paths          {chains} exhaust chains for {cylinders} cylinders -- {per_cyl}")

    depth, conservation = wiebe_report(preset)
    print(f"wiebe          pulse depth {depth:4.1f}x, energy conservation {conservation:5.3f} "
          f"(stage 7: depth > 1 pulses, conservation == 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
