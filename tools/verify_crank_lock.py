"""Crank-lock acceptance test for the synthesiser.

Run:  python tools/verify_crank_lock.py

What the lock has to prove:

1. without it, the event grid drifts against a supplied phase (this is the
   behaviour the lock replaces);
2. with it, the grid tracks the supplied phase to a small fraction of a cycle;
3. locking never introduces a discontinuity -- a click would be worse than the
   drift it fixes.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(_ROOT, ".pycache"))

import numpy as np  # noqa: E402

import simulator.bootstrap  # noqa: F401,E402
from soundgen.config import discover_presets  # noqa: E402
from simulator.bootstrap import preset_dir  # noqa: E402
from simulator.view.repro import load_spec  # noqa: E402
from soundgen.voice import EngineVoice, calibrate_gain  # noqa: E402

SR = 48000
BLOCK = 1024


def preset(stem="v8_flatplane"):
    for p in discover_presets(preset_dir()):
        if p.stem == stem:
            return load_spec(p)[0]
    raise SystemExit(f"no preset {stem}")


def phase_error(voice, cycles):
    """Absolute distance between the grid and the supplied cycle count."""
    return abs(float(cycles) - float(voice._abs_cycles))


def run(locked: bool, blocks: int = 120, rpm_drift: float = 3.0):
    pr = preset()
    gain = calibrate_gain(pr, sr=SR, block=BLOCK)
    v = EngineVoice(pr, rpm=3000.0, sr=SR, block=BLOCK, output_gain=gain)
    v.set_params(rpm=3000.0, load=0.5, throttle=0.5)
    for _ in range(8):
        v.process(BLOCK)

    # a crank that advances at a slightly different rpm than the audio is told --
    # exactly the condition that makes a free-running grid drift
    cycles = 0.0
    errors = []
    clicks = []
    prev_out = None
    for i in range(blocks):
        rpm_true = 3000.0 + rpm_drift * np.sin(i / 20.0)
        cycles += rpm_true / 60.0 / max(v.cyc_revs, 1.0) * BLOCK / SR
        v.set_params(rpm=3000.0, crank_cycles=cycles if locked else None)
        out = v.process(BLOCK)
        errors.append(phase_error(v, cycles))
        # a click shows up as a jump between the last sample of one block and the
        # first of the next, well above the signal's own sample-to-sample step
        if prev_out is not None:
            clicks.append(abs(float(out[0]) - float(prev_out[-1])))
        prev_out = out
    return np.array(errors), np.array(clicks), v


def main() -> int:
    free_err, free_clicks, _ = run(locked=False)
    lock_err, lock_clicks, v = run(locked=True)

    print(f"preset {v.preset.name}, {v.n_cyl} cyl, {len(free_err)} blocks\n")
    print(f"{'':22} {'mean err':>10} {'max err':>10} {'max step':>10}")
    print(f"{'free-running':22} {free_err.mean():10.3f} {free_err.max():10.3f} "
          f"{free_clicks.max():10.4f}")
    print(f"{'crank-locked':22} {lock_err.mean():10.3f} {lock_err.max():10.3f} "
          f"{lock_clicks.max():10.4f}")

    tail = lock_err[len(lock_err) // 2 :]
    print(f"\nlocked error after settling: mean {tail.mean():.4f} cycles "
          f"({tail.mean() * 360:.1f} deg), max {tail.max():.4f} cycles")

    ok = True
    if tail.mean() > 0.02:
        print("FAIL: grid does not track the supplied phase")
        ok = False
    if free_err.mean() < tail.mean() * 5:
        print("FAIL: the free-running grid should drift much more than the locked one")
        ok = False
    if lock_clicks.max() > max(free_clicks.max() * 3.0, 0.05):
        print("FAIL: locking introduces a discontinuity")
        ok = False
    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
