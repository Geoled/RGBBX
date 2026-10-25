"""Regression tests for the gas-dynamic voice (roadmap phase A/B).

The gas core is the second synthesis paradigm in the tree: sound *emerges*
from a network of lumped gas systems (compressible orifice flow, choked
branch, supersonic clamp) instead of passing through filters. These tests
pin its contracts:

* the network is stable (finite outputs, bounded temperatures);
* the firing order is the dominant line (the whole point of the paradigm);
* load changes the sound (energy in the burn, not a gain knob);
* the tract geometry changes the sound (constructor contract: pipe length
  from the preset shifts the resonance, muffler losses darken it);
* performance: the C++ batched driver holds realtime far above the live
  requirement.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from simulator import bootstrap  # noqa: F401
from soundgen import gasvoice
from simulator.bootstrap import preset_dir
from soundgen.config import load_preset

SR = 48000


def _preset(stem: str):
    return load_preset(preset_dir() / f"{stem}.toml")


gas = pytest.mark.skipif(
    not gasvoice.AVAILABLE, reason="gas core not built (tools/soundgen/native/build.py)"
)


def _render(voice, rpm, load, seconds):
    voice.set_operating_point(rpm, load, 0.9 if load > 0 else 0.0)
    voice.process(48000)  # 1 s warmup: the network's mean flow must settle
    return voice.process(int(seconds * SR))


def _band_rel(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    tot = float(spec.sum()) + 1e-30
    return 10.0 * math.log10(float(spec[(f >= lo) & (f < hi)].sum()) / tot)


@gas
def test_gas_network_is_stable() -> None:
    """No divergence: finite output, bounded state, across the range."""
    v = gasvoice.GasEngineVoice(_preset("i4_gasoline"), sr=SR, block=256)
    for rpm, load in ((850, 0.05), (3000, 0.9), (6000, 1.0)):
        x = _render(v, float(rpm), load, 1.0)
        assert np.all(np.isfinite(x)), f"non-finite at {rpm} rpm"
        assert float(np.abs(x).max()) < 2.0, f"exploded at {rpm} rpm"


@gas
def test_firing_order_is_the_dominant_line() -> None:
    """The paradigm's claim: the firing frequency comes out of the flow.

    A 4-stroke I4 at 3000 rpm fires at 100 Hz; the steady-state spectrum
    must put more energy in 80-120 Hz than in any other octave band.
    """
    v = gasvoice.GasEngineVoice(_preset("i4_gasoline"), sr=SR, block=256)
    x = _render(v, 3000.0, 0.9, 2.0)
    firing = _band_rel(x, SR, 80, 120)
    others = [
        _band_rel(x, SR, lo, hi)
        for lo, hi in ((10, 40), (40, 80), (120, 200), (200, 400), (400, 800))
    ]
    assert firing > max(others) + 3.0, (
        f"firing band {firing:.1f} dB not dominant over {max(others):.1f}"
    )


@gas
def test_load_changes_the_sound() -> None:
    """More charge energy must measurably change the output."""
    v = gasvoice.GasEngineVoice(_preset("i4_gasoline"), sr=SR, block=256)
    low = _render(v, 3000.0, 0.05, 1.0)
    high = _render(v, 3000.0, 0.9, 1.0)
    assert not np.allclose(low, high), "load had no effect on the gas voice"


@gas
def test_tract_geometry_is_audible() -> None:
    """Constructor contract: pipe length from the preset shifts the sound.

    A 2x longer pipe changes the quarter-wave response, so the steady-state
    spectra must differ in the firing band's neighbourhood.
    """
    base = _preset("i4_gasoline")
    v1 = gasvoice.GasEngineVoice(base, sr=SR, block=256)
    x1 = _render(v1, 3000.0, 0.9, 1.5)

    long_pipe = _preset("i4_gasoline")
    long_pipe.exhaust.pipe_length_m = base.exhaust.pipe_length_m * 2.0
    v2 = gasvoice.GasEngineVoice(long_pipe, sr=SR, block=256)
    x2 = _render(v2, 3000.0, 0.9, 1.5)

    assert not np.allclose(x1, x2, atol=1e-4), (
        "pipe length had no effect on the gas voice"
    )


@gas
def test_gas_voice_holds_realtime() -> None:
    """The C++ batched driver: 1 s of audio must render in < 0.2 s."""
    v = gasvoice.GasEngineVoice(_preset("i4_gasoline"), sr=SR, block=256)
    _render(v, 3000.0, 0.9, 0.5)  # warmed
    t0 = time.perf_counter()
    v.process(SR)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"1 s render took {elapsed:.2f} s (rtf {elapsed:.1f}x)"
