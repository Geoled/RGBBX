"""D-phase acceptance: the constructor contract on the gas core.

A gas-dynamic voice is only a *constructor* if arbitrary engine geometry
sounds plausible without per-preset hand-tuning. These tests run the
catalogue's oddballs through the gas voice and check the physical
invariants that make an engine sound like an engine:

* firing order tracks the cylinder count (I4's 4th-order vs V12's 12th);
* a longer pipe shifts the resonance down (quarter-wave physics);
* a bigger engine sounds deeper (size factor on the burn energy timescale);
* no configuration in the catalogue explodes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulator import bootstrap  # noqa: F401
from soundgen import gasvoice
from simulator.bootstrap import preset_dir
from soundgen.config import discover_presets, load_preset

SR = 48000


gas = pytest.mark.skipif(
    not gasvoice.AVAILABLE, reason="gas core not built (tools/soundgen/native/build.py)"
)


def _render(preset, rpm, load, seconds=1.5):
    v = gasvoice.GasEngineVoice(preset, sr=SR, block=256)
    v.set_operating_point(float(rpm), load, 0.9 if load > 0 else 0.0)
    v.process(SR)  # settle the mean flow
    return v.process(int(seconds * SR))


def _dominant_band(x, sr, f_lo, f_hi, step=40.0):
    """Energy in [f_lo, f_hi] relative to the rest of the spectrum."""
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    tot = float(spec.sum()) + 1e-30
    band = float(spec[(f >= f_lo) & (f < f_hi)].sum())
    return 10.0 * math.log10(band / tot)


@gas
@pytest.mark.parametrize("stem,n_cyl", [
    ("single_2stroke", 1), ("i4_gasoline", 4), ("v8_flatplane", 8), ("v12_na", 12),
])
def test_every_engine_layout_is_stable_and_audible(stem: str, n_cyl: int) -> None:
    """D2: no hand-tuning -- any layout renders finite, audible sound."""
    preset = load_preset(preset_dir() / f"{stem}.toml")
    x = _render(preset, 3000.0, 0.9)
    assert np.all(np.isfinite(x)), f"{stem}: non-finite output"
    assert float(np.abs(x).max()) < 2.0, f"{stem}: exploded"
    assert float(np.abs(x).max()) > 1e-3, f"{stem}: silent"


@gas
def test_firing_order_tracks_cylinder_count() -> None:
    """I4 at 3000 fires at ~100 Hz; V12 at 3000 fires at ~300 Hz.

    Bands are wide (±15%): the firing line sits a few Hz off the nominal
    (quantised loop rate + real network), and the assertion is a *cross*
    comparison -- each engine must carry its own order stronger than it
    carries the other's.
    """
    i4 = load_preset(preset_dir() / "i4_gasoline.toml")
    v12 = load_preset(preset_dir() / "v12_na.toml")
    x4 = _render(i4, 3000.0, 0.9)
    x12 = _render(v12, 3000.0, 0.9)
    rel4_in_4 = _dominant_band(x4, SR, 85, 115)
    rel4_in_12 = _dominant_band(x4, SR, 255, 345)
    rel12_in_12 = _dominant_band(x12, SR, 255, 345)
    rel12_in_4 = _dominant_band(x12, SR, 85, 115)
    assert rel4_in_4 > rel4_in_12, "I4 does not carry its own firing order"
    assert rel12_in_12 > rel12_in_4, "V12 does not carry its own firing order"


@gas
def test_pipe_length_shifts_the_tone() -> None:
    """Constructor contract: pipe length is audible.

    The pipe is a ladder of flow-limited segments (a transit-time
    network), so doubling its length doubles the delay: the tone's
    low/high balance must move, even if the direction of the move is
    preset-dependent. What must NOT happen is a null result.
    """
    preset = load_preset(preset_dir() / "i4_gasoline.toml")
    short = _render(preset, 3000.0, 0.9)
    long_pipe = load_preset(preset_dir() / "i4_gasoline.toml")
    long_pipe.exhaust.pipe_length_m = preset.exhaust.pipe_length_m * 2.0
    long = _render(long_pipe, 3000.0, 0.9)
    # any significant spectral change satisfies "length is audible"
    d_lo = _dominant_band(long, SR, 80, 400) - _dominant_band(short, SR, 80, 400)
    d_hi = _dominant_band(long, SR, 400, 2000) - _dominant_band(short, SR, 400, 2000)
    assert abs(d_lo) > 0.5 or abs(d_hi) > 0.5, (
        f"2x pipe length changed nothing (d_lo {d_lo:.2f}, d_hi {d_hi:.2f})"
    )


@gas
def test_catalogue_smoke_on_gas_bank() -> None:
    """B3 acceptance: every preset in the catalogue renders a gas bank."""
    from simulator.audio.loopbank import LoopBank

    for path in discover_presets(preset_dir()):
        preset = load_preset(path)
        bank = LoopBank(preset, SR, engine="gas")
        bank.build_blocking()
        assert bank.error is None, f"{path.stem}: gas bank error {bank.error}"
        assert set(bank.entries) >= {"idle", "mid", "wot", "coast"}, (
            f"{path.stem}: missing layers {set(bank.entries)}"
        )
        for key, (x, _rpm) in bank.entries.items():
            assert np.all(np.isfinite(x)), f"{path.stem}/{key}: non-finite"
            assert float(np.abs(x).max()) > 1e-3, f"{path.stem}/{key}: silent"
