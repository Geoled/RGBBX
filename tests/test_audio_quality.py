"""Regression tests for the library-backed audio and knock paths."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from simulator import bootstrap  # noqa: F401  (sets sys.path up for soundgen)
from simulator.audio.driver import EngineAudio
from simulator.physics import EngineModel, EngineSpec
from simulator.selftest import start_engine
from soundgen import dsp
from soundgen.config import load_preset
from soundgen.wavio import read_wav, write_wav

PRESETS = Path(__file__).resolve().parents[1] / "tools" / "soundgen" / "presets"


def test_fft_convolver_matches_direct_convolution() -> None:
    rng = np.random.default_rng(7)
    kernel = rng.standard_normal(37)
    x = rng.standard_normal(512)
    conv = dsp.FFTConvolver(kernel, block_size=128)
    got = conv.process(x)
    expected = np.convolve(x, kernel)[: x.size]
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_wav_24_bit_roundtrip(tmp_path: Path) -> None:
    sr = 48000
    t = np.arange(sr // 10, dtype=np.float64) / sr
    x = 0.45 * np.sin(2.0 * math.pi * 440.0 * t)
    path = write_wav(tmp_path / "tone.wav", x, sr, 24)
    y, got_sr = read_wav(path)
    assert got_sr == sr
    assert y.shape == x.shape
    assert float(np.max(np.abs(y - x))) < 2.0 / 8388607.0


def test_lufs_measurement_is_finite() -> None:
    pytest.importorskip("pyloudnorm")
    from soundgen.render import _measure

    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * math.pi * 440.0 * t)
    _peak, _rms, lufs = _measure(x, sr)
    assert lufs is not None
    assert np.isfinite(lufs)
    assert -40.0 < lufs < -10.0


def test_knock_sensitivity_to_octane() -> None:
    def peak_knock(octane: float) -> float:
        preset = load_preset(PRESETS / "i3_turbo.toml")
        preset.block.octane = octane
        model = EngineModel(EngineSpec.from_preset(preset))
        start_engine(model)
        model.start_dyno_pull()
        model.dyno.ramp_rpm_s = 0.0
        model.dyno.target_rpm = 2500.0
        model.dyno.end_rpm = max(model.spec.redline_rpm, 2500.0)
        peak = 0.0
        for _ in range(int(3.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
            peak = max(peak, model.knock_level)
        return peak

    low = peak_knock(80.0)
    high = peak_knock(105.0)
    assert low > 0.10, f"low-octane fuel should knock, got {low:.2f}"
    assert low > high + 0.05, f"octane should suppress knock: {low:.2f} vs {high:.2f}"


def test_loop_mixer_is_finite_and_uses_spline() -> None:
    preset = load_preset(PRESETS / "i3_turbo.toml")
    audio = EngineAudio(preset, enabled=False)
    audio._render_shots({"turbo_spool": 1, "knock": 1})
    assert set(audio._loops) == {"turbo_spool", "knock"}
    assert audio._loops["turbo_spool"][3] is not None, "turbo loop should use cubic interpolation"
    audio.update(3000.0, 1.0, 1.0, boost=1.0, knock=1.0)
    buffer = np.zeros(512, dtype=np.float64)
    audio._mix_loops(buffer, 1.0, 1.0)
    assert np.all(np.isfinite(buffer))
    assert float(np.max(np.abs(buffer))) > 1e-4
