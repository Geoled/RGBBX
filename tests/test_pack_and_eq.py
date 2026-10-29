"""Regression tests for the master EQ, the game pack and loop periodicity.

Three contracts are pinned here:

* the ``[eq]`` stage shapes the spectrum it says it shapes, and an inactive
  stage is *bit-identical* to having no stage at all;
* a loop rendered with the EQ (and with the rotating flow-burst bank) is still
  exactly periodic -- the seam is what a game engine loops on, and it used to
  be broken at -32 dB by a block-order bug;
* ``soundgen pack`` writes the folder a game project expects: per-engine
  ``layers/``/``events/`` subfolders, a manifest with load and throttle, a
  README and the zip.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simulator import bootstrap  # noqa: F401  (sets sys.path up before soundgen)
from soundgen import dsp, layers as layers_mod, pack, voice
from soundgen.config import Eq, load_preset
from simulator.bootstrap import preset_dir

SR = 24000
BLOCK = 512


def _preset(stem: str = "i4_gasoline"):
    return load_preset(preset_dir() / f"{stem}.toml")


def _band_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    return 10.0 * float(np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-30))


def _response(kernel: np.ndarray, sr: int, freq: float) -> float:
    """Magnitude of an FIR at one frequency, in dB."""
    n = 1 << 16
    h = np.zeros(n)
    h[: kernel.size] = kernel
    spec = np.abs(np.fft.rfft(h))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return dsp.lin_to_db(float(spec[int(np.argmin(np.abs(f - freq)))] ) + 1e-30)


# --------------------------------------------------------------------------- #
# EQ
# --------------------------------------------------------------------------- #
def test_eq_kernel_boosts_and_cuts_the_requested_bands() -> None:
    kernel = dsp.eq_kernel(
        SR,
        low_hz=100.0,
        low_db=6.0,
        bands=((3000.0, 1.0, -6.0),),
        taps=1025,
    )
    assert _response(kernel, SR, 50.0) == pytest.approx(6.0, abs=1.0)
    assert _response(kernel, SR, 3000.0) == pytest.approx(-6.0, abs=1.0)
    # a bell is local: two octaves below its centre it is back to the shelf
    assert _response(kernel, SR, 3000.0 / 4) == pytest.approx(0.0, abs=1.0)


def test_eq_kernel_is_long_enough_for_its_lowest_corner() -> None:
    kernel = dsp.eq_kernel(SR, low_hz=60.0, low_db=3.0, taps=1025)
    # >= ~6 periods of 60 Hz, otherwise the shelf edge is not resolved
    assert kernel.size >= int(6.0 * SR / 60.0) - 2


def test_eq_stage_changes_the_rendered_spectrum() -> None:
    preset = _preset()
    spec = layers_mod.DEFAULT_LAYERS[1]
    flat = replace(preset, eq=replace(preset.eq, enabled=False))
    shaped = replace(
        preset,
        eq=Eq(enabled=True, low_hz=120.0, low_db=4.0, bands=[[3000.0, 0.9, -4.0]]),
    )
    a = layers_mod.render_layer(flat, spec, 1.0, sr=SR, block=BLOCK)
    b = layers_mod.render_layer(shaped, spec, 1.0, sr=SR, block=BLOCK)
    assert _band_db(b, SR, 60.0, 160.0) > _band_db(a, SR, 60.0, 160.0) + 1.5
    assert _band_db(b, SR, 2500.0, 3600.0) < _band_db(a, SR, 2500.0, 3600.0) - 1.5


def test_inactive_eq_is_bit_identical_to_no_stage() -> None:
    preset = _preset()
    spec = layers_mod.DEFAULT_LAYERS[1]
    disabled = replace(preset, eq=replace(preset.eq, enabled=False))
    zeroed = replace(preset, eq=Eq(enabled=True))
    a = layers_mod.render_layer(disabled, spec, 1.0, sr=SR, block=BLOCK)
    b = layers_mod.render_layer(zeroed, spec, 1.0, sr=SR, block=BLOCK)
    assert np.array_equal(a, b)


def test_loop_with_eq_is_exactly_periodic() -> None:
    """The seam contract, with the EQ stage and the rotated burst bank active."""
    preset = _preset()
    spec = layers_mod.DEFAULT_LAYERS[1]
    rpm = layers_mod.quantized_rpm(preset, layers_mod.layer_rpm(preset, spec), SR)
    total = layers_mod.loop_length_samples(preset, rpm, SR)

    probe = voice.EngineVoice(
        preset,
        rpm=rpm,
        load=spec.load,
        throttle=spec.throttle,
        sr=SR,
        block=BLOCK,
        output_gain=1.0,
        quantize_period=True,
        loop_samples=total,
    )
    probe.process(layers_mod.warmup_samples(preset, rpm, SR))
    first = probe.process(total)
    following = probe.process(BLOCK)
    seam = dsp.rms(first[:BLOCK] - following)
    assert seam < 1e-6 * dsp.rms(first), f"seam {seam:.2e} ({dsp.lin_to_db(seam / dsp.rms(first)):.1f} dB)"


# --------------------------------------------------------------------------- #
# game pack
# --------------------------------------------------------------------------- #
def test_pack_writes_the_game_layout(tmp_path: Path) -> None:
    preset = _preset()
    opts = pack.PackOptions(
        sample_rate=SR,
        bit_depth=16,
        channels=1,
        include_events=True,
        archive=True,
        dither=True,
    )
    result = pack.export_pack([preset], tmp_path / "pack", opts, quiet=True)

    slug = preset.slug
    assert (tmp_path / "pack" / slug / "layers").is_dir()
    assert (tmp_path / "pack" / slug / "events").is_dir()
    assert (tmp_path / "pack" / "README.md").is_file()
    layers = sorted((tmp_path / "pack" / slug / "layers").glob("*.wav"))
    assert {p.stem.split("__")[-1] for p in layers} == {
        spec.key for spec in layers_mod.DEFAULT_LAYERS
    }
    assert any((tmp_path / "pack" / slug / "events").glob("*.wav"))
    assert result.archive is not None and result.archive.is_file()
    with zipfile.ZipFile(result.archive) as zf:
        names = zf.namelist()
    assert any(n.endswith("README.md") for n in names)
    assert any("/layers/" in n.replace("\\", "/") for n in names)

    doc = json.loads((tmp_path / "pack" / "manifest.json").read_text(encoding="utf-8"))
    entry = doc["presets"][preset.slug]
    idle = next(layer for layer in entry["layers"] if layer["file"].endswith("__idle.wav"))
    assert idle["file"].startswith(f"{slug}/layers/")
    assert idle["looping"] is True
    assert idle["referenceRpm"] > 0.0
    assert idle["load"] == pytest.approx(0.03)
    assert idle["throttle"] == pytest.approx(0.0)


def test_pack_readme_documents_how_to_blend(tmp_path: Path) -> None:
    preset = _preset()
    opts = pack.PackOptions(sample_rate=SR, channels=1, archive=False)
    pack.export_pack([preset], tmp_path / "pack", opts, quiet=True)
    text = (tmp_path / "pack" / "README.md").read_text(encoding="utf-8")
    assert "playbackRate" in text
    assert "referenceRpm" in text
    assert f"{preset.slug}/layers/{preset.slug}__idle.wav" in text


def test_pack_cli_parser_accepts_a_pack_command() -> None:
    from soundgen.cli import build_parser

    args = build_parser().parse_args(["pack", "--out", "x", "--zip"])
    assert args.func.__name__ == "cmd_pack"
    assert args.zip is True
