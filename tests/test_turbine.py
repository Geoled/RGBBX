"""Regression tests for the air-standard gas-turbine model."""

from __future__ import annotations

from pathlib import Path

from soundgen.config import load_preset

from simulator import bootstrap  # noqa: F401  (sets sys.path up for soundgen)
from simulator.physics import EngineModel, EngineSpec
from simulator.selftest import dyno_pull, start_engine, summarise

PRESETS = Path(__file__).resolve().parents[1] / "tools" / "soundgen" / "presets"


def _turbine() -> tuple[EngineModel, EngineSpec]:
    preset = load_preset(PRESETS / "turbine.toml")
    spec = EngineSpec.from_preset(preset)
    return EngineModel(spec), spec


def test_turbine_starts_and_holds_idle() -> None:
    model, spec = _turbine()
    start_engine(model)
    assert model._turbine_started
    assert model.running
    assert abs(model.rpm - spec.idle_rpm) < max(0.05 * spec.idle_rpm, 60.0)
    assert model.indicated_torque > 0.0
    assert model.turbine_fuel_kg_s > 0.0
    assert model.turbine_pressure_ratio > 1.0


def test_turbine_cycle_responds_to_speed_and_fuel() -> None:
    model, spec = _turbine()
    idle_torque, idle_power, _, _, _ = model._turbine_cycle(spec.idle_rpm, 0.06)
    max_torque, max_power, max_fuel, pr, tit = model._turbine_cycle(
        spec.redline_rpm, 1.0
    )

    assert max_power > idle_power > 0.0
    assert max_torque > idle_torque > 0.0
    assert max_fuel > 0.0
    assert 8.0 < pr < 20.0
    assert tit > 1400.0
    assert 100.0 < max_power / 1000.0 / (spec.displacement * 1000.0) < 1500.0

    # The compressor and turbine work terms must both be finite and positive;
    # a zero here means the cycle silently collapsed to a constant.
    half_torque, half_power, _, half_pr, _ = model._turbine_cycle(
        spec.redline_rpm * 0.5, 1.0
    )
    assert 0.0 < half_power < max_power
    assert 1.0 < half_pr < pr
    assert half_torque > 0.0


def test_turbine_dyno_sweep_reaches_redline() -> None:
    model, spec = _turbine()
    start_engine(model)
    points = dyno_pull(model)
    summary = summarise(points)

    assert summary is not None
    assert summary["max_rpm"] >= spec.redline_rpm * 0.95
    assert summary["peak_power_kw"] > 1000.0
    assert summary["peak_power_rpm"] <= spec.redline_rpm * 1.05
    assert summary["peak_torque"] > 300.0


def test_turbine_winds_down_when_ignition_is_cut() -> None:
    model, spec = _turbine()
    start_engine(model)
    model.ignition_on = False
    for _ in range(int(5.0 / 0.002)):
        model.step(0.002)
    assert not model._turbine_started
    assert model.rpm < spec.idle_rpm * 0.1
    assert model.indicated_torque == 0.0
    assert model.turbine_power_kw == 0.0
