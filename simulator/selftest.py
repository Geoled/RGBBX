"""Headless checks for the engine model.

Run with ``python -m simulator --selftest``. Nothing here opens a window or
touches the audio device, so it is the fast way to tell whether the physics is
sane before looking at anything else.

The important checks are *scale free*: brake mean effective pressure and
specific power, not absolute numbers, so they stay meaningful for a 1.3 L rotary
and a 6.0 L V8 alike.
"""

from __future__ import annotations

import math

import numpy as np

from . import bootstrap  # noqa: F401  (sets sys.path up before soundgen import)
from .physics import TWO_PI, EngineModel, EngineSpec, summarise_curve

from soundgen.config import load_preset


def _run(model: EngineModel, seconds: float, dt: float = 0.002, pedal=0.0, crank: bool = False):
    model.cranking = crank
    steps = max(int(seconds / dt), 1)
    for _ in range(steps):
        if callable(pedal):
            model.set_pedal(pedal())
        else:
            model.set_pedal(pedal)
        model.step(dt)


def start_engine(model: EngineModel, crank_s: float = 1.4, settle_s: float = 6.0) -> None:
    """Crank the starter, let go, and wait for the idle to settle."""
    model.reset()
    _run(model, crank_s, crank=True, pedal=0.0)
    model.cranking = False
    _run(model, settle_s, pedal=0.0)


def dyno_pull(model: EngineModel, dt: float = 0.002, timeout: float = 40.0):
    """Hold the throttle wide open and ramp the brake from idle to the redline.

    A speed-holding dyno means the brake torque *is* the engine's torque, so the
    points this returns are a real torque curve rather than a simulation of one.
    The brake is allowed to settle at the starting speed before the ramp begins,
    otherwise the initial catch shows up as a fake torque spike.
    """
    model.start_dyno_pull()
    ramp = model.dyno.ramp_rpm_s
    model.dyno.ramp_rpm_s = 0.0
    _run(model, 1.5, pedal=1.0)

    model.dyno.ramp_rpm_s = ramp
    points: list[tuple[float, float]] = []
    t = 0.0
    while t < timeout:
        model.step(dt)
        t += dt
        points.append((model.rpm, model.brake_torque))
        if model.rpm >= model.dyno.end_rpm * 0.985 and model.dyno.target_rpm >= model.dyno.end_rpm * 0.985:
            break
    model.stop_dyno()
    return points


def summarise(points: list[tuple[float, float]]) -> dict:
    return summarise_curve(points)


def bmep_bar(torque_nm: float, displacement_m3: float, cycle_revs: float = 2.0) -> float:
    """Brake mean effective pressure, bar. Sign-agnostic magnitude."""
    return torque_nm * TWO_PI * cycle_revs / max(displacement_m3, 1e-9) / 1e5


def _preset_paths():
    from .bootstrap import preset_dir
    from soundgen.config import discover_presets

    return discover_presets(preset_dir())


def main(argv: list[str] | None = None) -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    paths = _preset_paths()
    if not paths:
        print("no presets found")
        return 1

    print("Engine simulator self test\n")
    from soundgen import native as _native

    print(f"  DSP core: {_native.describe()}")
    print()
    rows = []
    for path in paths:
        preset = load_preset(path)
        spec = EngineSpec.from_preset(preset)
        model = EngineModel(spec)

        # --- spec sanity -------------------------------------------------- #
        # These used to be collected into a list and never read, so a preset with
        # a zero displacement or a zero throttle area would sail through and
        # fail somewhere much later, in the middle of a dyno pull.
        check(f"{preset.name}: displacement > 0", spec.displacement > 0.0)
        check(f"{preset.name}: throttle area > 0", spec.throttle_area > 0.0)
        check(f"{preset.name}: inertia > 0", spec.inertia > 0.0)

        # --- idle ---------------------------------------------------------- #
        start_engine(model)
        idle = model.rpm
        _run(model, 2.0, pedal=0.0)
        idle_drift = abs(model.rpm - idle)

        # --- dyno pull ----------------------------------------------------- #
        points = dyno_pull(model)
        st = summarise(points)
        finite = all(math.isfinite(p[0]) and math.isfinite(p[1]) for p in points)

        rows.append((preset.name, spec, idle, idle_drift, st, finite))

        check(f"{preset.name}: finite throughout", finite)
        check(f"{preset.name}: displacement {spec.displacement * 1000:.2f} L", spec.displacement > 0)

    print("\n  --- measured curves ---")
    print(
        f"  {'engine':26} {'L':>5} {'idle':>6} {'rpm':>6} {'T Nm':>7} {'@rpm':>6} "
        f"{'kW':>6} {'@rpm':>6} {'hp':>6} {'BMEP':>6} {'kW/L':>6}"
    )
    for name, spec, idle, drift, st, _ in rows:
        if not st:
            continue
        bmep = bmep_bar(st["peak_torque"], spec.displacement, spec.cycle_revs)
        kwl = st["peak_power_kw"] / (spec.displacement * 1000.0)
        print(
            f"  {name:26} {spec.displacement * 1000:5.2f} {idle:6.0f} {drift:6.1f} "
            f"{st['peak_torque']:7.1f} {st['peak_torque_rpm']:6.0f} "
            f"{st['peak_power_kw']:6.1f} {st['peak_power_rpm']:6.0f} "
            f"{st['peak_power_hp']:6.1f} {bmep:6.2f} {kwl:6.1f}"
        )

    # --- every engine, not just the reference four ------------------------- #
    print()
    names = [row[0] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    check(
        "preset names are unique",
        not duplicates,
        f"duplicates: {', '.join(duplicates)}" if duplicates else f"{len(names)} presets",
    )
    for name, spec, idle, drift, st, finite in rows:
        if not finite or not st:
            continue
        bmep = bmep_bar(st["peak_torque"], spec.displacement, spec.cycle_revs)
        kwl = st["peak_power_kw"] / (spec.displacement * 1000.0)
        if spec.kind == "turbine":
            check(
                f"{name}: idle follows its setpoint",
                abs(idle - spec.idle_rpm) < max(0.25 * spec.idle_rpm, 60.0)
                and drift < 150.0,
                f"{idle:.0f} vs {spec.idle_rpm:.0f} rpm (drift {drift:.1f})",
            )
            check(
                f"{name}: shaft power in a broad turboshaft band",
                100.0 < st["peak_power_kw"] < 20_000.0,
                f"{st['peak_power_kw']:.0f} kW",
            )
            check(
                f"{name}: turbine power density is plausible",
                20.0 < kwl < 1500.0,
                f"{kwl:.1f} kW/L",
            )
            check(
                f"{name}: peak power below the redline",
                st["peak_power_rpm"] <= spec.redline_rpm * 1.05,
                f"{st['peak_power_rpm']:.0f} vs {spec.redline_rpm:.0f} rpm",
            )
            continue
        check(
            f"{name}: idle follows its setpoint",
            abs(idle - spec.idle_rpm) < max(0.25 * spec.idle_rpm, 60.0) and drift < 150.0,
            f"{idle:.0f} vs {spec.idle_rpm:.0f} rpm (drift {drift:.1f})",
        )
        check(
            f"{name}: BMEP in a broad piston-engine band",
            4.0 < bmep < 60.0,
            f"{bmep:.2f} bar",
        )
        check(
            f"{name}: specific power in a broad band",
            5.0 < kwl < 200.0,
            f"{kwl:.1f} kW/L",
        )
        check(
            f"{name}: peak power below the redline",
            st["peak_power_rpm"] <= spec.redline_rpm * 1.05,
            f"{st['peak_power_rpm']:.0f} vs {spec.redline_rpm:.0f} rpm",
        )
        if bmep > 30.0 or kwl > 150.0:
            print(
                f"  [note] {name}: extreme output ({bmep:.1f} bar, {kwl:.1f} kW/L); "
                "check that the preset is intentional"
            )

    # --- character checks -------------------------------------------------- #
    print()
    # These checks are scale free, but they are written around one plain
    # naturally aspirated petrol four. Pick that one explicitly instead of
    # whichever four cylinder happens to sort first: the roster also contains
    # boosted fours (and flat ones), whose numbers belong to a different class.
    gasoline = next((r for r in rows if r[0] == "I4 Gasoline 2.0 NA"), None)
    if gasoline is None:
        gasoline = next(
            (
                r
                for r in rows
                if r[1].kind == "petrol" and r[1].cylinders == 4 and r[1].boost_bar <= 0.0
            ),
            None,
        )
    if gasoline:
        name, spec, idle, drift, st, _ = gasoline
        bmep = bmep_bar(st["peak_torque"], spec.displacement, spec.cycle_revs)
        kwl = st["peak_power_kw"] / (spec.displacement * 1000.0)
        check(
            "idle holds near the setpoint",
            abs(idle - spec.idle_rpm) < 180.0 and drift < 90.0,
            f"{idle:.0f} rpm vs target {spec.idle_rpm:.0f} (drift {drift:.1f})",
        )
        check(
            "peak BMEP in the range of a real petrol engine",
            8.0 < bmep < 14.0,
            f"{bmep:.2f} bar",
        )
        check(
            "specific power in the range of a real petrol engine",
            40.0 < kwl < 90.0,
            f"{kwl:.1f} kW/L",
        )

        # --- rev limiter ---------------------------------------------------- #
        model = EngineModel(spec)
        start_engine(model)
        model.limiter_enabled = True
        peak = 0.0
        for _ in range(int(6.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
            peak = max(peak, model.rpm)
        check(
            "rev limiter holds the engine near the redline",
            spec.redline_rpm * 0.97 < peak < spec.redline_rpm * 1.09,
            f"peak {peak:.0f} rpm, redline {spec.redline_rpm:.0f}",
        )

        # --- dyno ramp ------------------------------------------------------ #
        # The brake is a speed command. If its torque is applied twice per step
        # the measured ramp runs at twice the configured rate.
        model = EngineModel(spec)
        start_engine(model)
        model.start_dyno_pull()
        ramp = model.dyno.ramp_rpm_s
        model.dyno.ramp_rpm_s = 0.0
        _run(model, 1.5, pedal=1.0)
        model.dyno.ramp_rpm_s = ramp
        start_rpm = model.rpm
        _run(model, 1.0, pedal=1.0)
        measured = model.rpm - start_rpm
        model.stop_dyno()
        check(
            "dyno follows its configured ramp rate",
            abs(measured - ramp) < 0.10 * ramp,
            f"{measured:.0f} rpm/s vs {ramp:.0f} configured",
        )

        # --- ignition ------------------------------------------------------- #
        model = EngineModel(spec)
        start_engine(model)
        model.ignition_on = False
        _run(model, 3.0, pedal=0.0)
        check(
            "switching the ignition off stops the engine",
            model.rpm < spec.idle_rpm * 0.6,
            f"{model.rpm:.0f} rpm after 3 s",
        )

        # --- engine braking ------------------------------------------------- #
        model = EngineModel(spec)
        start_engine(model)
        for _ in range(int(4.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
        rpm_start = model.rpm
        t = 0.0
        while model.rpm > spec.idle_rpm * 1.5 and t < 5.0:
            model.set_pedal(0.0)
            model.step(0.002)
            t += 0.002
        decel = (rpm_start - model.rpm) / max(t, 1e-6)
        check(
            "lifting off decelerates at a realistic rate",
            800.0 < decel < 12000.0,
            f"{decel:.0f} rpm/s from {rpm_start:.0f} rpm",
        )

        # --- manifold pressure ---------------------------------------------- #
        model = EngineModel(spec)
        start_engine(model)
        idle_map = model.map_bar
        for _ in range(int(5.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
        load_map = model.map_bar
        check(
            "idle vacuum and wide-open pressure are both realistic",
            0.15 < idle_map < 0.75 and 0.80 < load_map < 1.06,
            f"idle {idle_map:.2f} bar, WOT {load_map:.2f} bar",
        )
        check("naturally aspirated engine makes no boost", model.boost_bar < 0.02, f"{model.boost_bar:.3f} bar")

    # --- forced induction --------------------------------------------------- #
    boosted = next((r for r in rows if r[1].boost_bar > 0.0 and r[1].kind == "petrol"), None)
    if boosted:
        name, spec, *_ = boosted
        model = EngineModel(spec)
        start_engine(model)
        for _ in range(int(6.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
        check(
            f"{name}: boost reaches the wastegate limit under load",
            model.boost_bar > 0.75 * spec.boost_bar,
            f"{model.boost_bar:.2f} bar of {spec.boost_bar:.2f}",
        )
        model.drain_events()
        for _ in range(int(0.4 / 0.002)):
            model.set_pedal(0.0)
            model.step(0.002)
        events = model.drain_events()
        check(
            f"{name}: lifting off dumps the boost",
            "blow_off" in events,
            f"events: {', '.join(events) or 'none'}",
        )
        # ... and an unloaded engine at idle must not be on boost.
        idle = EngineModel(spec)
        start_engine(idle)
        for _ in range(int(4.0 / 0.002)):
            idle.set_pedal(0.0)
            idle.step(0.002)
        check(
            f"{name}: no boost off the throttle",
            idle.boost_bar < 0.15 * spec.boost_bar,
            f"{idle.boost_bar:.2f} bar at {idle.rpm:.0f} rpm",
        )

    # --- backfire logic ---------------------------------------------------- #
    petrol = next((r for r in rows if r[1].kind == "petrol"), None)
    if petrol:
        spec = petrol[1]
        fired = 0
        for attempt in range(6):
            model = EngineModel(spec, seed=1000 + attempt)
            start_engine(model)
            for _ in range(int(5.0 / 0.002)):
                model.set_pedal(1.0)
                model.step(0.002)
            model.drain_events()
            for _ in range(int(2.0 / 0.002)):
                model.set_pedal(0.0)
                model.step(0.002)
            fired += model.drain_events().count("backfire")
        check(
            "lifting off at high rpm can backfire, but not every time",
            1 <= fired <= 6,
            f"{fired} backfires over 6 lift-offs",
        )

    # --- detonation -------------------------------------------------------- #
    # A knock demo needs an engine whose end gas is actually marginal. The
    # generic search below used to land on whichever boosted engine came
    # first -- a modern premium-fuel engine with a good intercooler is
    # *correctly* knock-free, and the check failed for the right physics.
    # The dedicated research mule (9.8:1, 1.4 bar, 87 octane, weak cooler)
    # is the honest demo; fall back to any marginal engine if it is absent.
    knocker = next(
        (r for r in rows if r[0].startswith("Knock Mule")),
        None,
    )
    if knocker is None:
        knocker = next(
            (
                r
                for r in rows
                if r[1].kind == "petrol"
                and r[1].boost_bar > 0.0
                and r[1].compression >= 10.0
                and r[1].octane <= 93.0
            ),
            None,
        )
    if knocker:
        name, spec, *_ = knocker
        model = EngineModel(spec)
        start_engine(model)
        model.start_dyno_pull()
        model.dyno.ramp_rpm_s = 0.0
        model.dyno.target_rpm = 2500.0
        model.dyno.end_rpm = max(spec.redline_rpm, 2500.0)
        peak_knock = 0.0
        peak_retard = 0.0
        for _ in range(int(4.0 / 0.002)):
            model.set_pedal(1.0)
            model.step(0.002)
            peak_knock = max(peak_knock, model.knock_level)
            peak_retard = max(peak_retard, model.knock_retard)
        check(
            f"{name}: low-speed high-load knock appears",
            peak_knock > 0.20,
            f"level {peak_knock:.2f}",
        )
        check(
            f"{name}: knock pulls timing",
            peak_retard > 0.10,
            f"retard {peak_retard * 100:.0f} %",
        )
        model.stop_dyno()
        _run(model, 4.0, pedal=0.0)
        check(
            f"{name}: timing recovers after the knock stops",
            model.knock_retard < 0.05,
            f"retard {model.knock_retard * 100:.0f} % after 4 s",
        )

    # --- live loop mixing --------------------------------------------------- #
    loop_row = next(
        (row for row in rows if row[1].kind == "petrol" and row[1].boost_bar > 0.0),
        None,
    )
    if loop_row:
        from .audio.driver import EngineAudio

        loop_path = next(p for p in paths if load_preset(p).name == loop_row[0])
        loop_audio = EngineAudio(load_preset(loop_path), enabled=False)
        loop_audio._render_shots({"turbo_spool": 1, "knock": 1})
        check(
            "turbo spool and knock are mixed as loops, not one-shots",
            "turbo_spool" in loop_audio._loops
            and "knock" in loop_audio._loops
            and "turbo_spool" not in loop_audio._shots
            and "knock" not in loop_audio._shots,
            f"loops: {', '.join(sorted(loop_audio._loops))}",
        )
        buffer = np.zeros(512, dtype=np.float64)
        loop_audio._mix_loops(buffer, 1.0, 1.0)
        peak = float(np.max(np.abs(buffer)))
        check("loop mixer contributes signal", peak > 1e-4, f"peak {peak:.4f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0
