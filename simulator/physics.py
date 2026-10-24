"""Real-time engine and drivetrain model.

The model is *mean value*: it tracks crank speed, manifold pressure and crank
angle rather than individual cylinder pressures. That is enough for a
convincing dyno, a convincing sound and a convincing 3D animation, and it runs
thousands of times faster than real time.

Signal flow of a single integration step::

    pedal -> throttle area -> orifice flow -> manifold pressure
                  ^                                |
                  |                                v
           engine pumping                    fuel + air -> cycle work
                                                   |
                                                   v
                                            indicated torque
                                                   |
                     friction + pumping + load/brake + starter
                                                   |
                                                   v
                                          crank torque balance

Nothing here knows about sound, 3D or the UI, so it can be tested headless.

Two choices matter for realism:

* **The torque curve is not authored.** It falls out of volumetric efficiency,
  friction, pumping loss and the pressure dropped across the throttle plate, so
  it responds correctly when the engine is resized.
* **Friction rises with load as well as speed.** The rings are pushed against
  the bore by cylinder pressure and the bearings take the firing load, so a hard
  working engine rubs harder than the same engine at light load. Without that
  term the model overstates part-load efficiency and gives a boosted engine
  something for nothing.
* **The throttle plate leaks.** A truly sealed plate would give a wrong idle and
  absurd intake vacuum on the overrun, so a small idle/bypass area is always
  open. It is deliberately tiny: the idle air has to come from the governor, or
  a big engine -- which moves far more air per revolution than a small one for
  the same plate gap -- over-speeds at idle and the governor has no authority to
  bring it back down.

Forced induction follows the same rule. A compressor raises the pressure
*upstream* of the throttle plate, the plate still has to pass the extra air, and
the charge arrives hotter than ambient by however much the intercooler failed to
remove. A turbo also has to spin up from exhaust energy, so it lags, makes
nothing below its spool speed and lets go slowly; a supercharger is geared to the
crank and answers immediately.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .bootstrap import preset_dir  # noqa: F401  (ensures sys.path is set up)

from soundgen import layout  # noqa: E402
from soundgen.config import Preset, Turbine  # noqa: E402

# --- air and fuel ---------------------------------------------------------- #
R_AIR = 287.05  # J/(kg*K)
GAMMA = 1.4
CP_AIR = 1005.0  # J/(kg*K), compressor-side specific heat
CP_GAS = 1050.0  # J/(kg*K), hot-section specific heat (mean value)
P_AMB = 101_325.0  # Pa
T_AMB = 293.15  # K
CHOKE_PR = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))  # ~0.528, sonic
BAR = 1.0e5
TWO_PI = 2.0 * math.pi

LHV_PETROL = 44.0e6  # J/kg
LHV_DIESEL = 42.7e6
LHV_KEROSENE = 42.8e6  # J/kg, Jet A-1 / diesel-like turbine fuel
AFR_PETROL = 14.7
DIESEL_LAMBDA_FULL = 1.45  # full-load lambda of a diesel: always lean

# Fraction of the maximum throttle area that is open even with the plate shut
# (plate leakage plus a closed idle bypass). See the module docstring. This is
# deliberately small: the idle air has to come from the governor, or a big
# engine -- which moves far more air per revolution than a small one for the
# same plate gap -- idles high and the governor has no authority to bring it
# back down. Real ECUs size the bypass the same way.
IDLE_LEAK_FRACTION = 0.0025

# The manifold cannot be pumped arbitrarily empty: during valve overlap the
# exhaust pushes back in, so intake vacuum saturates. Without this floor a
# closed throttle at speed would pull the manifold towards zero, which would
# give absurd engine braking and a nonsense gauge reading.
MAP_FLOOR_FRACTION = 0.12

# --- forced induction ------------------------------------------------------ #
COMPRESSOR_EFF = 0.70  # adiabatic efficiency of a turbo/supercharger compressor
INTERCOOLER_LOSS = 0.95  # pressure kept after the intercooler and plumbing

# Volumetric efficiency falls as the manifold is throttled: more residual gas
# is left in the cylinder, so less fresh charge fits.
def _charge_fraction(map_pa: float) -> float:
    pr = max(map_pa, 0.0) / P_AMB
    return 0.42 + 0.58 * math.sqrt(min(pr, 1.0))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _charge_state(boost: float, cooling: float) -> tuple[float, float]:
    """Manifold (pressure ratio, charge temperature) after compressor and cooler.

    Rising from ambient to ``1 + boost`` atmospheres heats the charge by
    ``(pr^((g-1)/g) - 1) / eta_c`` at a compressor efficiency ``eta_c``; an
    intercooler then removes ``cooling`` of that rise, and the plumbing loses a
    few percent of the pressure on the way to the throttle. Without modelling
    the heating a boosted engine is treated as breathing hot, thin air and makes
    far less power than its boost implies; without the loss it makes too much.
    """
    if boost <= 0.0:
        return 1.0, T_AMB
    pr = (1.0 + boost) * INTERCOOLER_LOSS
    t_comp = T_AMB * (1.0 + (pr ** ((GAMMA - 1.0) / GAMMA) - 1.0) / COMPRESSOR_EFF)
    return pr, t_comp - cooling * (t_comp - T_AMB)


def _charge_density_ratio(boost: float, cooling: float) -> float:
    """Manifold charge density relative to ambient (absolute, not gauge)."""
    pr, t_man = _charge_state(boost, cooling)
    return pr * T_AMB / t_man


@dataclass
class EngineSpec:
    """Physical description of the engine plus model coefficients."""

    name: str = "engine"
    kind: str = "petrol"  # petrol | diesel | turbine
    layout: str = "inline"  # inline | v | boxer | radial | rotary | turbine
    cylinders: int = 4
    cycle_revs: float = 2.0
    displacement: float = 2.0e-3  # m^3, total swept volume
    compression: float = 10.5
    octane: float = 95.0  # petrol knock resistance (RON)
    turbine: Turbine = field(default_factory=Turbine)
    idle_rpm: float = 850.0
    redline_rpm: float = 7000.0
    starter_rpm: float = 260.0
    rod_ratio: float = 1.6
    stroke: float = 0.087  # m
    bore: float = 0.087  # m
    bank_angle: float = 0.0
    firing_offsets: list[float] = field(default_factory=list)

    ve_peak: float = 0.95
    ve_peak_rpm_frac: float = 0.55
    ve_droop: float = 0.30  # fraction of VE lost between the peak and the redline
    afr: float = AFR_PETROL
    lhv: float = LHV_PETROL
    eta_i: float = 0.40
    inertia: float = 0.30  # kg*m^2, engine plus flywheel
    throttle_area: float = 7.0e-4  # m^2, effective (discharge coefficient folded in)
    exhaust_back_pressure: float = 1.03  # multiple of ambient
    manifold_volume: float = 1.2e-3  # m^3

    # --- forced induction --------------------------------------------------- #
    # The compressor sits *upstream* of the throttle plate, so boost raises the
    # pressure the plate sees rather than the manifold directly. ``boost_bar`` is
    # the wastegate-limited gauge boost at full load; 0 means naturally aspirated,
    # in which case none of this is evaluated.
    boost_bar: float = 0.0
    spool_rpm: float = 0.0  # below this the turbine makes nothing
    spool_span_rpm: float = 1.0
    supercharged: bool = False  # geared to the crank, so it spools instantly
    charge_cooling: float = 0.70  # intercooler effectiveness, 0..1

    @property
    def cylinder_displacement(self) -> float:
        return self.displacement / max(self.cylinders, 1)

    @property
    def cycles_per_second(self) -> float:
        """Engine cycles per second at 1 rpm."""
        return 1.0 / (60.0 * self.cycle_revs)

    @property
    def crank_radius(self) -> float:
        return self.stroke * 0.5

    @property
    def rod_length(self) -> float:
        return self.rod_ratio * self.crank_radius

    @property
    def ve_peak_rpm(self) -> float:
        return self.ve_peak_rpm_frac * self.redline_rpm

    @classmethod
    def from_preset(cls, preset: Preset) -> EngineSpec:
        block = preset.block
        disp = layout.displacement_cc(block) * 1e-6  # cc -> m^3
        kind = "diesel" if block.ignition == "compression" else "petrol"
        if block.layout == "turbine":
            kind = "turbine"

        spec = cls(
            name=preset.name,
            kind=kind,
            layout=str(block.layout),
            cylinders=max(int(block.cylinders), 1),
            cycle_revs=layout.cycle_revolutions(block),
            displacement=disp,
            compression=float(block.compression),
            octane=float(block.octane),
            turbine=preset.turbine,
            idle_rpm=float(preset.engine.idle_rpm),
            redline_rpm=float(preset.engine.redline_rpm),
            starter_rpm=float(preset.engine.starter_rpm),
            rod_ratio=float(block.rod_ratio),
            stroke=float(block.stroke_mm) * 1e-3,
            bore=float(block.bore_mm) * 1e-3,
            bank_angle=math.radians(float(block.bank_angle_deg)),
            firing_offsets=layout.firing_offsets(block),
        )

        # A free-breathing engine peaks its volumetric efficiency high in the
        # rev range; a lazy one peaks low. A preset may describe its own curve
        # instead: a race-derived engine holds VE almost to the limiter, and
        # without that it can never make power at the rpm it is famous for.
        spec.ve_peak_rpm_frac = 0.75 if kind == "turbine" else 0.55
        if block.ve_peak_rpm_frac > 0.0:
            spec.ve_peak_rpm_frac = float(block.ve_peak_rpm_frac)
        if block.ve_droop > 0.0:
            spec.ve_droop = min(float(block.ve_droop), 0.60)
        spec.ve_peak = 0.93 + min(0.07, 0.04 * math.log10(max(disp, 1e-9) / 5e-4) + 0.03)
        if block.ve_peak_value > 0.0:
            spec.ve_peak = float(block.ve_peak_value)

        if kind == "diesel":
            spec.lhv = LHV_DIESEL
            spec.eta_i = min(0.50, 0.45 * (1.0 + 0.010 * (spec.compression - 10.0)))
            spec.exhaust_back_pressure = 1.06
        else:
            spec.lhv = LHV_PETROL
            spec.eta_i = min(0.46, 0.40 * (1.0 + 0.012 * (spec.compression - 10.0)))

        # Rotating inertia: a bigger engine carries a heavier crank and flywheel,
        # which is why a large engine hangs on the throttle and a small one
        # snaps. Calibrated so a 2.0 L four free-revs at roughly 6000 rpm/s.
        litres = disp * 1000.0
        spec.inertia = (0.05 + 0.02 * litres) if kind == "turbine" else (0.12 + 0.09 * litres)

        spec.manifold_volume = max(0.35 * disp, 6.0e-4)

        # Forced induction comes straight from the sound preset's turbo section,
        # so the dyno and the audio describe the same hardware.
        t = preset.turbo
        if t.enabled and kind != "turbine":
            spec.boost_bar = max(0.0, float(t.max_boost_bar))
            spec.spool_rpm = float(t.spool_rpm)
            spec.spool_span_rpm = max(float(t.spool_span_rpm), 1.0)
            spec.supercharged = str(t.kind) == "supercharger"
            # The intercooler is a physical part of the induction system and a
            # strong lever on the knock model; the mule engine (87-octane pump
            # gas with a token cooler) needs a worse one than the 0.70 default.
            spec.charge_cooling = 0.70
            extra = getattr(t, "charge_cooling", None)
            if extra is not None:
                spec.charge_cooling = min(max(float(extra), 0.0), 0.95)

        if kind == "diesel":
            # no throttle plate at all: only the air cleaner and inlet restrict
            # the manifold, so it sits within a few percent of ambient
            return _size_throttle(spec, target_pr=0.97)
        return _size_throttle(spec)


# --------------------------------------------------------------------------- #
# gas flow
# --------------------------------------------------------------------------- #
def _chebyshev_flux(pr: float) -> float:
    """Compressible mass-flow function of pressure ratio (0 at pr >= 1)."""
    if pr >= 1.0:
        return 0.0
    if pr < CHOKE_PR:
        pr = CHOKE_PR  # choked: flow no longer depends on downstream pressure
    return math.sqrt(max(0.0, pr ** (2.0 / GAMMA) - pr ** ((GAMMA + 1.0) / GAMMA)))


def _orifice_mdot(area: float, p_up: float, t_up: float, p_down: float) -> float:
    """Steady isentropic flow through an orifice, kg/s."""
    if area <= 0.0 or p_up <= 0.0:
        return 0.0
    flux = _chebyshev_flux(max(p_down, 0.0) / p_up)
    if flux <= 0.0:
        return 0.0
    return area * p_up / math.sqrt(R_AIR * t_up) * math.sqrt(2.0 * GAMMA / (GAMMA - 1.0)) * flux


def _ve_open(spec: EngineSpec, rpm: float) -> float:
    """Wide-open-throttle volumetric efficiency at a given speed.

    The low-speed branch is deliberately steep: a real engine cannot fill its
    cylinders at 1200 rpm the way it can at 4000, and without that the model
    would show a torque curve that is far too flat down low.
    """
    peak = max(spec.ve_peak_rpm, 1.0)
    if rpm <= peak:
        shape = 0.55 + 0.45 * (max(rpm, 0.0) / peak) ** 1.2
    else:
        frac = (rpm - peak) / max(spec.redline_rpm - peak, 1.0)
        shape = 1.0 - spec.ve_droop * frac * frac
    return spec.ve_peak * max(0.40, shape)


def _ve(spec: EngineSpec, rpm: float, map_pa: float) -> float:
    """Volumetric efficiency at the current speed *and* manifold pressure."""
    return _ve_open(spec, rpm) * _charge_fraction(map_pa)


def _fmep_bar(spec: EngineSpec, rpm: float, load: float = 0.0) -> float:
    """Friction mean effective pressure, Chen-Flynn form (bar).

    ``FMEP = a + c*N + d*N^2`` is the usual Willans line, but it has no load
    term, and real friction is strongly load dependent: the rings are pushed
    against the bore by cylinder pressure, and the bearings take the firing
    load. That is the ``b*P_max`` term. Without it a model overstates part-load
    efficiency and understates the extra drag of a working engine -- a boosted
    engine at full load rubs hard enough that it is worth a few percent of the
    torque it just made.
    """
    n = max(rpm, 0.0) / 1000.0
    base = 0.70 + 0.11 * n + 0.013 * n * n
    base += 0.60 * _clamp(load, 0.0, 1.25)
    if spec.kind == "diesel":
        base *= 1.15  # heavier rotating assembly, higher cylinder pressure
    return base


def _fmep_torque(spec: EngineSpec, rpm: float, load: float = 0.0) -> float:
    return _fmep_bar(spec, rpm, load) * BAR * spec.displacement / (TWO_PI * spec.cycle_revs)


def _size_throttle(spec: EngineSpec, target_pr: float = 0.92) -> EngineSpec:
    """Size the inlet so the manifold droops by a set fraction at the redline.

    That droop is what makes a naturally aspirated torque curve fall away at the
    top end, so it should come from geometry, not from a lookup table.

    Sizing also has to keep the manifold filling equation well behaved: an
    absurdly large inlet makes the pressure step explode and the manifold just
    rattles against its clamp, which is how the diesel inlet used to behave.

    A boosted engine is sized against the *compressor outlet*, not against
    ambient: the throttle body and airbox have to pass the whole boosted mass
    flow or the compressor ends up throttled by its own inlet.
    """
    boost = max(spec.boost_bar, 0.0)
    pr, t_man = _charge_state(boost, spec.charge_cooling)
    p_in = P_AMB * pr
    ve = _ve(spec, spec.redline_rpm, target_pr * p_in)
    rho_man = (target_pr * p_in) / (R_AIR * t_man)
    mdot = ve * spec.displacement * rho_man * spec.redline_rpm * spec.cycles_per_second
    flux = _chebyshev_flux(target_pr)
    if flux <= 0.0 or mdot <= 0.0:
        return spec
    denominator = p_in / math.sqrt(R_AIR * t_man) * math.sqrt(2.0 * GAMMA / (GAMMA - 1.0)) * flux
    spec.throttle_area = mdot / denominator
    return spec


# --------------------------------------------------------------------------- #
# dynamometer
# --------------------------------------------------------------------------- #
@dataclass
class DynoConfig:
    """Brake settings for a dynamometer pull."""

    active: bool = False
    target_rpm: float = 0.0
    ramp_rpm_s: float = 550.0
    end_rpm: float = 0.0


def summarise_curve(points: list[tuple[float, float]]) -> dict:
    """Peak torque and power from a list of ``(rpm, brake torque Nm)`` points.

    Power is ``rpm * torque * 2*pi/60`` in watts; it is computed from the torque
    and speed rather than read off the model, so it is exactly what a brake would
    measure on the same curve.
    """
    if not points:
        return {}

    def power_w(point: tuple[float, float]) -> float:
        return point[0] * point[1] * TWO_PI / 60.0

    peak_t_rpm, peak_t = max(points, key=lambda p: p[1])
    peak_p_rpm, peak_p = max(points, key=power_w)
    return {
        "peak_torque": peak_t,
        "peak_torque_rpm": peak_t_rpm,
        "peak_power_kw": power_w((peak_p_rpm, peak_p)) / 1000.0,
        "peak_power_rpm": peak_p_rpm,
        "peak_power_hp": power_w((peak_p_rpm, peak_p)) / 745.7,
        "max_rpm": max(p[0] for p in points),
    }


class EngineModel:
    """Integrates the engine forward in time.

    Call :meth:`step` with the wall-clock delta; the model substeps internally
    at a fixed 0.25 ms so behaviour does not depend on the frame rate.
    """

    FIXED_DT = 0.00025  # s
    # idle governor: proportional gain (throttle per rpm) and integral rate
    GOV_KP = 0.00045
    GOV_KI = 0.00028
    BACKFIRE_MIN_INTERVAL = 0.15  # s, between overrun bangs
    BACKFIRE_LIMITER_INTERVAL = 0.60  # s, the limiter must not machine-gun
    BLOW_OFF_MIN_INTERVAL = 0.50  # s, between the pshh of successive lift-offs
    KNOCK_ONSET = 0.55  # Livengood-Wu index at which the end gas ignites
    KNOCK_TORQUE_LOSS = 0.30  # fraction of indicated torque lost at full retard

    def __init__(self, spec: EngineSpec, seed: int = 12345):
        self.spec = spec
        self.rng = random.Random(seed)

        self.rpm = 0.0
        self.theta = 0.0  # crank angle, rad, wrapped to one full cycle
        # Unwrapped count of firing cycles, the clock the audio locks to. Kept
        # separately from ``theta`` because ``theta`` wraps and the synthesiser
        # needs to know *how many* cycles passed between two calls, not just
        # where in a cycle the model happens to be.
        self._firing_cycles = 0.0
        self.map_pa = P_AMB
        self.map_temp_k = T_AMB  # manifold charge temperature
        self.boost = 0.0  # 0..1, compressor speed as a fraction of full spool
        self.pedal = 0.0
        self.throttle = 0.0
        self.cranking = False
        self.ignition_on = True
        self.limiter_enabled = True
        self.limiter_cut = False
        self.overrun_cut = False
        self.fuel_cut = False
        self.knock_level = 0.0  # 0..1, current detonation intensity
        self.knock_retard = 0.0  # 0..1, timing pulled by the knock sensor
        self.brake_torque = 0.0
        self.dyno = DynoConfig()
        self.dyno_integral = 0.0
        self._manual_load = 0.0
        self._gov_integral = 0.0

        self.indicated_torque = 0.0
        self.torque_envelope = 0.0
        self._wiebe_mean = 0.0
        self._display_mean = 0.0
        self._wiebe_prev_theta = 0.0
        self._wiebe_dtheta = 0.0
        self._wiebe_norm_cache = None
        # Exhaust thermal state (audit item 3): the pipe's speed of sound --
        # and with it every quarter-wave resonance -- rises as the collector
        # heats up under load and falls on the overrun. 0 = stone cold,
        # 1 = fully hot. Feeds `pipe_temp` for the synthesiser.
        self.pipe_heat = 0.0
        self.friction_torque = 0.0
        self.pumping_torque = 0.0
        self.net_torque = 0.0
        self.governor = 0.0
        self.events: list[str] = []

        self._prev_overrun_cut = False
        self._prev_limiter_cut = False
        self._backfire_cooldown = 0.0
        self._limiter_cooldown = 0.0
        self._blow_off_cooldown = 0.0
        self._prev_pedal = 0.0
        self._load_smoothed = 0.0
        self._load_peak = 0.0

        # Gas-turbine state.  The flags are separate from ``running`` because a
        # turbine can be above cranking speed while still accelerating to idle;
        # once the starter has lit the combustor it must stay alive until the
        # ignition is switched off.
        self._turbine_started = False
        self.turbine_power_kw = 0.0
        self.turbine_fuel_kg_s = 0.0
        self.turbine_pressure_ratio = 1.0
        self.turbine_inlet_k = T_AMB

        # Full-load indicated torque never changes for a given spec, so work it
        # out once: it is both the friction load reference and the denominator
        # the sound load is measured against.
        self._peak_indicated = self._peak_torque_estimate()

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def set_pedal(self, value: float) -> None:
        self.pedal = min(1.0, max(0.0, float(value)))

    @property
    def load(self) -> float:
        """Manual brake setting, 0..1."""
        return self._manual_load

    @load.setter
    def load(self, value: float) -> None:
        self._manual_load = min(1.0, max(0.0, float(value)))

    def start_dyno_pull(self, end_rpm: float | None = None) -> None:
        self.dyno.active = True
        self.dyno.target_rpm = self.spec.idle_rpm + 400.0
        self.dyno.end_rpm = float(end_rpm if end_rpm is not None else self.spec.redline_rpm)
        if self.spec.kind == "turbine":
            # The turbine lives at 12 000-42 000 rpm, so the piston-engine
            # 550 rpm/s sweep would take almost a minute.  A 20 s pull keeps the
            # same shape while staying inside the headless test timeout.
            self.dyno.ramp_rpm_s = max(
                self.dyno.ramp_rpm_s,
                (self.dyno.end_rpm - self.dyno.target_rpm) / 20.0,
            )
        self.dyno_integral = 0.0
        self.pedal = 1.0

    def stop_dyno(self) -> None:
        self.dyno.active = False
        self.dyno_integral = 0.0

    # ------------------------------------------------------------------ #
    # integration
    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        dt = min(max(dt, 0.0), 0.10)
        remaining = dt
        while remaining > 1e-9:
            h = min(self.FIXED_DT, remaining)
            self._step_once(h)
            remaining -= h

    def _step_once(self, dt: float) -> None:
        if self.spec.kind == "turbine":
            self._step_turbine(dt)
            return

        spec = self.spec
        self._backfire_cooldown = max(0.0, self._backfire_cooldown - dt)
        self._limiter_cooldown = max(0.0, self._limiter_cooldown - dt)
        self._blow_off_cooldown = max(0.0, self._blow_off_cooldown - dt)

        # --- rev limiter: cut fuel above the redline, with hysteresis ------ #
        if self.limiter_enabled:
            if self.limiter_cut:
                if self.rpm < spec.redline_rpm - 150.0:
                    self.limiter_cut = False
            elif self.rpm > spec.redline_rpm:
                self.limiter_cut = True
        else:
            self.limiter_cut = False

        # --- idle governor ------------------------------------------------- #
        error = spec.idle_rpm - self.rpm
        self.governor = 0.0
        if self.running and self.pedal < 0.02 and not self.dyno.active:
            self._gov_integral = min(0.14, max(0.0, self._gov_integral + self.GOV_KI * error * dt))
            self.governor = min(0.16, max(0.0, self.GOV_KP * error + self._gov_integral))

        # --- throttle plate ------------------------------------------------ #
        if self.dyno.active:
            self.pedal = 1.0
        demand = max(self.pedal, self.governor)
        if spec.kind == "diesel":
            # the plate only acts as a smoke limiter; the inlet stays open
            self.throttle = 0.35 + 0.65 * demand
        else:
            self.throttle = demand

        # --- forced induction ---------------------------------------------- #
        # The compressor raises the pressure *upstream* of the throttle plate; the
        # plate still has to pass the extra air, which is why boost cannot simply
        # be written into the manifold. A turbocharger takes its energy from the
        # exhaust, so it lags and it makes nothing below its spool speed; a
        # supercharger is geared to the crank and responds immediately.
        rel_redline = _clamp(self.rpm / max(spec.redline_rpm, 1.0), 0.0, 1.2)
        boost_pa = 0.0
        if spec.boost_bar > 0.0:
            if spec.supercharged:
                # Geared to the crank, so the capability arrives with speed.
                cap = _clamp(0.28 + 0.72 * rel_redline, 0.0, 1.0)
            else:
                # Below the threshold the exhaust does not carry enough enthalpy
                # to spin the turbine. Above it the *wastegate* sets the boost,
                # not the speed, so the capability has to saturate (a smoothstep)
                # rather than keep climbing with rpm until it runs out of span.
                span = max(spec.spool_span_rpm * 0.60, 1.0)
                x = _clamp((self.rpm - spec.spool_rpm) / span, 0.0, 1.0)
                cap = x * x * (3.0 - 2.0 * x)
            want = cap * demand
            # What the wastegate is doing, not what the driver asked for. The
            # rotor has to be accelerated by exhaust energy, so the turbo lights
            # in a fraction of a second when the engine is working and stays
            # lazy at light load; blowing off is the wastegate opening and is
            # always quicker than spooling.
            tau = (0.16 + 0.34 * (1.0 - demand)) if want > self.boost else 0.26
            self.boost += (want - self.boost) * (1.0 - math.exp(-dt / tau))
            boost_pa = spec.boost_bar * BAR * _clamp(self.boost, 0.0, 1.0)

            # A lifted throttle with a spinning compressor blows off. Detect the
            # pedal *release*, not the boost falling, so the pshh lands at the
            # moment the driver lets go rather than a fraction of a second later.
            if (
                not spec.supercharged
                and self.ignition_on
                and self._blow_off_cooldown <= 0.0
                and self.pedal < 0.05
                and self._prev_pedal >= 0.05
                and self.boost > 0.25
            ):
                self.events.append("blow_off")
                self._blow_off_cooldown = self.BLOW_OFF_MIN_INTERVAL
            self._prev_pedal = self.pedal
        # Compressing the charge heats it. An intercooler takes most of that back
        # out again, which is the difference between a boosted engine that makes
        # power and one that only makes detonation.
        pr_charge, self.map_temp_k = _charge_state(boost_pa / BAR, spec.charge_cooling)
        p_inlet = P_AMB * pr_charge

        # --- manifold filling ---------------------------------------------- #
        fr = IDLE_LEAK_FRACTION
        area = spec.throttle_area * (fr + (1.0 - fr) * self.throttle ** 1.8)
        mdot_in = _orifice_mdot(area, p_inlet, self.map_temp_k, self.map_pa)
        ve = _ve(spec, self.rpm, self.map_pa)
        rho_man = self.map_pa / (R_AIR * self.map_temp_k)
        cycles_s = max(self.rpm, 0.0) * spec.cycles_per_second
        mdot_out = ve * spec.displacement * rho_man * cycles_s
        self.map_pa += (R_AIR * self.map_temp_k / spec.manifold_volume) * (mdot_in - mdot_out) * dt
        self.map_pa = min(
            max(self.map_pa, MAP_FLOOR_FRACTION * P_AMB), max(1.05 * P_AMB, 0.98 * p_inlet)
        )

        # --- fuelling ------------------------------------------------------- #
        rpm = self.rpm
        self._wiebe_dtheta = rpm * TWO_PI / 60.0 * dt
        # deceleration fuel cut: off the pedal at speed the injectors stop
        self.overrun_cut = self.pedal < 0.02 and rpm > spec.idle_rpm * 1.8 and not self.dyno.active
        self.fuel_cut = self.limiter_cut or self.overrun_cut
        can_burn = rpm > spec.starter_rpm * 0.55
        have_fuel = (
            can_burn
            and self.ignition_on
            and not self.fuel_cut
            and (self.cranking or self.running)
        )

        indicated = 0.0
        self._update_knock(dt, demand)
        if have_fuel:
            rho = self.map_pa / (R_AIR * self.map_temp_k)
            air_per_cycle = ve * spec.cylinder_displacement * rho
            if spec.kind == "diesel":
                # fuel led: the pedal commands fuel, air is always in excess.
                # Pick the fuel mass so that at sea-level air density the diesel
                # cannot make more torque than the same-size petrol engine.
                fuel = spec.cylinder_displacement * rho * demand / (AFR_PETROL * DIESEL_LAMBDA_FULL)
            else:
                fuel = air_per_cycle / spec.afr

            # --- per-cylinder burn (stage 7: Wiebe) ------------------------- #
            # The mean-value model applied one "average combustion" per step.
            # Now every cylinder carries its own burn window: a Wiebe function
            # xb = 1 - exp(-a * x^(m+1)) over the burn angle, started at that
            # cylinder's own firing TDC. The step integrates d xb/dtheta, so a
            # cylinder contributes torque only while it is actually burning --
            # which is what makes the output at low rpm pulse at the firing
            # frequency instead of being a constant pull.
            #
            # The angle-integral of the multiplier over a cycle is exactly 1,
            # so the dyno curve and every torque figure are unchanged; what is
            # new is the *shape within the cycle*. The scatter term keeps the
            # cycle-to-cycle variation the old model had.
            burn = self._step_cylinders(dt, omega_hz=rpm * TWO_PI / 60.0)
            scatter = 0.02 + 0.05 * max(0.0, 1.0 - rpm / 2500.0)
            indicated = (
                spec.eta_i
                * fuel
                * spec.lhv
                / (TWO_PI * spec.cycle_revs)
                * burn
                * (1.0 + self.rng.uniform(-scatter, scatter))
            )

            # The knock sensor pulls timing before the ping becomes destructive.
            # That costs torque, which is exactly why a boosted engine on a hot
            # day feels weaker than the same engine on a cold one.
            indicated *= 1.0 - self.KNOCK_TORQUE_LOSS * self.knock_retard
        else:
            # No fuel this substep: the smoother must not keep delivering the
            # previous (burning) cycle's energy, or a cut engine would coast
            # on stale work. Zero in -> decays to zero within one window.
            self._wiebe_mean = 0.0

        # Sampled torque is the *cycle mean*: a single sample of the Wiebe pulse
        # train lands on a firing peak or a gap between firings, so every
        # mean-value consumer (the reported figure, the dyno plot, the
        # friction load term) must read the average. The raw pulse train stays
        # on ``torque_envelope`` for per-cylinder instrumentation.
        self.torque_envelope = indicated
        mean_indicated = self._cycle_mean_indicated(indicated)
        self.indicated_torque = mean_indicated
        # Reported figure is for the whole engine. Friction is evaluated at the
        # load the engine is actually under, not at the load it could be: the
        # rings and bearings know the cylinder pressure, not the pedal.
        load_frac = _clamp(
            mean_indicated * spec.cylinders / max(self._peak_indicated, 1e-6), 0.0, 1.25
        )
        self.friction_torque = _fmep_torque(spec, rpm, load_frac)
        # A turbine in the exhaust is a restriction, so boost raises the back
        # pressure roughly in step with the manifold and eats part of the gain.
        p_exh = P_AMB * (spec.exhaust_back_pressure + 0.9 * boost_pa / P_AMB)
        self.pumping_torque = max(0.0, p_exh - self.map_pa) * spec.displacement / (TWO_PI * spec.cycle_revs)

        starter = 0.0
        if self.cranking:
            fade = max(0.0, 1.0 - rpm / (spec.starter_rpm * 3.0))
            starter = self._starter_torque() * fade

        # --- crank balance --------------------------------------------------- #
        omega = rpm * TWO_PI / 60.0
        # The crank integrates the *cycle-mean* torque. Integrating the raw
        # Wiebe envelope instead would be physically defensible (its angle
        # integral is the same) but breaks two contracts in practice:
        #
        # * the idle governor is a feedback loop around indicated torque; a
        #   one-cycle-lagged mean was what stalled the 13.8 L three-cylinder
        #   diesel through its running threshold, but the *pulsed* torque has
        #   the opposite failure -- between firings the net goes negative and
        #   the speed ratchets through the limiter's hysteresis. The mean is
        #   what the original model's governor, limiter and stall boundary
        #   were tuned against.
        # * the speed-holding dyno is a torque *sampler*: with pulses, its
        #   integral ratchets on the between-firing error and reads BMEP 50 %
        #   high (measured on the same diesel).
        #
        # The pulse train itself is not lost -- ``torque_envelope`` keeps it
        # for the audio layer and any per-cylinder instrumentation.
        total_indicated = mean_indicated * spec.cylinders
        engine_torque = total_indicated - self.friction_torque - self.pumping_torque + starter
        brake = self._brake_torque(dt)
        # A speed-holding dyno reports the torque the engine delivers to it. While
        # the target speed is ramping, part of that torque goes into accelerating
        # the rotating mass rather than into the brake, so the brake alone reads
        # low by I*dw/dt -- about 10% of a four's peak at 550 rpm/s. Real dynos
        # correct for this (or ramp slowly enough that it does not matter); this
        # one reports the engine's own torque and keeps the physical brake for
        # the integration.
        self.brake_torque = engine_torque if self.dyno.active else brake
        net = engine_torque - brake
        # Reported quantities get a second, display-only smoothing pole in the
        # angle domain (tau = one engine cycle). The dyno plot, the telemetry
        # and the charts all *sample* torque, and any firing-frequency residue
        # left by the two-period mean above shows up as scatter (measured: the
        # I4's dyno peak read 13.6 bar BMEP against a mean-value 8.1 before
        # this pole). The integration above keeps the faster value, which is
        # what the governor and the energy balance need.
        mean_engine_torque = (
            mean_indicated * spec.cylinders
            - self.friction_torque
            - self.pumping_torque
            + starter
        )
        disp_tau = TWO_PI * spec.cycle_revs
        dth = max(self._wiebe_dtheta, 0.0)
        if disp_tau > 0.0 and dth > 0.0:
            alpha = 1.0 - math.exp(-dth / disp_tau)
            prev_disp = getattr(self, "_display_mean", mean_engine_torque)
            self._display_mean = prev_disp + alpha * (mean_engine_torque - prev_disp)
            mean_engine_torque = self._display_mean
        if self.dyno.active:
            self.brake_torque = mean_engine_torque
        self.net_torque = (
            mean_engine_torque - (brake if not self.dyno.active else 0.0)
        )
        omega += net / max(spec.inertia, 1e-6) * dt
        omega = max(0.0, min(omega, spec.redline_rpm * 1.15 * TWO_PI / 60.0))
        self.rpm = omega * 60.0 / TWO_PI

        period = TWO_PI * spec.cycle_revs
        self.theta = (self.theta + omega * dt) % period
        self._firing_cycles += omega * dt / period

        # --- exhaust heat soak (audit item 3) ------------------------------ #
        # The collector's temperature follows the *average* working condition
        # with a time constant of seconds: heat goes in at a rate set by how
        # hard the engine is burning (the smoothed load AND the mass flow --
        # revs push more gas through the same fire), and radiates out
        # constantly. ``pipe_temp`` exposes the resulting gas temperature to
        # the synthesiser, whose quarter-wave resonances then drift up by
        # 10-15 % over a hard pull and sag on the overrun -- the "the pipe
        # wakes up" effect every dyno video has.
        rel_rpm = _clamp(self.rpm / max(spec.redline_rpm, 1.0), 0.0, 1.1)
        heat_in = min(1.0, self._load_smoothed * (0.45 + 0.75 * rel_rpm))
        # faster than it cools: a working pipe heats in ~8 s, an idling one
        # sheds its heat over ~25 s (asymmetric tau)
        tau = 8.0 if heat_in > self.pipe_heat else 25.0
        self.pipe_heat += (heat_in - self.pipe_heat) * (1.0 - math.exp(-dt / tau))

        # order matters: detect the cut while the load still reflects the
        # previous, working condition
        self._detect_backfire()
        self._update_load_state(dt)

    def _update_knock(self, dt: float, demand: float) -> None:
        """Detonation from an ignition-delay correlation, plus timing retard.

        Rather than a hand-tuned "compression plus boost" score, this uses the
        Douaud-Eyzat ignition-delay correlation for the end gas::

            tau = 17.68 * (ON/100)^3.402 * p^-1.7 * exp(3800/T)   [ms]

        with ``p`` in bar and ``T`` in kelvin.  The available time is the crank
        angle around TDC at the current speed, so the dimensionless ratio
        ``t_window / tau`` behaves like the Livengood-Wu integral: it rises with
        boost and compression, falls with octane, and falls as the engine revs
        up and the cycle gets shorter.  A real ECU pulls timing when the knock
        sensor hears the ping; here that retard also feeds back into the end-gas
        estimate, which prevents a permanent full-retard state.

        This is still a mean-value model -- it does not resolve the pressure
        wave or the flame front -- but every term now has a physical meaning and
        the octane rating in a preset actually matters.
        """
        spec = self.spec
        if (
            spec.kind != "petrol"
            or self.rpm < max(600.0, spec.idle_rpm * 0.6)
            or not self.ignition_on
            or not (self.running or self.cranking)
            or self.fuel_cut
        ):
            target = 0.0
        else:
            # End-gas state at TDC, using a polytropic compression estimate.
            p_end_bar = max((self.map_pa / BAR) * spec.compression, 1.0)
            t_end_k = max(self.map_temp_k * spec.compression**0.35, 300.0)
            tau_ms = (
                17.68
                * (max(spec.octane, 70.0) / 100.0) ** 3.402
                * p_end_bar**-1.7
                * math.exp(3800.0 / t_end_k)
            )
            # Roughly 20 crank degrees around TDC is the window in which the
            # end gas can autoignite before the flame front arrives.
            window_ms = 60_000.0 / max(self.rpm * 3.0, 1.0)
            load_term = _clamp((demand - 0.15) / 0.85, 0.0, 1.0)
            tendency = (window_ms / max(tau_ms, 1e-3)) * load_term
            # Retard lowers the pressure peak and its temperature, so it
            # suppresses the very tendency that caused it.
            tendency *= 1.0 - 0.75 * self.knock_retard
            target = _clamp((tendency - self.KNOCK_ONSET) / 0.45, 0.0, 1.0)

        # Knock appears quickly and lingers for a few cycles.
        tau = 0.02 if target > self.knock_level else 0.20
        self.knock_level += (target - self.knock_level) * (1.0 - math.exp(-dt / tau))

        target_retard = _clamp(self.knock_level * 1.3, 0.0, 1.0)
        tau = 0.12 if target_retard > self.knock_retard else 0.80
        self.knock_retard += (target_retard - self.knock_retard) * (
            1.0 - math.exp(-dt / tau)
        )

    # ------------------------------------------------------------------ #
    # per-cylinder combustion (stage 7)
    # ------------------------------------------------------------------ #
    # Wiebe shape parameters. ``a`` is the completeness constant: xb at the end
    # of the window is 1 - exp(-a) = 99.3 %, the usual engineering closure.
    # ``m`` sets how front-loaded the burn is: 2.0 puts the peak burn rate near
    # a quarter of the way into the window, a real spark-ignition profile (fast
    # initial flame, long slow tail).
    WIEBE_A = 5.0
    WIEBE_M = 2.0
    # Burn window in crank angle: ignition 15 deg BTDC to 60 deg ATDC.
    BURN_START_RAD = math.radians(-15.0)
    BURN_SPAN_RAD = math.radians(75.0)
    # Below this crank speed the burn window spans more substeps than the fixed
    # step count can afford; the mean-value result (1.0) is used, which is the
    # correct limit -- the pulses are slower than the model's own resolution.
    WIEBE_MIN_HZ = 8.0

    def _cycle_mean_indicated(self, indicated: float) -> float:
        """Cycle-mean indicated torque: an angle-domain one-pole average.

        The Wiebe release is an angle-domain pulse train whose angle-integral
        equals the mean-value torque, so averaging it *over crank angle*
        recovers that mean exactly. A time average would be wrong while the
        engine accelerates (the burn window sweeps faster than the sample
        grid), and a block average over a fixed window has a reset artifact at
        exactly the wrong moment (measured: the post-crank flare of the 13.8 L
        three-cylinder diesel stalls when the window starts empty on the
        firings that matter).

        The one-pole smoother in the crank-angle domain has neither problem:

        * ``mean += (x - mean) * (1 - exp(-swept / tau))`` with ``tau`` equal
          to one burn window settles within one firing period, so the idle
          governor keeps its original authority;
        * it never resets, so there is no window-boundary artifact;
        * for anything slower than the firing frequency -- the governor, the
          dyno ramp, the whole torque curve -- it is effectively transparent,
          and its steady state is exactly the cycle mean by the integral
          argument above.

        Below the Wiebe speed floor the input is already the constant
        mean-value torque; a one-pole on a constant converges to the same
        constant, so the path is exact for both branches.
        """
        cycle_rad = TWO_PI * self.spec.cycle_revs
        dtheta = getattr(self, "_wiebe_dtheta", 0.0)
        if dtheta is None or dtheta <= 0.0:
            return indicated
        prev = getattr(self, "_wiebe_prev_theta", self.theta)
        swept = (self.theta - prev) % cycle_rad
        # a reset() between calls would sweep a huge false angle; clamp to a
        # substep's worth so the smoother cannot be poisoned
        swept = min(swept, dtheta * 2.0)
        self._wiebe_prev_theta = self.theta
        if swept <= 0.0:
            return getattr(self, "_wiebe_mean", indicated)
        # One-pole in the crank-angle domain. tau = two firing periods: the
        # step response settles well inside one engine cycle, which is the
        # widest window the idle governor tolerates (3+ periods was measured
        # to stall the 13.8 L three-cylinder diesel on the post-crank flare;
        # the lag crosses the governor's authority at the running threshold).
        # The residual firing-frequency ripple is removed *at the consumer*:
        # the dyno plot averages, and the reported brake torque below decays
        # it further. Keeping tau here short is what keeps the governor, the
        # limiter and the stall boundary on their original mean-value tuning.
        tau = 2.0 * cycle_rad / max(self.spec.cylinders, 1)
        mean = getattr(self, "_wiebe_mean", indicated)
        mean += (indicated - mean) * (1.0 - math.exp(-swept / tau))
        self._wiebe_mean = mean
        return mean

    def _step_cylinders(self, dt: float, omega_hz: float) -> float:
        """Advance every cylinder's burn and return this substep's energy fraction.

        The return value is a dimensionless multiplier for the mean-value
        indicated torque of this substep: 1.0 delivers exactly the old model's
        energy, 0.0 means no cylinder was in its burn window. Integrated over a
        full cycle the value is exactly 1.0 per cylinder on average, so the dyno
        curve, power figures and BMEP are all preserved; only the *shape within
        the cycle* changes. At 850 rpm a 75-degree window spans ~17 substeps and
        the release curve is resolved properly; at 14 500 rpm it spans less than
        one substep and the result approaches the mean-value pull, which is
        physically what happens -- the firing pulses merge as the engine revs.
        """
        spec = self.spec
        if omega_hz < self.WIEBE_MIN_HZ or not spec.firing_offsets:
            return 1.0

        cycle_rad = TWO_PI * spec.cycle_revs
        dtheta = omega_hz * dt  # crank radians this substep
        if dtheta <= 0.0:
            return 1.0
        n = spec.cylinders
        frac = 0.0
        step = dtheta / self.BURN_SPAN_RAD
        for cyl in range(n):
            off = spec.firing_offsets[cyl % len(spec.firing_offsets)]
            # angle of this cylinder within its own cycle, 0 = its firing TDC
            theta_c = (self.theta - off * cycle_rad) % cycle_rad
            # Window progress in a coordinate that is *continuous across the
            # firing TDC*: the window opens 15 deg before TDC, which in raw
            # theta_c lies just below the cycle wrap. Mapping through the
            # modulo first folds that piece back onto the window start, so the
            # progress runs smoothly 0 -> 1 instead of jumping from ~9 to 0.2
            # and losing the pre-TDC release -- which is a real part of the
            # burn (the flame starts before the piston reaches the top).
            x2 = ((theta_c - self.BURN_START_RAD) % cycle_rad) / self.BURN_SPAN_RAD
            if x2 <= 0.0 or x2 > 1.0 + step:
                continue
            x2 = min(x2, 1.0)
            x1 = x2 - step
            if x1 < 0.0:
                # ignition fell inside this substep: release only from ignition
                x1 = 0.0
            xb1 = 1.0 - math.exp(-self.WIEBE_A * x1 ** (self.WIEBE_M + 1.0))
            xb2 = 1.0 - math.exp(-self.WIEBE_A * x2 ** (self.WIEBE_M + 1.0))
            frac += xb2 - xb1

        # Normalisation. The mean-value model delivers a *constant* indicated
        # torque per substep -- energy E * dtheta/cycle_rad per cylinder, where
        # E is the per-window energy. The Wiebe release instead delivers E*frac
        # in this substep. The multiplier that reproduces the same total over
        # the cycle while keeping the pulses is therefore
        #   frac * cycle_rad / (n * dtheta)
        # with the per-window release measured the same way the integrator
        # discretises it (see _wiebe_window_sum), so the angle-integral of the
        # multiplier is 1.000 at every speed and phase. That conservation is
        # the contract the dyno curve depends on: peak torque, power and BMEP
        # are mean-value quantities and must not move when the pulses appear.
        norm = self._wiebe_window_sum(step) if step > 0.0 else 1.0
        return frac * cycle_rad / (norm * n * dtheta) if frac > 0.0 else 0.0

    def _wiebe_window_sum(self, step: float) -> float:
        """Exact discrete release of one window sampled at ``step`` intervals.

        Mirrors the increment arithmetic the integrator performs -- a substep
        covers [x-step, x] and releases xb(x) - xb(x-step) -- starting exactly
        at ignition (x = step from 0) and stopping exactly at the window end
        (x = 1), including the final partial substep. Because the arithmetic is
        identical, a fully swept window releases exactly this amount, so the
        pulse train conserves the mean-value energy by construction. Cached per
        step size: it only changes when the crank speed or the fixed step
        changes, i.e. essentially never inside a run.
        """
        cached = getattr(self, "_wiebe_norm_cache", None)
        if cached is not None and abs(cached[0] - step) < 1e-12:
            return cached[1]
        if step >= 1.0:
            total = 1.0 - math.exp(-self.WIEBE_A)  # one substep spans the window
        else:
            total = 0.0
            k = 1
            while k * step < 1.0:
                xb1 = 1.0 - math.exp(-self.WIEBE_A * ((k - 1) * step) ** (self.WIEBE_M + 1.0))
                xb2 = 1.0 - math.exp(-self.WIEBE_A * (k * step) ** (self.WIEBE_M + 1.0))
                total += xb2 - xb1
                k += 1
            xb1 = 1.0 - math.exp(-self.WIEBE_A * ((k - 1) * step) ** (self.WIEBE_M + 1.0))
            xb2 = 1.0 - math.exp(-self.WIEBE_A)  # close at the window end
            total += xb2 - xb1
        self._wiebe_norm_cache = (step, total)
        return total

    def _compression_torque(self) -> float:
        """Torque needed to push one full charge through compression.

        Polytropic compression work from BDC to TDC,
        ``W = P1*V1/(n-1) * (1 - CR^(1-n))``, converted to a mean torque over the
        cycle. This is the term that makes a high-compression diesel need a much
        larger starter than a petrol engine of the same size.
        """
        spec = self.spec
        cr = max(spec.compression, 2.0)
        n = 1.3
        v1 = spec.cylinder_displacement * cr / (cr - 1.0)
        work = P_AMB * v1 / (n - 1.0) * (1.0 - cr ** (1.0 - n))
        return work / (TWO_PI * spec.cycle_revs) * spec.cylinders

    def _starter_torque(self) -> float:
        """Cranking torque, scaled to what this engine actually needs."""
        spec = self.spec
        friction = _fmep_torque(spec, spec.starter_rpm)
        return max(25.0, friction + self._compression_torque())

    def _brake_torque(self, dt: float) -> float:
        spec = self.spec
        if self.dyno.active:
            # A real eddy-current dyno holds a target speed; the torque it needs
            # to do that *is* the engine's torque, which is what we plot.
            limit = 4.0 * self._peak_indicated
            self.dyno.target_rpm = min(self.dyno.end_rpm, self.dyno.target_rpm + self.dyno.ramp_rpm_s * dt)
            error = self.rpm - self.dyno.target_rpm
            torque = 1.6 * error + self.dyno_integral
            # Conditional integration. When the dyno drags the engine down from a
            # speed well above the target, the proportional term alone is already
            # far past the brake's ceiling; an integrator that keeps counting
            # through that takes seconds to bleed off, and on the chart it reads
            # as a torque spike at the start of the pull that the engine never
            # made. So integrate only when doing so can still change the output.
            saturated = (torque >= limit and error > 0.0) or (torque <= 0.0 and error < 0.0)
            if not saturated:
                self.dyno_integral = min(max(self.dyno_integral + error * dt * 2.0, 0.0), limit)
            torque = 1.6 * error + self.dyno_integral
            return min(max(torque, 0.0), limit)
        if self._manual_load < 1e-4:
            return 0.0
        # Viscous brake: unlike a constant-torque brake it is stable on the
        # falling side of the torque curve, so the engine settles instead of
        # hunting between two equilibrium points.
        omega = max(self.rpm, 1.0) * TWO_PI / 60.0
        coeff = 2.0 * self._peak_indicated / max(spec.redline_rpm * TWO_PI / 60.0, 1e-6)
        return self._manual_load * coeff * omega

    def _peak_torque_estimate(self) -> float:
        """Rough full-load torque, used only to size brakes and normalise load."""
        spec = self.spec
        if spec.kind == "turbine":
            torque, _, _, _, _ = self._turbine_cycle(spec.redline_rpm, 1.0)
            return max(torque, 1.0)
        ve = _ve(spec, spec.ve_peak_rpm, 0.97 * P_AMB)
        rho = P_AMB / (R_AIR * T_AMB)
        air = ve * spec.cylinder_displacement * rho
        if spec.boost_bar > 0.0:
            # full boost, after the intercooler
            air *= _charge_density_ratio(spec.boost_bar, spec.charge_cooling)
        if spec.kind == "diesel":
            fuel = spec.cylinder_displacement * rho / (AFR_PETROL * DIESEL_LAMBDA_FULL)
        else:
            fuel = air / spec.afr
        return spec.eta_i * fuel * spec.lhv / (TWO_PI * spec.cycle_revs) * spec.cylinders

    def _turbine_cycle(
        self, rpm: float, fuel: float
    ) -> tuple[float, float, float, float, float]:
        """Steady air-standard Brayton cycle at one shaft speed and fuel setting.

        Returns ``(shaft_torque_nm, shaft_power_w, fuel_kg_s, pressure_ratio,
        turbine_inlet_k)``.  The compressor pressure ratio follows speed squared,
        which is the usual first approximation for a fixed-geometry compressor
        map; mass flow follows speed, and the firing temperature is scheduled
        between the idle and maximum turbine-inlet values.  This is deliberately
        mean-value: no surge line, no bleed, no variable stators, but the cycle
        responds correctly to pressure ratio, ambient temperature, component
        efficiency and mass flow.

        Torque is the *shaft* torque after the compressor has taken its share:
        that is what a dyno or a gearbox sees, not the raw turbine work.
        """
        spec = self.spec
        gt = spec.turbine
        speed = _clamp(rpm / max(spec.redline_rpm, 1e-6), 0.0, 1.02)
        pr_c = 1.0 + (gt.pressure_ratio - 1.0) * speed * speed
        t2 = T_AMB * (
            1.0
            + (pr_c ** ((GAMMA - 1.0) / GAMMA) - 1.0)
            / max(gt.compressor_efficiency, 1e-6)
        )
        mdot = gt.mass_flow_kg_s * max(speed, 0.0)

        # No fuel means no combustion: the compressor is still being turned by
        # the residual rotor speed, but it produces no shaft torque.  Without
        # this the model would keep making hundreds of kilowatts while the
        # engine winds down with the ignition off.
        if fuel <= 1e-6 or mdot <= 0.0:
            return 0.0, 0.0, 0.0, pr_c, T_AMB

        t_idle = gt.idle_turbine_inlet_c + 273.15
        t_max = gt.max_turbine_inlet_c + 273.15
        t3 = t_idle + (t_max - t_idle) * _clamp(fuel, 0.0, 1.0)

        # The turbine expands almost the whole compressor pressure ratio; the
        # few percent lost in the combustor and ducts is not recovered.
        pr_t = max(pr_c * gt.pressure_loss, 1.001)
        t4 = t3 * (
            1.0
            - gt.turbine_efficiency
            * (1.0 - (1.0 / pr_t) ** ((GAMMA - 1.0) / GAMMA))
        )

        compressor_work = CP_AIR * max(t2 - T_AMB, 0.0)
        turbine_work = CP_GAS * max(t3 - t4, 0.0)
        net_work = max(turbine_work - compressor_work, 0.0) * gt.mechanical_efficiency
        power = mdot * net_work
        omega = max(rpm, 1.0) * TWO_PI / 60.0
        torque = power / omega if rpm > 1.0 else 0.0

        fuel_kg_s = (
            mdot
            * CP_AIR
            * max(t3 - t2, 0.0)
            / max(gt.combustor_efficiency * LHV_KEROSENE, 1e-6)
        )
        return torque, power, fuel_kg_s, pr_c, t3

    def _step_turbine(self, dt: float) -> None:
        """A gas turbine is not a reciprocating engine: it spools, it does not fire.

        The shaft speed is governed by a first-order spool response rather than
        by integrating the instantaneous torque.  That is what a real fuel
        control unit does: it schedules fuel to hold a requested speed, and the
        shaft follows.  The Brayton cycle then supplies the torque and power at
        that operating point, so the dyno curve, the load meter and the sound
        all see the same machine.
        """
        spec = self.spec
        gt = spec.turbine
        if self.dyno.active:
            self.pedal = 1.0
        if not self.ignition_on:
            self._turbine_started = False

        alive = self.ignition_on and (
            self.cranking or self.running or self._turbine_started
        )
        if self.cranking:
            # The starter only has to reach a light-off speed; the combustor
            # takes over from there.
            target = max(spec.starter_rpm, spec.idle_rpm * 0.20)
        elif alive:
            demand = self.pedal
            target = spec.idle_rpm + demand * (spec.redline_rpm - spec.idle_rpm)
        else:
            target = 0.0

        if self.dyno.active and alive:
            # A speed-holding dyno commands the shaft directly.  Unlike a
            # piston engine, a turbine does not need the brake to pull it down:
            # its fuel control follows the same target.  The ramp lives here
            # because the piston path advances it inside ``_brake_torque``.
            self.dyno.target_rpm = min(
                self.dyno.end_rpm,
                self.dyno.target_rpm + self.dyno.ramp_rpm_s * dt,
            )
            target = self.dyno.target_rpm

        target = _clamp(target, 0.0, spec.redline_rpm * 1.02)
        tau = gt.spool_tau_s if target > self.rpm else gt.wind_down_tau_s
        blend = 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-6))
        self.rpm += (target - self.rpm) * blend
        self.rpm = _clamp(self.rpm, 0.0, spec.redline_rpm * 1.02)

        if self.cranking and self.rpm > spec.starter_rpm * 0.5:
            self._turbine_started = True

        fuel = max(self.pedal, 0.06) if alive else 0.0
        torque, power, fuel_kg_s, pr_c, t3 = self._turbine_cycle(self.rpm, fuel)
        self.turbine_power_kw = power / 1000.0
        self.turbine_fuel_kg_s = fuel_kg_s
        self.turbine_pressure_ratio = pr_c
        self.turbine_inlet_k = t3

        self.map_pa = P_AMB * pr_c
        self.map_temp_k = T_AMB * (
            1.0
            + (pr_c ** ((GAMMA - 1.0) / GAMMA) - 1.0)
            / max(gt.compressor_efficiency, 1e-6)
        )
        self.boost = _clamp(
            (pr_c - 1.0) / max(gt.pressure_ratio - 1.0, 1e-6), 0.0, 1.0
        )
        self.throttle = fuel
        self.indicated_torque = torque
        self.friction_torque = 0.0
        self.pumping_torque = 0.0
        self.limiter_cut = False
        self.overrun_cut = False
        self.fuel_cut = False
        self.brake_torque = (
            self.indicated_torque
            if self.dyno.active
            else self.load * self._peak_indicated
        )
        self.net_torque = self.indicated_torque - self.brake_torque
        omega = self.rpm * TWO_PI / 60.0
        self.theta = (self.theta + omega * dt) % TWO_PI
        self._update_load_state(dt)

    # ------------------------------------------------------------------ #
    # derived outputs
    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self.rpm > self.spec.starter_rpm * 1.2

    @property
    def full_load_torque(self) -> float:
        """Roughly the torque this engine can make wide open, for gauge scales."""
        return self._peak_indicated

    @property
    def power_w(self) -> float:
        return self.net_torque * self.rpm * TWO_PI / 60.0

    @property
    def power_hp(self) -> float:
        return self.power_w / 745.7

    @property
    def power_kw(self) -> float:
        return self.power_w / 1000.0

    @property
    def map_bar(self) -> float:
        return self.map_pa / BAR

    @property
    def boost_bar(self) -> float:
        """Gauge pressure in the manifold (0 for a naturally aspirated engine)."""
        return max(0.0, self.map_pa / P_AMB - 1.0) * P_AMB / BAR

    @property
    def sound_load(self) -> float:
        """Engine load the way the synthesiser wants it: 0..1.

        Measured against the torque this engine could make at this speed, not
        against the pedal, so lifting off at high rpm drops it to zero and the
        sound leans out exactly as it should.
        """
        return self._load_smoothed

    def _update_load_state(self, dt: float) -> None:
        peak = max(self._peak_indicated, 1e-6)
        raw = max(0.0, min(1.0, self.indicated_torque * self.spec.cylinders / peak))
        # Only the overrun cut silences the engine as far as the sound is
        # concerned. Chopping on the limiter should *not* drag the load to zero a
        # few times a second: the synthesiser has its own limiter layer for that,
        # and a 4 Hz warble in the load parameter would sound like a fault.
        if self.overrun_cut or self.indicated_torque <= 0.0:
            raw = 0.0
        # smooth it: the sound should not chatter at the firing frequency
        self._load_smoothed += (raw - self._load_smoothed) * min(1.0, dt / 0.045)
        # peak-hold for the backfire decision: what matters is whether the engine
        # was working hard a moment ago, not what it is doing right now
        self._load_peak = max(raw, self._load_peak - dt / 1.5)

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def _detect_backfire(self) -> None:
        """Unburnt charge igniting in the exhaust, the way it happens for real.

        Two distinct mechanisms, both tied to a *rising edge* of the fuel cut
        rather than to a timer:

        * **Overrun** -- the classic one. Fuel was burning, the driver lifts off
          above a few thousand rpm, and the deceleration cut leaves a charge in a
          hot, still-flowing exhaust. Probability rises with rpm and with how
          hard the engine was working.
        * **Rev limiter** -- a hard cut also dumps raw fuel into a glowing
          manifold, but a bouncing limiter must not machine-gun bangs, so it gets
          a longer cooldown and a much lower chance.

        Being edge-triggered is what keeps this honest: holding the throttle at
        the limiter for ten seconds bangs a couple of times, not two hundred.
        """
        just_overrun = self.overrun_cut and not self._prev_overrun_cut
        just_limiter = self.limiter_cut and not self._prev_limiter_cut
        self._prev_overrun_cut = self.overrun_cut
        self._prev_limiter_cut = self.limiter_cut
        if not (just_overrun or just_limiter):
            return

        # Two independent cause-specific cooldowns, so a long spell bouncing on
        # the limiter cannot silence the bang that belongs to the lift-off.
        if just_overrun:
            if self._backfire_cooldown > 0.0 or self.rpm <= 3000.0:
                return
            chance = min(0.85, (self.rpm - 3000.0) / 3500.0) * (0.35 + 0.65 * self._load_peak)
            if self.rng.random() < chance:
                self.events.append("backfire")
                self._backfire_cooldown = self.BACKFIRE_MIN_INTERVAL
        else:
            if self._limiter_cooldown > 0.0:
                return
            if self.rng.random() < 0.25:
                self.events.append("backfire")
                self._limiter_cooldown = self.BACKFIRE_LIMITER_INTERVAL

    def drain_events(self) -> list[str]:
        out = self.events
        self.events = []
        return out

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.rpm = 0.0
        self.theta = 0.0
        self._firing_cycles = 0.0
        self.map_pa = P_AMB
        self.map_temp_k = T_AMB
        self.boost = 0.0
        self.limiter_cut = False
        self.overrun_cut = False
        self.fuel_cut = False
        self.knock_level = 0.0
        self.knock_retard = 0.0
        self.dyno_integral = 0.0
        self._gov_integral = 0.0
        self._load_smoothed = 0.0
        self._load_peak = 0.0
        self._blow_off_cooldown = 0.0
        self._prev_pedal = 0.0
        self._turbine_started = False
        self.pipe_heat = 0.0
        # Wiebe accumulators: a half-finished cycle from before the reset must
        # not leak into the next run's first reported torque
        self.torque_envelope = 0.0
        self._wiebe_mean = 0.0
        self._display_mean = 0.0
        self._wiebe_prev_theta = 0.0
        self._wiebe_dtheta = 0.0
        self._wiebe_norm_cache = None
        self.turbine_power_kw = 0.0
        self.turbine_fuel_kg_s = 0.0
        self.turbine_pressure_ratio = 1.0
        self.turbine_inlet_k = T_AMB
        self.events.clear()

    @property
    def crank_angle(self) -> float:
        """Crank angle within one revolution, radians."""
        return self.theta % TWO_PI

    @property
    def torque_pulse(self) -> float | None:
        """Wiebe pulse train, normalised to full load, for the synthesiser.

        ``torque_envelope`` is the raw per-cylinder instantaneous torque whose
        cycle-mean is ``indicated_torque``. This property rescales it against
        the engine's own full-load estimate so the audio gets 0..1+ regardless
        of engine size: 0 between firings, ~1 at a full-load firing peak, and a
        *sagging* value at light load -- which is the point, the firing of a
        closed-throttle engine is quieter than a working one.

        ``None`` when the engine is not on the Wiebe path (turbine, below the
        speed floor, or no fuel): the synthesiser then keeps its own
        load/throttle amplitude schedule.
        """
        spec = self.spec
        if spec.kind == "turbine":
            return None
        peak = max(self._peak_indicated * max(spec.cylinders, 1), 1e-6)
        return max(0.0, self.torque_envelope * spec.cylinders / peak)

    @property
    def pipe_temp(self) -> float | None:
        """Exhaust gas temperature the synthesiser should hear, Celsius.

        The preset's ``pipe_temp_c`` is the *fully hot* value. ``pipe_heat``
        (0 = stone cold, 1 = heat-soaked) interpolates from a cold-start
        ~120 C to that value, so the quarter-wave resonances start low and
        climb over the first seconds of running. ``None`` for a turbine (no
        pipes to the synthesiser's knowledge) or when the model has not run.
        """
        spec = self.spec
        if spec.kind == "turbine":
            return None
        # Cold start: barely-warm pipe. Full soak: the preset family's working
        # temperature. The synthesiser's comb delay scales with sqrt(T), so the
        # quarter-wave resonance spans ~1.3x over the whole range -- the
        # audible "pipe waking up".
        cold_c = 60.0
        hot_c = 600.0
        return cold_c + (hot_c - cold_c) * max(0.0, min(1.0, self.pipe_heat))

    @property
    def firing_phase(self) -> float:
        """Position in the firing cycle as a fraction, 0..1.

        ``theta`` advances at crank speed and wraps at the end of a full cycle
        (two revolutions for a four-stroke, one for a two-stroke), so the ratio
        is the fraction of the cycle elapsed; 0 means the reference cylinder is
        at its firing TDC.

        Use :attr:`firing_cycles` to drive the synthesiser: a fraction alias
        every time a block spans more than half a cycle, which at 48 kHz blocks
        is anything above ~1400 rpm.
        """
        period = TWO_PI * max(self.spec.cycle_revs, 1e-9)
        return (self.theta % period) / period

    @property
    def firing_cycles(self) -> float:
        """Monotonic count of firing cycles elapsed since start, fractional.

        This -- not :attr:`firing_phase` -- is what locks the audio to the
        model. A phase in 0..1 aliases: at 3000 rpm one 1024-sample block spans
        0.53 cycles and at 14500 rpm it spans 2.6, so a wrapped phase cannot say
        which cycle it is on. An unwrapped counter can, and the synthesiser
        simply interpolates from the previous value to this one across the block.

        Being monotonic also makes the contract obvious: the difference between
        two calls is exactly how many firings happened between them.
        """
        return self._firing_cycles

    def cylinder_angle(self, cylinder: int) -> float:
        """Crank angle of one cylinder within the cycle, radians."""
        spec = self.spec
        offsets = spec.firing_offsets
        phase = offsets[cylinder % len(offsets)] * TWO_PI * spec.cycle_revs if offsets else 0.0
        return (self.theta + phase) % (TWO_PI * spec.cycle_revs)

    def piston_position(self, cylinder: int, bank_sign: float = 1.0) -> float:
        """Distance from the crank axis to the pin, metres.

        Exact slider-crank kinematics, including the secondary term that makes a
        real engine vibrate at twice crank speed.
        """
        spec = self.spec
        r = spec.crank_radius
        l = spec.rod_length
        a = self.cylinder_angle(cylinder)
        if bank_sign < 0.0:
            a += spec.bank_angle
        s = r * math.sin(a)
        return r * math.cos(a) + math.sqrt(max(l * l - s * s, 1e-9))
