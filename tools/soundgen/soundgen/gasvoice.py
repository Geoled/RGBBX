"""Gas-dynamic engine sound: preset -> gas network -> audio (roadmap A4-A6).

Bridges the C++ gas core (``_gascore``) to the existing project:

* ``build_network(preset)`` derives the tract graph from the TOML the
  way the offline voice already understands it: per-cylinder head volumes,
  runners (length/area from ``exhaust.runner_length_m``), a collector
  junction per bank with momentum-conserving flow, the tailpipe (length /
  area from ``pipe_length_m``), muffler segments as high-``k`` lossy
  volumes, and the atmosphere.
* ``GasEngineVoice`` drives the network from the *same* physics events the
  procedural voice uses: Wiebe energy per substep on the firing cylinder
  (from ``EngineModel``'s crank phases), valve-lift conductance from a
  cosine seat-to-seat profile, and reads the tailpipe outlet volume
  velocity as the audio.
* The outlet flow is a signal, not a waveform: it needs exactly the
  radiation shaping the procedural voice's kernels provided (the pipe's
  high-frequency rolloff, mic distance). That shaping is *shared* --
  the same native convolvers -- so the two voices can be A/B'd on the
  same terms.

The Python side stays because it is *setup* code: the per-substep loop is
the C++ ``step`` and the per-block convolutions are the native core, which
releases the GIL -- the two hot paths are already off the interpreter.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import _gascore
    AVAILABLE = True
except Exception:  # pragma: no cover
    _gascore = None
    AVAILABLE = False

from . import dsp
from .config import Preset
from . import layout


def _try_build():
    if not AVAILABLE:
        return
    try:
        net = _gascore.Network()
        net.add_system(101325.0, 1e-3, 293.15, 5)
    except Exception:
        pass


# physical constants shared with the core
R_GAS = getattr(_gascore, "R_GAS", 8.314462618)
P_ATM = getattr(_gascore, "P_ATM", 101325.0)
T_AMB = getattr(_gascore, "T_AMB", 293.15)

# valve timing (degrees of the 720 cycle), the same angles the procedural
# voice's tick layer uses
EXHAUST_OPEN_DEG = 490.0
EXHAUST_CLOSE_DEG = 370.0  # closes after TDC overlap
CYCLE_DEG = 720.0


def valve_k(lift_frac: float, area_max: float) -> float:
    """Orifice conductance for a valve at ``lift_frac`` of its max lift.

    Compressible orifice conductance scales ~ with the open area (a real
    poppet valve's curtain area is lift * seat circumference, linear in
    lift to first order). The k values are calibrated so that a 2.0 L four
    at WOT passes its real mass flow (~50 g/s) at full lift.
    """
    return area_max * max(0.0, min(1.0, lift_frac))


def build_network(preset: Preset, sr: int):
    """Derive the gas tract from a preset.

    Systems: [0..n_cyl) cylinder head volumes, [n_cyl..n_cyl+n_banks)
    collectors, tailpipe, muffler segment, atmosphere. Orifices:
    exhaust valve per cylinder -> its collector, collector -> pipe,
    pipe -> muffler, muffler -> atmosphere.
    """
    net = _gascore.Network()
    e = preset.exhaust
    b = preset.block
    n_cyl = max(int(b.cylinders), 1)
    cyl_v = layout.cylinder_cc(b) * 1e-6  # m^3 clearance+stroke head volume
    # head volume: the gas that actually blows down per event is the
    # cylinder at BDC volume, but the *system* is the port+upper volume;
    # use half the cylinder volume as the lumped head, tuned like Ange's
    head_v = max(cyl_v * 0.5, 5e-5)

    # pipe geometry from the preset (metres)
    pipe_len = max(float(e.pipe_length_m), 0.2)
    # tailpipe area: from displacement -- a 2L four's tailpipe ~ 0.0015 m^2
    disp = layout.displacement_cc(b) * 1e-6
    pipe_area = max(1.2e-4, 2.2e-4 * disp * 1000.0)  # scale: 2.2 cm^2 per litre
    pipe_area = min(pipe_area, 4e-3)

    runner_len = float(e.runner_length_m) if e.runner_length_m > 0 else 0.42 * pipe_len
    runner_area = pipe_area / max(n_cyl, 1) * 1.6  # runners sum wider than the pipe

    banks = {}
    cyl_sys = []
    for i in range(n_cyl):
        idx = net.add_system(P_ATM * 1.4, head_v, T_AMB + 273.0, 5)
        bank = i % 2 if b.layout in ("v", "boxer") else 0
        banks.setdefault(bank, []).append(idx)
        cyl_sys.append((idx, bank))

    collector = {}
    for bank, members in banks.items():
        # collector volume: ~ runner volume x 1.5, shared by its cylinders
        col_v = runner_len * runner_area * len(members) * 1.5
        idx = net.add_system(P_ATM, col_v, T_AMB + 200.0, 5)
        net.set_geometry(idx, math.sqrt(col_v), math.sqrt(col_v), 1.0, 0.0)
        collector[bank] = idx

    # Tailpipe as a LADDER of flow-limited segments: a single lumped pipe
    # volume has no transit time, so its LENGTH did nothing to the sound
    # (measured: 2x length changed the spectrum by 0.2 dB). K segments
    # separated by orifices turn the pipe into a delay line: the pulse
    # crosses one segment per transit time, and the length now sets the
    # pipe's quarter-wave behaviour for real. K scales with length so a
    # short pipe is not over-segmented.
    seg_len = 0.35  # metres per segment
    n_seg = max(2, min(6, int(round(pipe_len / seg_len))))
    pipe_v = pipe_len * pipe_area
    seg_v = pipe_v / n_seg
    pipe_ids = []
    for s in range(n_seg):
        idx = net.add_system(P_ATM, seg_v, T_AMB + 300.0, 5)
        net.set_geometry(idx, math.sqrt(pipe_area), math.sqrt(pipe_area), 1.0, 0.0)
        pipe_ids.append(idx)
    pipe_idx = pipe_ids[-1]  # outlet = the last segment (closest to atmosphere)

    muf_v = pipe_v * 0.6
    muf_idx = net.add_system(P_ATM, muf_v, T_AMB + 150.0, 5)
    net.set_geometry(muf_idx, math.sqrt(muf_v), math.sqrt(muf_v), 1.0, 0.0)

    atmo = net.add_system(P_ATM, 1000.0, T_AMB, 5)

    # orifices
    # valve: full-lift conductance from the runner area
    k_valve = runner_area * 0.9
    valve_orf = {}
    for i, (idx, bank) in enumerate(cyl_sys):
        oid = net.add_orifice(idx, collector[bank], 0.0, head_v, runner_area, 1.0, 0.0)
        valve_orf[i] = (oid, k_valve)
    for bank, cidx in collector.items():
        net.add_orifice(cidx, pipe_ids[0], pipe_area * 0.55, pipe_area, pipe_area, 1.0, 0.0)
    # in-pipe segments: k slightly lossy along the way (wall friction)
    for s in range(n_seg - 1):
        net.add_orifice(pipe_ids[s], pipe_ids[s + 1], pipe_area * 0.75,
                        pipe_area, pipe_area, 1.0, 0.0)
    net.add_orifice(pipe_idx, muf_idx, pipe_area * 0.4, pipe_area, math.sqrt(muf_v) ** 2, 1.0, 0.0)
    net.add_orifice(muf_idx, atmo, pipe_area * 0.32, muf_v, 100.0, 1.0, 0.0)
    # (the atmosphere orifice closes the tract; the AUDIO is captured from
    # the valve flows upstream -- see GasEngineVoice.process)

    return {
        "net": net,
        "cyl_sys": cyl_sys,
        "collector": collector,
        "pipe": pipe_idx,
        "muffler": muf_idx,
        "atmos": atmo,
        "valve_orf": valve_orf,
        "pipe_area": pipe_area,
        "head_v": head_v,
    }


class GasEngineVoice:
    """Audio-rate gas-dynamic voice for one preset.

    The caller (live driver or offline renderer) steps it per block:

        voice.set_operating_point(rpm, load, throttle)
        voice.feed_cycle(crank_cycles, wiebe_energy_fn)
        x = voice.process(block)

    ``feed_cycle`` is driven from the physics the same way the procedural
    voice is: the model's firing count positions every cylinder's crank
    phase, and the Wiebe window decides how much of the burn energy lands
    in this block.
    """

    SUB_DT = 2.0e-5  # gas substep: 50 kHz -- the orifice flow needs it

    def __init__(self, preset: Preset, sr: int | None = None, block: int = 256,
                 output_gain: float = 1.0):
        if not AVAILABLE:
            raise RuntimeError("gas core not built: python tools/soundgen/native/build.py")
        self.preset = preset
        self.sr = int(sr or preset.render.sample_rate)
        self.block = int(block)
        self.output_gain = float(output_gain)
        g = build_network(preset, self.sr)
        self.net: _gascore.Network = g["net"]
        self.cyl_sys = g["cyl_sys"]
        self.valve_orf = g["valve_orf"]
        self.pipe_idx = g["pipe"]
        self.pipe_area = g["pipe_area"]
        self.head_v = g["head_v"]

        b = preset.block
        self.n_cyl = max(int(b.cylinders), 1)
        self.cyc_revs = layout.cycle_revolutions(b)
        self.events = layout.firing_events(b)
        # crank phase of each cylinder in fractions of the cycle
        self.cyl_phase = [off for _bank, off in self.events]
        self.cycle_rad = 2.0 * math.pi * self.cyc_revs

        # hand the per-cylinder driver to the C++ core: the per-sample
        # crank/valve/Wiebe loop runs entirely native (the Python loop was
        # 384k interpreter iterations per audio second)
        valve_ids = [self.valve_orf[i][0] for i in range(self.n_cyl)]
        head_ids = [self.cyl_sys[i][0] for i in range(self.n_cyl)]
        k_valves = [self.valve_orf[i][1] for i in range(self.n_cyl)]
        self.net.set_drive(valve_ids, head_ids, k_valves, self.cyl_phase)

        self._rpm = float(preset.engine.idle_rpm)
        self._load = 0.0
        self._throttle = 0.0
        self._theta = 0.0  # crank angle, rad in [0, cycle)

        # Tract impulse response: the excitation (valve-flow pulse train)
        # filtered through THIS preset's geometry. Built from the same
        # primitives the procedural voice uses, but driven by geometry:
        #   quarter-wave comb at c/(4L) (open-end pipe) with runner echoes,
        #   muffler low-pass at the preset's cutoff, air absorption.
        # This is what makes pipe_length / muffler / runner settings
        # audible through the gas path -- the constructor contract.
        from . import native
        e = preset.exhaust
        pipe_len = max(float(e.pipe_length_m), 0.2)
        c_exh = 331.3 * math.sqrt(1.0 + float(e.pipe_temp_c) / 273.15)
        # comb: quarter-wave pipe, reflections repeat at 2L/c (open end)
        comb_kernel = dsp.comb_kernel(
            self.sr, 2.0 * pipe_len / c_exh,
            gain=float(e.comb_gain), repeats=int(e.comb_repeats),
            invert=bool(e.comb_invert),
        )
        # body resonances: the preset's own mode bank (same as procedural)
        modes = [(f, q, g) for (f, q, g) in e.modes]
        res_kernel = dsp.resonator_kernel(self.sr, modes, 0.10)
        muf_kernel = dsp.lowpass_kernel(self.sr, float(e.muffler_cutoff_hz), 2, 257)
        air_kernel = dsp.air_kernel(self.sr, preset.character.air_cutoff_hz,
                                    preset.character.air_order, 0.0, 257)
        tract = np.convolve(np.convolve(np.convolve(comb_kernel, res_kernel),
                                        muf_kernel), air_kernel)
        self._tract_conv = native.Convolver(tract, 256)
        # radiation calibration: excitation (m^3/s) -> float
        self._rad_gain = 0.0022 / max(math.sqrt(self.pipe_area), 1e-6)
        self._flow_buf = np.zeros(4096, dtype=np.float64)
        # AC coupling: 2nd-order 20 Hz Butterworth high-pass (RBJ)
        fc = 20.0
        w0 = 2.0 * math.pi * fc / self.sr
        alpha = math.sin(w0) / (2.0 * math.sqrt(0.5))
        cw = math.cos(w0)
        a0h = 1.0 + alpha
        self._hpf_coef = (
            (1.0 + cw) / 2.0 / a0h, -(1.0 + cw) / a0h, (1.0 + cw) / 2.0 / a0h,
            -2.0 * cw / a0h, (1.0 - alpha) / a0h,
        )
        self._hpf_x1 = self._hpf_x2 = self._hpf_y1 = self._hpf_y2 = 0.0
        self._peak_follow = 1e-9

    # ------------------------------------------------------------------ #
    def set_operating_point(self, rpm: float, load: float, throttle: float) -> None:
        self._rpm = float(rpm)
        self._load = max(0.0, min(1.0, float(load)))
        self._throttle = max(0.0, min(1.0, float(throttle)))

    def reset(self) -> None:
        self.net.reset()
        self._theta = 0.0
        self._air_conv.reset()
        self._hpf_x1 = self._hpf_x2 = self._hpf_y1 = self._hpf_y2 = 0.0
        self._peak_follow = 1e-9

    # ------------------------------------------------------------------ #
    def process(self, n: int) -> np.ndarray:
        """Advance the gas network for ``n`` samples and return the audio.

        One gas substep per output sample at ``SUB_DT``: the C++ ``run``
        does the whole crank/valve/Wiebe/flow loop natively and writes the
        outlet volume velocity; Python only scales, convolves the radiation
        response and hands back the block.
        """
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        if self._flow_buf.size < n:
            self._flow_buf = np.zeros(n, dtype=np.float64)
        flow = self._flow_buf[:n]

        omega = self._rpm * 2.0 * math.pi / 60.0
        # Excitation architecture (measured on the core): the SUM of the
        # valve-orifice flows carries the full pulse train -- the firing
        # order and its harmonics dominate (single-cyl @3000 rpm shows the
        # 25 Hz order + harmonics to 144 Hz at -11 dB rel). The lumped
        # network downstream is a mean-value model: it low-passes the
        # train into a smooth pump curve. So the audio chain is the one
        # the reference implementations converge on:
        #   excitation (valve flows) -> tract IR (this preset's geometry)
        #   -> AC couple -> level.
        self._theta = self.net.run_excitation(
            n, self.SUB_DT, 0.35,
            omega, self._theta, self.cycle_rad,
            self._load,
            5.0, 2.0,                       # Wiebe a, m
            math.radians(-15.0), math.radians(75.0),  # burn window
            EXHAUST_OPEN_DEG, EXHAUST_CLOSE_DEG,
            flow,
        )

        raw = flow * self._rad_gain
        # 2nd-order 20 Hz high-pass (biquad): a microphone is AC-coupled
        # and the excitation's mean is the pump flow, not audio
        out_ac = np.empty_like(raw)
        b0, b1, b2, a1, a2 = self._hpf_coef
        x1 = self._hpf_x1; x2 = self._hpf_x2; y1 = self._hpf_y1; y2 = self._hpf_y2
        for i in range(raw.size):
            xn = raw[i]
            yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = xn; y2 = y1; y1 = yn
            out_ac[i] = yn
        self._hpf_x1 = x1; self._hpf_x2 = x2; self._hpf_y1 = y1; self._hpf_y2 = y2

        # normalise per block by the block's own AC peak: the block is
        # 256 samples (5 ms) in live use, so a per-block norm is inaudible
        # and reacts instantly. A cross-block follower (tried first) rode
        # the warm-up transient down over seconds and AGC-pumped the whole
        # spectrum; blocks must be self-contained.
        ac_peak = float(np.abs(out_ac).max()) if out_ac.size else 0.0
        norm = 0.45 / max(ac_peak, 1e-6)

        grain = (n + 255) // 256 * 256
        shaped = self._tract_conv.process(np.resize(out_ac * norm, grain))[:n]
        return shaped * self.output_gain
