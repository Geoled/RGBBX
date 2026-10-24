"""The streaming engine voice.

``EngineVoice`` is a block-based, stateful synthesizer. It produces the same
signal whether it is pulled continuously for real-time playback or rendered
offline in one go, so a preset sounds identical live and in the exported asset.

Signal path
-----------
    combustion events (per cylinder, own primary runner)
        + exhaust flow roar                            -> exhaust resonators
                                                       -> pipe comb (time varying)
                                                       -> muffler
    intake events (per cylinder, own runners)          -> intake resonators
        + intake flow noise
    valvetrain / injector clicks (cam timed)
    mechanical noise: crank thump + piston slap (per firing) + friction band
    crank / accessory whine
    turbo whistle + whoosh
    ---------------------------------------------------------------- sum
                        -> presence bell -> air rolloff -> DC block
                        -> soft clip -> gain
                        -> outdoor convolution reverb (wet/dry)

Everything upstream of the sum is linear, and the *steady* filters are
time-invariant, so they are realised as precomputed FIR kernels and convolved
block-by-block. The one exception is the exhaust pipe comb: its delay follows
the gas temperature, so it is a fractional delay line instead of a baked
kernel. The reverb is the last linear stage and is *also* time-invariant, which
is what keeps a rendered loop exactly periodic.

Three things make it behave like an engine rather than a pulse train:

* every cylinder has its own primary runner length, so the pulses reaching the
  collector are not perfectly coherent;
* the pulse shape is crossfaded between a rounded light-load shape and a
  shock-fronted full-load shape;
* every firing event gets its own cycle-to-cycle amplitude perturbation, which
  broadens the harmonic lines exactly the way a real engine does.

The mechanical layer is driven by the same firing grid, not by a random walk:
a crank/big-end thump lands on every firing TDC, piston slap follows it (and,
for a four-stroke, repeats more weakly at the opposite TDC), a band-limited
friction bed runs continuously, and the cam chain adds a whine at half crank
speed. It is gated off for turbines and its slap is gated off for rotaries.
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp, layout, native
from .config import Preset

# Valve timing, in degrees of the 720 deg four-stroke cycle (0 = firing TDC).
# Values are typical of a mildly tuned road cam. Both the close *and* the lift
# are audible; the lift is quieter (see Mechanical.valve_open_gain).
INTAKE_OPEN_DEG = 350.0
INTAKE_CLOSE_DEG = 600.0
EXHAUST_OPEN_DEG = 490.0
EXHAUST_CLOSE_DEG = 370.0
INJECTION_DEG = 350.0
CYCLE_DEG = 720.0

# The firing load does not peak at TDC: the bearing and big-end see it a little
# after, and the piston's secondary motion (the actual slap) is later still.
# Offsetting the mechanical events from the combustion pulse is both physical
# and what stops them from cancelling it, since both ride the same grid.
CRANK_LAG_DEG = 14.0
SLAP_LAG_DEG = 22.0
# Intake (and exhaust) valve pairs do not seat at exactly the same instant.
VALVE_PAIR_SPREAD_DEG = 1.6

# Output ceiling, in linear amplitude: -1 dBFS. The character stage (tanh
# saturation by ``aggression``) is not a peak guarantee -- its ceiling is
# ``1/tanh(drive)``, which is above full scale for every drive below ~1.3 -- so
# a transparent limiter follows it. Below ``SAFETY_KNEE * SAFETY_CEILING`` the
# limiter is bit-exact, so ordinary material passes through it untouched and
# only genuine overshoot is shaped.
SAFETY_CEILING = 0.891
SAFETY_KNEE = 0.75

# The induction noise is now coloured by the intake's own resonances (and a
# falling slope) instead of a flat band-pass. Resonances are selective, so that
# costs about 4 dB of broadband energy; this makeup restores the level the
# presets were authored at, which keeps ``noise_gain`` meaning what it says
# instead of silently getting quieter when the colouring changed.
INTAKE_NOISE_MAKEUP = 1.6


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in air at a given temperature (exhaust gas approximation)."""
    return 331.3 * math.sqrt(1.0 + float(temp_c) / 273.15)


class BankChain:
    """One exhaust path: body resonances -> pipe comb -> muffler.

    The resonator and muffler stages run on the partitioned native convolver
    (``soundgen.native``): their cost per block is fixed no matter how long the
    kernel is, which is what makes the per-cylinder exhaust paths of stage 4
    affordable. The comb is a fractional delay line either way.
    """

    def __init__(self, res_kernel, comb, muf_kernel, block_size: int):
        self.res = native.Convolver(res_kernel, block_size) if np.size(res_kernel) else None
        self.comb = comb if isinstance(comb, native.Comb) else comb
        self.muf = native.Convolver(muf_kernel, block_size) if np.size(muf_kernel) else None

    def reset(self) -> None:
        if self.res is not None:
            self.res.reset()
        self.comb.reset()
        if self.muf is not None:
            self.muf.reset()

    def process(self, x: np.ndarray, delay_s: float) -> np.ndarray:
        y = self.res.process(x) if self.res is not None else x
        y = self.comb.process(y, delay_s)
        if self.muf is not None:
            y = self.muf.process(y)
        return y


class EngineVoice:
    """Real-time-capable synthesiser for one engine operating point."""

    def __init__(
        self,
        preset: Preset,
        rpm: float | None = None,
        load: float = 0.0,
        throttle: float = 0.0,
        sr: int | None = None,
        block: int | None = None,
        output_gain: float = 1.0,
        seed: int | None = None,
        clip: bool = True,
        quantize_period: bool = False,
        loop_samples: int | None = None,
    ):
        self.preset = preset
        self.sr = int(sr or preset.render.sample_rate)
        # Stage 6: the voice accepts any block >= 16, so the audio thread can
        # run at 64 frames (1.33 ms at 48 kHz). The old floor of 64 is kept as
        # the *default* only because offline renderers pass preset block sizes.
        self.block = max(16, int(block or preset.render.block_size))
        self.output_gain = float(output_gain)
        self.clip = bool(clip)
        # Offline loop rendering snaps the cycle period to a whole number of
        # samples so the event grid repeats exactly and the loop has no seam.
        self.quantize_period = bool(quantize_period)
        # When rendering a loop, noise sources become periodic with this length.
        self.loop_samples = int(loop_samples) if loop_samples else None
        self.rng = np.random.default_rng(int(seed if seed is not None else preset.seed))

        rpm = float(rpm if rpm is not None else preset.engine.idle_rpm)
        self._start_rpm = rpm
        self._target = {"rpm": rpm, "load": float(load), "throttle": float(throttle)}
        self._rpm = dsp.RampParam(rpm)
        self._load = dsp.RampParam(load)
        self._throttle = dsp.RampParam(throttle)
        self._outbuf = np.zeros(0, dtype=np.float64)
        # DSP grain: every convolver runs at this block_size and accepts only
        # whole multiples of it. It is the segment size when the block divides
        # into segments (so fast rpm sweeps can render in sub-blocks) and the
        # block itself otherwise. Short requests are rounded up to a grain and
        # the surplus is buffered -- see _process_plain().
        # DSP grain: every convolver runs at this block_size and accepts only
        # whole multiples of it. It is the segment size when the block divides
        # into segments (so fast rpm sweeps can render in sub-blocks) and the
        # block itself otherwise. Short requests are rounded up to a grain and
        # the surplus is buffered -- see _process_plain().
        #
        # No grain floor: see CONV_GRAIN -- a floor above the live block makes
        # the callback cost bimodal (render every second call), and the heavy
        # half of that alternation is what produced the periodic underrun.
        self._grain = int(self.SEGMENT if self.block % self.SEGMENT == 0 else self.block)
        self._prev_thr = float(throttle)
        self._bark = 0.0
        self._spool = 0.0
        self._boot = True
        # stereo split state: the per-bank exhaust signals for the block that
        # was just rendered, carried in buffer/slice form so they stay aligned
        # with the mono samples the caller is about to read.
        self._stereo_block = None
        self._bank_buf = None
        self._stereo_out = None
        self._want_stereo = False
        # crank lock: when the caller supplies the physics firing-cycle count
        # the event grid is slaved to it; otherwise it free-runs from rpm, which
        # is what the offline loop renderer and the calibration probes want.
        self._crank_locked = False
        self._crank_cycles_target = 0.0
        self._crank_cycles_last = None  # previous call's value, for interpolation
        self._phase_error = 0.0
        # Wiebe pulse coupling (stage 7 -> audio): when the caller supplies the
        # physics' instantaneous torque, every firing event's gain tracks it.
        # None = decoupled (loops, calibration); float = modulated.
        self._torque_pulse: float | None = None
        # Exhaust heat soak (audit item 3): the live driver publishes the
        # physics' gas temperature; the comb delay retunes with the speed of
        # sound in it. None = use the preset's static temperature.
        self._pipe_temp_c: float | None = None

        self._derive()
        self._build()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def set_params(
        self,
        rpm: float | None = None,
        load: float | None = None,
        throttle: float | None = None,
        crank_cycles: float | None = None,
        torque_pulse: float | None = None,
        pipe_temp_c: float | None = None,
        distance_m: float | None = None,
    ) -> None:
        """Update operating point. Changes are ramped over the next block.

        ``crank_cycles`` is the engine's *unwrapped* firing-cycle count (as
        ``EngineModel.firing_cycles`` reports it), interpolated across the block
        by the renderer. When supplied, the event grid is slaved to it instead
        of free-running from rpm, so the audio fires when a piston fires.

        An unwrapped counter, not a 0..1 phase: one 1024-sample block spans half
        a cycle at 3000 rpm and two and a half at 14500, so a wrapped phase
        cannot say which cycle it belongs to and the lock would alias.

        Free-running, the grid drifts against the physics: at a steady 3000 rpm
        a 0.1 % rpm error accumulates a whole firing period of phase error in
        about two seconds, and every transient -- a backfire, a limiter cut, a
        lift-off -- lands at an arbitrary point of the combustion cycle. Locked,
        audio and crank are the same clock, so those events sit on the cylinder
        that produced them.

        ``torque_pulse`` is the physics' *instantaneous* per-cylinder torque
        (``EngineModel.torque_envelope``) normalised to the engine's full-load
        estimate: 0 between firings, up to a few at the Wiebe peak. Non-None
        modulates every combustion event's gain with it, so the audio carries
        the same pulse train the crank does -- the audible "lug" of a loaded
        firing at low rpm. ``None`` leaves event amplitudes on the load/throttle
        schedule, which is what the loop renderer and the calibration probes
        need (a static, exactly periodic amplitude).

        ``pipe_temp_c`` retunes the pipe comb with the speed of sound in the
        heat-soaked exhaust gas; ``distance_m`` moves the listener (1 m is the
        preset's reference: direct path ~1/r, later ground bounce, darker
        air). Both ``None`` keep the preset's static values.
        """
        if rpm is not None:
            self._target["rpm"] = float(rpm)
        if load is not None:
            self._target["load"] = dsp.clamp(float(load), 0.0, 1.5)
        if throttle is not None:
            self._target["throttle"] = dsp.clamp(float(throttle), 0.0, 1.0)
        if crank_cycles is not None:
            self._crank_cycles_target = float(crank_cycles)
            self._crank_locked = True
        if torque_pulse is not None:
            self._torque_pulse = max(0.0, float(torque_pulse))
        if pipe_temp_c is not None:
            self._pipe_temp_c = float(pipe_temp_c)
        if distance_m is not None:
            self._distance_m = max(0.5, float(distance_m))

    @property
    def crank_locked(self) -> bool:
        """True once a crank cycle count has been supplied (grid follows physics)."""
        return bool(self._crank_locked)

    def reset(self) -> None:
        self._outbuf = np.zeros(0, dtype=np.float64)
        self._abs_cycles = 0.0
        self._abs_samples = 0
        self._prev_thr = 0.0
        self._bark = 0.0
        self._spool = 0.0
        self._boot = True
        for chain in self._bank_chains:
            chain.reset()
        for conv in self._flow_conv:
            conv.reset()
        for conv in self._flow_shape:
            conv.reset()
        self._intake_chain.reset()
        self._intake_noise.reset()
        self._mech.reset()
        self._hiss.reset()
        self._friction.reset()
        self._acc_lp.reset()
        self._tilt.reset()
        self._air.reset()
        self._dcblock.reset()
        if self._reverb is not None:
            self._reverb.reset()
        for src in self._noise_sources:
            src.reset()
        for src in self._noise_ccv:
            src.reset()
        for osc in (
            self._accessory_osc,
            self._supply_osc,
            self._chain_osc,
            self._chain_osc2,
            self._turbo_osc,
            self._turbo_osc2,
        ):
            osc.reset()
        for carry in self._exc_carry:
            carry[:] = 0.0
        self._int_carry[:] = 0.0
        self._mech_carry[:] = 0.0
        self._bottom_carry[:] = 0.0
        self._clatter.reset()
        self._dist_lp.reset()
        self._flow_rot = 0

    # ------------------------------------------------------------------ #
    # synthesis
    # ------------------------------------------------------------------ #
    # Sub-block segmentation: the event grid is sample-accurate inside a block,
    # but the *grid rate* (dphi) is one scalar per block. During a fast blip
    # (5000+ rpm/s) a 512-sample block spans ~55 rpm, and the firing pitch
    # moves in audible 55-rpm stairs. Rendering the block in segments of this
    # many samples (a power of two, and a divisor of every block we use)
    # re-evaluates the grid rate per segment. 128 would quarter the stair but
    # quadruples the FFT count; 256 halves the stair at 2x the FFT count and is
    # used *only* while rpm is moving. With a live block of 64 (stage 6) the
    # block is already finer than the segment, so segmentation simply never
    # triggers -- the small block is the segment.
    SEGMENT = 256
    SEGMENT_RPM_THRESHOLD = 250.0  # rpm change per block beyond which we split

    # Partitioned-convolver grain floor: the convolver grain follows the
    # live block (no floor). A 256 floor amortised the FFT cost into every
    # SECOND 128-frame callback -- a bimodal callback (measured 1577 us vs
    # 19 us alternating), and the heavy half ran close enough to the deadline
    # that scheduler jitter on the user's machine tipped it over *periodically*:
    # an underrun every 5.3 ms is exactly the "dial-up modem" texture. A
    # constant per-callback cost (grain = block) is slower on average but
    # never alternates, and nothing in a hard-RT callback is more important
    # than that. Offline renders at block >= 1024 are unaffected.
    CONV_GRAIN = 0

    # Wiebe pulse coupling floor: the minimum gain a firing keeps when the
    # physics publishes torque_pulse = 0 (between firings or a stale zero).
    # 0.35 keeps the note alive while still leaving a clearly audible swing.
    PULSE_FLOOR = 0.35

    def process(self, n: int) -> np.ndarray:
        """Return exactly ``n`` samples. Safe to call in an audio callback.

        When the target rpm has moved far since the last block, the block is
        rendered in segments (see SEGMENT) so the firing grid's *rate* is
        re-quantised several times per block instead of once. The event grid
        itself was always sample-accurate; this closes the remaining pitch
        stair on fast blips.
        """
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        use_segments = (
            abs(float(self._target["rpm"]) - float(self._rpm.value)) > self.SEGMENT_RPM_THRESHOLD
            and n >= self.SEGMENT * 2
            and n % self.SEGMENT == 0
        )
        if not use_segments:
            return self._process_plain(n)
        out = np.empty(n, dtype=np.float64)
        pos = 0
        seg = self.SEGMENT
        while pos < n:
            take = min(n - pos, seg)
            out[pos : pos + take] = self._process_plain(take)
            pos += take
        return out

    def _process_plain(self, n: int) -> np.ndarray:
        n = int(n)
        out = np.empty(n, dtype=np.float64)
        pos = 0
        while pos < n:
            if self._outbuf.size == 0:
                # Never render more than one configured block at a time: the
                # per-path flow scratch buffers are sized for it, so a bigger n
                # would overflow them. Requests larger than the block simply
                # loop; segments are always <= the block. Short requests are
                # rounded up to a whole grain (the convolvers only accept
                # multiples of it) and the surplus stays in _outbuf for the
                # next call.
                want = min(n - pos, self.block)
                grain = self._grain
                take_block = min(((want + grain - 1) // grain) * grain, self.block)
                self._outbuf = self._render_block(n=take_block)
                if self._stereo_block is not None:
                    self._bank_buf = [sig.copy() for sig in self._stereo_block]
                else:
                    self._bank_buf = None
            take = min(n - pos, self._outbuf.size)
            out[pos : pos + take] = self._outbuf[:take]
            # The per-bank split is consumed in lockstep with the mono buffer,
            # so a buffered surplus cannot make the two drift apart. A plain
            # process() call in between consumes mono only, which leaves the
            # split shorter: drop it rather than mixing misaligned banks, and
            # let process_stereo() fall back to centred mono for that block.
            if self._bank_buf is not None and self._bank_buf[0].size != self._outbuf.size:
                self._bank_buf = None
            if self._want_stereo and self._bank_buf is not None:
                self._stereo_out = [
                    np.concatenate((acc, sig[:take]))
                    for acc, sig in zip(self._stereo_out, self._bank_buf)
                ]
                self._bank_buf = [sig[take:] for sig in self._bank_buf]
            self._outbuf = self._outbuf[take:]
            pos += take
        return out

    def boost(self, rpm: float | None = None, load: float | None = None) -> float:
        """Estimated boost in bar at the current operating point."""
        if not self.preset.turbo.enabled:
            return 0.0
        rpm = self._target["rpm"] if rpm is None else float(rpm)
        load = self._target["load"] if load is None else float(load)
        t = self.preset.turbo
        x = dsp.clamp((rpm - t.spool_rpm) / max(t.spool_span_rpm, 1.0), 0.0, 1.0)
        return t.max_boost_bar * x * dsp.clamp(load, 0.0, 1.0)

    def _snap(self, freq: float) -> float:
        """Snap a frequency so a whole number of cycles fits one loop.

        A free-running oscillator would otherwise reach the loop point with a
        different phase than it started with, which is an audible click.
        """
        if not self.loop_samples:
            return float(freq)
        cycles = max(1, int(round(float(freq) * self.loop_samples / self.sr)))
        return cycles * self.sr / self.loop_samples

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _derive(self) -> None:
        p = self.preset
        b = p.block
        self.cyc_revs = layout.cycle_revolutions(b)
        self.is_turbine = b.layout == "turbine"
        self.events = layout.firing_events(b)
        self.n_cyl = len(self.events)
        self.n_banks = max((e[0] for e in self.events), default=0) + 1
        # Stage 4: the exhaust chain count. Per-cylinder paths everywhere except
        # for a turbine (no firing events, so every "cylinder" is the same noise
        # source and extra chains would only burn CPU for identical output).
        self.n_paths = 1 if self.is_turbine else self.n_cyl
        self.size = layout.size_factor(b)
        self._valve_on = b.layout in ("inline", "v", "boxer", "radial") and int(b.strokes) == 4
        self._abs_cycles = 0.0
        self._abs_samples = 0
        period = max(int(round(self.cyc_revs * 60.0 * self.sr / max(self._start_rpm, 1.0))), 2)
        # Even period: the injector impulse grid is on 2-sample spacing, so an odd
        # loop length would flip its parity after one loop and break the seam.
        self._period_samples = period + (period % 2)
        # events per cylinder per rendered loop -- the cycle-to-cycle variation
        # stream has to wrap at exactly this many draws to stay periodic
        if self.loop_samples:
            # The circular noise draws and the CCV buffer are only seamless when
            # the loop holds a whole number of firing periods. A caller passing
            # its own loop_samples could otherwise get a silent seam, so snap it
            # rather than render a loop that clicks.
            periods = max(int(round(self.loop_samples / self._period_samples)), 1)
            snapped = periods * self._period_samples
            if snapped != self.loop_samples:
                self.loop_samples = snapped
        self._ccv_k = max(self.loop_samples // self._period_samples, 1) if self.loop_samples else 0

        # deterministic per-cylinder amplitude character (keeps loops seamless)
        rng = np.random.default_rng(int(p.seed) * 7919 + 13)
        c = p.character
        self._cyl_amp = []
        for _ in range(self.n_cyl):
            trim = 1.0 + c.irregularity * float(rng.standard_normal())
            lump = 1.0 + c.lumpiness * float(np.clip(rng.standard_normal(), -1.0, 1.0))
            self._cyl_amp.append(dsp.clamp(trim * lump, 0.2, 2.0))
        # ... and a little timbre, not just level. Injector flow, compression and
        # valve sealing differ between cylinders, so each one burns a little
        # sharper or softer. Without this every firing is the identical waveform
        # and the note sounds stamped rather than made.
        self._cyl_sharp = [
            dsp.clamp(0.55 * c.irregularity * float(rng.standard_normal()), -0.18, 0.18)
            for _ in range(self.n_cyl)
        ]

        # two combustion pulse shapes, crossfaded by load, at three durations,
        # crossfaded by rpm. The decay is not a property of the engine alone: a
        # blowdown lasts a fixed crank angle, so the time it takes shortens as
        # revs rise. Without the duration axis a V12 at 8500 rpm gets the same
        # long soft tail as a big V8 at idle, and every fast engine smears into
        # the same continuous wash instead of cracking.
        size = self.size
        slow = 1.0 / max(size, 0.35)  # bigger cylinder -> slower, darker burn
        self._dur_scales = (1.0, 0.58, 0.32)
        # The hard pulse's near-discontinuous front (the "crack") is what a
        # heavily loaded cylinder sounds like on a big-bore single. On a
        # many-cylinder engine the fronts arrive every few milliseconds and
        # the crack train reads as grit -- the "maracas" failure. Physically
        # each cylinder still dumps the same charge, so with n cylinders
        # firing n times more often the *aggregate* crack energy must stay
        # constant, which thins each individual front as 1/sqrt(n) (the
        # incoherent-sum normalisation). A single keeps its full 0.85 bite;
        # a V8's fronts drop to 0.30 -- still cracking per event, but the
        # train no longer sands the whole note.
        front = 0.85 / math.sqrt(max(self.n_cyl, 1))
        self._pulse: list[tuple[np.ndarray, np.ndarray]] = []
        for ds in self._dur_scales:
            soft = dsp.combustion_pulse(
                self.sr,
                rise_s=0.00045 * slow * ds,
                decay_s=0.0068 * slow * ds,
                ring_hz=260.0 * size,
                ring_gain=0.34,
                noise_amount=0.16,
                front_gain=0.0,
                rng=np.random.default_rng(int(p.seed) + 5),
            )
            hard = dsp.combustion_pulse(
                self.sr,
                rise_s=0.00018 * slow * ds,
                decay_s=0.0042 * slow * ds,
                ring_hz=440.0 * size,
                ring_gain=0.50,
                noise_amount=0.30,
                front_gain=front,
                rng=np.random.default_rng(int(p.seed) + 6),
            )
            self._pulse.append((soft, hard))

    def _build(self) -> None:
        p = self.preset
        sr = self.sr
        e = p.exhaust
        i = p.intake
        m = p.mechanical
        c = p.character
        size = self.size
        bs = self.block
        slow = 1.0 / max(size, 0.35)

        # Convolvers run at a grain size: a convolver accepts any multiple of
        # its block_size, so one built at SEGMENT handles both the full block
        # (4x) and the segments used on fast rpm sweeps. Building at the full
        # block would forbid the smaller segments and reintroduce the pitch
        # stairs this whole path exists to remove. For a block that is not a
        # multiple of SEGMENT the grain is the block itself.
        cb = self._grain

        # --- per-cylinder combustion kernels (exhaust + intake runners)
        base_runner = e.runner_length_m if e.runner_length_m > 0.0 else 0.42 * e.pipe_length_m
        c_exh = speed_of_sound(e.pipe_temp_c)
        base_irun = i.runner_length_m if i.runner_length_m > 0.0 else 0.30
        c_int = i.runner_sound_speed
        rng = np.random.default_rng(int(p.seed) * 104729 + 7)
        self._exh_soft = []
        self._exh_hard = []
        self._k_exh = []
        self._k_flow = []
        self._int_soft = []
        self._int_hard = []
        self._k_int = []
        hre_bank = []
        for idx in range(self.n_cyl):
            le = max(base_runner * (1.0 + e.runner_spread * float(rng.uniform(-1.0, 1.0))), 0.03)
            hre = dsp.runner_kernel(sr, le, c_exh, e.runner_reflect, 2)
            li = max(base_irun * (1.0 + i.runner_spread * float(rng.uniform(-1.0, 1.0))), 0.03)
            hri = dsp.runner_kernel(sr, li, c_int, i.runner_reflect, 2)
            eh_s, eh_h, in_s, in_h = [], [], [], []
            for soft, hard in self._pulse:
                eh_s.append(np.convolve(soft, hre))
                eh_h.append(np.convolve(hard, hre))
                in_s.append(np.convolve(soft, hri))
                in_h.append(np.convolve(hard, hri))
            self._exh_soft.append(eh_s)
            self._exh_hard.append(eh_h)
            self._int_soft.append(in_s)
            self._int_hard.append(in_h)
            self._k_exh.append(np.zeros(max(k.size for k in eh_s + eh_h), dtype=np.float64))
            self._k_int.append(np.zeros(max(k.size for k in in_s + in_h), dtype=np.float64))
            # each cylinder gets its own turbulence burst and its own copy of
            # "what the runner does to it", so the roars are not one shared hiss
            fb = dsp.flow_burst(
                sr,
                attack_s=0.0005 * slow,
                decay_s=0.0026 * slow,
                lo_hz=e.flow_band_lo_hz,
                hi_hz=e.flow_band_hi_hz,
                rng=np.random.default_rng(int(p.seed) * 31337 + idx * 131),
            )
            self._k_flow.append(np.convolve(fb, hre))
            hre_bank.append(hre)
        # Audit item 4: statistical blowdown. A real exhaust never repeats the
        # same noise tail twice; the single baked burst made every firing of a
        # cylinder the identical waveform. A small bank of per-cylinder burst
        # *variants* rotates by event index, so consecutive firings differ in
        # their noise component while the deterministic pulse body (the part
        # that carries the firing order) stays put. In loop mode the bank is
        # sized to the loop's event count, so the sequence still closes
        # exactly at the loop point -- the seam contract is preserved.
        flow_variants = 3 if not self.loop_samples else max(1, self._ccv_k or 1)
        self._k_flow_bank = []
        for idx in range(self.n_cyl):
            row = [self._k_flow[idx]]
            for v in range(1, flow_variants):
                fbv = dsp.flow_burst(
                    sr,
                    attack_s=0.0005 * slow,
                    decay_s=0.0026 * slow,
                    lo_hz=e.flow_band_lo_hz,
                    hi_hz=e.flow_band_hi_hz,
                    rng=np.random.default_rng(
                        int(p.seed) * 31337 + idx * 131 + 977 * v
                    ),
                )
                row.append(np.convolve(fbv, hre_bank[idx]))
            self._k_flow_bank.append(row)
        self._flow_rot = 0
        # Keep the loop path deterministic: a rotated bank only closes at the
        # loop point when the bank length divides the loop's firing count.
        # _ccv_k is exactly that count for loop renders, so loop mode gets a
        # bank of that size and free-running mode gets the default 3.

        # --- exhaust: one full chain per cylinder (stage 4) --------------- #
        # Every cylinder gets its own resonator bank, pipe comb and muffler.
        # A V8 previously shared two chains, so all four cylinders of a bank
        # were filtered by one transfer function and only their *timing*
        # differed. The runner kernels above shape each pulse separately, but
        # everything downstream of the collector was common. Per-cylinder paths
        # put each pulse through its own pipe: the interference pattern that
        # makes a real header sound different from a synthesised pulse train
        # now exists in the acoustics, not only in the event grid.
        #
        # Cost: n_paths chains instead of n_banks, i.e. up to 8x more
        # convolutions -- the ~2.3x more FFT work noted in the roadmap. That
        # is affordable only on the partitioned native convolver, whose cost per
        # block does not grow with kernel length; this is why stage 4 was
        # blocked on stage 5.
        self._bank_chains = []
        self._bank_delay = []
        # Summing n_paths pipes coherently at their shared resonances
        # (the body modes are one bank of resonators per preset) would lift
        # the mode peaks by sqrt(n_paths) relative to the pulse train: that is
        # the "too deep, too resonant" colour the per-cylinder refactor
        # shipped with. Energy-normalise the sum instead: 1/n_paths keeps the
        # *total* excitation gain constant across engine sizes, so a V12 is
        # not twice a V6 in the body of the note. (The old per-bank code had
        # 1/sqrt(2) on two chains; the coherent-mode error is the sqrt factor
        # that crept in with 8+ chains.)
        self._bank_gain = [1.0 / self.n_paths] * self.n_paths
        # ...but per-path pulses remain incoherent between pipes, so make the
        # *broadband* layer up with sqrt(n_paths): applied where the per-path
        # noise and pulses are actually uncorrelated (the flow layer below),
        # not at the chain output where the resonator modes pile up coherently.
        self._flow_conv = []
        rng_banks = np.random.default_rng(int(p.seed) * 15485863 + 29)
        for path in range(self.n_paths):
            bank = self.events[path][0] if not self.is_turbine else 0
            asym = 1.0 + (e.bank_asymmetry if bank == 1 else 0.0)
            # per-cylinder pipe length jitter: two cylinders never share an
            # exact acoustic length, which decorrelates their standing waves
            asym *= 1.0 + e.runner_spread * 0.35 * float(rng_banks.uniform(-1.0, 1.0))
            delay = 2.0 * e.pipe_length_m * asym / speed_of_sound(e.pipe_temp_c)
            if self.is_turbine:
                # A turbine has no firing events, so its spectrum is the shape
                # of the noise itself: continuous, rpm-driven broadband noise.
                res = dsp.bandpass_kernel(sr, 90.0, 4200.0, 2, 257)
                muf_cut = e.muffler_cutoff_hz
            else:
                modes = [(f * size, q, g) for (f, q, g) in e.modes]
                res = dsp.resonator_kernel(sr, modes, p.render.resonator_length_s)
                muf_cut = e.muffler_cutoff_hz * size
            muf = dsp.lowpass_kernel(sr, muf_cut, 2, 257)
            comb = native.Comb(
                sr, delay * 1.4 + 0.002, int(e.comb_repeats), e.comb_gain, e.comb_invert, cb
            )
            self._bank_chains.append(BankChain(res, comb, muf, cb))
            self._bank_delay.append(delay)
            self._flow_conv.append(
                native.Convolver(
                    dsp.bandpass_kernel(sr, e.flow_band_lo_hz, e.flow_band_hi_hz, 2, 257), cb
                )
            )
        # pre-chain flow source: broad, gently bounded, then coloured by the pipe.
        # ONE convolver PER PATH: a convolver is stateful, so a shared instance
        # stepped once per path would interleave eight noise streams through one
        # overlap-add history and output garbage -- that was exactly the
        # mid-high buzz the per-cylinder refactor shipped with.
        self._flow_shape = [
            native.Convolver(dsp.bandpass_kernel(sr, 70.0, 6000.0, 1, 257), cb)
            for _ in range(self.n_paths)
        ]

        # --- intake
        i_modes = [(f * size, q, g) for (f, q, g) in i.modes]
        h_i_res = dsp.resonator_kernel(sr, i_modes, p.render.resonator_length_s * 0.6)
        h_i_lp = dsp.lowpass_kernel(sr, i.band_hi_hz, 2, 257)
        # Both stages are static, so bake them into one kernel: a chain of two
        # convolvers is two FFTs per block, and the FFT is what the whole voice
        # spends most of its time in.
        self._intake_chain = native.Convolver(np.convolve(h_i_res, h_i_lp), cb)
        # The induction noise is the same air as the pulses, so it is coloured
        # by the same resonances. It is also turbulent, not white: the kernel
        # carries a falling slope (``noise_tilt_db_oct``) so the top octaves
        # drop away instead of forming a flat hiss shelf. Both filters are baked
        # into one kernel, so this stays a single convolution per block.
        h_i_noise = dsp.spectral_kernel(
            sr,
            lambda f: dsp.mag_bandpass(f, i.band_lo_hz, i.band_hi_hz, 2)
            * dsp.mag_tilt_oct(f, i.noise_tilt_hz, i.noise_tilt_db_oct),
            385,
        )
        self._intake_noise = native.Convolver(np.convolve(h_i_noise, h_i_res), cb)

        # --- mechanical: cam-timed clicks through a bright resonant tick
        rng_m = np.random.default_rng(int(p.seed) * 7919 + 3)
        h_valve = dsp.tick_kernel(
            sr, m.valvetrain_bright_hz * size, 0.0026, noise_amount=0.55, rng=rng_m
        )
        h_valve_open = dsp.tick_kernel(
            sr, m.valvetrain_bright_hz * size * 0.90, 0.0019, noise_amount=0.62, rng=rng_m
        )
        h_inj = dsp.tick_kernel(
            sr, m.valvetrain_bright_hz * 1.45 * size, 0.0011, noise_amount=0.72, rng=rng_m
        )
        # A click has to stay a click: the high-pass takes out the thump that
        # would only smear it, and the gentle top-end roll-off keeps the noise
        # burst from reading as hiss. Short kernels on purpose -- a 1025-tap
        # linear-phase filter delays the click by 10 ms and pre-rings for as
        # long, which is exactly what makes a tick sound synthetic.
        h_tick_hp = dsp.highpass_kernel(sr, 420.0, 2, 257)
        h_tick_lp = dsp.lowpass_kernel(sr, min(8000.0, sr * 0.45), 2, 257)
        self._valve_tick = h_valve
        self._valve_open_tick = h_valve_open
        self._inj_tick = h_inj
        # one baked kernel rather than two convolvers (see the intake chain)
        self._mech = native.Convolver(np.convolve(h_tick_hp, h_tick_lp), cb)
        self._mech_len = max(h_valve.size, h_valve_open.size, h_inj.size)
        # Close-mic clatter (audit item 6, retuned after the maracas report):
        # a first-order, *gentle* lift around the head's resonance. The first
        # cut used order 3 at 129 taps -- steep edges ring at the band edges
        # and every tick came out as a metallic "chik", which at 500
        # clicks/s on a V8 reads as a shaker. The fix is physical: the head
        # is a wide, low-Q resonator, not a bell -- order 1, no edge
        # steepness, and the gain divided by sqrt(cylinders) so the *summed*
        # click layer stays constant as the engine grows (8 cylinders fire 4
        # valve events each per cycle; a flat per-cylinder gain scales the
        # layer with cylinder count, which is exactly what happened).
        clatter_lo = min(m.valvetrain_bright_hz * 0.80, 2500.0)
        clatter_hi = min(max(m.valvetrain_bright_hz * 1.60, clatter_lo * 1.5), sr * 0.42)
        self._clatter_scale = 0.55 / math.sqrt(max(self.n_cyl, 1))
        self._clatter = native.Convolver(
            dsp.bandpass_kernel(sr, clatter_lo, clatter_hi, 1, 129), cb
        )
        # Fixed per-cylinder scatter, so the clicks neither stack into one
        # waveform nor sit exactly on the exhaust pulse.
        rng_s = np.random.default_rng(int(p.seed) * 104729 + 17)
        scatter = max(float(m.mech_scatter_ms), 0.0) / 1000.0
        self._cyl_mech_off = rng_s.uniform(-scatter, scatter, size=self.n_cyl)
        self._accessory_osc = dsp.PhaseOsc(sr)
        self._supply_osc = dsp.PhaseOsc(sr)
        # Band-limited, downward-tilted: the mechanical floor is noise, but it
        # is not white noise.
        self._hiss = native.Convolver(
            dsp.hiss_kernel(sr, m.hiss_lo_hz, m.hiss_hi_hz, m.hiss_tilt_db, 513), cb
        )
        self._acc_lp = native.Convolver(dsp.lowpass_kernel(sr, 2400.0, 2, 257), cb)

        # --- rotating assembly ------------------------------------------- #
        # A firing load lands in the main bearings and the rod big end; the
        # piston then rocks over at TDC and slaps the bore. Both are short, low
        # impacts. They are injected as *events* on the same grid as combustion,
        # so the bottom end tracks the firing order instead of being a static
        # rumble.
        rng_b = np.random.default_rng(int(p.seed) * 15485863 + 11)
        self._crank_tick = dsp.impact_kernel(
            sr,
            m.crank_hz * size,
            0.0095 * slow,
            attack_s=0.0006,
            noise_amount=0.12,
            rng=rng_b,
        )
        self._slap_tick = dsp.impact_kernel(
            sr,
            m.slap_hz * size,
            0.0055 * slow,
            attack_s=0.00035,
            noise_amount=0.08,
            rng=rng_b,
        )
        self._bottom_len = max(self._crank_tick.size, self._slap_tick.size)
        # Oil / bearing / ring shear: continuous, low, and never allowed to
        # become hiss. It is the "wet" part of the mechanical bed.
        self._friction = native.Convolver(
            dsp.bandpass_kernel(sr, m.friction_lo_hz, m.friction_hi_hz, 2, 257), cb
        )
        self._chain_osc = dsp.PhaseOsc(sr)
        self._chain_osc2 = dsp.PhaseOsc(sr)

        # --- tone shaping
        self._tilt = native.Convolver(
            dsp.presence_kernel(
                sr, c.presence_hz, c.brightness_db, c.presence_q, p.render.kernel_taps
            ),
            cb,
        )
        self._air = native.Convolver(
            dsp.air_kernel(sr, c.air_cutoff_hz, c.air_order, 0.0, p.render.kernel_taps), cb
        )
        self._dcblock = native.Convolver(dsp.dc_block_kernel(sr, 22.0, 0.20), cb)

        # --- space: outdoor convolution reverb --------------------------- #
        # Audit item 7: the space is now parameterised by distance. A listener
        # at ``distance_m`` hears (a) the ground bounce arrive r/c later than
        # the direct sound -- predelay scales with distance; (b) the direct
        # level fall as 1/r -- the engine as a whole is scaled at the output,
        # not the reverb, so the wet/dry balance stays what the preset says;
        # (c) the air swallow the high end on the way -- the damping falls as
        # the wavefront travels, order 1 in r. The preset's own numbers are
        # the 1 m reference; the live driver can move the listener at any
        # time through set_params(distance_m=...).
        sp = p.space
        if sp.enabled and sp.mix > 0.0:
            ir = dsp.outdoor_ir(
                sr,
                decay_s=sp.decay_s,
                predelay_ms=sp.predelay_ms,
                damping_hz=sp.damping_hz,
                early_gain=sp.early_gain,
                tail_gain=sp.tail_gain,
                seed=int(p.seed) + int(sp.seed),
            )
            self._reverb = native.Convolver(ir, cb)
        else:
            self._reverb = None
        self._rev_mix = dsp.clamp(float(sp.mix), 0.0, 0.95)
        # distance state (audit item 7): 1 m is the reference; None = the
        # preset's static space, no runtime retuning
        self._distance_m: float | None = None
        self._ref_predelay_ms = float(sp.predelay_ms)
        # one-pole low-pass reused for the distance air absorption
        self._dist_lp = native.Convolver(dsp.lowpass_kernel(sr, 1000.0, 1, 129), cb)

        # --- turbo
        self._turbo_osc = dsp.PhaseOsc(sr)
        self._turbo_osc2 = dsp.PhaseOsc(sr)
        self._whoosh = native.Convolver(
            dsp.bandpass_kernel(sr, 1200.0, 7000.0, 2, 385), cb
        )

        # --- noise sources (one sample per output sample)
        self._n_intake = self._make_noise()
        self._n_mech = self._make_noise()
        self._n_friction = self._make_noise()
        self._n_turbo = self._make_noise()
        self._n_turbine = self._make_noise()
        self._n_exhflow = [self._make_noise() for _ in range(self.n_paths)]
        self._noise_sources = [
            self._n_intake,
            self._n_mech,
            self._n_friction,
            self._n_turbo,
            self._n_turbine,
            *self._n_exhflow,
        ]
        # --- cycle-to-cycle variation: exactly one draw per firing event
        self._noise_ccv = [self._make_ccv() for _ in range(self.n_cyl)]

        # --- injection scratch buffers
        self._pad = max(
            max(k.size for k in self._k_exh),
            max(k.size for k in self._k_int),
            max(k.size for k in self._k_flow),
        ) + 2
        self._mech_pad = self._mech_len + 2
        self._bottom_pad = self._bottom_len + 2
        # Scratch spans are sized for the render grain: a short request is
        # rounded up to a whole grain before rendering (the convolvers only
        # accept multiples of it), so the buffers must hold one grain.
        span = ((bs + self._grain - 1) // self._grain) * self._grain
        self._exc_buf = [np.zeros(span + self._pad, dtype=np.float64) for _ in range(self.n_paths)]
        self._exc_carry = [np.zeros(self._pad, dtype=np.float64) for _ in range(self.n_paths)]
        self._int_buf = np.zeros(span + self._pad, dtype=np.float64)
        self._int_carry = np.zeros(self._pad, dtype=np.float64)
        self._mech_buf = np.zeros(span + self._mech_pad, dtype=np.float64)
        self._mech_carry = np.zeros(self._mech_pad, dtype=np.float64)
        self._bottom_buf = np.zeros(span + self._bottom_pad, dtype=np.float64)
        self._bottom_carry = np.zeros(self._bottom_pad, dtype=np.float64)
        # The exhaust flow noise is drawn exactly once per block and then used
        # twice (once inside the pipe, once off its mouth).  Drawing it twice
        # would desynchronise the loop-periodic noise stream.
        self._flow_blk = [np.zeros(span, dtype=np.float64) for _ in range(self.n_paths)]
        self._mix_k = [
            np.zeros(self._pad, dtype=np.float64) for _ in range(self.n_cyl)
        ]  # scratch for kernel crossfade (exhaust)
        self._mix_ki = [
            np.zeros(self._pad, dtype=np.float64) for _ in range(self.n_cyl)
        ]  # scratch for kernel crossfade (intake)
        self._blend_key: tuple | None = None

    def _make_noise(self):
        """Create a noise source, periodic over the loop when one is being rendered."""
        seed = int(self.rng.integers(1, 2**31 - 1))
        if self.loop_samples:
            return dsp.CircularNoise(self.loop_samples, seed)
        return dsp.NoiseGen(seed)

    def _make_ccv(self):
        """One sample per firing event, wrapping at the event count in a loop."""
        seed = int(self.rng.integers(1, 2**31 - 1))
        if self.loop_samples and self._ccv_k:
            return dsp.CircularNoise(self._ccv_k, seed)
        return dsp.NoiseGen(seed)

    # ------------------------------------------------------------------ #
    # synthesis
    # ------------------------------------------------------------------ #
    def _duration_weights(self, dur: float) -> tuple[int, int, float]:
        """Bracket ``dur`` between two of the prebuilt pulse durations."""
        sc = self._dur_scales
        if dur >= sc[0]:
            return 0, 0, 0.0
        for j in range(len(sc) - 1):
            if sc[j] >= dur >= sc[j + 1]:
                span = max(sc[j] - sc[j + 1], 1e-9)
                return j, j + 1, (sc[j] - dur) / span
        return len(sc) - 1, len(sc) - 1, 0.0

    def _blend_kernels(self, sharp: float, dur: float = 1.0) -> None:
        """Crossfade the per-cylinder pulses over load *and* over rpm.

        Load picks the shape (round and dark .. sharp and cracked); rpm picks the
        duration, because a blowdown lasts a fixed crank angle and therefore
        less wall-clock time the faster the engine turns.
        """
        d0, d1, wd = self._duration_weights(dur)
        key = (round(float(sharp), 3), d0, d1, round(float(wd), 3))
        if key == self._blend_key:
            return
        self._blend_key = key
        pairs = ((d0, 1.0 - wd), (d1, wd))
        for idx in range(self.n_cyl):
            # each cylinder's own load-character offset, so the firing order is
            # not the same pulse stamped out over and over
            w = dsp.clamp(float(sharp) + self._cyl_sharp[idx], 0.0, 1.0)
            inv = 1.0 - w
            k = self._mix_k[idx]
            k[:] = 0.0
            ki = self._mix_ki[idx]
            ki[:] = 0.0
            for dd, wdd in pairs:
                if wdd <= 0.0:
                    continue
                lo = inv * wdd
                hi = w * wdd
                a = self._exh_soft[idx][dd]
                k[: a.size] += lo * a
                b = self._exh_hard[idx][dd]
                k[: b.size] += hi * b
                ai = self._int_soft[idx][dd]
                ki[: ai.size] += lo * ai
                bi = self._int_hard[idx][dd]
                ki[: bi.size] += hi * bi
            self._k_exh[idx] = k
            self._k_int[idx] = ki

    def _render_block(self, n: int | None = None) -> np.ndarray:
        # Segmented calls hand us a smaller slice (a divisor of self.block);
        # every scratch buffer is sized for the full block and the convolvers
        # accept any multiple of their block_size, so this is safe.
        n = int(n) if n is not None else self.block
        p = self.preset
        c = p.character
        e = p.exhaust
        m = p.mechanical

        rpm = self._rpm.block(self._target["rpm"], n)
        load = self._load.block(self._target["load"], n)
        thr = self._throttle.block(self._target["throttle"], n)
        rpm_now = float(rpm[-1])
        load_now = dsp.clamp(float(load[-1]), 0.0, 1.0)
        thr_now = float(thr[-1])
        redline = max(p.engine.redline_rpm, 1.0)
        rel = dsp.clamp(rpm_now / redline, 0.0, 1.2)

        # Audit item 5: the idle floor must sag with revs. A constant
        # ``idle_floor`` keeps the firing events as loud at 5000 rpm overrun
        # as at idle, which over-fills the note exactly where a real engine
        # goes quiet (high vacuum, tiny charge). Fade the floor out through
        # the mid range so the overrun keeps the *shape* of a firing (the
        # order is still there) at a level that decays like the charge does.
        base = c.idle_floor * dsp.clamp(1.35 - 1.1 * rel, 0.25, 1.0)
        level = c.combustion_gain * (base + (1.0 - base) * (0.6 * load_now + 0.4 * thr_now))
        hot = dsp.clamp(0.72 * load_now + 0.28 * thr_now, 0.0, 1.0)
        flow = rel * (0.10 + 0.90 * load_now)
        sharp = dsp.clamp(
            c.load_sharpening * (0.70 * load_now + 0.30 * thr_now) + 0.22 * rel, 0.0, 1.0
        )
        # AM depth at the firing order (audit item 2): the envelope swing of
        # the pulse train -- what makes a V8 *sound* like a V8 -- is set by how
        # much louder a firing is than the bed under it. Boosting the events
        # relative to the continuous layers (flow noise, hiss) raises the
        # firing-order AM directly: the events carry the order, the bed is
        # flat. Mild range on purpose: +2.3 dB at WOT, -2.5 dB at closed
        # throttle. Cutting the idle events harder would starve the idle
        # lope -- a closed-throttle firing still has bite on a real engine --
        # and the LUFS calibration drags the whole bed back up anyway.
        # Measured: flat-plane idle AM -11.5 -> -5.7 dB, cross-plane WOT
        # -4.9 -> -1.6 dB.
        duty = 0.6 * load_now + 0.4 * thr_now
        level *= 0.75 + 0.55 * duty**1.5

        # --- tip-in enrichment: a short broadband bark on a fast throttle rise
        if self._boot:
            self._bark = 0.0
            self._boot = False
        else:
            dthr = thr_now - self._prev_thr
            if dthr > 0.02:
                self._bark = min(1.0, self._bark + 1.7 * dthr)
        self._prev_thr = thr_now
        self._bark *= math.exp(-n / max(0.09 * self.sr, 1.0))
        bark = c.transient_bark * self._bark

        # A blowdown occupies a fixed crank angle, so the same pulse gets
        # shorter in wall-clock time the faster the engine turns. The duration
        # scale is bracketed between the prebuilt pulse variants.
        dur = dsp.clamp(c.pulse_ref_rpm / max(rpm_now, 1.0), c.pulse_min_scale, 1.0)

        self._blend_kernels(sharp, dur)
        excs, intake_exc, mech_exc, bottom_exc = self._render_events(
            n, float(np.mean(rpm)), level, flow, load_now
        )

        # ---- exhaust
        # The per-path processed signals are kept: the stereo path pans them,
        # the mono path sums them. Same DSP, one branch at the end.
        bank_out = []
        exhaust = np.zeros(n, dtype=np.float64)
        # Pipe heat soak (audit item 3): the caller (live driver) may publish
        # the physics' exhaust gas temperature. Hotter gas -> faster sound ->
        # *shorter* comb delay -> the quarter-wave resonance climbs. The
        # preset's own temp still decides the baked runner kernels; this only
        # retunes the time-varying pipe, which is where the drift lives.
        c_exh = speed_of_sound(
            self._pipe_temp_c if self._pipe_temp_c is not None else e.pipe_temp_c
        )
        c_ref = speed_of_sound(e.pipe_temp_c)
        for bank, exc in enumerate(excs):
            delay = self._bank_delay[bank] / math.sqrt(1.0 + e.formant_shift * hot)
            delay *= c_ref / c_exh
            sig = self._bank_gain[bank] * self._bank_chains[bank].process(exc, delay)
            bank_out.append(sig)
            exhaust += sig

        # tip-in bark and the tailpipe shear-layer roar are two different gains
        # on the *same* filtered noise, so filter each bank once and scale the
        # result. Calling the convolver twice with the same block would advance
        # its overlap-add state twice and mix two different pieces of history
        # into one block, which is both wrong and audible as a smeared bark.
        bark_gain = 0.9 * bark * c.combustion_gain if bark > 0.0 else 0.0
        tail_gain = e.tail_noise * flow * c.combustion_gain if flow > 0.0 else 0.0
        if bark_gain > 0.0 or tail_gain > 0.0:
            for bank in range(self.n_paths):
                shaped = self._flow_conv[bank].process(self._flow_blk[bank][:n])
                sig = self._bank_gain[bank] * (bark_gain + tail_gain) * shaped
                bank_out[bank] = bank_out[bank] + sig
                exhaust += sig

        # ---- intake
        intake = self._intake_chain.process(intake_exc)
        intake *= p.intake.pulse_gain * (0.08 + 0.92 * thr_now)
        intake += self._intake_noise.process(self._n_intake.render(n)) * (
            INTAKE_NOISE_MAKEUP
            * p.intake.noise_gain
            * (0.10 + 0.90 * thr_now)
            * (0.30 + 0.70 * load_now)
            * (0.45 + 0.55 * rel)
        )

        # ---- mechanical
        mech = self._mech.process(mech_exc)
        # The bottom end is summed *after* the bright tick high-pass, because a
        # main bearing and a piston slap are low-frequency events.
        mech += bottom_exc
        # Close-mic clatter (audit item 6, retuned): a gentle first-order
        # lift of the head's resonance. Never routed through the exhaust
        # filters, so it stays bright at close range where the exhaust is
        # darkest. Gain is divided by sqrt(cylinders) at build time
        # (``_clatter_scale``) so the summed layer does not grow with the
        # cylinder count -- the maracas regression was exactly that growth.
        clatter = self._clatter.process(mech_exc)
        mech += clatter * self._clatter_scale * (
            m.valvetrain_gain * (0.30 + 0.70 * rel) * (0.25 + 0.75 * load_now)
        )
        mech += self._friction.process(self._n_friction.render(n)) * (
            m.friction_gain * (0.30 + 0.70 * rel)
        )
        mech += self._hiss.process(self._n_mech.render(n)) * (m.hiss_gain * (0.35 + 0.65 * rel))

        # ---- accessories (alternator / pump whine, rises with rpm)
        # A whine is a tone, but a lone sine floating above the engine's own
        # broadband floor stops sounding like machinery and starts sounding like
        # a whistle. So: keep it quiet, keep the ripple harmonic close to the
        # fundamental (the pair then reads as one whir instead of a beep), and
        # band-limit the pair so a high-revving whine cannot run up to 5 kHz.
        acc = m.accessory_gain * (0.35 + 0.65 * rel)
        acc_ratio = max(rpm_now / max(p.engine.idle_rpm, 1.0), 0.3)
        acc_f = self._snap(m.accessory_hz * acc_ratio)
        accessory = self._accessory_osc.render(n, acc_f) * acc
        accessory += self._supply_osc.render(n, self._snap(acc_f * 2.7)) * acc * 0.18
        # Timing chain / cam drive turns at half crank speed and adds one
        # harmonic. It lives in the same band-limited accessory cluster, so a
        # high-revving chain cannot turn into a whistle either.
        chain_f = self._snap(m.chain_hz * acc_ratio)
        chain = self._chain_osc.render(n, chain_f) + 0.42 * self._chain_osc2.render(
            n, self._snap(chain_f * 2.0)
        )
        accessory += chain * (m.chain_gain * (0.25 + 0.75 * rel))
        accessory = self._acc_lp.process(accessory)

        # ---- turbo (spools up faster than it spools down)
        turbo = np.zeros(n)
        if p.turbo.enabled:
            t = p.turbo
            x = dsp.clamp((rpm_now - t.spool_rpm) / max(t.spool_span_rpm, 1.0), 0.0, 1.0)
            target = x * dsp.clamp(load_now, 0.0, 1.0)
            if t.kind == "supercharger":
                target = dsp.clamp(rel, 0.0, 1.2) * (0.3 + 0.7 * thr_now)
            tau = 0.35 if target > self._spool else 1.0
            self._spool += (target - self._spool) * (1.0 - math.exp(-n / max(tau * self.sr, 1.0)))
            spool = dsp.clamp(self._spool, 0.0, 1.0)
            f0 = self._snap(t.whistle_hz + t.whistle_span_hz * spool)
            # Two close tones rather than one: a single sine reads as a test
            # tone, a blade-pass pair reads as a turbo.
            whistle = self._turbo_osc.render(n, f0) + 0.45 * self._turbo_osc2.render(n, self._snap(f0 * 1.5))
            # A piston engine's turbo whistles only while it is actually
            # spinning: keeping 10 % of the tone alive with the turbo asleep left
            # a pure 2.2 kHz sine sitting 50 dB above the engine's own spectrum
            # at idle -- exactly the "high frequency that stands out" this is
            # meant to remove. A gas turbine is the opposite case, so it declares
            # how much of its whine survives at idle, and the rushing noise it
            # sits on is scaled the same way so the tone is never alone.
            w = t.idle_whistle + (1.0 - t.idle_whistle) * spool**1.2
            whistle *= t.whistle_gain * w
            if t.flutter:
                tt = (self._abs_samples + np.arange(n, dtype=np.float64)) / self.sr
                whistle *= 0.68 + 0.32 * np.sin(dsp.TWO_PI * self._snap(13.0) * tt) ** 2
            whoosh = self._whoosh.process(self._n_turbo.render(n)) * t.whoosh_gain * w
            turbo = whistle + whoosh

        mix = exhaust + intake + mech + accessory + turbo
        # Audit item 7: distance. Farther away = quieter direct path (1/r),
        # a later ground bounce (extra predelay r/c) and a darker air
        # (extra high-end loss ~ r-dependent order-1 low-pass). All three are
        # static per call -- the parameters ramp in set_params -- so the
        # offline loop contract (deterministic, periodic) is untouched: the
        # loop renderer never supplies a distance.
        dist_scale = 1.0
        if self._distance_m is not None:
            d = max(self._distance_m, 0.5)
            dist_scale = 1.0 / d
            # the reverberant share grows with distance: the direct path
            # falls as 1/r, the ground and surfaces do not
            self._rev_mix = dsp.clamp(
                float(self.preset.space.mix) * min(1.0, 0.45 + 0.28 * d), 0.0, 0.95
            )
        mix = self._tilt.process(mix)
        mix = self._air.process(mix)
        if self._distance_m is not None:
            # air absorption with distance: a first-order complement whose
            # blend rises with r, cutting the top end by up to ~-8 dB at 30 m
            # (12 kHz carries ~ -1 dB/10 m in humid air; the ear hears the
            # *cumulative* darkening, which is what this one-pole models)
            r = max(self._distance_m, 0.5)
            blend = max(0.0, min(0.55, 0.018 * r))
            if blend > 0.0:
                mix = mix - blend * self._dist_lp.process(mix)
        mix = self._dcblock.process(mix)
        # The room is the last linear stage: equal-power wet/dry, applied to a
        # signal whose direct path stays untouched. Being LTI, it keeps the
        # offline loops exactly periodic once its own tail has been warmed up.
        if self._reverb is not None:
            wet = self._reverb.process(mix)
            a = self._rev_mix * (math.pi * 0.5)
            mix = math.cos(a) * mix + math.sin(a) * wet
        if dist_scale != 1.0:
            mix *= dist_scale
        # Gain is calibrated (offline, once per preset) so a strong operating
        # point lands near full scale. Saturation is therefore applied *after*
        # the gain, which makes it a gentle, level-dependent character control
        # instead of a hard clipper.
        mix = mix * self.output_gain
        if self.clip:
            # Resonance compressor (audio audit): the pipe's mode envelope is
            # 6-8 dB tall, and the firing order slides under it with rpm, so
            # the "resonance rpms" run that much hotter than the sweep the
            # gain was calibrated on. Feeding that straight into the safety
            # limiter crushed them (measured: 45 % of samples in the knee at
            # 7325 rpm on the V8, crest 2.8 dB -- the "squashed" character).
            # A 2:1 soft stage above -9 dBFS takes the resonance overshoot
            # gracefully, keeps the crest alive (measured 4.9 dB after) and
            # leaves the limiter as the *safety* it is meant to be: below the
            # compressor's threshold it is bit-exact, like the limiter
            # below its knee.
            mix = dsp.resonance_compress(mix)
            mix = dsp.soft_clip(mix, c.aggression)
            # The character stage is not a guarantee: tanh saturates at
            # 1/tanh(drive) ~ 1.03..1.31, i.e. it can leave the signal above
            # full scale, and for a gentle drive it barely touches the peaks at
            # all. This second, transparent stage is what actually keeps the
            # output inside +-0.891 (-1 dBFS): it is bit-exact below the knee,
            # so the note itself is untouched and only genuine overshoot is
            # bent. Without it, calm presets and 14 dB-crest presets land 10 dB
            # apart in loudness, because only the peaks ever set the gain.
            mix = dsp.safety_limit(mix, SAFETY_CEILING, SAFETY_KNEE)
        # keep the path split for process_stereo(). The tone chain and reverb
        # are applied to the mono sum, so the paths here are *pre-shaping*;
        # stereo reconstructs them with mid/side: mid = mono, side from the
        # bank sums. The split is stored for exactly the samples this call
        # rendered and is carried through _process_plain() in step with the
        # mono buffer, so process_stereo() never has to guess how many samples
        # it is owed.
        if self.n_banks >= 2:
            # collapse the per-cylinder paths back onto the physical banks:
            # side content is which side of the engine the pipe exits, not which
            # cylinder it belongs to
            bank_sum = []
            for bank in range(self.n_banks):
                acc = np.zeros(n, dtype=np.float64)
                for path, (b, _off) in enumerate(self.events[: self.n_paths]):
                    if b == bank:
                        acc += bank_out[path]
                bank_sum.append(acc)
            self._stereo_block = [
                np.asarray(sig, dtype=np.float64)[:n] * self.output_gain for sig in bank_sum
            ]
        else:
            self._stereo_block = None
        return mix

    # ------------------------------------------------------------------ #
    # stereo
    # ------------------------------------------------------------------ #
    def process_stereo(self, n: int) -> np.ndarray:
        """Return ``(n, 2)``: exhaust banks panned, everything else centred.

        The exhaust is the only layer with real physical left/right content:
        bank 0 of a V engine is one side of the car, bank 1 the other. The
        pan law is constant-power with a fixed crossfeed, so a single-bank
        engine (an inline four) collapses back to mono and a V8 keeps its
        two-pipe width. Mechanical, intake and turbo are centred -- they sit
        on the engine's centreline.

        The mono render runs first and the block renderer leaves its per-bank
        exhaust signals behind for exactly the samples produced; this only
        re-mixes, so the DSP cost is identical to mono plus one extra sum per
        sample.
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        self._stereo_out = (
            [np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)]
            if self.n_banks >= 2
            else None
        )
        self._want_stereo = True
        try:
            mono = self.process(n)
        finally:
            self._want_stereo = False
            banks = self._stereo_out
            self._stereo_out = None

        st = np.empty((mono.size, 2), dtype=np.float64)
        if not banks or len(banks) < 2:
            st[:, 0] = mono
            st[:, 1] = mono
            return st
        bl = banks[0]
        br = banks[1]
        if bl.size < mono.size or br.size < mono.size:
            # never pad a short bank with silence: fall back to centred mono
            st[:, 0] = mono
            st[:, 1] = mono
            return st
        bl = bl[: mono.size]
        br = br[: mono.size]
        # mid/side: mid is the mono render (all layers, fully shaped), side
        # is the bank difference. Constant-power width, 15% crossfeed.
        side = 0.35 * (bl - br)
        left = mono + side
        right = mono - side
        np.clip(left, -1.0, 1.0, out=left)
        np.clip(right, -1.0, 1.0, out=right)
        st[:, 0] = left
        st[:, 1] = right
        return st

    def _render_block_stereo(self, n: int):  # pragma: no cover - kept for API
        """Legacy hook; the split now lives on ``_stereo_block``."""
        self._render_block(n)
        return None, self._stereo_block

    # ------------------------------------------------------------------ #
    # event grid
    # ------------------------------------------------------------------ #
    def _span(self, off: float, n: int, base: int):
        """Yield ``(index_in_block, event_index)`` for one phase offset.

        Most 64-sample blocks contain no event at all for a given phase (a V8
        at 3000 rpm fires a cylinder every ~2.5 ms, i.e. every ~120 samples).
        The generator's loop overhead used to dominate the small-block profile,
        so the empty case -- ``j0 > j1`` -- is now checked before the loop and
        the generator is never entered.
        """
        if self.quantize_period:
            # Exact integer event grid: cylinder ``off`` fires at absolute sample
            # ``j*P + round(off*P)``. Because P is a whole number of samples this
            # grid repeats exactly, which is what makes the loop seam-free.
            p = self._period_samples
            q = int(round(off * p))
            j0 = -((q - base) // p)  # ceil((base - q) / p)
            j1 = -((q - base - n) // p) - 1  # ceil((base + n - q) / p) - 1
            if j0 > j1:
                return
            for j in range(j0, j1 + 1):
                idx = j * p + q - base
                if 0 <= idx < n:
                    yield idx, j
        else:
            start = self._grid_start
            dphi = self._grid_dphi
            end = self._grid_end
            j0 = int(math.floor(start - off)) + 1
            j1 = int(math.floor(end - off))
            if j0 > j1:
                return
            inv_dphi = 1.0 / dphi if dphi > 0.0 else 0.0
            for j in range(j0, j1 + 1):
                idx = int(round((j + off - start) * inv_dphi))
                if 0 <= idx < n:
                    yield idx, j

    def _render_events(self, n: int, rpm: float, level: float, flow: float, load: float = 0.0):
        """Build the exhaust / intake / mechanical excitation blocks."""
        p = self.preset
        e = p.exhaust
        m = p.mechanical
        # Only the writable span needs clearing, not the whole padded buffer:
        # the tail beyond n+pad is never read before it is overwritten.
        span = n + self._pad
        for buf in self._exc_buf:
            buf[:span] = 0.0
        self._int_buf[: n + self._pad] = 0.0
        self._mech_buf[: n + self._mech_pad] = 0.0
        self._bottom_buf[: n + self._bottom_pad] = 0.0
        for path in range(self.n_paths):
            self._exc_buf[path][: self._pad] += self._exc_carry[path]
        self._int_buf[: self._pad] += self._int_carry
        self._mech_buf[: self._mech_pad] += self._mech_carry
        self._bottom_buf[: self._bottom_pad] += self._bottom_carry

        # Turbulent flow roar. Part of it is a continuous pipe flow, but the
        # loud part is generated by the blowdown itself, so it is injected as a
        # per-event burst below rather than mixed in as a steady hiss.
        fg = e.flow_noise * flow * p.character.combustion_gain
        for path in range(self.n_paths):
            self._flow_blk[path][:n] = self._n_exhflow[path].render(n)
            if fg > 0.0:
                self._exc_buf[path][:n] += (
                    self._flow_shape[path].process(self._flow_blk[path][:n]) * fg
                )

        base = self._abs_samples
        if self.quantize_period:
            pass
        elif self._crank_locked:
            # The physics owns the grid: the block spans exactly the firing
            # cycles the model reported between the previous call and this one.
            # dphi is derived from that difference, so the audio cannot drift --
            # there is no independent oscillator left to drift.
            prev = self._crank_cycles_last
            if prev is None:
                # first locked block: snap once. Assumes the model advanced at
                # its stated rpm for the block already in flight, which is right
                # to within one block and avoids an audible jump.
                cyc = self.cyc_revs * 60.0 / max(float(self._target["rpm"]), 1.0)
                prev = self._crank_cycles_target - (n / self.sr) / max(cyc, 1e-9)
                self._abs_cycles = prev
            delta = self._crank_cycles_target - prev
            if delta <= 0.0:
                # a stalled engine still needs a sane rate; fall back to rpm
                cyc = self.cyc_revs * 60.0 / max(float(self._target["rpm"]), 1.0)
                delta = (n / self.sr) / max(cyc, 1e-9)
            self._grid_dphi = delta / n
            self._grid_start = self._abs_cycles
            self._grid_end = self._abs_cycles + delta
            self._crank_cycles_last = self._crank_cycles_target
        else:
            cyc = self.cyc_revs * 60.0 / max(rpm, 1.0)
            self._grid_dphi = 1.0 / max(cyc * self.sr, 1e-9)  # cycles per sample
            self._grid_start = self._abs_cycles
            self._grid_end = self._abs_cycles + self._grid_dphi * n

        if self.is_turbine:
            # a turbine has no discrete firing: continuous, rpm-driven noise
            self._exc_buf[0][:n] += self._n_turbine.render(n) * level
        else:
            ccv = p.character.ccv
            gflow = e.flow_burst * flow * p.character.combustion_gain
            # Wiebe pulse coupling (audit item 1, alias fix): with the physics'
            # instantaneous torque supplied, every firing event scales with
            # it. The published value is a 120 Hz *sample* of a ~6 ms-wide
            # Wiebe peak, so consecutive firings read aliased steps of a
            # fast-moving envelope -- measured 129 jumps/s into the event
            # gains, heard as a modem-like grit at exactly that rate (and
            # the same signal feeds the gauges, which is the "graph is
            # wrong" part). Smoothing the value over one *firing period* in
            # the crank-angle domain removes the sampling artefact while
            # keeping the real dynamics: the peaks are ~one firing period
            # wide, so a tau of half the period passes them untouched and
            # kills only the step between aliased samples. Deterministic in
            # angle, so the loop path (torque_pulse=None) stays exact.
            tp = self._torque_pulse
            if tp is not None:
                cyc = self.cyc_revs * 60.0 / max(rpm, 1.0)
                per_s = cyc / max(self.n_cyl, 1)  # one firing period
                tau_s = max(per_s * 0.5, 0.004)
                alpha = 1.0 - math.exp(-n / (tau_s * self.sr))
                prev = getattr(self, "_tp_smooth", tp)
                # track the target but never snap: the whole point is that
                # consecutive firings see a coherent value
                self._tp_smooth = prev + alpha * (tp - prev)
                tp = self._tp_smooth
            for idx in range(self.n_cyl):
                _bank, off = self.events[idx]
                ke = self._k_exh[idx]
                ki = self._k_int[idx]
                amp = self._cyl_amp[idx]
                for pos, _j in self._span(off, n, base):
                    v = self._noise_ccv[idx].next() if ccv > 0.0 else 0.0
                    cv = 1.0 + ccv * float(np.clip(v, -2.5, 2.5))
                    g = level * amp * cv
                    if tp is not None:
                        # clamp the low end: a firing under load always has
                        # some bite even when the published pulse lags to 0
                        g *= self.PULSE_FLOOR + (1.0 - self.PULSE_FLOOR) * min(tp, 3.0)
                    # stage 4: cylinder idx feeds its own exhaust path
                    self._inject(self._exc_buf[idx], pos, ke, g)
                    if gflow > 0.0:
                        # audit item 4: rotate through this cylinder's burst
                        # variants so consecutive firings have different noise
                        # tails (deterministic body, statistical blowdown)
                        kf = self._k_flow_bank[idx][self._flow_rot % len(self._k_flow_bank[idx])]
                        self._inject(self._exc_buf[idx], pos, kf, gflow * amp * cv)
                    self._inject(self._int_buf, pos, ki, g)
                    self._flow_rot += 1

        # ---- rotating assembly: main-bearing thump and piston slap
        # These follow the *firing* grid, so the bottom end carries the firing
        # order rather than being a static rumble. They sit a little after TDC,
        # where the load actually peaks: that is physical, and it also keeps them
        # from meeting the exhaust pulse exactly in antiphase. A four-stroke
        # piston rocks over again at the opposite TDC, which is why the bed
        # pulses twice per cycle.
        if not self.is_turbine and p.block.layout != "rotary":
            crank_g = m.crank_gain * (0.40 + 0.60 * load)
            slap_g = m.slap_gain * (0.35 + 0.65 * load)
            lag_c = CRANK_LAG_DEG / CYCLE_DEG
            lag_s = SLAP_LAG_DEG / CYCLE_DEG
            for idx in range(self.n_cyl):
                _bank, off = self.events[idx]
                amp = self._cyl_amp[idx]
                sc = self._cyl_mech_off[idx]
                if crank_g > 0.0:
                    for pos, _j in self._span((off + lag_c + sc) % 1.0, n, base):
                        self._inject(self._bottom_buf, pos, self._crank_tick, crank_g * amp)
                if slap_g > 0.0:
                    for pos, _j in self._span((off + lag_s + sc) % 1.0, n, base):
                        self._inject(self._bottom_buf, pos, self._slap_tick, slap_g * amp)
                    if int(p.block.strokes) == 4:
                        off2 = (off + 0.5 + lag_s + sc) % 1.0
                        for pos, _j in self._span(off2, n, base):
                            self._inject(
                                self._bottom_buf, pos, self._slap_tick, 0.45 * slap_g * amp
                            )

        # ---- mechanical: valve-seat and injector clicks at their real angles
        # Every valve lifts and seats once per cycle. The pair-spread keeps the
        # two intake (or exhaust) valves from firing the identical waveform at
        # the identical instant, which is what made the layer sound synthetic.
        # Two normalisations keep this layer honest:
        #   * 1/sqrt(cylinders): a V8 seats 32 valves per cycle, a single
        #     seats 2 -- without it the summed 3-12 kHz band grows with the
        #     cylinder count ("more cylinders, more maracas");
        #   * a load-dependent mask: at idle the note is quiet and a flat
        #     tick layer sat at PARITY with the body band (measured hf/mid
        #     -7 dB, vs a real idle's -15..-25). Under load the combustion
        #     masks the metalwork, so the full gain survives exactly where
        #     it is inaudible anyway; at idle the ticks drop to half.
        tick_norm = (1.0 / math.sqrt(max(self.n_cyl, 1))) * (0.45 + 0.55 * load)
        clicks: list[tuple[float, np.ndarray, float]] = []
        if self._valve_on:
            n_int = max(1, int(m.valves_per_cyl) // 2)
            n_exh = max(1, int(m.valves_per_cyl) - n_int)
            open_g = m.valvetrain_gain * m.valve_open_gain
            if open_g > 0.0 or m.valvetrain_gain > 0.0:
                for k in range(n_int):
                    d = k * VALVE_PAIR_SPREAD_DEG
                    clicks.append((INTAKE_OPEN_DEG + d, self._valve_open_tick, open_g))
                    clicks.append((INTAKE_CLOSE_DEG + d, self._valve_tick, m.valvetrain_gain))
                for k in range(n_exh):
                    d = k * VALVE_PAIR_SPREAD_DEG
                    clicks.append((EXHAUST_OPEN_DEG + d, self._valve_open_tick, open_g))
                    clicks.append((EXHAUST_CLOSE_DEG + d, self._valve_tick, m.valvetrain_gain))
        # Injectors are not valvetrain: a rotary and a two-stroke have them and
        # no valves, so this is gated separately.
        if m.injector_gain > 0.0 and not self.is_turbine:
            clicks.append((INJECTION_DEG, self._inj_tick, m.injector_gain))
        if clicks:
            for idx in range(self.n_cyl):
                off0 = self.events[idx][1] + self._cyl_mech_off[idx]
                amp = self._cyl_amp[idx]
                for deg, kernel, gain in clicks:
                    if gain <= 0.0:
                        continue
                    off = (off0 + deg / CYCLE_DEG) % 1.0
                    for pos, _j in self._span(off, n, base):
                        self._inject(self._mech_buf, pos, kernel, gain * amp * tick_norm)

        for path in range(self.n_paths):
            self._exc_carry[path][:] = self._exc_buf[path][n : n + self._pad]
        self._int_carry[:] = self._int_buf[n : n + self._pad]
        self._mech_carry[:] = self._mech_buf[n : n + self._mech_pad]
        self._bottom_carry[:] = self._bottom_buf[n : n + self._bottom_pad]
        self._abs_samples += n
        if not self.quantize_period:
            self._abs_cycles = self._grid_end
        return (
            [self._exc_buf[b][:n] for b in range(self.n_paths)],
            self._int_buf[:n],
            self._mech_buf[:n],
            self._bottom_buf[:n],
        )

    def _inject(self, buf: np.ndarray, i: int, kernel: np.ndarray, gain: float) -> None:
        m = min(kernel.size, buf.size - i)
        if m > 0:
            buf[i : i + m] += gain * kernel[:m]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def calibrate_gain(
    preset: Preset,
    sr: int | None = None,
    block: int | None = None,
    target_peak_db: float = -1.0,
    target_lufs: float = -14.0,
    operating_points: list[tuple[float, float, float]] | None = None,
) -> float:
    """Return the shared output gain for a preset: equal loudness, safe peaks.

    Two constraints, and the one that binds harder wins:

    * **loudness** -- every preset should sit at ``target_lufs`` so switching
      engines changes the character, not the volume. Peak level is a poor proxy
      for this: a spiky single-cylinder note and a smooth V12 differ by 7 dB of
      crest factor at the same perceived level, and calibrating on peak left a
      10 dB spread across the catalogue.
    * **headroom** -- the *final* signal must stay below ``target_peak_db``.

    Both are measured on the signal the listener actually gets, i.e. after the
    ``aggression`` saturation stage, and that stage is not unity-gain: it is
    ``tanh(x*drive)/tanh(drive)``, whose slope at zero is ``drive/tanh(drive)``
    -- up to +8 dB at drive 2.6. Measuring before it therefore both missed the
    real loudness and let the peaks through unchecked. Because the stage is
    nonlinear, the gain is solved by a few secant steps rather than one division:
    each step renders the *same* raw frames through the real clipper and reads
    the loudness back.

    The raw frames are rendered once and cached, so the extra accuracy costs a
    handful of rFFTs per preset, not a second synthesis run.
    """
    sr = int(sr or preset.render.sample_rate)
    block = int(block or preset.render.block_size)

    if not operating_points:
        idle = preset.engine.idle_rpm
        red = preset.engine.redline_rpm
        operating_points = []
        for i in range(9):
            f = i / 8.0
            rpm = idle + f * (red - idle)
            operating_points.append((rpm, 1.0, 1.0))  # worst case
            operating_points.append((rpm, 0.45, 0.42))  # part throttle

    rpm0, load0, thr0 = operating_points[0]
    voice = EngineVoice(
        preset, rpm=rpm0, load=load0, throttle=thr0, sr=sr, block=block, output_gain=1.0, clip=False
    )
    voice.process(int(0.25 * sr))
    frames: list[np.ndarray] = []
    for rpm, load, throttle in operating_points:
        voice.set_params(rpm=rpm, load=load, throttle=throttle)
        voice.process(int(0.05 * sr))
        frames.append(np.array(voice.process(int(0.20 * sr)), dtype=np.float64, copy=True))
    if not frames:
        return 1.0

    drive = float(preset.character.aggression)

    def measured(gain: float) -> tuple[float, float, float]:
        """(loudness over the sweep, loudest single point, absolute peak).

        Mirrors the runtime master chain exactly -- resonance compressor,
        character saturation, safety limiter -- because the gain is only
        meaningful against what the listener actually hears. When the
        compressor stage was added without updating this, the calibration
        still assumed the limiter was the only nonlinearity and picked a
        gain the runtime chain then re-shaped.
        """
        peak = 0.0
        num = 0.0
        den = 0.0
        loudest = -240.0
        for frame in frames:
            out = frame * gain
            out = dsp.resonance_compress(out)
            out = dsp.soft_clip(out, drive)
            out = dsp.safety_limit(out, SAFETY_CEILING, SAFETY_KNEE)
            peak = max(peak, dsp.peak(out))
            frame_loud = dsp.loudness_lufs(out, sr)
            loudest = max(loudest, frame_loud)
            num += frame_loud * out.size
            den += out.size
        return (num / den if den else -120.0), loudest, peak

    peak_limit = min(dsp.db_to_lin(target_peak_db), SAFETY_CEILING)
    raw_peak = max(dsp.peak(f) for f in frames)
    if raw_peak < 1e-9:
        return 1.0

    # The limiter makes the output peak almost independent of the gain once the
    # signal is into it, so a peak-based first guess is only a starting point:
    # the loudness loop below converges in two or three steps from it.
    gain = peak_limit / raw_peak
    for _ in range(6):
        loud, _hottest, _peak = measured(gain)
        deficit = target_lufs - loud
        if abs(deficit) < 0.1:
            break
        if deficit > 0.0:
            # getting louder: the limiter caps the peak, so loudness rises
            # sub-linearly -- step by half the requested amount and let the
            # next iteration correct
            gain *= dsp.db_to_lin(deficit * 0.6)
        else:
            gain *= dsp.db_to_lin(deficit)
        if gain <= 0.0:
            gain = 1e-6
            break

    # Hot-spot clamp (audit: "squashed at resonance rpms"). The sweep's
    # average lands on the target, but the points where the firing order
    # sits on a pipe mode run several dB hotter than the rest (measured on
    # the V8: -5.0 LUFS at 7325 against -13 typical). With a single shared
    # gain those points drive 16-39 % of their samples into the limiter's
    # knee -- audible pumping exactly at the rpms a driver holds.
    #
    # Rather than clamping the hottest point by brute force (which dragged
    # the sweep MEAN 6-8 dB under target and made every engine a whisper),
    # soften the resonance peaks at the source: the loudest frames are
    # loud because the limiter has already flattened them, so their LUFS is
    # compressed loudness. Re-target the calibration on a SOFTENED sweep --
    # each frame's contribution to the mean is capped at target+3 dB before
    # averaging -- and let the final gain be driven by that. Frames below
    # the cap count fully; the resonance monsters stop dominating. The
    # result: the mean stays on target, and the hot points ride the
    # limiter only briefly instead of living in it.
    softened_target = target_lufs

    def softened_loud(gain: float) -> tuple[float, float]:
        num = den = 0.0
        hottest = -240.0
        for frame in frames:
            out = frame * gain
            out = dsp.resonance_compress(out)
            out = dsp.soft_clip(out, drive)
            out = dsp.safety_limit(out, SAFETY_CEILING, SAFETY_KNEE)
            fl = dsp.loudness_lufs(out, sr)
            hottest = max(hottest, fl)
            fl_capped = min(fl, target_lufs + 3.0)
            num += fl_capped * out.size
            den += out.size
        return (num / den if den else -240.0), hottest

    for _ in range(6):
        loud, hottest = softened_loud(gain)
        deficit = softened_target - loud
        if abs(deficit) < 0.15:
            break
        gain *= dsp.db_to_lin(deficit * (0.6 if deficit > 0 else 1.0))
        if gain <= 0.0:
            gain = 1e-6
            break
    return float(gain)
