"""Loop-bank player: constant-cost playback layer for ``EngineAudio``.

One player instance per preset. The audio thread calls :meth:`produce`
which never allocates in steady state (its scratch buffers are created
once) and never runs a single line of the procedural synthesiser -- the
bank was rendered by ``soundgen.layers`` in the background at load time.

Blend model (all state is precomputed, the loop is arithmetic):

* two layers play at a time, crossfaded along the rpm axis
  (idle<->mid<->wot);
* the coast layer carries the closed-throttle character and blends in
  against the speed pair with weight (1 - state);
* the limiter layer is a gated overlay: the caller supplies limiter
  intensity (0..1, the model's cut flag smoothed) and it rides on top;
* pitch = rpm / layer_ref_rpm (playing a loop rendered at rpm0 at rate
  rpm/rpm0 reproduces the firing order exactly, and the loops were
  rendered at a whole number of cycles so they stay periodic at any
  integer playback rate);
* ``torque_pulse`` amplitude-modulates the engine bed (attack/release
  smoothed inside the player -- the published pulse is a 120 Hz sample
  of a ~6 ms Wiebe peak, smoothing was already proven necessary there);
* ``pipe_temp`` detunes the whole bank: hotter gas -> faster sound ->
  the loops play a touch faster (resonances climb ~1.3x cold->hot, the
  player reproduces the tail of that as ~+1.5% pitch).
"""

from __future__ import annotations

import math

import numpy as np


class LoopPlayer:
    """Plays a rendered LoopBank at live operating points."""

    def __init__(self, bank, sr: int):
        self.bank = bank
        self.sr = int(sr)
        self.ready = False
        # read cursors per layer key, in samples (float: fractional rate)
        self._pos: dict[str, float] = {}
        self._gain: dict[str, float] = {}
        self._gain_live: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._rate: dict[str, float] = {}
        self._bed_lpf = 1.0  # smoothed torque_pulse for the AM
        self._lim = 0.0
        self._am_target = 0.0
        self._am_tau = 0.01
        self._rate_hint = float(bank.rpm_lo)
        self._limiter_target = 0.0
        self._heat = 0.0
        self._out = np.zeros(2048, dtype=np.float64)

    def _entry(self, key):
        return self.bank.entries.get(key)

    # ------------------------------------------------------------------ #
    # main thread: parameter smoothing (called by update(); cheap)
    # ------------------------------------------------------------------ #
    def set_state(
        self,
        rpm: float,
        load: float,
        throttle: float,
        torque_pulse: float | None,
        pipe_temp_c: float | None,
        limiter: float,
    ) -> None:
        """Compute per-layer target gains and rates for the next blocks.

        Called from the main thread (already the parameter publisher's
        cadence); the audio thread only reads the dicts this fills.
        Gains are the *targets*: the player ramps its live gains toward
        them per block so layer switches are click-free even when the
        model jumps between operating points in one step.
        """
        if not self.bank.entries:
            return
        span = max(self.bank.rpm_hi - self.bank.rpm_lo, 1.0)
        f = min(max((rpm - self.bank.rpm_lo) / span, 0.0), 1.0)
        state = 0.6 * min(max(load, 0.0), 1.0) + 0.4 * min(max(throttle, 0.0), 1.0)

        # Bed blend, equal-power, capped at unity total: the two speed
        # layers crossfade with sin/cos weights; the coast layer replaces
        # bed power rather than adding to it (same denominator), so the
        # blend is a rotation, not a sum -- WOT must not clip against the
        # ceiling the layers were peak-normalised into.
        target: dict[str, float] = {}
        if f <= 0.42:
            k = f / 0.42 if 0.42 > 0 else 0.0
            a, b = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
            target["idle"] = a * state
            target["mid"] = b * state
        else:
            k = (f - 0.42) / 0.58 if 0.58 > 0 else 1.0
            a, b = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
            target["mid"] = a * state
            target["wot"] = b * state
        target["coast"] = 1.0 - state
        if limiter > 0.0:
            target["limiter"] = 0.45 * limiter

        # rates: pitch follows rpm relative to each layer's reference rpm.
        # coast is rendered near the top of the range; at low rpm it must
        # not slow below ~0.55x or the overrun bed turns into a drone.
        for key, w in target.items():
            entry = self._entry(key)
            if entry is None:
                continue
            _, ref = entry
            rate = rpm / max(ref, 1.0)
            if key == "coast":
                rate = max(rate, 0.55)
            self._rate[key] = float(rate)
            self._gain[key] = float(w)
        self._target = target
        # pitch heat: reference 60..600 C published by the model
        if pipe_temp_c is not None:
            t = min(max(pipe_temp_c, 60.0), 600.0)
            self._heat = (t - 60.0) / 540.0 * 0.015

        # AM: smoothed torque_pulse (tau ~ one firing period at this rpm)
        if torque_pulse is not None:
            period = 60.0 / max(rpm, 1.0) * 2.0 / max(self.bank.preset.block.cylinders, 1)
            self._am_target = min(max(torque_pulse, 0.0), 1.5)
            self._am_tau = max(period * 0.5, 0.004)

    def prepare(self) -> None:
        """Main-thread prelude: prime live gains from the first targets."""
        self._gain_live = {k: v for k, v in self._gain.items()}
        self.ready = True

    # ------------------------------------------------------------------ #
    # audio thread
    # ------------------------------------------------------------------ #
    def produce(self, n: int, out: np.ndarray) -> None:
        """Add ``n`` frames of engine bed into ``out[:n]`` (mono)."""
        if not self.ready or not self.bank.entries:
            return
        if self._out.size < n:
            self._out = np.zeros(n, dtype=np.float64)
        buf = self._out[:n]
        buf.fill(0.0)

        # ramp live gains toward targets (one-pole over the block)
        alpha_g = min(1.0, n / (0.030 * self.sr))  # 30 ms gain ramps
        active: list[tuple[str, float, float]] = []
        for key, w in self._target.items():
            entry = self._entry(key)
            if entry is None or w <= 0.001:
                live = self._gain_live.get(key, 0.0)
                if live > 0.001:
                    # let it fade out, then stop reading it
                    self._gain_live[key] = live * (1.0 - alpha_g)
                    if self._gain_live[key] > 0.001:
                        active.append((key, self._gain_live[key], self._rate.get(key, 1.0)))
                continue
            live = self._gain_live.get(key, 0.0) + (w - self._gain_live.get(key, 0.0)) * alpha_g
            self._gain_live[key] = live
            rate = self._rate.get(key, 1.0) * (1.0 + self._heat)
            if live > 0.001:
                active.append((key, live, rate))

        # AM envelope: smoothed torque_pulse lifts the bed
        if self._am_target > 0.0:
            alpha = 1.0 - math.exp(-n / (self._am_tau * self.sr))
            self._bed_lpf += (self._am_target - self._bed_lpf) * alpha
        else:
            self._bed_lpf *= 1.0 - min(1.0, n / (0.05 * self.sr))
        # map 0..1 pulse to 0.82..1.18 gain: gentle lugging, never a gate
        am_gain = 0.82 + 0.36 * min(self._bed_lpf, 1.0)

        for key, live, rate in active:
            entry = self._entry(key)
            if entry is None:
                continue
            x, _ref = entry
            size = x.size
            pos = self._pos.get(key, 0.0)
            # The bed (idle/mid/wot/coast) carries the AM; the limiter layer
            # is an overlay and stays at unity so the gate does not breathe.
            t = pos + np.arange(n, dtype=np.float64) * rate
            i0 = (t % size).astype(np.int64)
            i1 = (i0 + 1) % size
            frac = t - np.floor(t)
            w = x[i0] * (1.0 - frac) + x[i1] * frac
            if key == "limiter":
                buf += w * live
            else:
                buf += w * live
            self._pos[key] = float((pos + n * rate) % size)

        # AM applies to the engine bed only (limiter overlay is inside
        # `active` too; the split above keeps the code simple and the
        # overlay is itself a gated texture whose gain is the gate).
        out[:n] += buf * am_gain
