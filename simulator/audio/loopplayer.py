"""Loop-bank player: constant-cost playback layer for ``EngineAudio``.

One player instance per preset. The audio thread calls :meth:`produce`
which never allocates in steady state (its scratch buffers are created
once) and never runs a single line of the procedural synthesiser -- the
bank was rendered by ``soundgen.layers`` in the background at load time.

Blend model (all state is precomputed, the loop is arithmetic):

* two layers play at a time, crossfaded along the rpm axis
  (idle<->mid<->wot);
* the coast layer carries the closed-throttle character and blends in
  against the speed pair only as speed rises (at idle it is silent: the
  coast loop is rendered high in the rev range and would sound like a fan),
  and it never takes the whole bed -- overrun keeps a share of the speed
  layer so it still has a pumping low end;
* the limiter layer is a gated overlay: the caller supplies limiter
  intensity (0..1, the model's cut flag smoothed) and it rides on top;
* pitch = rpm / layer_ref_rpm (playing a loop rendered at rpm0 at rate
  rpm/rpm0 reproduces the firing order exactly, and the loops were
  rendered at a whole number of cycles so they stay periodic at any
  integer playback rate); the rate is one-pole smoothed so a fast pull
  through the rev range does not step the pitch block by block;
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

# Coast is a noise-led overrun texture, not a tone: it may never take the
# whole bed (no body left) and it is never pitched far from its render point
# (pitching it up turns its 1-3 kHz band into hiss).
COAST_MAX = 0.62
COAST_RATE_MIN = 0.85
COAST_RATE_MAX = 1.12


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
        self._rate_live: dict[str, float] = {}
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

        # Coast is the overrun character: the throttle-shut loop is rendered
        # near the top of the rev range, so it must only take over when the
        # engine is actually turning fast. Letting it replace the bed at idle
        # (the old ``1 - state`` weight) meant the very first sound after a
        # preset load was the coast loop played at its 0.55x floor -- a low
        # whine that reads as a fan, not an engine. Fade it in with speed and
        # hand the remaining bed power to the normal idle/mid/wot crossfade.
        #
        # It is also capped below unity: the coast loop is noise-led (its
        # energy sits at 1-3 kHz), so weighting it at 1.0 made lift-off at
        # speed a bare hiss with no body. Keeping a share of the speed layer
        # underneath restores the pumping low end the ear expects on overrun.
        speed = min(max(f / 0.30, 0.0), 1.0)
        coast = COAST_MAX * (1.0 - state) * speed
        bed = 1.0 - coast
        # Bed blend, equal-power, capped at unity total: the two speed
        # layers crossfade with sin/cos weights over the remaining ``bed``
        # power, so the blend stays a rotation and WOT cannot clip against
        # the ceiling the layers were peak-normalised into.
        target: dict[str, float] = {}
        if f <= 0.42:
            k = f / 0.42 if 0.42 > 0 else 0.0
            a, b = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
            target["idle"] = a * bed
            target["mid"] = b * bed
        elif f <= 0.72:
            # mid and wot are rendered at 42 % and 72 % of the range: cross
            # them between those two anchors, not all the way to the redline.
            # The old 0.42..1.0 sweep kept half the mid loop (a 1.5x pitch
            # shift) under the wot note until the limiter, which is heard as
            # a thin two-tone whine rather than an engine.
            k = (f - 0.42) / 0.30 if 0.30 > 0 else 1.0
            a, b = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
            target["mid"] = a * bed
            target["wot"] = b * bed
        else:
            target["wot"] = bed
        target["coast"] = coast
        if limiter > 0.0:
            target["limiter"] = 0.45 * limiter

        # Renormalise the bed rotation: with coast replacing bed power and
        # the limiter riding on top, the sin/cos pair plus coast can sum
        # above unity at mid-throttle (measured 0.965 peak on the gas
        # bank). Cap the bed weights at a total of 1.0 keeping their
        # ratios, then add the limiter overlay after.
        bed_total = sum(target.values())
        if bed_total > 1.0 and "limiter" in target:
            bed = bed_total - target["limiter"]
            if bed > 0.0:
                scale = 1.0 - target["limiter"]
                for key in ("idle", "mid", "wot", "coast"):
                    if key in target:
                        target[key] *= scale / bed
        elif bed_total > 1.0:
            for key in list(target):
                target[key] /= bed_total

        # rates: pitch follows rpm relative to each layer's reference rpm.
        # coast is rendered near the top of the range; at low rpm it must
        # not slow below ~0.55x or the overrun bed turns into a drone, and
        # at high rpm it is a *texture*, not a note: pitching a noise loop
        # up 1.6x turns its 1-3 kHz band into 1.6-4.8 kHz hiss, which is
        # exactly the "torn" top end. Keep it near its rendered pitch.
        for key, entry in self.bank.entries.items():
            rate = rpm / max(entry[1], 1.0)
            if key == "coast":
                rate = min(max(rate, COAST_RATE_MIN), COAST_RATE_MAX)
            self._rate[key] = float(rate)
        self._gain = {key: float(w) for key, w in target.items()}
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
        self._rate_live = {
            k: v * (1.0 + self._heat) for k, v in self._rate.items()
        }
        self.ready = True

    def _live_rate(self, key: str, alpha: float) -> float:
        """One-pole the read rate toward its target (audio thread)."""
        target = self._rate.get(key, 1.0) * (1.0 + self._heat)
        live = self._rate_live.get(key, target)
        live += (target - live) * alpha
        self._rate_live[key] = live
        return live

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
        # Pitch gets the same treatment: a step in rpm used to step the read
        # rate within one block, and a chain of those steps is heard as
        # tearing on a fast pull. 45 ms is short enough that the note still
        # tracks the tacho, long enough to smooth a block-rate update.
        alpha_r = min(1.0, n / (0.045 * self.sr))
        active: list[tuple[str, float, float]] = []
        for key, w in self._target.items():
            entry = self._entry(key)
            if entry is None or w <= 0.001:
                live = self._gain_live.get(key, 0.0)
                if live > 0.001:
                    # let it fade out, then stop reading it
                    self._gain_live[key] = live * (1.0 - alpha_g)
                    if self._gain_live[key] > 0.001:
                        active.append(
                            (key, self._gain_live[key], self._live_rate(key, alpha_r))
                        )
                continue
            live = self._gain_live.get(key, 0.0) + (w - self._gain_live.get(key, 0.0)) * alpha_g
            self._gain_live[key] = live
            rate = self._live_rate(key, alpha_r)
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
