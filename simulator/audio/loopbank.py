"""Loop-bank playback: the audio path without synthesis in the callback.

The procedural voice (``EngineVoice``) is the *offline* renderer: it is
excellent and expensive. Running it inside the PortAudio callback from
Python was the lag: the callback is pure Python paying ~40 Python<->C
crossings per block while the GUI thread holds the GIL for tens of
milliseconds at a time -- every heavy GUI frame became an underrun, no
matter how the DSP was tuned.

This module inverts the arrangement:

* at preset load, the loop bank (idle / part / WOT / coast / limiter) is
  rendered in a background thread by the *same* ``layers.render_layer``
  the offline assets use -- one-time cost, best quality the voice has;
* the audio callback only *plays*: a resampling read pointer per layer,
  an equal-power crossfade driven by the live (rpm, load, throttle), and
  the one-shots mixed on top. Cost per block is a few linear
  interpolations per output sample -- tens of microseconds, ~50x inside
  the deadline, with no GIL-sensitive work of our own left in it.

The modulation that made the live voice feel alive is preserved as
playback-time parameters rather than resynthesis: pitch follows rpm
exactly (a loop rendered at rpm0 played at rpm/rpm0 keeps the firing
order right), ``torque_pulse`` amplitude-modulates the engine bed, and
``pipe_temp`` detunes the whole bank as the pipe heats up.
"""

from __future__ import annotations

import threading

import numpy as np

from soundgen import layers as layers_mod


class LoopBank:
    """The five rendered loops for one preset, plus their reference rpms.

    The bank renders at unity gain and normalises to ``target_peak`` once:
    the offline calibration sweep (``voice_gain``) costs ~10 s on a W16 and
    is a loudness-equalisation tool for the *catalogue*, while the bank is
    peak-normalised internally and the master limiter in the callback keeps
    the ceiling. Skipping the sweep cuts the preset-switch silence path to
    the bank render alone (~9 s W16, ~2 s I4, threaded).
    """

    TARGET_PEAK = 0.79  # -2 dBFS

    def __init__(self, preset, sr: int, gain: float = 1.0, block: int = 1024,
                 engine: str = "procedural"):
        self.preset = preset
        self.sr = int(sr)
        self.ready = threading.Event()
        self.error: str | None = None
        self._build_thread: threading.Thread | None = None
        idle = float(preset.engine.idle_rpm)
        red = float(preset.engine.redline_rpm)
        self.rpm_lo = idle
        self.rpm_hi = red
        # (key, buffer, ref_rpm, loop length)
        self.entries: dict[str, tuple[np.ndarray, float]] = {}
        self._gain = float(gain)
        self._block = int(block)
        # Renderer: "procedural" (the offline voice) or "gas" (the gas-
        # dynamic core, roadmap B3). Both produce the same entry format,
        # so the player and the whole asset path are renderer-agnostic.
        self.engine = str(engine)

    # ------------------------------------------------------------------ #
    # construction (background thread)
    # ------------------------------------------------------------------ #
    def build_async(self) -> None:
        if self._build_thread is not None:
            return
        self._build_thread = threading.Thread(
            target=self._build, name="loop-bank", daemon=True
        )
        self._build_thread.start()

    def _build(self) -> None:
        try:
            # Layers are independent renders and the native convolver
            # releases the GIL inside its FFT passes, so a small pool of
            # worker threads genuinely overlaps them: the W16's bank drops
            # from ~22 s to ~9 s, which is the difference between a preset
            # switch that plays within seconds and one that sits in silence.
            import concurrent.futures as cf

            if self.engine == "gas":
                from soundgen import gasbank

                def _one(spec):
                    x, rpm = gasbank.render_gas_layer(self.preset, spec, sr=self.sr)
                    return spec.key, (np.asarray(x, dtype=np.float64), rpm)
            else:
                def _one(spec):
                    rpm = layers_mod.layer_rpm(self.preset, spec)
                    x = layers_mod.render_layer(
                        self.preset, spec, self._gain, sr=self.sr, block=self._block
                    )
                    return spec.key, (np.asarray(x, dtype=np.float64), rpm)

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                for key, entry in ex.map(_one, layers_mod.DEFAULT_LAYERS):
                    self.entries[key] = entry
            # one normalisation pass for the whole bank: the layers keep
            # their relative balance, the bank lands on the target peak
            peak = max(float(np.abs(x).max()) for x, _ in self.entries.values())
            if peak > 1e-9:
                scale = self.TARGET_PEAK * self._gain / peak
                for key in list(self.entries):
                    x, rpm = self.entries[key]
                    self.entries[key] = (x * scale, rpm)
        except Exception as exc:  # noqa: BLE001 - reported on the main thread
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ready.set()

    def build_blocking(self) -> None:
        """For tests: build synchronously."""
        if self._build_thread is None:
            self._build()

    # ------------------------------------------------------------------ #
    # playback helpers (audio thread, allocation-free)
    # ------------------------------------------------------------------ #
    def layer_weights(self, rpm: float, load: float, throttle: float):
        """Blend weights over the five layers for one operating point.

        The engine is two dimensions of sound: speed (which of idle/mid/wot
        by rpm) and state (throttle open vs overrun, by load/throttle). The
        limiter layer is added on top whenever the cut is active (the voice
        free-runs it in the loop, so it plays only when the caller says so).
        Returns ``(keys, weights)`` with 3 or fewer entries.
        """
        span = max(self.rpm_hi - self.rpm_lo, 1.0)
        f = float(np.clip((rpm - self.rpm_lo) / span, 0.0, 1.0))
        state = 0.6 * float(np.clip(load, 0.0, 1.0)) + 0.4 * float(np.clip(throttle, 0.0, 1.0))

        # speed axis: idle <-> mid <-> wot
        keys, w = [], []
        if f <= 0.42:
            keys.append(("idle", "mid"))
            k = f / 0.42 if 0.42 > 0 else 0.5
            w.append((1.0 - k, k))
        else:
            keys.append(("mid", "wot"))
            k = (f - 0.42) / 0.58 if 0.58 > 0 else 1.0
            w.append((1.0 - k, k))
        # state axis: blend each speed pair towards coast by (1 - state)
        pairs = []
        for (lo_key, hi_key), (lo_w, hi_w) in zip(keys, w):
            speed = lo_w * lo_key + hi_w * hi_key
            coast = (1.0 - state) * speed
            pairs.append((lo_key, hi_key, lo_w * state, hi_w * state, coast))
        return pairs

    def ref_rpm(self, key: str) -> float:
        entry = self.entries.get(key)
        return entry[1] if entry else float(self.preset.engine.idle_rpm)
