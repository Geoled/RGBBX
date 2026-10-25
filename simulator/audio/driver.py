"""Drives the sound synthesiser from the engine model.

The synthesiser is the same :class:`~soundgen.voice.EngineVoice` that renders the
offline WAV files, so what the simulator plays is the same engine that produced
the shipped assets -- there is no separate "real-time approximation" to drift out
of sync with the offline one.

The model lives on the main thread and the synthesiser on the audio thread. They
meet at exactly two places:

* a parameter snapshot (rpm, load, throttle), replaced under a short lock;
* a queue of one-shot triggers, drained by the audio thread.

One-shots are *rendered* by the audio thread from the same generators the offline
event files come from, then mixed over the running layers. That is what makes a
backfire land on top of the engine note instead of interrupting it.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import asdict, dataclass

import numpy as np

try:  # optional: a smoother read for the pitched loop assets
    from scipy.interpolate import CubicSpline as _CubicSpline  # type: ignore
except Exception:  # pragma: no cover - depends on the host
    _CubicSpline = None

from .. import bootstrap  # noqa: F401  (sets sys.path up before soundgen import)

from soundgen import layers as layers_mod  # noqa: E402
from soundgen.config import Preset  # noqa: E402
from soundgen.events import events_for, render_event  # noqa: E402
from soundgen.voice import EngineVoice, calibrate_gain  # noqa: E402

from .loopbank import LoopBank  # noqa: E402
from .loopplayer import LoopPlayer  # noqa: E402
from .sounddevice_backend import SoundDeviceStream  # noqa: E402
from .winmm import WaveOutStream  # noqa: E402

# Calibrating the shared gain costs about a second of audio per preset, and the
# answer never changes, so it is computed once and kept.
_GAIN_CACHE: dict[tuple[str, int, int], float] = {}

# A looping event asset is peak-normalised on its own, not against the engine
# layers, so it needs a mix gain here. These are deliberately conservative:
# the spool is a texture under the exhaust note, while knock has to cut through.
TURBO_LOOP_GAIN = 0.22
KNOCK_LOOP_GAIN = 1.00


def voice_gain(preset: Preset, sr: int, block: int, target_peak_db: float = -2.0) -> float:
    """Shared output gain for a preset, calibrated on its own layer points.

    The calibration sweep is rendered at the preset's *offline* block size
    (1024), never at the live one: at 64 frames the same sweep costs 2.6x more
    wall-clock (3.9 s vs 1.5 s for the W16 -- a multi-second GUI freeze on
    every engine switch, heard as the audio lagging the engine), and the gain
    is a property of the preset's spectrum, not of the streaming block size.
    The result is cached per (preset, sr, block) so a switch back is instant.
    """
    key = (
        hashlib.sha1(repr(asdict(preset)).encode("utf-8")).hexdigest(),
        int(sr),
        int(block),
    )
    cached = _GAIN_CACHE.get(key)
    if cached is None:
        points = [(rpm, spec.load, spec.throttle) for spec, rpm, _ in layers_mod.layer_descriptors(preset, sr)]
        cal_block = int(preset.render.block_size)
        cached = calibrate_gain(
            preset, sr=sr, block=cal_block, operating_points=points, target_peak_db=target_peak_db
        )
        _GAIN_CACHE[key] = cached
    return cached


@dataclass
class AudioStats:
    errors: int = 0
    starved: int = 0
    latency_ms: float = 0.0

    def __str__(self) -> str:
        state = "ok" if self.errors == 0 and self.starved == 0 else "problems"
        return f"audio {state}: {self.starved} starved, {self.errors} errors, {self.latency_ms:.0f} ms latency"


class EngineAudio:
    """Real-time sound for one engine.

    ``enabled=False`` builds the object but never opens a device, which is what
    ``--no-audio`` and the headless tests use.
    """

    def __init__(
        self,
        preset: Preset,
        sr: int = 48000,
        block_frames: int = 512,
        buffers: int = 5,
        master: float = 1.0,
        enabled: bool = True,
        preload: dict[str, int] | None = None,
        audio_backend: str = "auto",
        audio_device: int | str | None = None,
        mode: str = "assets",
        engine: str = "procedural",
    ):
        self.preset = preset
        self.sr = int(sr)
        self.block_frames = int(block_frames)
        self.buffers = int(buffers)
        self.master = float(master)
        self.enabled = bool(enabled)
        self.audio_backend = str(audio_backend or "auto").lower()
        self.audio_device = audio_device
        # Sound source: "assets" plays the background-rendered loop bank
        # (constant ~40 us callbacks, no synthesis in the audio thread);
        # "procedural" runs the full EngineVoice in the callback (offline
        # renders, tests, and whoever wants the live path back).
        # ``engine`` picks the bank's renderer: "procedural" (the offline
        # voice) or "gas" (the gas-dynamic core, roadmap B3).
        self.mode = str(mode)
        self.engine = str(engine)
        # 2 channels: the voice's process_stereo() pans the exhaust banks
        # (a V8 keeps its two-pipe width); mono devices get a duplicated
        # frame and lose only the width, nothing else.
        self.channels = 2

        self.voice: EngineVoice | None = None
        self.stream: SoundDeviceStream | WaveOutStream | None = None
        self.backend: str | None = None
        self.error: str | None = None
        self.warnings: list[str] = []
        self.gain = 1.0
        # asset-mode state
        self.bank: LoopBank | None = None
        self.player: LoopPlayer | None = None
        self._bank_warm = 0.0  # fade-in progress 0..1
        self._limiter_level = 0.0
        self._bed_scratch: np.ndarray | None = None

        self._lock = threading.Lock()
        # (rpm, load, throttle, boost, knock, crank_cycles, torque_pulse, pipe_temp)
        self._params: tuple[
            float, float, float, float, float, float | None, float | None, float | None
        ] = (
            float(preset.engine.idle_rpm),
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            None,
        )
        self._triggers: list[str] = []
        self._active: list[list] = []  # [samples, cursor] owned by the audio thread
        self._shots: dict[str, list[np.ndarray]] = {}
        self._loops: dict[str, list] = {}  # [samples, cursor, gain, optional spline]
        self._shot_index = 0

        default_preload = {
            "backfire": 2,
            "start_catch": 1,
            "start_crank": 1,
            "shutdown": 1,
            "blow_off": 1,
            "turbo_spool": 1,
            "knock": 1,
        }
        # ``enabled=False`` (headless tests, --no-audio) still builds the
        # sources: the test harness drives _produce directly against the
        # bank/voice without opening a device. Only the *stream* is skipped.
        self._open(preload if preload is not None else default_preload)
        if not self.enabled:
            self.stream = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _open(self, preload: dict[str, int]) -> None:
        try:
            if self.mode == "assets":
                # Asset mode: the bank renders in the background and the
                # player takes over the callback the moment it is ready.
                # The stream opens immediately (silence until then), so the
                # device exists and the GUI never waits on synthesis. The
                # bank peak-normalises itself (LoopBank.TARGET_PEAK), so the
                # ~10 s calibration sweep is NOT on the switch path: a preset
                # change costs bank-build only (~2 s I4, ~9 s W16).
                self.bank = LoopBank(self.preset, self.sr, engine=self.engine)
                self.player = LoopPlayer(self.bank, self.sr)
                if self.enabled:
                    # Bank AND one-shots both render off the audio and GUI
                    # threads. The shots are independent renders too; run
                    # them in a worker so _open returns immediately after
                    # opening the stream (the callback emits silence until
                    # the bank is ready -- a fade-in, not a freeze).
                    self.bank.build_async()
                    import threading as _th

                    _th.Thread(
                        target=self._render_shots,
                        args=(preload,),
                        name="shot-bank",
                        daemon=True,
                    ).start()
                else:
                    # headless tests drive _produce directly: entries must
                    # exist without a device
                    self.bank.build_blocking()
                    self._render_shots(preload)
            else:
                self.gain = voice_gain(self.preset, self.sr, self.block_frames)
                self.voice = EngineVoice(
                    self.preset,
                    rpm=float(self.preset.engine.idle_rpm),
                    load=0.0,
                    throttle=0.0,
                    sr=self.sr,
                    block=self.block_frames,
                    output_gain=self.gain,
                    clip=True,
                )
                self._render_shots(preload)
        except Exception as exc:  # keep the window alive even if synthesis breaks
            self.error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            self.voice = None
            self.stream = None
            return

        if not self.enabled:
            # headless: sources are built, no device is opened
            return

        # PortAudio first: lower latency, device selection and a cross-platform
        # path. WinMM is the safety net for machines without PortAudio or with a
        # device that cannot be opened by it.
        backends = []
        if self.audio_backend in ("auto", "sounddevice", "portaudio"):
            backends.append("sounddevice")
        if self.audio_backend in ("auto", "winmm", "waveout"):
            backends.append("winmm")

        for name in backends:
            # First try the low-latency WASAPI endpoint; if that particular
            # device refuses 48 kHz, fall back to PortAudio's own default
            # before giving up on sounddevice altogether.
            attempts = [True]
            if name == "sounddevice" and self.audio_device is None:
                attempts.append(False)
            for auto_low_latency in attempts:
                try:
                    if name == "sounddevice":
                        stream = SoundDeviceStream(
                            self.sr,
                            self.channels,
                            self.block_frames,
                            latency="low",
                            device=self.audio_device,
                            auto_low_latency=auto_low_latency,
                        )
                    else:
                        stream = WaveOutStream(
                            self.sr, self.channels, self.block_frames, buffers=self.buffers
                        )
                    stream.start(self._produce)
                except Exception as exc:
                    label = name if auto_low_latency else f"{name} (default device)"
                    self.warnings.append(f"{label}: {type(exc).__name__}: {exc}")
                    # a device refusing stereo is not worth losing audio over:
                    # fall back to mono and retry once before moving on
                    if self.channels > 1:
                        self.channels = 1
                        try:
                            if name == "sounddevice":
                                stream = SoundDeviceStream(
                                    self.sr, 1, self.block_frames, latency="low",
                                    device=self.audio_device,
                                    auto_low_latency=auto_low_latency,
                                )
                            else:
                                stream = WaveOutStream(
                                    self.sr, 1, self.block_frames, buffers=self.buffers
                                )
                            stream.start(self._produce)
                        except Exception as exc2:
                            self.warnings.append(f"{label} (mono): {type(exc2).__name__}: {exc2}")
                            continue
                        self.stream = stream
                        self.backend = name
                        self.error = None
                        return
                    continue
                self.stream = stream
                self.backend = name
                self.error = None
                return

        self.error = "; ".join(self.warnings) or "no audio backend available"
        self.enabled = False
        self.voice = None
        self.stream = None

    def _render_shots(self, preload: dict[str, int]) -> None:
        specs = {spec.key: spec for spec in events_for(self.preset)}
        for key, count in preload.items():
            spec = specs.get(key)
            if spec is None:
                continue
            variants = []
            for i in range(max(1, int(count))):
                # distinct seeds so repeated backfires are not literally identical
                seed = int(self.preset.seed) + 7919 * (i + 1) + sum(map(ord, key))
                variants.append(
                    render_event(
                        # Render the one-shots at the preset's offline block:
                        # the event assets are WAV-like buffers, not streamed,
                        # and at the live block the same renders cost several
                        # times more wall-clock (measured: 3.1 s vs 1.4 s for
                        # the W16's shot bank -- seconds of frozen audio on
                        # every engine switch, heard as the engine lagging).
                        self.preset, spec, 1.0, sr=self.sr,
                        block=int(self.preset.render.block_size), seed=seed
                    )
                )
            if not variants:
                continue
            if spec.looping:
                # A looping event is a continuous texture, not a one-shot: keep
                # one seamless period and mix it under the engine at a gain the
                # caller controls (boost for the turbo, knock intensity for the
                # detonation pings).
                samples = variants[0]
                spline = None
                if key == "turbo_spool" and _CubicSpline is not None and samples.size > 8:
                    # The spool asset is resampled continuously, so linear
                    # interpolation would add audible images to the whoosh.
                    # A periodic cubic spline is cheap at one evaluation per
                    # output frame and keeps the loop seam continuous.
                    grid = np.arange(samples.size + 1, dtype=np.float64)
                    values = np.concatenate([samples, samples[:1]])
                    spline = _CubicSpline(grid, values, bc_type="periodic")
                self._loops[key] = [samples, 0.0, 0.0, spline]
            else:
                self._shots[key] = variants

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream = None

    # ------------------------------------------------------------------ #
    # control (main thread)
    # ------------------------------------------------------------------ #
    def update(
        self,
        rpm: float,
        load: float,
        throttle: float,
        boost: float = 0.0,
        knock: float = 0.0,
        crank_cycles: float | None = None,
        torque_pulse: float | None = None,
        pipe_temp_c: float | None = None,
    ) -> None:
        """Publish the current operating point. Cheap and safe to call often.

        ``crank_cycles`` is the model's unwrapped firing-cycle count
        (``EngineModel.firing_cycles``). Supplying it locks the synthesiser's
        event grid to the physics crank instead of letting it free-run from rpm,
        so every firing lands on a real piston position and transients
        (backfire, limiter cut, lift-off) occur where the model says they did.
        ``None`` leaves the grid free-running, which is what the offline loop and
        event renderers want.

        ``torque_pulse`` is ``EngineModel.torque_envelope`` normalised against
        the model's full-load estimate -- the Wiebe pulse train of stage 7.
        Non-None makes every firing event in the synthesiser track the
        *instantaneous* physics torque: the audible "lugging" of a firing
        under load, the relaxation between firings at low rpm. ``None``
        disables the coupling (offline loop renders, calibration probes).

        ``pipe_temp_c`` is ``EngineModel.pipe_temp`` -- the heat-soaked exhaust
        gas temperature. The synthesiser retunes its pipe comb with the speed
        of sound in it, so the quarter-wave resonance climbs as the collector
        heats up over a hard pull and sags on the overrun. ``None`` keeps the
        preset's static temperature.

        An unwrapped count rather than a 0..1 phase: one audio block spans more
        than a cycle above ~1400 rpm, so a wrapped phase cannot be interpolated
        across a block without aliasing.
        """
        with self._lock:
            self._params = (
                float(rpm),
                float(load),
                float(throttle),
                min(1.0, max(0.0, float(boost))),
                min(1.0, max(0.0, float(knock))),
                None if crank_cycles is None else float(crank_cycles),
                None if torque_pulse is None else max(0.0, float(torque_pulse)),
                None if pipe_temp_c is None else float(pipe_temp_c),
            )

    def trigger(self, name: str) -> None:
        """Queue a one-shot by event key (for example ``"backfire"``)."""
        with self._lock:
            self._triggers.append(name)

    def set_volume(self, value: float) -> None:
        """Master gain. Read once per block by the audio thread."""
        self.master = max(0.0, min(1.5, float(value)))

    # ------------------------------------------------------------------ #
    # synthesis (audio thread)
    # ------------------------------------------------------------------ #
    def _produce(self, out: np.ndarray) -> None:
        if self.mode == "assets":
            self._produce_assets(out)
            return
        voice = self.voice
        if voice is None:
            out.fill(0)
            return

        with self._lock:
            (
                rpm, load, throttle, boost, knock, crank_cycles, torque_pulse, pipe_temp_c
            ) = self._params
            triggers = self._triggers
            self._triggers = []

        voice.set_params(
            rpm=rpm, load=load, throttle=throttle,
            crank_cycles=crank_cycles, torque_pulse=torque_pulse,
            pipe_temp_c=pipe_temp_c,
        )
        # PortAudio normally delivers exactly ``blocksize`` frames, but asking
        # the voice for the real output length keeps the callback correct if a
        # host API hands us a different block at startup or shutdown.
        stereo = self.channels >= 2 and hasattr(voice, "process_stereo")
        if stereo:
            frame = voice.process_stereo(int(out.shape[0]))
        else:
            frame = np.column_stack(
                [voice.process(int(out.shape[0]))] * max(1, out.shape[1])
            )

        for name in triggers:
            variants = self._shots.get(name)
            if not variants:
                continue
            variant = variants[self._shot_index % len(variants)]
            self._shot_index += 1
            self._active.append([variant, 0])

        n = frame.shape[0]
        for entry in self._active:
            samples, cursor = entry
            take = min(n, samples.shape[0] - cursor)
            if take > 0:
                frame[:take] += samples[cursor : cursor + take, None]
                entry[1] += take
        if self._active:
            self._active = [e for e in self._active if e[1] < e[0].shape[0]]

        # loop assets are mono ambience: they sit centred in both channels
        loop_buf = np.zeros(n, dtype=np.float64)
        self._mix_loops(loop_buf, boost, knock)

        if self.master != 1.0:
            frame *= self.master
        frame += loop_buf[:, None]
        np.clip(frame, -1.0, 1.0, out=frame)
        # Round rather than truncate: truncation toward zero adds a small
        # amount of odd-harmonic distortion at low levels.
        pcm = np.rint(frame * 32767.0).astype(np.int16)
        for ch in range(out.shape[1]):
            out[:, ch] = pcm[:, min(ch, pcm.shape[1] - 1)]

    def _produce_assets(self, out: np.ndarray) -> None:
        """Asset-mode callback: playback only, ~40 us, no synthesis.

        The loop bank renders in the background at preset load; until it
        finishes (a one-time 1-3 s per preset) this emits silence so the
        device exists and the GUI never blocks on synthesis. Once ready,
        the bed is the bank blend at the live operating point, one-shots
        mix on top exactly as in procedural mode, and the turbo/knock
        texture loops ride underneath.
        """
        n = int(out.shape[0])
        with self._lock:
            (
                rpm, load, throttle, boost, knock, _cycles, torque_pulse, pipe_temp_c
            ) = self._params
            triggers = self._triggers
            self._triggers = []

        player = self.player
        if player is None or not self.bank.entries:
            out.fill(0)
            return
        if not player.ready:
            # The bank just finished rendering in the background: the
            # callback is the first code that runs after it, so THIS is
            # where the player gets prepared. (The live path never calls
            # prepare() explicitly -- the silence bug was exactly this
            # check guarding an unprimed player forever.)
            if not self.bank.ready.is_set():
                out.fill(0)
                return
            player.set_state(
                rpm, load, throttle, torque_pulse, pipe_temp_c, self._limiter_level
            )
            player.prepare()
        # Bank just became ready: fade in over ~150 ms so the entry is not
        # a hard cut from silence.
        if self._bank_warm < 1.0:
            self._bank_warm = min(1.0, self._bank_warm + n / (0.15 * self.sr))

        # operating-point -> blend targets (main-thread side of the player,
        # but cheap enough to run inline; it is pure arithmetic)
        player.set_state(
            rpm, load, throttle, torque_pulse, pipe_temp_c, self._limiter_level
        )
        if self._bed_scratch is None or self._bed_scratch.size < n:
            self._bed_scratch = np.zeros(n, dtype=np.float64)
        bed = self._bed_scratch[:n]
        bed.fill(0.0)
        player.produce(n, bed)
        bed *= self._bank_warm

        for name in triggers:
            variants = self._shots.get(name)
            if not variants:
                continue
            variant = variants[self._shot_index % len(variants)]
            self._shot_index += 1
            self._active.append([variant, 0])
        for entry in self._active:
            samples, cursor = entry
            take = min(n, samples.shape[0] - cursor)
            if take > 0:
                bed[:take] += samples[cursor : cursor + take]
                entry[1] += take
        if self._active:
            self._active = [e for e in self._active if e[1] < e[0].shape[0]]

        self._mix_loops(bed, boost, knock)
        if self.master != 1.0:
            bed *= self.master
        np.clip(bed, -1.0, 1.0, out=bed)
        pcm = np.rint(bed * 32767.0).astype(np.int16)
        for ch in range(out.shape[1]):
            out[:, ch] = pcm[:n]

    def set_limiter(self, level: float) -> None:
        """Publish the rev-limiter cut state (0..1) for the limiter layer."""
        self._limiter_level = min(max(float(level), 0.0), 1.0)

    def _mix_loops(self, mono: np.ndarray, boost: float, knock: float) -> None:
        """Mix the seamless loop assets under the engine voice.

        ``turbo_spool`` tracks compressor speed: silent at atmospheric, louder
        and played faster as boost comes up. ``knock`` tracks the detonation
        intensity from the physics, with a faster ping rate at high intensity.
        Both gains are smoothed so a throttle transition does not click.
        """
        if not self._loops:
            return
        n = mono.shape[0]
        if n <= 0:
            return
        frames = np.arange(n, dtype=np.float64)
        for key, entry in self._loops.items():
            samples, cursor, gain, spline = entry
            if key == "turbo_spool":
                target = boost * TURBO_LOOP_GAIN
                rate = 0.65 + 1.00 * boost
            elif key == "knock":
                target = knock * KNOCK_LOOP_GAIN
                rate = 1.00 + 0.80 * knock
            else:
                continue
            tau = 0.05 if target > gain else 0.18
            gain += (target - gain) * (1.0 - math.exp(-n / (self.sr * tau)))
            entry[2] = gain
            if gain < 1e-4:
                continue
            length = samples.shape[0]
            if length == 0:
                continue
            pos = np.mod(cursor + frames * rate, length)
            if spline is not None:
                mono += spline(pos) * gain
            else:
                i0 = pos.astype(np.int64) % length
                frac = pos - np.floor(pos)
                i1 = (i0 + 1) % length
                mono += (samples[i0] * (1.0 - frac) + samples[i1] * frac) * gain
            entry[1] = (cursor + n * rate) % length

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> AudioStats:
        if self.stream is None:
            return AudioStats(errors=0, starved=0, latency_ms=0.0)
        return AudioStats(
            errors=self.stream.errors,
            starved=self.stream.starved,
            latency_ms=self.stream.latency_s * 1000.0,
        )

    def describe(self) -> str:
        if not self.enabled:
            return f"audio off ({self.error})" if self.error else "audio off"
        stream = self.stream
        if stream is None:
            return "audio stopping"
        return f"{self.stats} [{stream.error_report}]"
