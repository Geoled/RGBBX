"""Real-time playback and performance measurement.

The synthesiser is block-based, so feeding it from an audio callback gives
live sound with no extra machinery. ``sounddevice`` is an *optional*
dependency: it is only needed to actually open a speaker stream. Throughput
benchmarking works without it and proves the engine can run faster than
real time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .config import Preset
from .voice import EngineVoice, calibrate_gain


@dataclass
class BenchResult:
    seconds: float
    elapsed: float
    realtime_factor: float
    cpu_percent: float
    sample_rate: int
    block_size: int

    def __str__(self) -> str:
        return (
            f"{self.realtime_factor:5.2f}x real time "
            f"({self.cpu_percent:5.1f}% of one core) "
            f"@ {self.sample_rate} Hz, block {self.block_size}"
        )


def bench(
    preset: Preset,
    seconds: float = 3.0,
    sr: int | None = None,
    block: int | None = None,
    sweep: bool = True,
) -> BenchResult:
    """Measure synthesis throughput without touching audio hardware."""
    sr = int(sr or preset.render.sample_rate)
    block = int(block or 512)
    gain = calibrate_gain(preset, sr=sr, block=block)
    voice = EngineVoice(preset, rpm=preset.engine.idle_rpm, sr=sr, block=block, output_gain=gain)

    total = int(seconds * sr)
    done = 0
    start = time.perf_counter()
    while done < total:
        if sweep:
            frac = done / max(total, 1)
            rpm = preset.engine.idle_rpm + frac * (preset.engine.redline_rpm - preset.engine.idle_rpm)
            voice.set_params(rpm=rpm, load=min(frac * 1.4, 1.0), throttle=min(frac * 1.4, 1.0))
        voice.process(block)
        done += block
    elapsed = time.perf_counter() - start
    return BenchResult(
        seconds=total / sr,
        elapsed=elapsed,
        realtime_factor=(total / sr) / max(elapsed, 1e-9),
        cpu_percent=100.0 * elapsed / max(total / sr, 1e-9),
        sample_rate=sr,
        block_size=block,
    )


def _require_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Live playback needs the optional 'sounddevice' package.\n"
            "Install it with:  pip install sounddevice\n"
            "You can still render offline and benchmark with:  python -m soundgen bench"
        ) from exc
    return sd


def _default_output_device(sd, min_channels: int = 1) -> int | None:
    """WASAPI default output on Windows, otherwise PortAudio's choice."""
    try:
        for host in sd.query_hostapis():
            if "wasapi" not in str(host.get("name", "")).lower():
                continue
            index = int(host.get("default_output_device", -1))
            if index >= 0 and int(sd.query_devices(index).get("max_output_channels", 0)) >= min_channels:
                return index
    except Exception:
        pass
    return None


def play(
    preset: Preset,
    seconds: float | None = None,
    sr: int | None = None,
    block: int = 512,
    rpm: float | None = None,
    load: float = 0.5,
    throttle: float = 0.5,
    interactive: bool = False,
) -> None:
    """Stream the engine to the default output device.

    With ``interactive=True`` you can drive it from the keyboard (Windows
    first, then a portable fallback): ``W``/``S`` throttle, ``A``/``D`` rpm,
    ``Q`` quit.
    """
    sd = _require_sounddevice()
    sr = int(sr or preset.render.sample_rate)
    gain = calibrate_gain(preset, sr=sr, block=block)

    state = {
        "rpm": float(rpm if rpm is not None else preset.engine.idle_rpm),
        "load": float(load),
        "throttle": float(throttle),
        "quit": False,
    }
    voice = EngineVoice(
        preset,
        rpm=state["rpm"],
        load=state["load"],
        throttle=state["throttle"],
        sr=sr,
        block=block,
        output_gain=gain,
    )

    def callback(outdata, frames, time_info, status):  # noqa: ANN001
        voice.set_params(rpm=state["rpm"], load=state["load"], throttle=state["throttle"])
        x = voice.process(frames)
        if outdata.ndim == 2 and outdata.shape[1] > 1:
            outdata[:, 0] = x
            outdata[:, 1] = x * 0.97
        else:
            outdata[:] = x.reshape(-1, 1)

    thread = None
    if interactive:
        thread = threading.Thread(target=_keyboard_loop, args=(state, preset), daemon=True)
        thread.start()

    print(f"Streaming '{preset.name}' at {sr} Hz. " + ("W/S throttle, A/D rpm, Q quit." if interactive else "Ctrl+C to stop."))
    stream = sd.OutputStream(
        samplerate=sr,
        channels=2,
        blocksize=block,
        callback=callback,
        dtype="float32",
        latency="low",
        device=_default_output_device(sd, min_channels=2),
    )
    stream.start()
    try:
        deadline = None if seconds is None else time.time() + seconds
        while stream.active:
            if state["quit"]:
                break
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


def _keyboard_loop(state: dict, preset: Preset) -> None:  # pragma: no cover - interactive
    span = preset.engine.redline_rpm - preset.engine.idle_rpm
    try:
        import msvcrt  # type: ignore

        while not state["quit"]:
            if msvcrt.kbhit():
                ch = msvcrt.getch().lower()
                _apply_key(ch.decode(errors="ignore"), state, preset, span)
            time.sleep(0.01)
        return
    except ImportError:
        pass

    try:
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not state["quit"]:
                ch = sys.stdin.read(1).lower()
                _apply_key(ch, state, preset, span)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return


def _apply_key(ch: str, state: dict, preset: Preset, span: float) -> None:  # pragma: no cover
    if ch == "w":
        state["throttle"] = min(1.0, state["throttle"] + 0.08)
        state["rpm"] = min(preset.engine.redline_rpm, state["rpm"] + 0.04 * span)
        state["load"] = min(1.0, state["load"] + 0.06)
    elif ch == "s":
        state["throttle"] = max(0.0, state["throttle"] - 0.08)
        state["rpm"] = max(preset.engine.idle_rpm, state["rpm"] - 0.06 * span)
        state["load"] = max(0.0, state["load"] - 0.06)
    elif ch == "a":
        state["rpm"] = max(preset.engine.idle_rpm, state["rpm"] - 0.03 * span)
    elif ch == "d":
        state["rpm"] = min(preset.engine.redline_rpm, state["rpm"] + 0.03 * span)
    elif ch in ("q", "\x1b"):
        state["quit"] = True
