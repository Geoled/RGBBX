"""Low-latency audio output through PortAudio (``sounddevice``).

The WinMM backend in :mod:`simulator.audio.winmm` is a useful zero-dependency
fallback, but PortAudio gives us device selection, WASAPI/ASIO on Windows and a
much smaller buffer latency.  The rest of the audio path does not care which
backend is underneath: both expose ``start(fill)``/``stop()`` and call ``fill``
with an ``int16`` block of shape ``(frames, channels)``.

``sounddevice`` is intentionally optional.  If it is not installed, or no
output device can be opened, :class:`EngineAudio` falls back to WinMM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .winmm import AudioUnavailable

__all__ = ["SoundDeviceStream"]


def _default_low_latency_device(sd) -> int | None:
    """Prefer the WASAPI default output on Windows.

    PortAudio's global default on Windows is usually the MME mapper, which can
    add ~100 ms even with ``latency="low"``. WASAPI exposes the same physical
    endpoint with a much smaller buffer. If anything about the query fails we
    simply return ``None`` and let PortAudio choose.
    """
    try:
        for host in sd.query_hostapis():
            name = str(host.get("name", "")).lower()
            if "wasapi" not in name:
                continue
            index = int(host.get("default_output_device", -1))
            if index < 0:
                continue
            info = sd.query_devices(index)
            if int(info.get("max_output_channels", 0)) > 0:
                return index
    except Exception:
        pass
    return None


class SoundDeviceStream:
    """PortAudio output stream with the same contract as ``WaveOutStream``."""

    def __init__(
        self,
        samplerate: int = 48000,
        channels: int = 1,
        block_frames: int = 512,
        latency: str | float = "low",
        device: int | str | None = None,
        auto_low_latency: bool = True,
    ):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise AudioUnavailable(
                "sounddevice is not installed; install it with: pip install sounddevice"
            ) from exc

        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")

        self._sd = sd
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.block_frames = int(block_frames)
        self.latency = latency
        if device is None and auto_low_latency:
            device = _default_low_latency_device(sd)
        self.device = device
        self._fill: Callable[[Any], None] | None = None
        self._running = False
        self.errors = 0
        self.starved = 0

        try:
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.block_frames,
                dtype="int16",
                latency=self.latency,
                device=self.device,
                callback=self._callback,
            )
        except Exception as exc:
            raise AudioUnavailable(f"cannot open sounddevice output: {exc}") from exc

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.starved += 1
        fill = self._fill
        if fill is None:
            outdata.fill(0)
            return
        try:
            fill(outdata)
        except Exception:
            # Never let an exception escape into PortAudio's callback thread:
            # it would kill the stream and leave the UI without sound.
            self.errors += 1
            outdata.fill(0)

    def start(self, fill: Callable[[Any], None]) -> None:
        self._fill = fill
        try:
            self._stream.start()
        except Exception as exc:
            try:
                self._stream.close()
            except Exception:
                pass
            raise AudioUnavailable(f"cannot start sounddevice stream: {exc}") from exc
        self._running = True

    def stop(self) -> None:
        self._running = False
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    @property
    def latency_s(self) -> float:
        try:
            return float(self._stream.latency)
        except Exception:
            return 0.0

    @property
    def latency_ms(self) -> float:
        return self.latency_s * 1000.0

    @property
    def error_report(self) -> str:
        if not self.errors:
            return "no errors"
        return f"{self.errors} callback errors"
