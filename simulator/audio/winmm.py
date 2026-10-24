"""Streaming PCM output through WinMM ``waveOut``, via ``ctypes``.

This is the zero-dependency Windows fallback for when PortAudio/``sounddevice``
is unavailable. It is a small amount of code because the job is small: keep a
few buffers queued, refill each one as the device finishes it.

``waveOut`` is opened with ``CALLBACK_NULL`` and polled rather than with a
callback. That matters in Python: a callback fires on a system thread that must
not block, while polling lets a normal Python thread do the (comparatively slow)
synthesis and write buffers when they are free. ``timeBeginPeriod(1)`` raises the
timer resolution so a 1 ms sleep really is about 1 ms -- without it Windows can
round a short sleep up to ~15 ms and the queue runs dry.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes

import numpy as np

WAVE_MAPPER = ctypes.c_uint(-1).value
WAVE_FORMAT_PCM = 1
CALLBACK_NULL = 0x0000
WHDR_DONE = 0x00000001

__all__ = ["WaveOutStream", "AudioUnavailable"]


class AudioUnavailable(RuntimeError):
    """Raised when no output device can be opened."""


if sys.platform == "win32":
    class _WAVEFORMATEX(ctypes.Structure):
        # packed: the real structure is 18 bytes, not the 20 that default
        # alignment would give it
        _pack_ = 1
        _fields_ = [
            ("wFormatTag", wintypes.WORD),
            ("nChannels", wintypes.WORD),
            ("nSamplesPerSec", wintypes.DWORD),
            ("nAvgBytesPerSec", wintypes.DWORD),
            ("nBlockAlign", wintypes.WORD),
            ("wBitsPerSample", wintypes.WORD),
            ("cbSize", wintypes.WORD),
        ]

    class _WAVEHDR(ctypes.Structure):
        _fields_ = [
            ("lpData", ctypes.c_void_p),
            ("dwBufferLength", wintypes.DWORD),
            ("dwBytesRecorded", wintypes.DWORD),
            ("dwUser", ctypes.c_void_p),
            ("dwFlags", wintypes.DWORD),
            ("dwLoops", wintypes.DWORD),
            ("lpNext", ctypes.c_void_p),
            ("reserved", ctypes.c_void_p),
        ]


def _bind() -> ctypes.WinDLL:
    winmm = ctypes.WinDLL("winmm")
    winmm.waveOutOpen.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.POINTER(_WAVEFORMATEX),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    winmm.waveOutOpen.restype = ctypes.c_uint
    for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
        fn = getattr(winmm, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WAVEHDR), ctypes.c_uint]
        fn.restype = ctypes.c_uint
    winmm.waveOutClose.argtypes = [ctypes.c_void_p]
    winmm.waveOutClose.restype = ctypes.c_uint
    winmm.waveOutReset.argtypes = [ctypes.c_void_p]
    winmm.waveOutReset.restype = ctypes.c_uint
    return winmm


def _mmresult_text(code: int) -> str:
    known = {
        0: "no error",
        2: "bad device id",
        4: "cannot allocate memory",
        5: "generic error",
        6: "bad format",
        7: "device already open",
        11: "still playing",
        32: "no driver",
        33: "unsupported function",
        34: "no device",
        35: "not enabled",
    }
    return known.get(code, f"MMRESULT {code}")


class WaveOutStream:
    """A double-buffered 16-bit PCM output stream.

    ``fill`` is called on the stream's own thread with an ``int16`` array of
    shape ``(block_frames, channels)`` that it must overwrite with the next block
    of audio. The stream keeps ``buffers`` blocks in flight, so the worst-case
    latency is roughly ``(buffers - 1) * block_frames / samplerate``.
    """

    def __init__(
        self,
        samplerate: int = 48000,
        channels: int = 1,
        block_frames: int = 512,
        buffers: int = 5,
    ):
        if sys.platform != "win32":
            raise AudioUnavailable("the winmm audio backend is Windows-only")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")

        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.block_frames = int(block_frames)
        self.buffers = max(2, int(buffers))

        self._winmm = _bind()
        self._hwo = ctypes.c_void_p()
        self._running = False
        self._thread: threading.Thread | None = None
        self._timer_raised = False

        self._block_bytes = self.block_frames * self.channels * 2
        self._bytes = self._block_bytes * self.buffers
        self._raw = (ctypes.c_char * self._bytes)()
        self._base = ctypes.addressof(self._raw)
        self._headers = (_WAVEHDR * self.buffers)()
        self._queued = [False] * self.buffers

        self._errors = 0
        self._error_stages: dict[str, int] = {}
        self._last_fill_error: str | None = None
        self.starved = 0

        self._open()
        self._prepare()

    def _note_error(self, stage: str, code: int) -> None:
        self._errors += 1
        key = f"{stage}:{_mmresult_text(code)}" if code else stage
        self._error_stages[key] = self._error_stages.get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # device handling
    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        fmt = _WAVEFORMATEX(
            wFormatTag=WAVE_FORMAT_PCM,
            nChannels=self.channels,
            nSamplesPerSec=self.samplerate,
            nAvgBytesPerSec=self.samplerate * self.channels * 2,
            nBlockAlign=self.channels * 2,
            wBitsPerSample=16,
            cbSize=0,
        )
        rc = self._winmm.waveOutOpen(
            ctypes.byref(self._hwo), WAVE_MAPPER, ctypes.byref(fmt), None, None, CALLBACK_NULL
        )
        if rc != 0:
            raise AudioUnavailable(f"waveOutOpen failed: {_mmresult_text(rc)}")

    def _prepare(self) -> None:
        """Prepare every buffer once.

        A prepared header carries ``WHDR_PREPARED`` and must keep it: clearing
        ``dwFlags`` wholesale before re-queueing a used buffer drops that bit and
        ``waveOutWrite`` then rejects the header. Only ``WHDR_DONE`` is cleared
        when a buffer is recycled.
        """
        size = ctypes.sizeof(_WAVEHDR)
        for i in range(self.buffers):
            h = self._headers[i]
            h.lpData = self._base + i * self._block_bytes
            h.dwBufferLength = self._block_bytes
            h.dwFlags = 0
            rc = self._winmm.waveOutPrepareHeader(self._hwo, ctypes.byref(h), size)
            if rc != 0:
                self._note_error("prepare", rc)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self, fill) -> None:
        if self._running:
            return
        self._running = True
        # Without this, a 1 ms sleep can take 15 ms and the queue empties.
        try:
            self._winmm.timeBeginPeriod(1)
            self._timer_raised = True
        except Exception:  # pragma: no cover - depends on the host
            self._timer_raised = False
        self._thread = threading.Thread(target=self._loop, args=(fill,), name="engine-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.close()

    def close(self) -> None:
        try:
            self._winmm.waveOutReset(self._hwo)
        except Exception:
            pass
        size = ctypes.sizeof(_WAVEHDR)
        for i in range(self.buffers):
            try:
                self._winmm.waveOutUnprepareHeader(self._hwo, ctypes.byref(self._headers[i]), size)
            except Exception:
                pass
        try:
            self._winmm.waveOutClose(self._hwo)
        except Exception:
            pass
        if self._timer_raised:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._timer_raised = False

    # ------------------------------------------------------------------ #
    # producer thread
    # ------------------------------------------------------------------ #
    def _loop(self, fill) -> None:
        size = ctypes.sizeof(_WAVEHDR)
        out = np.zeros((self.block_frames, self.channels), dtype=np.int16)
        while self._running:
            wrote_any = False
            outstanding = 0
            for i in range(self.buffers):
                h = self._headers[i]
                done = bool(h.dwFlags & WHDR_DONE)
                if self._queued[i] and not done:
                    outstanding += 1
                    continue
                if self._queued[i]:
                    # the device has finished with this block; hand it back
                    h.dwFlags = h.dwFlags & ~WHDR_DONE
                    self._queued[i] = False

                try:
                    fill(out)
                except Exception as exc:  # never let a synthesis bug kill audio
                    self._note_error("fill", 0)
                    self._last_fill_error = f"{type(exc).__name__}: {exc}"
                    out.fill(0)

                ctypes.memmove(h.lpData, out.tobytes(), self._block_bytes)
                h.dwBufferLength = self._block_bytes
                rc = self._winmm.waveOutWrite(self._hwo, ctypes.byref(h), size)
                if rc != 0:
                    self._note_error("write", rc)
                    continue
                self._queued[i] = True
                wrote_any = True
                outstanding += 1

            if outstanding == 0 and self._running:
                # nothing is playing: the device ran dry between refills
                self.starved += 1
            if not wrote_any:
                time.sleep(0.001)

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    @property
    def errors(self) -> int:
        return self._errors

    @property
    def error_report(self) -> str:
        if not self._errors:
            return "no errors"
        parts = ", ".join(f"{k} x{v}" for k, v in sorted(self._error_stages.items()))
        if self._last_fill_error:
            parts += f"; last fill: {self._last_fill_error}"
        return parts

    @property
    def latency_s(self) -> float:
        return (self.buffers - 1) * self.block_frames / float(self.samplerate)

    def __enter__(self) -> WaveOutStream:
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
