"""Bridge to the native DSP core, with the numpy path as the reference.

The voice is written against this module, never against ``_dspcore`` directly,
so the same code runs with and without a compiler:

* :class:`Convolver` -- uniformly partitioned overlap-save FIR. The native build
  is 6-10x faster on long kernels because its cost per block does not grow with
  kernel length, while a single-shot overlap-add has to transform at
  ``next_pow2(block + taps)``.
* :class:`Comb` -- the fractional-delay feed-forward pipe.
* :func:`master` -- calibrated gain, character saturation and the safety
  ceiling in one pass over the block.
* :func:`rfft` / :func:`irfft` -- real transforms for the analysis helpers.

The fallback is *the same algorithm*, not a different one: the same partitions,
the same multiply order, the same window offset. That makes it a reference the
native build can be tested against sample for sample, and it guarantees the
offline renderer does not change character depending on whether a compiler
happened to be installed.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

AVAILABLE = False
REASON = "not loaded"
_core = None


def _load() -> None:
    global AVAILABLE, REASON, _core
    if os.environ.get("RGBBX_NO_NATIVE"):
        REASON = "disabled by RGBBX_NO_NATIVE"
        return
    try:
        import importlib

        _core = importlib.import_module("_dspcore")
    except Exception as exc:  # pragma: no cover - depends on the machine
        REASON = f"{type(exc).__name__}: {exc}"
        _core = None
        return
    AVAILABLE = True
    REASON = "native"


_load()


def describe() -> str:
    """One line for the status bar and the quality gate."""
    return "native" if AVAILABLE else f"numpy ({REASON})"


def next_pow2(n: int) -> int:
    p = 1
    n = max(int(n), 1)
    while p < n:
        p <<= 1
    return p


class _NumpyConvolver:
    """Reference implementation: same partitions, same order as the native core.

    Uniformly partitioned overlap-add (Gardner): the kernel is cut into
    ``block``-sized partitions, each pre-transformed once at ``2*block`` points
    zero-padded to the upper half. The frequency delay line keeps the spectra of
    past input blocks; partition ``p`` multiplies the block ``p`` steps back.
    The inverse transform of each product is ``2*block`` long: the first half is
    the current block's output, the second half is a tail carried into the next
    block. That tail carry is what makes a partitioned sum equal to the plain
    convolution.
    """

    def __init__(self, kernel: np.ndarray, block: int):
        k = np.asarray(kernel, dtype=np.float64).ravel()
        if k.size == 0:
            k = np.array([1.0])
        self.block_size = int(block)
        self.n_fft = next_pow2(2 * self.block_size)
        self.m = self.n_fft // 2
        self.partitions = max(1, (k.size + self.block_size - 1) // self.block_size)
        padded = np.zeros(self.partitions * self.block_size)
        padded[: k.size] = k
        self.H = [
            np.fft.rfft(padded[p * self.block_size : (p + 1) * self.block_size], self.n_fft)
            for p in range(self.partitions)
        ]
        self.hist = [np.zeros(self.m + 1, dtype=np.complex128) for _ in range(self.partitions)]
        self.pos = 0
        self.tail = np.zeros(self.block_size, dtype=np.float64)

    def reset(self) -> None:
        self.hist = [np.zeros(self.m + 1, dtype=np.complex128) for _ in range(self.partitions)]
        self.pos = 0
        self.tail[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        out = np.empty(n, dtype=np.float64)
        b = self.block_size
        for pos in range(0, n, b):
            blk = x[pos : pos + b]
            spec = np.fft.rfft(blk, self.n_fft)
            self.hist[self.pos] = spec
            acc = np.zeros(self.m + 1, dtype=np.complex128)
            idx = self.pos
            for p in range(self.partitions):
                acc += self.H[p] * self.hist[idx]
                idx = self.partitions - 1 if idx == 0 else idx - 1
            self.pos = (self.pos + 1) % self.partitions
            y = np.fft.irfft(acc, self.n_fft)
            out[pos : pos + b] = y[:b] + self.tail
            self.tail = y[b:].copy()
        return out


class Convolver:
    """Partitioned FIR convolver: native when built, numpy reference otherwise."""

    __slots__ = ("_impl", "block", "native", "partitions", "fft_size")

    def __init__(self, kernel, block: int):
        self.block = int(block)
        if AVAILABLE:
            arr = np.ascontiguousarray(np.asarray(kernel, dtype=np.float64).ravel())
            if arr.size == 0:
                arr = np.array([1.0])
            self._impl = _core.Convolver(arr, self.block)
            self.native = True
            self.partitions = int(self._impl.partitions)
            self.fft_size = int(self._impl.fft_size)
        else:
            self._impl = _NumpyConvolver(kernel, self.block)
            self.native = False
            self.partitions = int(self._impl.partitions)
            self.fft_size = int(self._impl.n_fft)

    def reset(self) -> None:
        self._impl.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        if x.size % self.block != 0:
            raise ValueError(f"Convolver expects a multiple of block={self.block}, got {x.size}")
        return self._impl.process(x)


class _NumpyComb:
    def __init__(self, sr: int, max_delay_s: float, repeats: int, gain: float, invert: bool, block: int):
        self.sr = int(sr)
        self.repeats = max(int(repeats), 1)
        self.gain = float(gain)
        self.invert = bool(invert)
        self.block = int(block)
        self.maxd = max(int(np.ceil(float(max_delay_s) * self.sr)), 2)
        self.size = self.block + self.maxd * self.repeats + 4
        self.buf = np.zeros(self.size, dtype=np.float64)
        self.w = 0

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.w = 0

    def process(self, x: np.ndarray, delay_s: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n == 0:
            return x
        if n + self.maxd * self.repeats + 4 > self.buf.size:
            self.buf = np.zeros(n + self.maxd * self.repeats + 4, dtype=np.float64)
            self.size = self.buf.size
            self.w = 0
        t = self.w + np.arange(n, dtype=np.float64)
        self.buf[(t % self.size).astype(np.int64)] = x
        y = x.copy()
        d = min(max(float(delay_s) * self.sr, 1.0), float(self.maxd))
        sign = -1.0 if self.invert else 1.0
        g = 1.0
        for k in range(1, self.repeats + 1):
            g *= self.gain * sign
            p = t - k * d
            i0 = np.floor(p).astype(np.int64)
            frac = p - i0
            a = self.buf[i0 % self.size]
            b = self.buf[(i0 + 1) % self.size]
            y += g * (a + frac * (b - a))
        self.w = int((self.w + n) % self.size)
        return y


class Comb:
    """Fractional-delay feed-forward comb (quarter-wave pipe model)."""

    __slots__ = ("_impl", "native")

    def __init__(self, sr: int, max_delay_s: float, repeats: int, gain: float, invert: bool, block: int):
        if AVAILABLE:
            self._impl = _core.Comb(
                int(sr), float(max_delay_s), int(repeats), float(gain), bool(invert), int(block)
            )
            self.native = True
        else:
            self._impl = _NumpyComb(sr, max_delay_s, repeats, gain, invert, block)
            self.native = False

    def reset(self) -> None:
        self._impl.reset()

    def process(self, x: np.ndarray, delay_s: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        return self._impl.process(x, float(delay_s))


def master(x: np.ndarray, gain: float, drive: float, ceiling: float, knee: float) -> np.ndarray:
    """Gain -> character saturation -> safety ceiling, in one pass."""
    x = np.asarray(x, dtype=np.float64)
    if AVAILABLE:
        return _core.master(np.ascontiguousarray(x), float(gain), float(drive), float(ceiling), float(knee))
    y = x * float(gain)
    if float(drive) > 1.0001:
        d = float(drive)
        y = np.tanh(y * d) / np.tanh(d)
    c = max(float(ceiling), 1e-3)
    k = max(min(float(knee), 0.999) * c, 1e-6)
    span = c - k
    over = np.abs(y) > k
    if np.any(over):
        m = np.abs(y[over])
        y[over] = np.sign(y[over]) * (k + span * np.tanh((m - k) / span))
    return y


def rfft(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if AVAILABLE:
        re, im = _core.rfft(np.ascontiguousarray(arr))
        return re + 1j * im
    return np.fft.rfft(arr)


def irfft(spec: np.ndarray, n: int) -> np.ndarray:
    spec = np.asarray(spec)
    if AVAILABLE:
        return _core.irfft(
            np.ascontiguousarray(spec.real), np.ascontiguousarray(spec.imag), int(n)
        )
    return np.fft.irfft(spec, int(n))
