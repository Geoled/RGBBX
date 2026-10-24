"""WAV I/O.

``soundfile``/libsndfile is used when available because it writes correct
24-bit PCM, handles multichannel layouts and reads more formats.  The standard
library fallback keeps the offline renderer usable in a bare Python install.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

try:  # optional, but strongly preferred
    import soundfile as _sf  # type: ignore
except Exception:  # pragma: no cover - depends on the host
    _sf = None


def _to_pcm_bytes(x: np.ndarray, bit_depth: int) -> bytes:
    if bit_depth == 16:
        v = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        return (v * 32767.0).astype("<i2").tobytes()
    if bit_depth == 24:
        v = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        i = np.round(v * 8388607.0).astype(np.int32)
        b = np.empty(i.size * 3, dtype=np.uint8)
        b[0::3] = (i & 0xFF).astype(np.uint8)
        b[1::3] = ((i >> 8) & 0xFF).astype(np.uint8)
        b[2::3] = ((i >> 16) & 0xFF).astype(np.uint8)
        return b.tobytes()
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def _dither(x: np.ndarray, bit_depth: int) -> np.ndarray:
    """TPDF dither at the target quantiser step.

    Dither is deliberately *not* on by default: it makes the low bits of a
    rendered loop non-periodic.  ``render --dither`` opts in for one-shot
    assets, where it removes the correlated quantisation distortion that a
    quiet tail can otherwise reveal.
    """
    if bit_depth not in (16, 24):
        return x
    lsb = 1.0 / float((1 << (bit_depth - 1)) - 1)
    rng = np.random.default_rng(0x5EED)
    noise = rng.random(x.shape) - rng.random(x.shape)
    return x + noise * lsb


def write_wav(
    path: str | Path,
    data: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
    dither: bool = False,
) -> Path:
    """Write ``data`` to a WAV file.

    ``data`` may be 1-D (mono) or 2-D shaped ``(frames, channels)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        channels = 1
    elif x.ndim == 2:
        channels = int(x.shape[1])
    else:
        raise ValueError("data must be 1-D or 2-D (frames, channels)")
    if dither:
        x = _dither(x, bit_depth)

    if _sf is not None:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth)
        if subtype is None:
            raise ValueError(f"unsupported bit depth: {bit_depth}")
        _sf.write(str(path), x, int(sample_rate), subtype=subtype, format="WAV")
        return path

    interleaved = x.reshape(-1)
    frames = _to_pcm_bytes(interleaved, bit_depth)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bit_depth // 8)
        w.setframerate(int(sample_rate))
        w.writeframes(frames)
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file. Returns ``(data, sample_rate)``.

    Mono data is 1-D; multichannel is ``(frames, channels)`` in [-1, 1].
    """
    if _sf is not None:
        x, sr = _sf.read(str(path), dtype="float64", always_2d=False)
        return np.asarray(x, dtype=np.float64), int(sr)

    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i >= 0x800000, i - 0x1000000, i)
        x = i.astype(np.float64) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if ch > 1:
        x = x.reshape(-1, ch)
    return x, sr
