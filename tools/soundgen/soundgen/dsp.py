"""Streaming DSP primitives for the engine-sound synthesizer.

Design notes
------------
The synthesizer must run in **real time** (streaming blocks) *and* render
**offline at high quality**. To make both paths produce the same sound, every
filter here is a stateful, block-based LTI processor built from FIR kernels:

* resonators  -> analytic damped-sinusoid impulse responses
* exhaust pipe -> comb (sum of delayed deltas)
* tone shaping -> linear-phase FIR designed from a magnitude function

All kernels are convolved with an overlap-add FFT convolver, which is fully
vectorised (numpy only, no scipy) and fast enough for real-time.

Pure numpy. No external dependencies.
"""

from __future__ import annotations

import functools
import math

import numpy as np

TWO_PI = 2.0 * np.pi

# The rate every preset's numbers were authored against. Kernel sizes below are
# written as tap counts *at this rate* and rescaled for any other rate, so a
# filter keeps its transition band in hertz instead of getting sharper as the
# device rate climbs.
REF_SR = 48000


def taps_for(sr: int, ref_taps: int) -> int:
    """Odd tap count that holds ``ref_taps`` worth of *duration* at any rate.

    A linear-phase FIR designed with a fixed tap count gets a narrower
    transition band -- and a different phase response -- as the sample rate
    rises: 513 taps is 10.7 ms at 48 kHz but 5.3 ms at 96 kHz, so the same
    cutoff leaks differently. Scaling the count by ``sr / REF_SR`` keeps the
    window length in seconds constant, which is what makes a preset sound the
    same on a 44.1 kHz, 48 kHz and 96 kHz device.
    """
    n = int(round(int(ref_taps) * float(sr) / float(REF_SR)))
    return max(n | 1, 3)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def lin_to_db(value: float) -> float:
    v = float(abs(value))
    return 20.0 * np.log10(v) if v > 1e-12 else -240.0


def next_pow2(n: int) -> int:
    n = max(1, int(n))
    p = 1
    while p < n:
        p <<= 1
    return p


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def peak(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


# --------------------------------------------------------------------------- #
# loudness
# --------------------------------------------------------------------------- #
# BS.1770 K-weighting: a +4 dB high shelf above ~1.5 kHz followed by a
# 2nd-order high-pass at ~38 Hz. The reference implementation gives biquad
# coefficients that only hold at 48 kHz, so both stages are re-derived here for
# whatever rate the device runs at (RBJ cookbook form) and evaluated
# analytically. That keeps the measurement rate-independent, which matters
# because loudness calibration must not itself drift with the sample rate.
_K_SHELF_HZ = 1681.974450955533
_K_SHELF_DB = 3.999843853973347
_K_SHELF_Q = 0.7071752369554196
_K_HP_HZ = 38.13547087602444
_K_HP_Q = 0.5003270373238773


def _biquad_high_shelf(sr: int, fc: float, gain_db: float, q: float):
    a = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * fc / float(sr)
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    sa = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1) + (a - 1) * cw + sa)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - sa)
    a0 = (a + 1) - (a - 1) * cw + sa
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - sa
    return (b0, b1, b2, a0, a1, a2)


def _biquad_high_pass(sr: int, fc: float, q: float):
    w0 = TWO_PI * fc / float(sr)
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = (1 + cw) / 2.0
    b1 = -(1 + cw)
    b2 = (1 + cw) / 2.0
    a0 = 1 + alpha
    a1 = -2 * cw
    a2 = 1 - alpha
    return (b0, b1, b2, a0, a1, a2)


def _biquad_mag(coeffs, f, sr: int):
    """Exact |H(e^{jw})| of a biquad at frequencies ``f``."""
    b0, b1, b2, a0, a1, a2 = coeffs
    w = TWO_PI * np.asarray(f, dtype=np.float64) / float(sr)
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    num = b0 + b1 * z1 + b2 * z2
    den = a0 + a1 * z1 + a2 * z2
    return np.abs(num / den)


def k_weight_mag(f, sr: int):
    """Magnitude of the BS.1770 K-weighting curve at frequencies ``f``."""
    shelf = _biquad_high_shelf(sr, _K_SHELF_HZ, _K_SHELF_DB, _K_SHELF_Q)
    hp = _biquad_high_pass(sr, _K_HP_HZ, _K_HP_Q)
    return _biquad_mag(shelf, f, sr) * _biquad_mag(hp, f, sr)


def octave_bands(
    x: np.ndarray,
    sr: int,
    edges: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Energy share per octave band, on an axis independent of the sample rate.

    Narrow (1/3-octave) bands are useless for comparing rates on this material:
    an engine is a *line* spectrum, so at 4000 rpm a V8 puts harmonics every
    267 Hz and a 1/3-octave band holds two or three of them. A few degrees of
    phase shift moves a line out of a band and reads as 3 dB of "timbre change"
    that is not audible at all. Octave bands hold 8-30 lines and are stable.
    """
    edges = edges or _OCTAVE_EDGES
    if x.size < 64:
        return np.zeros(len(edges))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))
    tot = float(spec.sum()) + 1e-30
    return np.array(
        [float(spec[(freqs >= lo) & (freqs < hi)].sum()) / tot for lo, hi in edges]
    )


_OCTAVE_EDGES: tuple[tuple[float, float], ...] = (
    (40.0, 80.0),
    (80.0, 160.0),
    (160.0, 320.0),
    (320.0, 640.0),
    (640.0, 1280.0),
    (1280.0, 2560.0),
    (2560.0, 5120.0),
    (5120.0, 10240.0),
)


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """K-weighted loudness in LUFS over the whole buffer.

    Full-length FFT with a rectangular window: Parseval then makes the result
    the exact K-weighted mean square, so no window or gate is needed for a
    stationary signal. This is what loudness calibration targets -- peak level
    is a poor proxy, because a spiky single-cylinder note and a smooth V12 can
    differ by 7 dB of crest at the same perceived level.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 16:
        return -120.0
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))
    w = k_weight_mag(freqs, sr)
    power = np.abs(spec) ** 2 * w**2
    # one-sided spectrum -> double every bin except DC and Nyquist
    if power.size > 2:
        power[1:-1] *= 2.0
    mean_square = float(power.sum()) / (x.size * x.size)
    return -0.691 + 10.0 * math.log10(max(mean_square, 1e-30))


def normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    p = peak(x)
    if p < 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) * (db_to_lin(peak_db) / p)


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """Smooth saturation. drive <= 1 is nearly linear."""
    drive = max(float(drive), 1e-6)
    if drive <= 1.0001:
        return np.asarray(x, dtype=np.float64)
    return np.tanh(np.asarray(x, dtype=np.float64) * drive) / np.tanh(drive)


def noise_burst(
    n: int,
    sr: int,
    rng: np.random.Generator,
    ref_sr: int = REF_SR,
) -> np.ndarray:
    """``n`` samples of white noise whose *realisation* does not depend on ``sr``.

    A plain ``rng.standard_normal(n)`` makes the sound depend on the output
    rate: 96 kHz draws twice as many samples as 48 kHz, so the two renders are
    different noise, not two renderings of one. Every kernel that embeds a noise
    burst then differs between rates even though its filter is rate-invariant,
    and a cross-rate comparison measures the random seed rather than the DSP.

    Drawing on the reference grid and resampling through the spectrum keeps the
    realisation (and therefore the timbre) the same at any rate. Resampling in
    the frequency domain rather than by interpolation matters for noise: linear
    interpolation is a low-pass, so upsampling would quietly darken the burst.
    """
    n = max(int(n), 1)
    if int(sr) == int(ref_sr):
        return rng.standard_normal(n)

    n_ref = max(int(round(n * float(ref_sr) / float(sr))), 2)
    base = rng.standard_normal(n_ref)
    if n == n_ref:
        return base

    spec = np.fft.rfft(base)
    target = n // 2 + 1
    if target <= spec.size:
        spec = spec[:target]
        # taper the cut so truncation does not ring across the whole burst
        if spec.size > 4:
            spec[-4:] *= np.linspace(1.0, 0.0, 4)
    else:
        spec = np.concatenate([spec, np.zeros(target - spec.size, dtype=spec.dtype)])
    out = np.fft.irfft(spec, n)
    # irfft of a resized spectrum loses amplitude proportional to the band it
    # spans; restore unit variance so ``noise_amount`` still means what it says
    sd = float(np.std(out))
    if sd > 1e-12:
        out *= 1.0 / sd
    return out


def resonance_compress(
    x: np.ndarray,
    threshold: float = 0.35,
    ratio: float = 2.0,
) -> np.ndarray:
    """2:1 soft stage above ``threshold``; bit-exact below it.

    The pipe's mode envelope makes the "resonance rpms" -- where the firing
    order slides onto a mode -- run 6-8 dB hotter than the operating sweep
    the shared calibration gain was measured on. Handing that swing to the
    safety limiter alone produced 45 % of samples in its knee at the V8's
    7325 rpm point and a 2.8 dB crest: audibly crushed, exactly at the rpm a
    driver holds. A single static gain cannot serve an 8 dB spread; this
    stage can:

    * below ``threshold`` (~-9 dBFS) nothing changes -- the limiter and this
      stage are both bit-exact in their linear regions, so the calibration's
      sweep average is untouched;
    * above it, excess magnitude is halved: the resonance peaks lose ~4 dB,
      the crest stays alive (measured 4.9 dB after), and the safety limiter
      returns to being a safety.

    No attack/release: the resonance is sustained, not transient, and a
    time-variant gain would both smear the firing-order AM the voice works
    to keep and break the offline loop's exact periodicity.
    """
    x = np.asarray(x, dtype=np.float64)
    thr = max(float(threshold), 1e-4)
    mag = np.abs(x)
    over = mag > thr
    if not np.any(over):
        return x
    excess = (mag[over] - thr) * (1.0 / max(float(ratio), 1.0))
    out = x.copy()
    out[over] = np.sign(x[over]) * (thr + excess)
    return out


def safety_limit(x: np.ndarray, ceiling: float = 0.891, knee: float = 0.70) -> np.ndarray:
    """Peak ceiling with a linear region, so quiet material passes untouched.

    ``soft_clip`` is a *character* stage: it distorts the whole waveform and
    saturates towards ``1/tanh(drive)``, which for a gentle drive is above full
    scale anyway. That makes it useless as a guarantee. This is the guarantee:

    * below ``knee * ceiling`` the signal is bit-exact (no colouring at all);
    * above it, a ``tanh`` whose slope matches at the knee bends the overshoot
      into the ceiling.

    Slope continuity at the knee is why there is no harmonic artefact on
    material that only occasionally crosses it -- the transfer function is C1,
    so the only distortion is on the peaks that need it. A catalogue-wide
    loudness target needs exactly this: an engine with a 14 dB crest factor has
    a handful of samples per firing that are far above the rest, and without a
    limiter those samples alone decide the gain for the whole preset, leaving it
    6 dB quieter than its neighbours.
    """
    x = np.asarray(x, dtype=np.float64)
    ceiling = max(float(ceiling), 1e-3)
    knee_abs = max(min(float(knee), 0.999) * ceiling, 1e-6)
    mag = np.abs(x)
    over = mag > knee_abs
    if not np.any(over):
        return x
    span = ceiling - knee_abs
    shaped = knee_abs + span * np.tanh((mag[over] - knee_abs) / span)
    out = x.copy()
    out[over] = np.sign(x[over]) * shaped
    return out


def fade_edges(x: np.ndarray, fade_in: int = 0, fade_out: int = 0) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    n = y.size
    fade_in = int(min(max(fade_in, 0), n))
    fade_out = int(min(max(fade_out, 0), n))
    if fade_in > 1:
        y[:fade_in] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in)))
    if fade_out > 1:
        y[-fade_out:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out)))
    return y


def equal_power_crossfade_loop(x: np.ndarray, fade_samples: int) -> np.ndarray:
    """Make a buffer seamlessly loopable.

    Blends the tail into the head with an equal-power crossfade, then trims the
    tail. The returned buffer's end connects smoothly to its start.
    """
    y = np.array(x, dtype=np.float64, copy=True)
    n = y.size
    fade = int(min(max(fade_samples, 0), n // 4))
    if fade <= 1:
        return y
    w = np.linspace(0.0, 1.0, fade)
    w_in = np.sin(w * np.pi / 2.0)
    w_out = np.cos(w * np.pi / 2.0)
    head = y[:fade].copy()
    tail = y[n - fade:].copy()
    y[:fade] = head * w_in + tail * w_out
    return y[: n - fade]


# --------------------------------------------------------------------------- #
# kernel builders (offline, run once per preset)
# --------------------------------------------------------------------------- #
def resonator_kernel(
    sr: int,
    modes,
    length_s: float = 0.12,
    tail_fade_s: float = 0.015,
    normalize: bool = True,
    q_taper: bool = True,
) -> np.ndarray:
    """Impulse response of a bank of damped resonators.

    ``modes`` is an iterable of ``(freq_hz, q, gain)``. Each mode contributes
    ``gain * exp(-pi*f*t/q) * sin(2*pi*f*t)``.

    ``normalize=True`` scales every mode so that its own spectral peak equals
    its ``gain``. Without that, a raw damped sinusoid has a resonance peak of
    roughly ``q*sr/(pi*f)`` -- 350x at 95 Hz, 18x at 2600 Hz -- so a mode list
    weighted 1.0/0.85/0.65/... silently turns into a 40 dB low-frequency shelf
    and the rest of the voice (intake, mechanical, flow) disappears under it.
    Normalising makes ``modes`` mean what it looks like: a frequency-response
    shape, with each ``gain`` the height of that formant.

    ``q_taper=True`` applies the physical loss profile of a real pipe around
    a 500 Hz hinge: wall/gas losses grow with frequency, so high modes are
    broad (Q scaled by ``(500/f)^0.25``) and weak (gain scaled by
    ``(500/f)^(1/3)``), while the low body modes keep only moderate Q --
    a tailpipe with a muffler is no organ pipe, so effective Q is clamped to
    3. The presets were authored with *rising* Q (1.6 at 84 Hz -> 2.85 at
    2.7 kHz), the opposite of physics; when a firing-order harmonic landed
    on one of those high-Q high modes the per-point loudness ran up to 8 dB
    over the sweep, and the shared calibration gain then crushed exactly
    those rpms (measured: 39 % of samples in the limiter knee at 7325 rpm on
    the V8). A one-sided taper made the low modes *more* resonant instead
    (42 % in the knee at 1500 rpm) -- the losses run both directions from
    the hinge, and the taper must too.
    """
    n = max(int(length_s * sr), 8)
    t = np.arange(n, dtype=np.float64) / sr
    h = np.zeros(n, dtype=np.float64)
    k = int(tail_fade_s * sr)
    window = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, k))) if 1 < k < n else None
    nf = next_pow2(n * 4)
    nyquist = sr * 0.5
    for mode in modes:
        f = max(float(mode[0]), 1.0)
        q = max(float(mode[1]), 0.5)
        g = float(mode[2])
        if f >= nyquist:
            # A resonator above Nyquist would fold back to a difference tone and
            # put a peak somewhere the preset never asked for. Modes are silent
            # above the band by definition, so drop them instead.
            continue
        if q_taper:
            # Physical loss profile, symmetric around 500 Hz (see
            # docstring): below it the exhaust's open end and muffler keep
            # Q moderate (a real tailpipe is no organ pipe), above it the
            # wall/gas losses make the modes broad *and* short. The first
            # attempt tapered only upward, which tripled the low modes' Q
            # and made the 1500 rpm hot point *worse* (42 % in the knee) --
            # the taper has to run both ways.
            q = q * (500.0 / f) ** 0.25
            q = min(max(q, 0.5), 3.0)
            g *= (500.0 / f) ** (1.0 / 3.0)
        decay = np.pi * f / q
        m = np.exp(-decay * t) * np.sin(TWO_PI * f * t)
        if window is not None:
            m[-k:] *= window
        if normalize:
            pk = float(np.max(np.abs(np.fft.rfft(m, nf))))
            if pk > 1e-12:
                g /= pk
        h += g * m
    h -= h.mean()
    return h


def comb_kernel(
    sr: int,
    delay_s: float,
    gain: float = 0.55,
    repeats: int = 4,
    invert: bool = False,
) -> np.ndarray:
    """Feed-forward comb: delta train at ``delay_s`` with decaying gain.

    With ``invert=True`` successive echoes alternate in sign, which is what a
    pipe open at one end does: the compression wave comes back from the open
    end as an expansion. That puts the resonance at ``c/(4L)`` (a quarter-wave
    tube) instead of ``c/(2L)``.
    """
    d = max(int(round(delay_s * sr)), 1)
    n = d * repeats + 1
    h = np.zeros(n, dtype=np.float64)
    g = 1.0
    s = -1.0 if invert else 1.0
    for k in range(repeats + 1):
        idx = k * d
        if idx >= n:
            break
        h[idx] += g
        g *= gain * s
    return h


def runner_kernel(
    sr: int,
    length_m: float,
    sound_speed: float,
    reflect: float = 0.42,
    repeats: int = 2,
    invert: bool = False,
) -> np.ndarray:
    """Primary-pipe reflection pattern for one cylinder.

    Each cylinder sits at the end of its own primary runner. The pulse leaves
    the valve, travels to the collector, partly reflects and comes back, so the
    collector sees the direct pulse plus decaying copies every ``2L/c``. Giving
    every cylinder a slightly different runner length is what stops the event
    train from being a perfectly coherent pulse train -- it is the difference
    between a synthesised "brrr" and a real header.
    """
    length_m = max(float(length_m), 0.01)
    d = max(int(round(2.0 * length_m / max(sound_speed, 1.0) * sr)), 1)
    n = d * repeats + 1
    h = np.zeros(n, dtype=np.float64)
    h[0] = 1.0
    g = 1.0
    s = -1.0 if invert else 1.0
    for k in range(1, repeats + 1):
        g *= reflect * s
        h[k * d] += g
    return h


class CombDelay:
    """Time-varying feed-forward comb built on a fractional-delay line.

    A baked FIR comb cannot follow the pipe: exhaust gas temperature rises with
    load, the speed of sound rises with it, and the quarter-wave resonance moves
    up by 10-15 %. A delay line can do that in place of a 2048-point FFT per
    block, so it is both more accurate and cheaper than the convolution it
    replaces.

    Reads are always at ``k * delay`` *before* the current sample, so the slot
    still holds the sample written at exactly that absolute time no matter how
    the write pointer wraps. Output is therefore bit-exactly periodic whenever
    the input is, which is what the offline loops rely on.
    """

    def __init__(self, sr: int, max_delay_s: float, repeats: int, gain: float, invert: bool = True, block: int = 1024):
        self.sr = int(sr)
        self.repeats = max(int(repeats), 1)
        self.gain = float(gain)
        self.invert = bool(invert)
        self.block = int(block)
        self.maxd = max(int(np.ceil(float(max_delay_s) * self.sr)), 2)
        self._alloc(self.block)

    def _alloc(self, block: int) -> None:
        need = int(block) + self.maxd * self.repeats + 4
        self.buf = np.zeros(need, dtype=np.float64)
        self.w = 0

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.w = 0

    def process(self, x: np.ndarray, delay_s: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n == 0:
            return x
        need = n + self.maxd * self.repeats + 4
        if self.buf.size < need:
            self._alloc(n)
        t = self.w + np.arange(n, dtype=np.float64)
        self.buf[(t % self.buf.size).astype(np.int64)] = x
        y = x.copy()
        d = min(max(float(delay_s) * self.sr, 1.0), float(self.maxd))
        sign = -1.0 if self.invert else 1.0
        g = 1.0
        size = self.buf.size
        for k in range(1, self.repeats + 1):
            g *= self.gain * sign
            p = t - k * d
            i0 = np.floor(p).astype(np.int64)
            frac = p - i0
            a = self.buf[i0 % size]
            b = self.buf[(i0 + 1) % size]
            y += g * (a + frac * (b - a))
        self.w = int((self.w + n) % size)
        return y


def mag_lowpass(f, fc, order: int = 2):
    fc = max(float(fc), 1.0)
    return 1.0 / np.sqrt(1.0 + (np.asarray(f, dtype=np.float64) / fc) ** (2 * order))


def mag_highpass(f, fc, order: int = 2):
    fc = max(float(fc), 1.0)
    r = np.asarray(f, dtype=np.float64) / fc
    return r**order / np.sqrt(1.0 + r ** (2 * order))


def mag_bandpass(f, lo, hi, order: int = 2):
    return mag_highpass(f, lo, order) * mag_lowpass(f, hi, order)


def mag_tilt(f, pivot: float, gain_db: float):
    """0 dB at DC, ``gain_db`` at high frequency. Controls brightness."""
    pivot = max(float(pivot), 1.0)
    r = np.asarray(f, dtype=np.float64) / pivot
    shape = r / (1.0 + r)
    return 10.0 ** (float(gain_db) * shape / 20.0)


def mag_tilt_oct(f, pivot: float, db_per_oct: float):
    """Constant ``db_per_oct`` slope above ``pivot``, flat below it.

    This is what broadband *noise* needs. ``mag_tilt`` saturates at a fixed
    shelf, which is right for a tone control but wrong for a noise bed: a shelf
    keeps the spectrum flat from the pivot upwards, so the top octaves stay as
    loud as the body. Real turbulent noise (intake, jet, shear layer) falls with
    roughly -3..-9 dB/octave, and that slope is most of what separates a hiss
    from a roar. The slope is smoothed with ``log2(1 + f/pivot)`` so there is no
    hard knee, and it approaches ``db_per_oct`` per octave well above the pivot.
    """
    pivot = max(float(pivot), 1.0)
    f = np.asarray(f, dtype=np.float64)
    shape = np.log2(1.0 + f / pivot)
    return 10.0 ** (float(db_per_oct) * shape / 20.0)


def spectral_kernel(sr: int, mag_fn, taps: int = 1025, dc_gain: float | None = None) -> np.ndarray:
    """Linear-phase FIR whose magnitude follows ``mag_fn(f)``.

    The impulse response is designed on a fine grid, rotated to be causal and
    windowed. Because ``rfft(h)[0] == sum(h)``, designing directly from the
    magnitude already yields the correct gain (normalising by the DC bin would
    annihilate high-pass kernels, where that bin is zero by design).

    ``taps`` is expressed **at 48 kHz** and rescaled by :func:`taps_for`, so the
    window is the same number of milliseconds at every sample rate and the
    filter's transition band stays put in hertz.

    ``dc_gain`` optionally forces the exact DC gain after windowing. Truncating
    a long response (e.g. a very low cut-off) leaves a residual DC error, so
    filters that must reject DC pass ``dc_gain=0.0``.
    """
    taps = taps_for(sr, taps)
    if taps % 2 == 0:
        taps += 1
    n = next_pow2(taps * 8)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.asarray(mag_fn(f), dtype=np.float64)
    h = np.fft.irfft(mag.astype(np.complex128), n)
    half = taps // 2
    h = np.concatenate([h[n - half :], h[: half + 1]])
    if h.size > 3:
        h *= np.hanning(h.size)
    if dc_gain is not None:
        h += (float(dc_gain) - float(np.sum(h))) / h.size
    return h


def bandpass_kernel(sr: int, lo: float, hi: float, order: int = 2, taps: int = 513):
    # Keep the band inside Nyquist, the same way hiss_kernel and air_kernel do.
    # Above it the magnitude response folds back, so a preset whose band was
    # written for 48 kHz would quietly change character at a lower rate.
    lo = max(float(lo), 20.0)
    hi = min(max(float(hi), lo * 1.2), sr / 2 * 0.999)
    return spectral_kernel(sr, lambda f: mag_bandpass(f, lo, hi, order), taps, dc_gain=0.0)


def lowpass_kernel(sr: int, fc: float, order: int = 2, taps: int = 513):
    if fc >= sr / 2 * 0.999:
        return np.array([1.0])
    return spectral_kernel(sr, lambda f: mag_lowpass(f, fc, order), taps, dc_gain=1.0)


def highpass_kernel(sr: int, fc: float, order: int = 2, taps: int = 513):
    if fc <= 1.0:
        return np.array([1.0])
    return spectral_kernel(sr, lambda f: mag_highpass(f, fc, order), taps, dc_gain=0.0)


def tilt_kernel(sr: int, pivot: float, gain_db: float, taps: int = 1025):
    if abs(gain_db) < 0.05:
        return np.array([1.0])
    return spectral_kernel(sr, lambda f: mag_tilt(f, pivot, gain_db), taps)


def presence_kernel(sr: int, centre: float, gain_db: float, q: float = 0.9, taps: int = 1025):
    """A bell (not a shelf) around ``centre``: the "rasp" band of an engine.

    A shelf is the wrong shape for an engine voice: lifting everything above a
    pivot keeps climbing all the way to Nyquist, so mechanical hiss and cymbal-
    like tick transients get boosted exactly as much as the useful 1-3 kHz
    body. A bell adds presence and then gets out of the way.
    """
    if abs(gain_db) < 0.05:
        return np.array([1.0])
    c = min(max(float(centre), 20.0), sr / 2 * 0.999)
    q = max(float(q), 0.05)

    def mag(f):
        r = np.asarray(f, dtype=np.float64) / c
        # bell peaks at exactly 1.0 when f == c, falls to 0 at DC and Nyquist
        den = np.sqrt((1.0 - r * r) ** 2 + (r / q) ** 2)
        bell = (r / q) / np.maximum(den, 1e-12)
        a = 10.0 ** (float(gain_db) / 20.0)
        return 1.0 + (a - 1.0) * bell

    return spectral_kernel(sr, mag, taps)


def mag_low_shelf(f, fc: float, gain_db: float, order: int = 1):
    """First-order shelf: ``gain_db`` at DC, 0 dB well above ``fc``.

    Built as ``1 + (a-1) * lowpass`` rather than a biquad so it composes with
    the other magnitude helpers and lands in the same FIR designer.
    """
    a = 10.0 ** (float(gain_db) / 20.0)
    return 1.0 + (a - 1.0) * mag_lowpass(f, max(float(fc), 1.0), order)


def mag_high_shelf(f, fc: float, gain_db: float, order: int = 1):
    """Mirror of :func:`mag_low_shelf`: 0 dB at DC, ``gain_db`` above ``fc``."""
    a = 10.0 ** (float(gain_db) / 20.0)
    return 1.0 + (a - 1.0) * mag_highpass(f, max(float(fc), 1.0), order)


def eq_kernel(
    sr: int,
    *,
    low_hz: float = 90.0,
    low_db: float = 0.0,
    low_order: int = 1,
    high_hz: float = 8000.0,
    high_db: float = 0.0,
    high_order: int = 1,
    bands: tuple[tuple[float, float, float], ...] = (),
    taps: int = 1025,
) -> np.ndarray:
    """Master tone EQ as one linear-phase FIR: two shelves plus peaking bells.

    This is the user-facing tone stage, not a voicing primitive: the shelves set
    weight and air, the bells move a band without disturbing its neighbours. All
    of it is one convolution, so it costs exactly as much as the ``air`` stage
    and -- being LTI -- leaves an already periodic loop periodic.

    The window has to be long enough to resolve the lowest band; a 21 ms kernel
    cannot shape 80 Hz. ``taps`` therefore grows with the lowest corner, and the
    cost stays bounded because the partitioned convolver does not care.

    The requirement is expressed in :data:`REF_SR` units like every other
    ``taps`` argument: :func:`spectral_kernel` rescales the count by
    ``sr / REF_SR`` so the window keeps its *duration*. Computing the need in
    samples at ``sr`` instead would silently halve it at 24 kHz and leave a
    60 Hz shelf unresolved.
    """
    corners = [hz for hz, _, _ in bands]
    if abs(low_db) > 0.05:
        corners.append(low_hz)
    low_corner = min(corners) if corners else 90.0
    # ~6 periods of the lowest corner, so the shelf edge is resolved
    need = int(6.0 * REF_SR / max(low_corner, 20.0))
    taps = max(int(taps), need)
    if taps % 2 == 0:
        taps += 1

    def mag(f):
        m = np.ones_like(np.asarray(f, dtype=np.float64))
        if abs(low_db) > 0.05:
            m = m * mag_low_shelf(f, low_hz, low_db, low_order)
        if abs(high_db) > 0.05:
            m = m * mag_high_shelf(f, high_hz, high_db, high_order)
        for hz, q, gain_db in bands:
            if abs(gain_db) <= 0.05:
                continue
            c = min(max(float(hz), 20.0), sr / 2 * 0.999)
            q = max(float(q), 0.05)
            r = np.asarray(f, dtype=np.float64) / c
            bell = (r / q) / np.maximum(np.sqrt((1.0 - r * r) ** 2 + (r / q) ** 2), 1e-12)
            a = 10.0 ** (float(gain_db) / 20.0)
            m = m * (1.0 + (a - 1.0) * bell)
        return m

    return spectral_kernel(sr, mag, taps)


def air_kernel(sr: int, cutoff: float = 6200.0, order: int = 2, tilt_db: float = 0.0, taps: int = 1025):
    """End-of-chain voicing: air absorption plus the natural downward slope.

    Real engine noise is never flat to Nyquist -- far-field air absorption and
    the source spectrum itself both fall with frequency. This is the single
    stage that keeps the top octave from turning into hiss.
    """
    fc = min(max(float(cutoff), 200.0), sr / 2 * 0.999)

    def mag(f):
        m = mag_lowpass(f, fc, order)
        if abs(tilt_db) > 0.05:
            m = m * mag_tilt(f, 250.0, tilt_db)
        return m

    return spectral_kernel(sr, mag, taps)


def hiss_kernel(sr: int, lo: float = 220.0, hi: float = 3200.0, tilt_db: float = -9.0, taps: int = 513):
    """Shape for broadband mechanical noise (tappets, chain, pump, injectors).

    Raw white noise is not a sound any engine makes: it puts as much energy into
    the 12-24 kHz octave as at 1 kHz, which is heard as tape hiss laid on top of
    the engine. Real mechanical noise is broadband but band-limited and falls
    with frequency, so this is a band-pass with a downward tilt.
    """
    lo = max(float(lo), 20.0)
    hi = min(max(float(hi), lo * 1.2), sr / 2 * 0.999)

    def mag(f):
        return mag_bandpass(f, lo, hi, 2) * mag_tilt(f, 700.0, tilt_db)

    return spectral_kernel(sr, mag, taps)


@functools.lru_cache(maxsize=32)
def _dc_block_kernel_cached(sr: int, cutoff_hz: float, length_s: float) -> np.ndarray:
    """Impulse response of a one-pole DC blocker: ``y = x - x1 + R*y1``.

    A dedicated short kernel is used instead of a low-cut-off FIR, because a
    24 Hz high-pass would need tens of thousands of taps to reject DC properly.
    """
    r = math.exp(-TWO_PI * max(cutoff_hz, 0.01) / sr)
    n = max(int(length_s * sr), 64)
    h = np.empty(n, dtype=np.float64)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(n):
        x = 1.0 if i == 0 else 0.0
        y = x - prev_x + r * prev_y
        h[i] = y
        prev_x = x
        prev_y = y
    h -= h.mean()  # exact zero DC gain even after truncation
    return h


def dc_block_kernel(sr: int, cutoff_hz: float = 20.0, length_s: float = 0.20) -> np.ndarray:
    return _dc_block_kernel_cached(int(sr), float(cutoff_hz), float(length_s)).copy()


def combustion_pulse(
    sr: int,
    rise_s: float = 0.0008,
    decay_s: float = 0.006,
    ring_hz: float = 320.0,
    ring_gain: float = 0.45,
    noise_amount: float = 0.25,
    rng: np.random.Generator | None = None,
    front_gain: float = 0.0,
    attack_s: float = 2.0e-5,
) -> np.ndarray:
    """One cylinder's combustion excitation.

    A fast pressure rise followed by an exponential decay, plus a resonating
    component and a short noise burst (the initial "bark").

    ``front_gain`` adds a very narrow spike on the leading edge. A heavily
    loaded cylinder dumps its charge almost instantly, and that near-discontinuity
    is what puts the hard high-frequency "crack" into the note; a light-load
    pulse is rounded and dark. The pair of pulse shapes the voice crossfades
    between is built from this parameter.
    """
    rng = rng or np.random.default_rng(1234)
    n = max(int((rise_s * 4.0 + decay_s * 7.0) * sr), 16)
    t = np.arange(n, dtype=np.float64) / sr
    body = (1.0 - np.exp(-t / max(rise_s, 1e-6))) * np.exp(-t / max(decay_s, 1e-6))
    ring = ring_gain * np.sin(TWO_PI * ring_hz * t) * np.exp(-t / max(decay_s * 0.6, 1e-6))
    pulse = body + ring
    if noise_amount > 0.0:
        # The blowdown's turbulent noise is NOT white. A white burst inside
        # the pulse was measured to put MORE energy in 4-8 kHz than in
        # 0.5-1 kHz (+7.7 dB), which flattened the firing-order harmonics to
        # ~1.7 dB/harmonic at 5000 rpm (a real exhaust rolls off -3..-6) and
        # gave the note its metallic, comb-flat buzz. Physically the gas is
        # generated at the valve and low-passed by the port and runner on its
        # way out: a one-pole at ~1.2 kHz on the noise reproduces exactly
        # that, and the harmonics return to a natural rolloff.
        raw = noise_burst(n, sr, rng)
        lp = math.exp(-TWO_PI * 1200.0 / float(sr))
        shaped = np.empty_like(raw)
        acc = 0.0
        for i in range(n):
            acc = lp * acc + (1.0 - lp) * raw[i]
            shaped[i] = acc
        # refit the burst's level: the lowpass removed energy, and the
        # noise_burst contract is "unit-ish amplitude for noise_amount"
        sd = float(np.std(shaped))
        if sd > 1e-12:
            shaped /= sd
        burst = shaped * np.exp(-t / max(decay_s * 0.25, 1e-6))
        pulse = pulse + noise_amount * burst
    if front_gain > 0.0:
        fw = max(rise_s * 0.20, 6e-6)
        pulse = pulse + front_gain * np.exp(-((t - fw) / fw) ** 2)
    # A couple of samples of attack is all that is wanted: the whole character of
    # an exhaust pulse is that its leading edge is nearly a step. A wider window
    # (the obvious "avoid a hard edge" reflex) costs 6 dB/octave of harmonics
    # above `1/(2*pi*window)` and turns a rasp into a hum.
    a = min(max(int(attack_s * sr), 1), n // 4)
    if a > 1:
        pulse[:a] *= np.linspace(0.0, 1.0, a)
    return pulse


def flow_burst(
    sr: int,
    attack_s: float = 0.0005,
    decay_s: float = 0.0026,
    lo_hz: float = 180.0,
    hi_hz: float = 5200.0,
    rng: np.random.Generator | None = None,
    taps: int = 257,
) -> np.ndarray:
    """One cylinder's blowdown turbulence burst.

    The roar in an exhaust is not a steady hiss: almost all of it is generated
    while the exhaust valve is open and the gas is moving fastest, so it arrives
    as a short noise burst synchronised with the blowdown. Turning the noise into
    an *event* rather than a constant is what makes it sound like gas moving in a
    pipe instead of tape hiss -- and it makes the noise pick up the firing order,
    which is a large part of why an engine sounds like an engine.

    Normalised to unit peak so the preset gain means "how loud is this burst".
    """
    rng = rng or np.random.default_rng(99)
    n = max(int((attack_s * 4.0 + decay_s * 6.0) * sr), 32)
    shape = bandpass_kernel(sr, lo_hz, hi_hz, 2, taps)
    noise = noise_burst(n + shape.size, sr, rng)
    filt = np.convolve(noise, shape, mode="full")[:n]
    t = np.arange(n, dtype=np.float64) / sr
    env = (1.0 - np.exp(-t / max(attack_s, 1e-6))) * np.exp(-t / max(decay_s, 1e-6))
    out = filt * env
    pk = float(np.max(np.abs(out)))
    if pk > 1e-12:
        out /= pk
    return out


def impact_kernel(
    sr: int,
    hz: float = 120.0,
    decay_s: float = 0.008,
    attack_s: float = 0.0005,
    noise_amount: float = 0.10,
    rng: np.random.Generator | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """A short, low, damped thump.

    This is the sound of metal stopping metal: a main bearing taking a firing
    load, a piston rocking over at TDC, a chain snatching. It is deliberately
    built from *one* low resonance plus a little noise -- stacking several
    modes here would turn a knock into a drum.
    """
    rng = rng or np.random.default_rng(7)
    n = max(int((attack_s * 3.0 + decay_s * 6.0) * sr), 16)
    t = np.arange(n, dtype=np.float64) / sr
    env = (1.0 - np.exp(-t / max(attack_s, 1e-6))) * np.exp(-t / max(decay_s, 1e-6))
    body = np.sin(TWO_PI * max(float(hz), 10.0) * t) * env
    if noise_amount > 0.0:
        # a couple of milliseconds of low noise gives the knock an "edge" that a
        # pure sine lacks, without adding audible hiss
        nk = lowpass_kernel(sr, max(float(hz) * 6.0, 400.0), 2, 129)
        burst = np.convolve(noise_burst(n + nk.size, sr, rng), nk, mode="full")[:n]
        body = body + noise_amount * burst * env
    a = min(max(int(attack_s * sr), 1), n // 4)
    if a > 1:
        body[:a] *= np.linspace(0.0, 1.0, a)
    if normalize:
        pk = float(np.max(np.abs(body)))
        if pk > 1e-12:
            body /= pk
    return body


def outdoor_ir(
    sr: int,
    decay_s: float = 0.38,
    predelay_ms: float = 15.0,
    damping_hz: float = 3800.0,
    early_gain: float = 0.55,
    tail_gain: float = 0.72,
    seed: int = 1,
    tail_start_ms: float = 9.0,
) -> np.ndarray:
    """Impulse response of an outdoor space (ground + a few hard surfaces).

    Outdoors is *not* a big hall. There is no dense late field and almost no
    high-frequency tail: what a listener hears is a handful of strong early
    reflections -- mostly the ground -- and a short, dark, diffuse decay from
    whatever walls happen to be nearby. That is what this builds:

    * sparse early taps between ``predelay`` and ``predelay + 80 ms``;
    * a band-limited noise tail that decays to -60 dB in ``decay_s``;
    * the whole thing normalised to unit energy, so the wet signal has the same
      RMS as the dry one and ``mix`` means what it says.

    Unit energy also keeps the stage loop-safe: a linear FIR applied to a
    periodic signal is periodic in its steady state, which the offline loops
    rely on.
    """
    sr = int(sr)
    predelay = max(float(predelay_ms), 0.0) / 1000.0
    decay = max(float(decay_s), 0.04)
    damping = min(max(float(damping_hz), 400.0), sr * 0.45)
    n = max(int((predelay + decay) * sr), 64)
    rng = np.random.default_rng(int(seed))

    h = np.zeros(n, dtype=np.float64)

    # --- early reflections ------------------------------------------------- #
    t0 = max(int(predelay * sr), 0)
    er_end = min(n, t0 + int(0.080 * sr))
    idx = t0 + 1
    while idx < er_end - 1:
        age = (idx - t0) / max(sr, 1)
        amp = early_gain * math.exp(-age / max(decay * 0.30, 0.005))
        amp *= float(rng.uniform(0.35, 1.0))
        h[idx] += amp * (-1.0 if rng.random() < 0.45 else 1.0)
        # a small cluster rather than a single sample: real surfaces are not
        # points, and a bare impulse train sounds like a delay, not a room
        for _ in range(int(rng.integers(0, 3))):
            j = idx + int(rng.integers(1, max(int(0.002 * sr), 2)))
            if j < er_end:
                h[j] += amp * float(rng.uniform(0.15, 0.45)) * (-1.0 if rng.random() < 0.5 else 1.0)
        idx += int(rng.integers(max(int(0.0015 * sr), 1), max(int(0.007 * sr), 2)))

    # soften the taps: a hard click is a recording artefact, not a reflection
    h = np.convolve(h, lowpass_kernel(sr, damping, 1, 129), mode="full")[:n]

    # --- diffuse tail ------------------------------------------------------ #
    start = max(int((predelay + float(tail_start_ms) / 1000.0) * sr), 0)
    if start < n:
        m = n - start
        tau = decay / 6.907755  # -60 dB at decay_s
        t = np.arange(m, dtype=np.float64) / sr
        env = np.exp(-t / max(tau, 1e-6))
        attack = min(int(0.004 * sr), max(m // 4, 1))
        if attack > 1:
            env[:attack] *= np.linspace(0.0, 1.0, attack)
        noise = noise_burst(m, sr, rng)
        tail = np.convolve(noise, lowpass_kernel(sr, damping * 1.25, 2, 257), mode="full")[:m]
        h[start:] += tail_gain * env * tail

    # --- normalise --------------------------------------------------------- #
    h -= h.mean()
    energy = float(np.sqrt(np.sum(h * h)))
    if energy > 1e-12:
        h /= energy
    return h


def tick_kernel(
    sr: int,
    bright_hz: float = 3800.0,
    decay_s: float = 0.0018,
    noise_amount: float = 0.5,
    rng: np.random.Generator | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Short metallic click used for valvetrain / injector noise.

    A real valve-seat impact is not a pure damped sine: it is a very short
    broadband transient that excites a ring at the head's resonance. So the
    kernel is a damped ring *plus* a band-limited noise burst, which is what
    gives a click its "tick" instead of a "ping".

    Normalised to unit **time-domain peak** so the preset gains mean "how loud
    is this click". (Normalising by the spectral peak, as this used to, made the
    waveform 30-70 times smaller than the gain implied -- the whole mechanical
    layer was inaudible because of it.)
    """
    rng = rng or np.random.default_rng(11)
    n = max(int(decay_s * 6.0 * sr), 8)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t / max(decay_s, 1e-6))
    ring = np.sin(TWO_PI * max(float(bright_hz), 20.0) * t) * env
    if noise_amount > 0.0:
        lo = max(float(bright_hz) * 0.60, 300.0)
        hi = min(float(bright_hz) * 1.70, sr * 0.45)
        if hi <= lo:
            hi = min(lo * 2.0, sr * 0.45)
        bp = bandpass_kernel(sr, lo, hi, 2, 129)
        burst = np.convolve(noise_burst(n + bp.size, sr, rng), bp, mode="full")[:n]
        pk = float(np.max(np.abs(burst)))
        if pk > 1e-12:
            burst /= pk
        body = (1.0 - noise_amount) * ring + noise_amount * burst * env
    else:
        body = ring
    a = min(max(int(2.0e-5 * sr), 1), n // 4)
    if a > 1:
        body[:a] *= np.linspace(0.0, 1.0, a)
    if normalize:
        pk = float(np.max(np.abs(body)))
        if pk > 1e-12:
            body /= pk
    return body


# --------------------------------------------------------------------------- #
# stateful processors
# --------------------------------------------------------------------------- #
class FFTConvolver:
    """Overlap-add FIR convolver.

    Input length must be an exact multiple of ``block_size``. The voice layer
    guarantees this by buffering into whole blocks, which keeps sample timing
    exact and avoids any drift between the offline and real-time paths.
    """

    def __init__(self, kernel: np.ndarray, block_size: int = 1024):
        kernel = np.asarray(kernel, dtype=np.float64).ravel()
        if kernel.size == 0:
            kernel = np.array([1.0])
        self.block_size = int(block_size)
        self.kernel_len = int(kernel.size)
        self.n_fft = next_pow2(self.block_size + self.kernel_len - 1)
        self.H = np.fft.rfft(kernel, self.n_fft)
        self._acc = np.zeros(self.n_fft, dtype=np.float64)

    def reset(self) -> None:
        self._acc[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        b = self.block_size
        if x.size % b != 0:
            raise ValueError(f"FFTConvolver expects a multiple of block_size={b}, got {x.size}")
        out = np.empty(x.size, dtype=np.float64)
        for pos in range(0, x.size, b):
            blk = x[pos : pos + b]
            y = np.fft.irfft(np.fft.rfft(blk, self.n_fft) * self.H, self.n_fft)
            self._acc += y
            out[pos : pos + b] = self._acc[:b]
            # Shift the overlap-add accumulator in place. ``np.roll`` allocates
            # a new array and normalises the axis tuple on every block, which is
            # measurable in the audio thread; this slice copy does neither.
            self._acc[:-b] = self._acc[b:]
            self._acc[-b:] = 0.0
        return out


class ConvolverChain:
    """A series of FIR stages applied to every sample."""

    def __init__(self, kernels, block_size: int = 1024):
        self.stages = [FFTConvolver(k, block_size) for k in kernels if k is not None and np.size(k)]

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).ravel()
        for s in self.stages:
            y = s.process(y)
        return y


class NoiseGen:
    """White noise source with a reproducible stream."""

    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)

    def render(self, n: int) -> np.ndarray:
        return self.rng.standard_normal(int(n))

    def next(self) -> float:
        """One scalar sample. Used where the draw rate must track events."""
        return float(self.rng.standard_normal())


class CircularNoise:
    """White noise read from a fixed buffer, wrapping at its length.

    Continuous noise would otherwise make a rendered loop non-repeating: fresh
    random samples every block can never line up with the loop point. Reading
    from a fixed buffer indexed modulo the loop length makes the noise itself
    exactly periodic, so a loop containing noise still closes perfectly.
    """

    def __init__(self, length: int, seed: int = 0):
        self.n = max(int(length), 1)
        self.buf = np.random.default_rng(seed).standard_normal(self.n)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def render(self, count: int) -> np.ndarray:
        count = int(count)
        idx = (self.pos + np.arange(count)) % self.n
        self.pos = (self.pos + count) % self.n
        return self.buf[idx]

    def next(self) -> float:
        v = float(self.buf[self.pos])
        self.pos += 1
        if self.pos >= self.n:
            self.pos = 0
        return v


class PhaseOsc:
    """Band-limited-ish sine oscillator with continuous phase."""

    def __init__(self, sr: int):
        self.sr = int(sr)
        self.phase = 0.0

    def reset(self) -> None:
        self.phase = 0.0

    def render(self, n: int, freq) -> np.ndarray:
        f = np.asarray(freq, dtype=np.float64)
        if f.ndim == 0:
            f = np.full(int(n), float(f))
        ph = self.phase + np.cumsum(TWO_PI * f / self.sr)
        self.phase = float(ph[-1] % TWO_PI)
        return np.sin(ph)


class OnePole:
    """Exponential smoother for click-free parameter changes."""

    def __init__(self, sr: int, time_s: float = 0.02, init: float = 0.0):
        self.a = float(np.exp(-1.0 / max(time_s * sr, 1.0)))
        self.value = float(init)

    def process(self, target, n: int) -> np.ndarray:
        t = float(target)
        if t == self.value:
            return np.full(int(n), self.value)
        # closed-form ramp for a constant target
        k = np.arange(1, int(n) + 1, dtype=np.float64)
        vals = t + (self.value - t) * (self.a**k)
        self.value = float(vals[-1])
        return vals


class RampParam:
    """Linear ramp from the previous value to a new target over one block."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self._prev = float(value)

    def block(self, target, n: int) -> np.ndarray:
        t = float(target)
        prev = self._prev
        out = np.linspace(prev, t, int(n), endpoint=False) if n > 1 else np.array([prev])
        self.value = t
        self._prev = t
        return out
