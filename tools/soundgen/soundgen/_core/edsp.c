/* Real-time DSP core for the engine voice.
 *
 * Everything in here runs on the audio callback, so the rules are:
 *
 *   - no allocation, no locks, no syscalls after init;
 *   - every buffer is sized once, at construction;
 *   - one pass over the block per stage, no temporary arrays.
 *
 * The Python synthesiser spends ~48% of its time in pocketfft: 10 000 rFFTs
 * per three seconds of audio, each on a 256..2048 point block, each wrapped in
 * a Python call. This core does the same overlap-add work in C with the
 * transform plans precomputed, which is what frees the budget for per-cylinder
 * exhausts and low latency.
 *
 * The API is deliberately flat: a convolver is a pointer plus a gain, a chain
 * is an array of stages, and everything is addressed by index. No callbacks
 * cross the boundary -- the Python side calls one function per block.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)

/* ------------------------------------------------------------------ */
/* FFT: iterative radix-2, real input, precomputed twiddles           */
/* ------------------------------------------------------------------ */
typedef struct {
    int n;          /* transform length (power of two) */
    int n_half;     /* n / 2 */
    double *re;     /* cos table, n/2 entries */
    double *im;     /* -sin table, n/2 entries */
    int *rev;       /* bit-reversal permutation */
    double *buf;    /* scratch, n doubles */
} cfft_plan;

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static cfft_plan *plan_create(int n) {
    if (n < 2) n = 2;
    n = next_pow2(n);
    cfft_plan *p = (cfft_plan *)calloc(1, sizeof(cfft_plan));
    if (!p) return NULL;
    p->n = n;
    p->n_half = n / 2;
    p->re = (double *)malloc(sizeof(double) * (size_t)p->n_half);
    p->im = (double *)malloc(sizeof(double) * (size_t)p->n_half);
    p->rev = (int *)malloc(sizeof(int) * (size_t)n);
    p->buf = (double *)malloc(sizeof(double) * (size_t)(n * 2));
    if (!p->re || !p->im || !p->rev || !p->buf) {
        free(p->re); free(p->im); free(p->rev); free(p->buf); free(p);
        return NULL;
    }
    for (int i = 0; i < p->n_half; ++i) {
        double a = -TWO_PI * (double)i / (double)n;
        p->re[i] = cos(a);
        p->im[i] = sin(a);
    }
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int r = 0, x = i;
        for (int b = 0; b < bits; ++b) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        p->rev[i] = r;
    }
    return p;
}

static void plan_destroy(cfft_plan *p) {
    if (!p) return;
    free(p->re); free(p->im); free(p->rev); free(p->buf);
    free(p);
}

/* In-place complex FFT on interleaved re/im arrays of length n. */
static void fft_run(cfft_plan *p, double *re, double *im, int inverse) {
    const int n = p->n;
    for (int i = 0; i < n; ++i) {
        int j = p->rev[i];
        if (j > i) {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                double wr = p->re[k * step];
                double wi = p->im[k * step];
                if (inverse) wi = -wi;
                int a = i + k;
                int b = a + half;
                double tr = re[b] * wr - im[b] * wi;
                double ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
    if (inverse) {
        double s = 1.0 / (double)n;
        for (int i = 0; i < n; ++i) { re[i] *= s; im[i] *= s; }
    }
}

/* ------------------------------------------------------------------ */
/* overlap-add FIR convolver                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    int block;      /* grain: input must be a whole multiple of this */
    int taps;       /* kernel length */
    int n_fft;      /* transform size */
    double *Hr;     /* kernel spectrum, real part */
    double *Hi;     /* kernel spectrum, imaginary part */
    double *acc;    /* overlap-add accumulator, n_fft doubles */
    double *re;     /* scratch */
    double *im;     /* scratch */
    cfft_plan *plan;
    double gain;
} convolver;

EXPORT convolver *edsp_convolver_new(const double *kernel, int taps, int block) {
    if (!kernel || taps <= 0 || block <= 0) return NULL;
    convolver *c = (convolver *)calloc(1, sizeof(convolver));
    if (!c) return NULL;
    c->block = block;
    c->taps = taps;
    c->n_fft = next_pow2(block + taps - 1);
    c->gain = 1.0;
    c->plan = plan_create(c->n_fft);
    c->Hr = (double *)calloc((size_t)c->n_fft, sizeof(double));
    c->Hi = (double *)calloc((size_t)c->n_fft, sizeof(double));
    c->acc = (double *)calloc((size_t)c->n_fft, sizeof(double));
    c->re = (double *)calloc((size_t)c->n_fft, sizeof(double));
    c->im = (double *)calloc((size_t)c->n_fft, sizeof(double));
    if (!c->plan || !c->Hr || !c->Hi || !c->acc || !c->re || !c->im) {
        if (c->plan) plan_destroy(c->plan);
        free(c->Hr); free(c->Hi); free(c->acc); free(c->re); free(c->im);
        free(c);
        return NULL;
    }
    for (int i = 0; i < taps; ++i) c->re[i] = kernel[i];
    for (int i = taps; i < c->n_fft; ++i) c->re[i] = 0.0;
    memset(c->im, 0, sizeof(double) * (size_t)c->n_fft);
    fft_run(c->plan, c->re, c->im, 0);
    memcpy(c->Hr, c->re, sizeof(double) * (size_t)c->n_fft);
    memcpy(c->Hi, c->im, sizeof(double) * (size_t)c->n_fft);
    memset(c->re, 0, sizeof(double) * (size_t)c->n_fft);
    memset(c->im, 0, sizeof(double) * (size_t)c->n_fft);
    return c;
}

EXPORT void edsp_convolver_free(convolver *c) {
    if (!c) return;
    plan_destroy(c->plan);
    free(c->Hr); free(c->Hi); free(c->acc); free(c->re); free(c->im);
    free(c);
}

EXPORT void edsp_convolver_reset(convolver *c) {
    if (!c) return;
    memset(c->acc, 0, sizeof(double) * (size_t)c->n_fft);
}

EXPORT void edsp_convolver_set_gain(convolver *c, double g) { if (c) c->gain = g; }

/* Process `n` samples in place-shaped output. n must be a multiple of block. */
EXPORT int edsp_convolver_process(convolver *c, const double *in, double *out, int n) {
    if (!c || !in || !out) return -1;
    const int b = c->block;
    const int nf = c->n_fft;
    if (n <= 0) return 0;
    if (n % b != 0) return -1;
    for (int pos = 0; pos < n; pos += b) {
        const double *src = in + pos;
        for (int i = 0; i < b; ++i) c->re[i] = src[i];
        for (int i = b; i < nf; ++i) c->re[i] = 0.0;
        memset(c->im, 0, sizeof(double) * (size_t)nf);
        fft_run(c->plan, c->re, c->im, 0);
        for (int i = 0; i < nf; ++i) {
            double a = c->re[i], bb = c->im[i];
            c->re[i] = a * c->Hr[i] - bb * c->Hi[i];
            c->im[i] = a * c->Hi[i] + bb * c->Hr[i];
        }
        fft_run(c->plan, c->re, c->im, 1);
        for (int i = 0; i < b; ++i) {
            double v = c->acc[i] + c->re[i];
            out[pos + i] = v * c->gain;
        }
        /* shift the accumulator: consume b samples, keep the tail */
        for (int i = 0; i < nf - b; ++i) c->acc[i] = c->acc[i + b] + c->re[i + b];
        for (int i = nf - b; i < nf; ++i) c->acc[i] = 0.0;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* chain of convolvers, applied in series                              */
/* ------------------------------------------------------------------ */
EXPORT int edsp_chain_process(convolver **stages, int count, const double *in,
                              double *scratch, double *out, int n) {
    if (count <= 0) {
        memcpy(out, in, sizeof(double) * (size_t)n);
        return 0;
    }
    const double *src = in;
    for (int s = 0; s < count; ++s) {
        double *dst = (s == count - 1) ? out : scratch;
        int rc = edsp_convolver_process(stages[s], src, dst, n);
        if (rc != 0) return rc;
        src = dst;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* fractional-delay comb, used for the exhaust pipe                   */
/* ------------------------------------------------------------------ */
typedef struct {
    int sr;
    int repeats;
    double gain;
    int invert;
    int maxd;
    int size;
    double *buf;
    int w;
} combdelay;

EXPORT combdelay *edsp_comb_new(int sr, double max_delay_s, int repeats, double gain,
                                int invert, int block) {
    combdelay *c = (combdelay *)calloc(1, sizeof(combdelay));
    if (!c) return NULL;
    c->sr = sr;
    c->repeats = repeats > 0 ? repeats : 1;
    c->gain = gain;
    c->invert = invert ? 1 : 0;
    c->maxd = (int)ceil(max_delay_s * sr);
    if (c->maxd < 2) c->maxd = 2;
    c->size = block + c->maxd * c->repeats + 4;
    c->buf = (double *)calloc((size_t)c->size, sizeof(double));
    if (!c->buf) { free(c); return NULL; }
    c->w = 0;
    return c;
}

EXPORT void edsp_comb_free(combdelay *c) {
    if (!c) return;
    free(c->buf);
    free(c);
}

EXPORT void edsp_comb_reset(combdelay *c) {
    if (!c) return;
    memset(c->buf, 0, sizeof(double) * (size_t)c->size);
    c->w = 0;
}

EXPORT void edsp_comb_process(combdelay *c, const double *in, double *out,
                              double delay_s, int n) {
    if (!c || n <= 0) return;
    const int size = c->size;
    double d = delay_s * c->sr;
    if (d < 1.0) d = 1.0;
    if (d > (double)c->maxd) d = (double)c->maxd;
    for (int i = 0; i < n; ++i) {
        double x = in[i];
        int idx = (c->w + i) % size;
        c->buf[idx] = x;
        double y = x;
        double g = 1.0;
        double t = (double)(c->w + i);
        for (int k = 1; k <= c->repeats; ++k) {
            g *= c->gain * (c->invert ? -1.0 : 1.0);
            double p = t - (double)k * d;
            double i0f = floor(p);
            int i0 = (int)i0f;
            double frac = p - i0f;
            int a = ((i0 % size) + size) % size;
            int b = (((i0 + 1) % size) + size) % size;
            double v = c->buf[a] + frac * (c->buf[b] - c->buf[a]);
            y += g * v;
        }
        out[i] = y;
    }
    c->w = (c->w + n) % size;
}

/* ------------------------------------------------------------------ */
/* soft clip and peak limiter                                         */
/* ------------------------------------------------------------------ */
EXPORT void edsp_soft_clip(double *x, int n, double drive) {
    if (drive <= 1.0001) return;
    const double norm = 1.0 / tanh(drive);
    for (int i = 0; i < n; ++i) x[i] = tanh(x[i] * drive) * norm;
}

EXPORT double edsp_safety_limit(double *x, int n, double ceiling, double knee) {
    double knee_abs = ceiling * knee;
    if (knee_abs < 1e-6) knee_abs = 1e-6;
    const double span = ceiling - knee_abs;
    double peak = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = x[i];
        double m = fabs(v);
        if (m > knee_abs) {
            double shaped = knee_abs + span * tanh((m - knee_abs) / span);
            v = (v < 0.0) ? -shaped : shaped;
            x[i] = v;
        }
        m = fabs(v);
        if (m > peak) peak = m;
    }
    return peak;
}

/* ------------------------------------------------------------------ */
/* block helpers used by the per-cylinder mixer                       */
/* ------------------------------------------------------------------ */
EXPORT void edsp_mul_add(double *dst, const double *src, double g, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i] * g;
}

EXPORT void edsp_scale(double *x, double g, int n) {
    for (int i = 0; i < n; ++i) x[i] *= g;
}

EXPORT double edsp_peak(const double *x, int n) {
    double p = 0.0;
    for (int i = 0; i < n; ++i) {
        double m = fabs(x[i]);
        if (m > p) p = m;
    }
    return p;
}

EXPORT double edsp_rms(const double *x, int n) {
    if (n <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    return sqrt(s / (double)n);
}

/* One-pole smoothing, used for gains that must not step. */
typedef struct { double a; double y; } onepole;

EXPORT void edsp_onepole_init(onepole *p, double tau_s, int sr) {
    if (!p) return;
    p->a = (tau_s > 0.0) ? exp(-1.0 / (tau_s * (double)sr)) : 0.0;
    p->y = 0.0;
}

EXPORT void edsp_onepole_run(onepole *p, double *x, int n) {
    if (!p) return;
    double a = p->a, y = p->y;
    for (int i = 0; i < n; ++i) {
        y = a * y + (1.0 - a) * x[i];
        x[i] = y;
    }
    p->y = y;
}

/* ------------------------------------------------------------------ */
/* version / self-check                                               */
/* ------------------------------------------------------------------ */
EXPORT int edsp_version(void) { return 1; }

/* Impulse response of a convolver, for the Python-side verification. */
EXPORT int edsp_convolver_impulse(convolver *c, const double *kernel_in, double *out, int n) {
    if (!c || !out || n <= 0) return -1;
    (void)kernel_in;
    double *inp = (double *)calloc((size_t)((n / c->block + 1) * c->block), sizeof(double));
    if (!inp) return -1;
    inp[0] = 1.0;
    int pad = ((n + c->block - 1) / c->block) * c->block;
    int rc = edsp_convolver_process(c, inp, out, pad);
    free(inp);
    return rc;
}
