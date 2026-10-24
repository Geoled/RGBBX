// Native DSP core for the engine voice.
//
// The Python layer spends most of its time in numpy's pocketfft: every block
// convolved each stage with a transform sized for the whole kernel, so a
// 5760-tap resonator cost an 8192-point FFT and the reverb a 32768-point one.
//
// This module replaces that with uniformly partitioned overlap-save
// convolution: the kernel is cut into `block`-sized slices, each slice is
// pre-transformed once, and a block costs one forward transform of 2*block, one
// inverse transform of the same size and P complex multiplies -- independent of
// how long the kernel is. The reverb's 32768-point transform becomes a
// 2048-point one and a long kernel stops dominating the profile.
//
// Everything is double precision and matches soundgen.native's numpy reference
// sample for sample, because the offline renderer and the live audio thread must
// produce the same signal and a test asserts it (tests/test_native_dsp.py).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double PI = 3.14159265358979323846;
using cplx = std::complex<double>;

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// --------------------------------------------------------------------------- //
// complex FFT: iterative radix-2, precomputed bit reversal and twiddles
// --------------------------------------------------------------------------- //
struct Plan {
    int n = 0;
    std::vector<int> rev;
    std::vector<cplx> tw;

    void init(int n_) {
        n = std::max(n_, 2);
        rev.resize(size_t(n));
        int lg = 0;
        while ((1 << lg) < n) ++lg;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < lg; ++b) {
                if (i & (1 << b)) r |= 1 << (lg - 1 - b);
            }
            rev[size_t(i)] = r;
        }
        tw.resize(size_t(n) / 2);
        for (int i = 0; i < n / 2; ++i) {
            const double a = -2.0 * PI * double(i) / double(n);
            tw[size_t(i)] = cplx(std::cos(a), std::sin(a));
        }
    }

    void fft(cplx* a, bool inverse) const {
        for (int i = 0; i < n; ++i) {
            const int r = rev[size_t(i)];
            if (i < r) std::swap(a[i], a[r]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; ++j) {
                    cplx t = tw[size_t(j * step)];
                    if (inverse) t = std::conj(t);
                    const cplx u = a[i + j];
                    const cplx v = a[i + j + half] * t;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
        if (inverse) {
            const double s = 1.0 / double(n);
            for (int i = 0; i < n; ++i) a[i] *= s;
        }
    }
};

const Plan& plan_for(int n) {
    static std::map<int, Plan> cache;
    auto it = cache.find(n);
    if (it == cache.end()) {
        Plan p;
        p.init(n);
        it = cache.emplace(n, std::move(p)).first;
    }
    return it->second;
}

// Real transform of n real samples through a complex transform of the same
// size. Packing two samples per complex value would halve the work, but the
// unpacking formula is where a subtle conjugate error silently doubles a
// spectrum; the plain version is exact, obviously so, and the grouped FFT cost
// is not what dominates this core.
void rfft_into(const Plan& plan, const double* x, cplx* out, std::vector<cplx>& scratch) {
    const int n = plan.n;
    if (scratch.size() < size_t(n)) scratch.resize(size_t(n));
    for (int i = 0; i < n; ++i) scratch[size_t(i)] = cplx(x[i], 0.0);
    plan.fft(scratch.data(), false);
    for (int k = 0; k <= n / 2; ++k) out[k] = scratch[size_t(k)];
}

// Inverse of rfft_into: n/2+1 bins -> n real samples.
void irfft_into(const Plan& plan, const cplx* spec, double* out, std::vector<cplx>& scratch) {
    const int n = plan.n;
    if (scratch.size() < size_t(n)) scratch.resize(size_t(n));
    scratch[0] = cplx(spec[0].real(), 0.0);
    for (int k = 1; k < n / 2; ++k) {
        scratch[size_t(k)] = spec[k];
        scratch[size_t(n - k)] = std::conj(spec[k]);
    }
    if (n >= 2) {
        const cplx nyq = spec[size_t(n / 2)];
        scratch[size_t(n / 2)] = cplx(nyq.real(), 0.0);
    }
    plan.fft(scratch.data(), true);
    for (int i = 0; i < n; ++i) out[i] = scratch[size_t(i)].real();
}

// --------------------------------------------------------------------------- //
// uniformly partitioned overlap-save convolver
// --------------------------------------------------------------------------- //
class Convolver {
public:
    Convolver(const double* k, int klen, int block) {
        if (block <= 0) throw std::invalid_argument("block must be > 0");
        B_ = block;
        N_ = next_pow2(2 * B_);
        M_ = N_ / 2;
        const Plan& plan = plan_for(N_);
        P_ = std::max(1, (std::max(klen, 1) + B_ - 1) / B_);
        std::vector<double> padded(size_t(P_) * size_t(B_), 0.0);
        for (int i = 0; i < klen; ++i) padded[size_t(i)] = k[i];
        H_.assign(size_t(P_) * size_t(M_ + 1), cplx(0.0, 0.0));
        std::vector<double> slice(size_t(N_), 0.0);
        std::vector<cplx> spec(size_t(M_ + 1));
        std::vector<cplx> scratch;
        for (int p = 0; p < P_; ++p) {
            std::fill(slice.begin(), slice.end(), 0.0);
            std::copy(padded.begin() + size_t(p) * size_t(B_),
                      padded.begin() + size_t(p + 1) * size_t(B_), slice.begin());
            rfft_into(plan, slice.data(), spec.data(), scratch);
            std::copy(spec.begin(), spec.end(), &H_[size_t(p) * size_t(M_ + 1)]);
        }
        fdl_.assign(size_t(P_) * size_t(M_ + 1), cplx(0.0, 0.0));
        acc_.assign(size_t(M_ + 1), cplx(0.0, 0.0));
        cur_.assign(size_t(M_ + 1), cplx(0.0, 0.0));
        xbuf_.assign(size_t(N_), 0.0);
        ybuf_.assign(size_t(N_), 0.0);
        tail_.assign(size_t(B_), 0.0);
        scratch_.reserve(size_t(N_));
        pos_ = 0;
    }

    int partitions() const { return P_; }
    int fft_size() const { return N_; }
    int block() const { return B_; }

    void reset() {
        std::fill(fdl_.begin(), fdl_.end(), cplx(0.0, 0.0));
        std::fill(xbuf_.begin(), xbuf_.end(), 0.0);
        std::fill(tail_.begin(), tail_.end(), 0.0);
        pos_ = 0;
    }

    // One block in, one block out. Uniformly partitioned overlap-add (Gardner):
    // the frequency delay line holds the spectra of past input blocks, partition
    // ``p`` of the kernel pairs with the block ``p`` steps back, and the second
    // half of each inverse transform is a tail that belongs to the *next*
    // output block -- kept and added there.
    void process(const double* x, double* y) {
        const Plan& plan = plan_for(N_);
        std::fill(xbuf_.begin(), xbuf_.end(), 0.0);
        std::memcpy(xbuf_.data(), x, sizeof(double) * size_t(B_));
        rfft_into(plan, xbuf_.data(), cur_.data(), scratch_);

        std::copy(cur_.begin(), cur_.end(), &fdl_[size_t(pos_) * size_t(M_ + 1)]);
        for (int i = 0; i <= M_; ++i) acc_[size_t(i)] = cplx(0.0, 0.0);
        int idx = pos_;
        for (int p = 0; p < P_; ++p) {
            const cplx* h = &H_[size_t(p) * size_t(M_ + 1)];
            const cplx* xs = &fdl_[size_t(idx) * size_t(M_ + 1)];
            for (int i = 0; i <= M_; ++i) acc_[size_t(i)] += h[i] * xs[i];
            idx = (idx == 0) ? P_ - 1 : idx - 1;
        }
        pos_ = (pos_ + 1) % P_;

        irfft_into(plan, acc_.data(), ybuf_.data(), scratch_);
        for (int i = 0; i < B_; ++i) y[i] = ybuf_[size_t(i)] + tail_[size_t(i)];
        std::memcpy(tail_.data(), ybuf_.data() + B_, sizeof(double) * size_t(B_));
    }

private:
    int B_ = 0, N_ = 0, M_ = 0, P_ = 1, pos_ = 0, H_keep_ = 0;
    std::vector<cplx> H_;
    std::vector<cplx> fdl_;
    std::vector<cplx> acc_;
    std::vector<cplx> cur_;
    std::vector<double> xbuf_;
    std::vector<double> ybuf_;
    std::vector<double> tail_;
    std::vector<cplx> scratch_;
};

// --------------------------------------------------------------------------- //
// fractional-delay feed-forward comb (port of dsp.CombDelay, same indexing)
// --------------------------------------------------------------------------- //
class Comb {
public:
    void configure(int sr, double max_delay_s, int repeats, double gain, bool invert, int block) {
        sr_ = std::max(sr, 1);
        block_ = std::max(block, 1);
        repeats_ = std::max(repeats, 1);
        gain_ = gain;
        invert_ = invert;
        maxd_ = std::max(int(std::ceil(max_delay_s * double(sr_))), 2);
        size_ = block_ + maxd_ * repeats_ + 4;
        buf_.assign(size_t(size_), 0.0);
        w_ = 0;
    }

    void reset() {
        std::fill(buf_.begin(), buf_.end(), 0.0);
        w_ = 0;
    }

    void process(const double* x, double* y, int n, double delay_s) {
        if (n <= 0) return;
        if (size_t(n) + size_t(maxd_ * repeats_ + 4) > buf_.size()) {
            buf_.resize(size_t(n + maxd_ * repeats_ + 4), 0.0);
            size_ = int(buf_.size());
        }
        for (int i = 0; i < n; ++i) buf_[size_t((w_ + i) % size_)] = x[i];
        const double d = std::min(std::max(delay_s * double(sr_), 1.0), double(maxd_));
        const double sign = invert_ ? -1.0 : 1.0;
        for (int i = 0; i < n; ++i) y[i] = x[i];
        double g = 1.0;
        for (int k = 1; k <= repeats_; ++k) {
            g *= gain_ * sign;
            const double kd = double(k) * d;
            for (int i = 0; i < n; ++i) {
                const double p = double(w_ + i) - kd;
                const long long base = (long long)std::floor(p);
                const double frac = p - double(base);
                const long long a = ((base % size_) + size_) % size_;
                const long long b = (((base + 1) % size_) + size_) % size_;
                y[i] += g * (buf_[size_t(a)] + frac * (buf_[size_t(b)] - buf_[size_t(a)]));
            }
        }
        w_ = (w_ + n) % size_;
    }

private:
    int sr_ = 48000, block_ = 1024, repeats_ = 1, maxd_ = 2, w_ = 0, size_ = 0;
    double gain_ = 0.5;
    bool invert_ = true;
    std::vector<double> buf_;
};

// --------------------------------------------------------------------------- //
// output stage: calibrated gain -> character saturation -> safety ceiling
// --------------------------------------------------------------------------- //
void master(double* x, int n, double gain, double drive, double ceiling, double knee) {
    for (int i = 0; i < n; ++i) x[i] *= gain;
    if (drive > 1.0001) {
        const double norm = std::tanh(drive);
        for (int i = 0; i < n; ++i) x[i] = std::tanh(x[i] * drive) / norm;
    }
    const double c = std::max(ceiling, 1e-3);
    const double k = std::max(std::min(knee, 0.999) * c, 1e-6);
    const double span = c - k;
    for (int i = 0; i < n; ++i) {
        const double m = std::abs(x[i]);
        if (m > k) {
            const double shaped = k + span * std::tanh((m - k) / span);
            x[i] = (x[i] < 0.0 ? -shaped : shaped);
        }
    }
}

// --------------------------------------------------------------------------- //
// python bindings
// --------------------------------------------------------------------------- //
py::array_t<double> as_f64(const py::array& a) {
    return py::array_t<double, py::array::c_style | py::array::forcecast>(a);
}

using Kernel = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct PyConvolver {
    Convolver conv;
    int block;
    PyConvolver(const Kernel& kernel, int block_)
        : conv(static_cast<const double*>(kernel.data()), int(kernel.size()), block_), block(block_) {}

    py::array_t<double> process(const py::array& xin) {
        auto x = as_f64(xin);
        const int n = int(x.size());
        if (n % block != 0) throw std::invalid_argument("input must be a multiple of block");
        py::array_t<double> out(n);
        const double* xp = x.data();
        double* yp = out.mutable_data();
        for (int pos = 0; pos < n; pos += block) conv.process(xp + pos, yp + pos);
        return out;
    }
};

struct PyComb {
    Comb comb;
    PyComb(int sr_, double max_delay_s, int repeats, double gain, bool invert, int block_) {
        comb.configure(sr_, max_delay_s, repeats, gain, invert, block_);
    }
    void reset() { comb.reset(); }
    py::array_t<double> process(const py::array& xin, double delay_s) {
        auto x = as_f64(xin);
        const int n = int(x.size());
        py::array_t<double> out(n);
        comb.process(x.data(), out.mutable_data(), n, delay_s);
        return out;
    }
};

}  // namespace

PYBIND11_MODULE(_dspcore, m) {
    m.doc() = "Native DSP core: partitioned FFT convolution, fractional comb, master chain";

    m.def("rfft", [](const py::array& xin) {
        auto x = as_f64(xin);
        const int n = int(x.size());
        const Plan& plan = plan_for(n);
        py::array_t<double> re(n / 2 + 1), im(n / 2 + 1);
        std::vector<cplx> spec(size_t(n / 2 + 1));
        std::vector<cplx> scratch;
        rfft_into(plan, x.data(), spec.data(), scratch);
        for (int i = 0; i <= n / 2; ++i) {
            re.mutable_data()[i] = spec[size_t(i)].real();
            im.mutable_data()[i] = spec[size_t(i)].imag();
        }
        return py::make_tuple(re, im);
    });

    m.def("irfft", [](const py::array& rein, const py::array& imin, int n) {
        auto re = as_f64(rein);
        auto im = as_f64(imin);
        const Plan& plan = plan_for(n);
        std::vector<cplx> spec(size_t(n / 2 + 1));
        for (int i = 0; i <= n / 2; ++i) spec[size_t(i)] = cplx(re.data()[i], im.data()[i]);
        py::array_t<double> out(n);
        std::vector<cplx> scratch;
        irfft_into(plan, spec.data(), out.mutable_data(), scratch);
        return out;
    });

    m.def("master", [](py::array_t<double, py::array::c_style> x, double gain, double drive,
                       double ceiling, double knee) {
        master(x.mutable_data(), int(x.size()), gain, drive, ceiling, knee);
        return x;
    });

    py::class_<PyConvolver>(m, "Convolver")
        .def(py::init<const Kernel&, int>())
        .def("process", &PyConvolver::process)
        .def("reset", [](PyConvolver& c) { c.conv.reset(); })
        .def_property_readonly("partitions", [](PyConvolver& c) { return c.conv.partitions(); })
        .def_property_readonly("fft_size", [](PyConvolver& c) { return c.conv.fft_size(); });

    py::class_<PyComb>(m, "Comb")
        .def(py::init<int, double, int, double, bool, int>())
        .def("process", &PyComb::process)
        .def("reset", &PyComb::reset);

    m.attr("AVAILABLE") = true;
}
