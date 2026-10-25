// Gas-dynamic engine sound core (roadmap phase A).
//
// Paradigm shift from the filter bank: the sound *emerges* from a network of
// lumped gas systems (Ange Yaghi's engine-sim model, ported from
// gas_system.cpp / exhaust_system.cpp) rather than passing through filters.
// Every element -- cylinder head, runner, collector, pipe, muffler segment,
// atmosphere -- is a GasSystem with {n_mol, V, E_k, momentum[2], mix}.
// Flow between systems is compressible orifice flow with a choked branch;
// momentum carried with the flow and dissipated as supersonic velocity is
// clamped back into heat. The outlet volume velocity of the tailpipe IS the
// audio: rasp on load comes from the choked-flow nonlinearity, cross-plane
// burble from real pulse interference at the collector junction.
//
// This file implements the core physics (phase A1-A3) plus the network
// runner (A5): the Python side builds the graph from a preset, the C++ side
// integrates it at the audio rate.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double R_GAS = 8.314462618;
constexpr double P_ATM = 101325.0;
constexpr double T_AMB = 293.15;

struct Mix {
    double p_fuel = 0.0;
    double p_o2 = 0.0;
    double p_inert = 1.0;
};

// --------------------------------------------------------------------------- //
// GasSystem: one lumped thermodynamic volume with 2D bulk momentum
// --------------------------------------------------------------------------- //
struct GasSystem {
    double n = 0.0;          // mol
    double V = 1.0;          // m^3
    double E_k = 0.0;        // J (thermal)
    double mom[2] = {0, 0};  // bulk momentum, kg*m/s
    Mix mix;

    // geometry for the pressure-on-faces momentum update (A3)
    double width = 1.0, height = 1.0;
    double dx = 1.0, dy = 0.0;

    int dof = 5;             // degrees of freedom (3 transl + 2 rot)
    double choked_limit = 0.0;
    double choked_rate = 0.0;

    void initialize(double P, double V_, double T, int dof_ = 5) {
        dof = dof_;
        V = V_;
        n = P * V / (R_GAS * T);
        E_k = T * (0.5 * dof * n * R_GAS);
        mom[0] = mom[1] = 0.0;
        mix = Mix{};
        cache_flow();
    }

    void cache_flow() {
        const double hcr = heat_capacity_ratio();
        choked_limit = std::pow(2.0 / (hcr + 1.0), hcr / (hcr - 1.0));
        // cached choked mass-flux factor: sqrt(hcr) * (2/(hcr+1))^((hcr+1)/(2(hcr-1)))
        choked_rate = std::sqrt(hcr) *
            std::pow(2.0 / (hcr + 1.0), (hcr + 1.0) / (2.0 * (hcr - 1.0)));
    }

    void set_geometry(double w, double h, double dx_, double dy_) {
        width = w; height = h; dx = dx_; dy = dy_;
    }

    inline double heat_capacity_ratio() const { return 1.0 + 2.0 / dof; }
    inline double temperature() const {
        return n > 1e-12 ? E_k / (0.5 * dof * n * R_GAS) : 0.0;
    }
    inline double pressure() const {
        if (V <= 0.0) return 0.0;
        // bound the head temperature: combustion dumps can spike T past
        // any physical value in one substep; the clamp keeps c() and the
        // flow sane while preserving the energy trend
        const double T = temperature();
        return n * R_GAS * (T > 4000.0 ? 4000.0 : T) / V;
    }
    inline double mass() const { return n * 0.0289647; }  // air molar mass
    inline double c() const {
        const double T = temperature();
        return std::sqrt(heat_capacity_ratio() * R_GAS * (T > 4000.0 ? 4000.0 : T) / 0.0289647);
    }
    inline double kinetic_energy_per_mol() const { return n > 0.0 ? E_k / n : 0.0; }
    inline double velocity_x() const { const double m = mass(); return m > 0 ? mom[0] / m : 0.0; }
    inline double velocity_y() const { const double m = mass(); return m > 0 ? mom[1] / m : 0.0; }

    inline double dynamic_pressure(double fx, double fy) const {
        // component of bulk kinetic energy along direction (fx, fy)
        const double m = mass();
        if (m <= 0.0) return 0.0;
        const double vx = velocity_x(), vy = velocity_y();
        return 0.5 * m * (vx * fx + vy * fy) * (vx * fx + vy * fy) / V;
    }

    void change_volume(double dV) {
        // quasi-cubic expansion work (Ange's approximation, ported)
        const double V0 = V;
        const double L = std::cbrt(V0 + dV);
        const double dL = -dV / (L * L);
        V = V0 + dV;
        E_k += dL * pressure() * (L * L);
        if (E_k < 0.0) E_k = 0.0;
    }

    void add_energy(double dE) {
        E_k += dE;
        // bound runaway: a mis-tuned Wiebe dump must not push T to millions.
        // The cap is far above any real exhaust temperature; hitting it
        // means the event feed is wrong, and clamping keeps the network
        // stable instead of exploding to 1e18 like the first render did.
        const double T_max = 4000.0;
        const double E_max = T_max * (0.5 * dof * std::max(n, 1e-12) * R_GAS);
        if (E_k > E_max) E_k = E_max;
    }
    void add_n(double dn, double e_per_mol) {
        E_k += e_per_mol * dn;
        n += dn;
        if (n < 0.0) n = 0.0;
    }

    // 2-argument legacy probe kept for the forward-declared definition
    inline double valve_flow_capture_impl(double k_flow, double p_sink, double t_sink) const;

    double sink_p_probe = P_ATM;
    double sink_T_probe = T_AMB;

    // signed mol/s this system would push through the given orifice THIS
    // substep (positive = out of this system). Used by run_excitation to
    // sum the valve blowdowns without disturbing the state: a probe, not
    // a move. The FlowParams type is declared below GasSystem (it carries
    // no state of its own), so the probe is defined after both.
    double valve_flow_capture(const struct FlowParams& fp) const;

    // A3: clamp any supersonic bulk velocity back to sonic, dumping the
    // excess into heat. This is the nonlinearity that makes the rasp: a
    // hard-blowing pulse cannot propagate faster than sound, and the energy
    // that would have gone into a sharper front becomes pressure instead.
    void dissipate_excess_velocity() {
        const double vx = velocity_x(), vy = velocity_y();
        const double v2 = vx * vx + vy * vy;
        const double cc = c() * c();
        if (cc >= v2 || v2 == 0.0) return;
        const double k = std::sqrt(cc / v2);
        mom[0] *= k; mom[1] *= k;
        E_k += 0.5 * mass() * (v2 - cc);
        if (E_k < 0.0) E_k = 0.0;
    }

    // A3: pressure forces on the box faces accelerate/decelerate the bulk
    // gas (Ange's updateVelocity, ported and simplified for the exhaust
    // direction). beta < 1 is wall friction (velocityDecay in the preset).
    void update_velocity(double dt, double beta) {
        if (n == 0.0) return;
        const double depth = V / (width * height);
        const double vx0 = velocity_x(), vy0 = velocity_y();
        const double m = mass();

        // pressure on the two faces normal to the flow axis
        const double p0 = dynamic_pressure(dx, dy);
        const double p1 = dynamic_pressure(-dx, -dy);
        const double A = height * depth;
        mom[0] -= (p0 - p1) * A * dx * dt * beta;
        mom[1] -= (p0 - p1) * A * dy * dt * beta;
        // wall drag along the axis
        mom[0] *= (1.0 - beta * dt * 0.5);
        mom[1] *= (1.0 - beta * dt * 0.5);

        const double vx1 = velocity_x(), vy1 = velocity_y();
        E_k -= 0.5 * m * ((vx1 * vx1 - vx0 * vx0) + (vy1 * vy1 - vy0 * vy0));
        if (E_k < 0.0) E_k = 0.0;
    }
};

// --------------------------------------------------------------------------- //
// flow(): compressible orifice flow between two systems (A2)
// --------------------------------------------------------------------------- //
struct FlowParams {
    double k_flow = 0.0;        // orifice conductance
    double area_0 = 0.0, area_1 = 0.0;
    double dx = 1.0, dy = 0.0;
};

// probe definition (declared inside GasSystem before FlowParams existed).
// flow_rate is declared below; declare it here so the probe can call it.
double flow_rate(double k, double P0, double P1, double T0, double T1,
                 double hcr, double choked_limit, double choked_rate);

inline double GasSystem::valve_flow_capture(const FlowParams& fp) const {
    const double hcr = heat_capacity_ratio();
    const double P0 = pressure();
    const double rate = flow_rate(fp.k_flow, P0, sink_p_probe,
                                  temperature(), sink_T_probe,
                                  hcr, choked_limit, choked_rate);
    return rate;  // mol/s, signed
}

// flow rate in mol/s (signed: positive 0 -> 1). Ported from GasSystem::flow.
double flow_rate(
    double k, double P0, double P1, double T0, double T1,
    double hcr, double choked_limit, double choked_rate)
{
    if (k == 0.0) return 0.0;
    double direction, T_up, p_up, p_dn;
    if (P0 > P1) { direction = 1.0; T_up = T0; p_up = P0; p_dn = P1; }
    else         { direction = -1.0; T_up = T1; p_up = P1; p_dn = P0; }
    const double ratio = p_dn / std::max(p_up, 1e-9);
    double rate;
    if (ratio <= choked_limit) {
        rate = choked_rate / std::sqrt(R_GAS * std::max(T_up, 1e-9));
    } else {
        const double s = std::pow(ratio, 1.0 / hcr);
        rate = (2.0 * hcr) / (hcr - 1.0);
        rate *= s * (s - ratio);
        rate = std::sqrt(std::max(rate, 0.0) / (R_GAS * std::max(T_up, 1e-9)));
    }
    return direction * p_up * rate * k;
}

// equilibrium clamp: never move more than equalises the pressures
double pressure_eq_max_flow(const GasSystem& a, const GasSystem& b) {
    if (a.pressure() > b.pressure()) {
        const double d = (b.V * a.E_k - a.V * b.E_k) /
            (b.V * a.kinetic_energy_per_mol() + a.V * b.kinetic_energy_per_mol());
        return std::max(0.0, std::min(d, a.n));
    }
    return 0.0;
}

void do_flow(GasSystem& s0, GasSystem& s1, const FlowParams& fp, double dt,
             double* outlet_capture = nullptr) {
    const double P0 = s0.pressure();
    const double P1 = s1.pressure();
    if (P0 == P1) return;
    GasSystem* src = &s0;
    GasSystem* dst = &s1;
    double fdir = 1.0;
    if (P0 < P1) { src = &s1; dst = &s0; fdir = -1.0; }

    const double hcr = src->heat_capacity_ratio();
    double flow = dt * flow_rate(fp.k_flow, src->pressure(), dst->pressure(),
                                src->temperature(), dst->temperature(),
                                hcr, src->choked_limit, src->choked_rate);
    // clamp to pressure equilibrium: never overshoot into inversion
    const double max_flow = pressure_eq_max_flow(*src, *dst);
    if (max_flow > 0.0) flow = std::min(flow, max_flow);
    // and never empty more than half the source in one substep (the 0.9 in
    // the original allowed a near-empty head, whose pressure then spiked on
    // the next combustion event -- the divergence this replaced)
    if (flow <= 0.0) return;
    flow = std::min(flow, 0.5 * src->n);
    if (flow <= 0.0) return;

    const double e_per_mol = src->kinetic_energy_per_mol();
    const double src_n0 = src->n;
    src->add_n(-flow, -e_per_mol);
    dst->add_n(flow, e_per_mol);

    // radiated capture: volume velocity of the gas crossing THIS orifice
    // this substep (m^3/s at the destination's temperature). The atmosphere
    // orifice's capture is the audio: a rate, so pulse timing survives the
    // lumped pipe that low-passes volume-velocity states.
    if (outlet_capture != nullptr) {
        const double T_dst = dst->temperature();
        *outlet_capture = fdir * flow / dt * (T_dst / P_ATM) * (R_GAS * T_dst / P_ATM);
        // n/dt [mol/s] -> m^3/s via ideal gas: V_dot = n_dot R T / P
        *outlet_capture = fdir * (flow / dt) * R_GAS * std::max(T_dst, 1.0) / P_ATM;
    }

    // momentum: the transferred mass carries the source's bulk velocity plus
    // an orifice-jet component from the conductance. The jet is bounded by
    // the destination's speed of sound; the excess lands as heat in
    // dissipate_excess_velocity, which is the rasp path.
    const double m_flow = flow * 0.0289647;
    const double frac = flow / std::max(src_n0, 1e-12);
    const double mvx = src->mom[0] * frac, mvy = src->mom[1] * frac;
    src->mom[0] -= mvx; src->mom[1] -= mvy;

    const double jet_v = std::min((flow / std::max(fp.area_1, 1e-9)) / dt, dst->c());
    const double jet_mom = jet_v * m_flow;
    dst->mom[0] += (mvx + jet_mom) * fp.dx * fdir;
    dst->mom[1] += (mvy + jet_mom) * fp.dy * fdir;

    // bulk kinetic energy the source lost is re-added as heat: the port and
    // the jet both thermalise downstream, keeping E total bounded
    if (src_n0 > 0.0) {
        const double v2_src = (mvx * mvx + mvy * mvy) / (m_flow * m_flow + 1e-30);
        src->E_k += 0.5 * m_flow * v2_src;
    }
    if (src->E_k < 0.0) src->E_k = 0.0;
    if (dst->E_k < 0.0) dst->E_k = 0.0;
}

// --------------------------------------------------------------------------- //
// GasNetwork: the runner (A5). Python builds {systems, orifices}; C++
// integrates, Python feeds cylinder events and reads the outlet flow.
// --------------------------------------------------------------------------- //
struct Orifice {
    int src = 0, dst = 0;
    FlowParams fp;
};

struct GasNetwork {
    std::vector<GasSystem> sys;
    std::vector<Orifice> orf;

    // event queue: {system index, energy J, duration s} -- a combustion
    // dump into the cylinder head volume
    struct Event { int sys; double energy; double n_add; };
    std::vector<Event> events;

    // valve command: orifice conductances can be retargeted by the caller
    // (valve lift profile). index into orf, k_flow value.
    std::vector<std::pair<int, double>> k_override;

    int add_system(double P, double V, double T, double dof) {
        GasSystem g;
        g.initialize(P, V, T, int(dof));
        sys.push_back(g);
        return int(sys.size()) - 1;
    }

    void set_geometry(int i, double w, double h, double dx, double dy) {
        sys[size_t(i)].set_geometry(w, h, dx, dy);
    }

    int add_orifice(int s, int d, double k, double a0, double a1, double dx, double dy) {
        Orifice o;
        o.src = s; o.dst = d;
        o.fp.k_flow = k; o.fp.area_0 = a0; o.fp.area_1 = a1;
        o.fp.dx = dx; o.fp.dy = dy;
        orf.push_back(o);
        return int(orf.size()) - 1;
    }

    void set_orifice_k(int i, double k) { orf[size_t(i)].fp.k_flow = k; }

    void add_event(int s, double energy, double n_add) {
        events.push_back({s, energy, n_add});
    }

    // one physics step. dt = substep. Returns the outlet volume velocity
    // (m^3/s) of `outlet` -- this IS the audio sample.
    double step(int outlet, double dt, double velocity_decay) {
        // apply queued combustion events (Wiebe energy per substep, fed by
        // the Python physics which already knows the phase)
        for (const Event& e : events) {
            sys[size_t(e.sys)].add_energy(e.energy);
            sys[size_t(e.sys)].add_n(e.n_add, 0.5 * 5 * R_GAS * 800.0);
        }
        events.clear();

        // valve overrides
        for (const auto& kv : k_override) orf[size_t(kv.first)].fp.k_flow = kv.second;
        k_override.clear();

        // flows
        for (const Orifice& o : orf) {
            do_flow(sys[size_t(o.src)], sys[size_t(o.dst)], o.fp, dt);
        }
        // velocity update + supersonic clamp on every system
        for (GasSystem& g : sys) {
            g.update_velocity(dt, velocity_decay);
            g.dissipate_excess_velocity();
        }

        const GasSystem& out = sys[size_t(outlet)];
        // volume velocity: bulk speed x cross-section (the depth factor in
        // the geometry was double-counting the pipe length)
        return out.velocity_x() * (out.width * out.height);
    }

    // Audio excitation: the SUM of the valve-orifice flows (mol/s), signed
    // by flow direction into the tract. The lumped network is a mean-value
    // model -- the atmosphere orifice flow is a smooth pump curve -- but
    // the VALVE flows pulse with every blowdown (the burn spikes the head
    // pressure and the valve dumps it). Summing the valves gives the
    // collector's input history: the excitation that a real tract's
    // acoustics filter into the radiated sound. Captured per substep; the
    // caller builds the tract IR from the same geometry and convolves.
    double valve_capture = 0.0;

    // capture valve flows during a step: called by run() per substep
    void capture_valves_reset() { valve_capture = 0.0; }

    double run_excitation(int n_samples, double dt, double velocity_decay,
                          double omega, double theta0, double cycle_rad,
                          double load, double wiebe_a, double wiebe_m,
                          double burn_start_rad, double burn_span_rad,
                          double exhaust_open_deg, double exhaust_close_deg,
                          py::array_t<double>& outv)
    {
        auto out = outv.mutable_data();
        const int ncyl = int(drive.size());
        const double dtheta = omega * dt;
        double theta = theta0;
        const double ex_open = exhaust_open_deg * PI / 180.0;
        const double ex_close = exhaust_close_deg * PI / 180.0;
        const double ex_total =
            (360.0 - (exhaust_open_deg - (exhaust_close_deg - 360.0))) * PI / 180.0;
        const double cyl_fuel = 1.8e3 * (load > 0.15 ? load : 0.15);
        const double e_mol_ref = 0.5 * 5 * R_GAS * 800.0;
        const double mol_to_m3 = R_GAS * 800.0 / P_ATM;  // hot gas m^3 per mol

        for (int s = 0; s < n_samples; ++s) {
            for (int i = 0; i < ncyl; ++i) {
                const CylDrive& d = drive[size_t(i)];
                double ca = theta - d.phase * cycle_rad;
                ca = std::fmod(ca, cycle_rad);
                if (ca < 0.0) ca += cycle_rad;

                double lift = 0.0;
                if (ca >= ex_open || ca < ex_close) {
                    double x;
                    if (ca >= ex_open) x = (ca - ex_open) / ex_total;
                    else x = (ca + cycle_rad - ex_open) / ex_total;
                    if (x > 1.0) x = 1.0;
                    lift = 0.5 * (1.0 - std::cos(2.0 * PI * x));
                }
                orf[size_t(d.valve_orifice)].fp.k_flow = d.k_valve * lift;

                double xx = ca - burn_start_rad;
                xx = std::fmod(xx, cycle_rad);
                if (xx < 0.0) xx += cycle_rad;
                xx /= burn_span_rad;
                if (xx > 0.0 && xx <= 1.0) {
                    const double x0 = xx - dtheta / burn_span_rad > 0.0
                                    ? xx - dtheta / burn_span_rad : 0.0;
                    const double xb0 = 1.0 - std::exp(-wiebe_a * std::pow(x0, wiebe_m + 1.0));
                    const double xb1 = 1.0 - std::exp(-wiebe_a * std::pow(xx, wiebe_m + 1.0));
                    const double dE = (xb1 - xb0) * cyl_fuel;
                    if (dE > 0.0) {
                        sys[size_t(d.head_sys)].add_energy(dE);
                        sys[size_t(d.head_sys)].add_n(dE / e_mol_ref * 0.06, e_mol_ref);
                    }
                }
            }

            // flow the valves and sum their signed rates (probe first with
            // fresh sink state, then commit through do_flow)
            double exc = 0.0;
            for (const CylDrive& d : drive) {
                const int dst_sys = orf[size_t(d.valve_orifice)].dst;
                sys[size_t(d.head_sys)].sink_p_probe = sys[size_t(dst_sys)].pressure();
                sys[size_t(d.head_sys)].sink_T_probe = sys[size_t(dst_sys)].temperature();
                exc += sys[size_t(d.head_sys)].valve_flow_capture(
                    orf[size_t(d.valve_orifice)].fp);
            }
            for (const Orifice& o : orf) {
                do_flow(sys[size_t(o.src)], sys[size_t(o.dst)], o.fp, dt);
            }
            for (GasSystem& g : sys) {
                g.update_velocity(dt, velocity_decay);
                g.dissipate_excess_velocity();
            }

            out[s] = exc * mol_to_m3;  // m^3/s of hot gas into the tract

            theta += dtheta;
            if (theta >= cycle_rad) theta -= cycle_rad;
        }
        return theta;
    }

    // ------------------------------------------------------------------ //
    // batched driver: the whole per-sample crank/valve/Wiebe loop in C++.
    // The Python per-sample loop was 384k interpreter iterations per audio
    // second (~2-4 s/s); this does the same arithmetic in one call.
    // ------------------------------------------------------------------ //
    // per-cylinder driver state, set up once by the caller
    struct CylDrive {
        int valve_orifice = -1;
        int head_sys = -1;
        double k_valve = 0.0;   // full-lift conductance
        double phase = 0.0;     // crank phase, fraction of the cycle
    };
    std::vector<CylDrive> drive;

    void set_drive(const std::vector<int>& valve_orf, const std::vector<int>& head_sys,
                   const std::vector<double>& k_valve, const std::vector<double>& phase) {
        drive.clear();
        for (size_t i = 0; i < valve_orf.size(); ++i) {
            CylDrive d;
            d.valve_orifice = valve_orf[i];
            d.head_sys = head_sys[i];
            d.k_valve = k_valve[i];
            d.phase = phase[i];
            drive.push_back(d);
        }
    }

    // Run `n_samples` gas substeps writing the outlet flow into `out`.
    // theta advances internally from the passed crank angle (rad, wrapped
    // by the caller's cycle bookkeeping). Returns the advanced theta.
    double run(int n_samples, double dt, double velocity_decay,
               double omega,          // crank speed, rad/s
               double theta0,         // crank angle at block start, rad
               double cycle_rad,      // full cycle, rad
               double load,           // 0..1, Wiebe charge scale
               double wiebe_a, double wiebe_m,
               double burn_start_rad, double burn_span_rad,
               double exhaust_open_deg, double exhaust_close_deg,
               py::array_t<double>& outv)
    {
        auto out = outv.mutable_data();
        const int ncyl = int(drive.size());
        const double dtheta = omega * dt;
        double theta = theta0;
        const double ex_open = exhaust_open_deg * PI / 180.0;
        const double ex_close = exhaust_close_deg * PI / 180.0;
        const double ex_total = (360.0 - (exhaust_open_deg - (exhaust_close_deg - 360.0))) * PI / 180.0;
        // blowdown charge: 1.8 kJ per 500 cc cylinder at full load
        const double cyl_fuel = 1.8e3 * (load > 0.15 ? load : 0.15);
        const double e_mol_ref = 0.5 * 5 * R_GAS * 800.0;
        const double mol_to_m3 = R_GAS * 800.0 / P_ATM;  // hot gas m^3 per mol

        for (int s = 0; s < n_samples; ++s) {
            for (int i = 0; i < ncyl; ++i) {
                const CylDrive& d = drive[size_t(i)];
                double ca = theta - d.phase * cycle_rad;
                ca = std::fmod(ca, cycle_rad);
                if (ca < 0.0) ca += cycle_rad;

                // exhaust valve lift profile (cosine seat to seat)
                double lift = 0.0;
                if (ca >= ex_open || ca < ex_close) {
                    double x;
                    if (ca >= ex_open) x = (ca - ex_open) / ex_total;
                    else x = (ca + cycle_rad - ex_open) / ex_total;
                    if (x > 1.0) x = 1.0;
                    lift = 0.5 * (1.0 - std::cos(2.0 * PI * x));
                }
                orf[size_t(d.valve_orifice)].fp.k_flow = d.k_valve * lift;

                // Wiebe burn window
                double xx = ca - burn_start_rad;
                xx = std::fmod(xx, cycle_rad);
                if (xx < 0.0) xx += cycle_rad;
                xx /= burn_span_rad;
                if (xx > 0.0 && xx <= 1.0) {
                    const double x0 = xx - dtheta / burn_span_rad > 0.0
                                    ? xx - dtheta / burn_span_rad : 0.0;
                    const double xb0 = 1.0 - std::exp(-wiebe_a * std::pow(x0, wiebe_m + 1.0));
                    const double xb1 = 1.0 - std::exp(-wiebe_a * std::pow(xx, wiebe_m + 1.0));
                    const double dE = (xb1 - xb0) * cyl_fuel;
                    if (dE > 0.0) {
                        sys[size_t(d.head_sys)].add_energy(dE);
                        sys[size_t(d.head_sys)].add_n(dE / e_mol_ref * 0.06, e_mol_ref);
                    }
                }
            }

            // flow the valves and sum their signed rates (probe first with
            // fresh sink state, then commit through do_flow)
            double exc = 0.0;
            for (const CylDrive& d : drive) {
                const int dst_sys = orf[size_t(d.valve_orifice)].dst;
                sys[size_t(d.head_sys)].sink_p_probe = sys[size_t(dst_sys)].pressure();
                sys[size_t(d.head_sys)].sink_T_probe = sys[size_t(dst_sys)].temperature();
                exc += sys[size_t(d.head_sys)].valve_flow_capture(
                    orf[size_t(d.valve_orifice)].fp);
            }
            for (const Orifice& o : orf) {
                do_flow(sys[size_t(o.src)], sys[size_t(o.dst)], o.fp, dt);
            }
            for (GasSystem& g : sys) {
                g.update_velocity(dt, velocity_decay);
                g.dissipate_excess_velocity();
            }

            out[s] = exc * mol_to_m3;  // m^3/s of hot gas into the tract

            theta += dtheta;
            if (theta >= cycle_rad) theta -= cycle_rad;
        }
        return theta;
    }
};

double flow_rate_binding(GasSystem& a, GasSystem& b, double k, double dt) {
    FlowParams fp; fp.k_flow = k;
    const double hcr = a.heat_capacity_ratio();
    return dt * flow_rate(k, a.pressure(), b.pressure(), a.temperature(), b.temperature(),
                          hcr, a.choked_limit, a.choked_rate);
}

}  // namespace

PYBIND11_MODULE(_gascore, m) {
    m.doc() = "Gas-dynamic engine sound core (lumped systems, compressible flow)";

    py::class_<GasNetwork>(m, "Network")
        .def(py::init<>())
        .def("add_system", &GasNetwork::add_system,
             py::arg("P"), py::arg("V"), py::arg("T"), py::arg("dof") = 5)
        .def("set_geometry", &GasNetwork::set_geometry)
        .def("add_orifice", &GasNetwork::add_orifice)
        .def("set_orifice_k", &GasNetwork::set_orifice_k)
        .def("add_event", &GasNetwork::add_event)
        .def("run_excitation", &GasNetwork::run_excitation,
             py::arg("n_samples"), py::arg("dt"), py::arg("velocity_decay"),
             py::arg("omega"), py::arg("theta0"), py::arg("cycle_rad"),
             py::arg("load"), py::arg("wiebe_a"), py::arg("wiebe_m"),
             py::arg("burn_start_rad"), py::arg("burn_span_rad"),
             py::arg("exhaust_open_deg"), py::arg("exhaust_close_deg"),
             py::arg("out"))
        .def("step", &GasNetwork::step,
             py::arg("outlet"), py::arg("dt"), py::arg("velocity_decay"))
        .def("set_drive", [](GasNetwork& net,
                             const std::vector<int>& valve_orf,
                             const std::vector<int>& head_sys,
                             const std::vector<double>& k_valve,
                             const std::vector<double>& phase) {
            net.set_drive(valve_orf, head_sys, k_valve, phase);
         })
        .def("run", &GasNetwork::run,
             py::arg("n_samples"), py::arg("dt"),
             py::arg("velocity_decay"), py::arg("omega"), py::arg("theta0"),
             py::arg("cycle_rad"), py::arg("load"),
             py::arg("wiebe_a"), py::arg("wiebe_m"),
             py::arg("burn_start_rad"), py::arg("burn_span_rad"),
             py::arg("exhaust_open_deg"), py::arg("exhaust_close_deg"),
             py::arg("out"))
        .def("get_n", [](GasNetwork& net, int i) { return net.sys[size_t(i)].n; })
        .def("get_p", [](GasNetwork& net, int i) { return net.sys[size_t(i)].pressure(); })
        .def("get_T", [](GasNetwork& net, int i) { return net.sys[size_t(i)].temperature(); })
        .def("get_vx", [](GasNetwork& net, int i) { return net.sys[size_t(i)].velocity_x(); })
        .def("reset", [](GasNetwork& net) {
            for (auto& g : net.sys) { g.mom[0] = g.mom[1] = 0.0; }
         });

    m.attr("R_GAS") = R_GAS;
    m.attr("P_ATM") = P_ATM;
    m.attr("T_AMB") = T_AMB;
}
