#ifndef IPPL_BH_DISORDER_HEATING_BH_MANAGER_HPP
#define IPPL_BH_DISORDER_HEATING_BH_MANAGER_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "NBody/BeamDiagnostics.hpp"
#include "NBody/NBodyManager.hpp"

#include "DihDiagnostics.hpp"

namespace ippl::nbody {

namespace dih_detail {

// Constant linear focusing toward the box origin, applied as an additive force
// on top of the BH-computed Ex/Ey/Ez. Mirrors the reference DIH driver at
// disorder_heating.cu:117 (`a -= strength · r / beamRad`).
//
// Sign: our integrator's full kick is `P -= dt · E`, so adding
// `+ strength · R / beamRad` to E produces `dP = -dt · strength · R / beamRad`,
// pushing P toward the origin — same effect as `a -= strength · r / bRad`
// followed by `v += dt · a`.
template <class T>
__global__ void focusingKernel(unsigned start, unsigned n,
                               T scale,
                               const T* __restrict__ Rx,
                               const T* __restrict__ Ry,
                               const T* __restrict__ Rz,
                               T* __restrict__ Ex,
                               T* __restrict__ Ey,
                               T* __restrict__ Ez) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Ex[i] += scale * Rx[i];
    Ey[i] += scale * Ry[i];
    Ez[i] += scale * Rz[i];
}

template <class T>
inline void applyFocusing(SphexaParticleContainer<T, 3>& pc, T strength, T beamRad) {
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    constexpr unsigned kBlockSize = 256;
    const T            scale      = strength / beamRad;
    const unsigned     grid       = (n + kBlockSize - 1) / kBlockSize;
    focusingKernel<T><<<grid, kBlockSize>>>(
        start, n, scale,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw());
}

}  // namespace dih_detail

// Binary format mirrors sphexa/.../disorder_heating.cu:205-219:
//   uint64_t N
//   double[N] x
//   double[N] y
//   double[N] z
inline void loadDihPositions(const std::string& path, std::vector<double>& x,
                             std::vector<double>& y, std::vector<double>& z) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open DIH initial-position file: " + path);
    }
    std::uint64_t N = 0;
    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    x.resize(N);
    y.resize(N);
    z.resize(N);
    in.read(reinterpret_cast<char*>(x.data()), N * sizeof(double));
    in.read(reinterpret_cast<char*>(y.data()), N * sizeof(double));
    in.read(reinterpret_cast<char*>(z.data()), N * sizeof(double));
    if (!in) {
        throw std::runtime_error("Failed to read all particle data from " + path);
    }
}

// Disorder-Induced-Heating reproduction. Open BCs, parameters locked to
// disorder_heating.cu:55-67. Sign flip on G absorbs the kick-convention
// difference between sphexa's `v += dt·a` and our `P -= dt·E`.
//
// Step ordering: Drift → BH → Focus → full Kick.
//
// Output: stdout (DIH log, parseable by sphexa's plot_dih_comparison.py)
// + data/DIH_IPPL_BH.csv with columns
//   step sigma_x sigma_y sigma_z emit_x emit_y emit_z T_x T_y T_z T_norm
// (step = -1 marks the initial-state row).
template <class T, unsigned Dim>
class DisorderHeatingBHManager : public NBodyManager<T, Dim> {
    static_assert(Dim == 3, "DisorderHeatingBHManager requires Dim == 3");

    using Base = NBodyManager<T, Dim>;
    using typename Base::Container;
    using typename Base::SolverParams;

public:
    struct Config {
        int Nt = 1000;
    };

    DisorderHeatingBHManager(std::vector<double> hx,
                             std::vector<double> hy,
                             std::vector<double> hz,
                             Config              cfg = {})
        : Base(hx.size(), cfg.Nt, /*dt=*/T(2.15623e-13))
        , cfg_m(cfg)
        , hx_m(std::move(hx))
        , hy_m(std::move(hy))
        , hz_m(std::move(hz)) {}

protected:
    void initializeContainer() override {
        using cstone::BoundaryType;
        auto pc = std::make_unique<Container>(
            /*rank=*/0, /*nRanks=*/1,
            /*bucketSize=*/64, /*bucketSizeFocus=*/64, theta_m,
            std::array<T, 6>{-boxHalf_m, boxHalf_m,
                             -boxHalf_m, boxHalf_m,
                             -boxHalf_m, boxHalf_m},
            std::array<BoundaryType, 3>{
                BoundaryType::open, BoundaryType::open, BoundaryType::open});
        this->setContainer(std::move(pc));
    }

    void initializeParticles() override {
        const std::size_t bytes = this->N() * sizeof(T);
        cudaMemcpy(this->pc().getRxRaw(), hx_m.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->pc().getRyRaw(), hy_m.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->pc().getRzRaw(), hz_m.data(), bytes, cudaMemcpyHostToDevice);

        thrust::fill_n(thrust::device_ptr<T>(this->pc().getChargeRaw()), this->N(), T(1));
        thrust::fill_n(thrust::device_ptr<T>(this->pc().getHRaw()),      this->N(), smoothH_m);
        thrust::fill_n(thrust::device_ptr<T>(this->pc().getPxRaw()),     this->N(), T(0));
        thrust::fill_n(thrust::device_ptr<T>(this->pc().getPyRaw()),     this->N(), T(0));
        thrust::fill_n(thrust::device_ptr<T>(this->pc().getPzRaw()),     this->N(), T(0));

        hx_m.clear(); hx_m.shrink_to_fit();
        hy_m.clear(); hy_m.shrink_to_fit();
        hz_m.clear(); hz_m.shrink_to_fit();
    }

    void initializeSolverParams(SolverParams& p) override {
        // Sign of G is flipped relative to sphexa DIH (`-ke`) — see class comment.
        p.G     = +ke_m;
        p.theta = theta_m;
    }

    void post_initial_solve() override {
        const auto avgAcc = reduceMeanAbsAccel(this->pc());
        const T avgForce  = std::sqrt(avgAcc.x * avgAcc.x +
                                      avgAcc.y * avgAcc.y +
                                      avgAcc.z * avgAcc.z);
        focusStrength_m = focusMul_m * avgForce;
        std::printf("applyConstantFocusing> Focusing Force %g\n",
                    static_cast<double>(focusStrength_m));
        std::printf("Pre Run finished\n");

        const std::string fnameCsv = "data/DIH_IPPL_BH.csv";
        csvPath_m = fnameCsv;
        std::ofstream header(csvPath_m, std::ios::out);
        header << "step sigma_x sigma_y sigma_z "
               << "emit_x emit_y emit_z "
               << "T_x T_y T_z T_norm\n";
    }

    void advanceImpl() override {
        this->drift();
        this->solve();
        focus();
        this->kickFull();
    }

    void focus() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("focus");
        ScopedIpplTimer scope(t);
        dih_detail::applyFocusing<T>(this->pc(), focusStrength_m, beamRad_m);
    }

    void dumpImpl() override {
        // it_m == 0: initial dump from pre_run(); it_m == k+1: post-step
        // dump after iteration k.
        const int step = this->it() - 1;
        const bool isInitial = (step == -1);
        printDihStats(this->pc(), csvPath_m, step, isInitial);

        if (!isInitial) {
            std::printf("LeapFrogStep> Step %d Finished.\n", step);
            std::fflush(stdout);
        }
    }

private:
    // Reproduces DIH's printStats (disorder_heating.cu:221-264) so that
    // sphexa's plot_dih_comparison.py parses our log unchanged. Also appends a
    // CSV row with the same metrics.
    static void printDihStats(SphexaParticleContainer<T, 3>& pc,
                              const std::string& fnameCsv,
                              int step, bool isInitial) {
        auto bs    = reduceBeamStats(pc);
        const T n  = static_cast<T>(pc.endIndex() - pc.startIndex());
        const T invN = T(1) / n;
        for (int i = 0; i < 12; ++i) { bs.vals[i] *= invN; }

        auto safeSqrt = [](T v) { return std::sqrt(std::max(T(0), v)); };
        const T sigma_x = safeSqrt(bs.vals[0] - bs.vals[9]  * bs.vals[9]);
        const T sigma_y = safeSqrt(bs.vals[3] - bs.vals[10] * bs.vals[10]);
        const T sigma_z = safeSqrt(bs.vals[6] - bs.vals[11] * bs.vals[11]);
        const T emit_x  = safeSqrt(bs.vals[0] * bs.vals[1] - bs.vals[2] * bs.vals[2]);
        const T emit_y  = safeSqrt(bs.vals[3] * bs.vals[4] - bs.vals[5] * bs.vals[5]);
        const T emit_z  = safeSqrt(bs.vals[6] * bs.vals[7] - bs.vals[8] * bs.vals[8]);

        std::printf("ComputeBeamStatistics> Beam Statistics: \n");
        std::printf("ComputeBeamStatistics> RMS Beam Size: %g , %g , %g\n",
                    static_cast<double>(sigma_x), static_cast<double>(sigma_y),
                    static_cast<double>(sigma_z));
        std::printf("ComputeBeamStatistics> RMS Emittance: %g , %g , %g\n",
                    static_cast<double>(emit_x), static_cast<double>(emit_y),
                    static_cast<double>(emit_z));

        auto avgV = reduceMeanVelocity(pc);
        std::printf("computeTemperature> Average Velocity: ( %g , %g , %g )\n",
                    static_cast<double>(avgV.x), static_cast<double>(avgV.y),
                    static_cast<double>(avgV.z));

        auto T3 = reduceTemperature(pc, avgV);
        const T Tnorm = std::sqrt(T3.x * T3.x + T3.y * T3.y + T3.z * T3.z);
        std::printf("computeTemperature> Temperature: ( %g , %g , %g )\n",
                    static_cast<double>(T3.x), static_cast<double>(T3.y),
                    static_cast<double>(T3.z));
        std::printf("computeTemperature> L2-Norm of Temperature: %g\n",
                    static_cast<double>(Tnorm));

        std::ofstream row(fnameCsv, std::ios::app);
        row.precision(16);
        row.setf(std::ios::scientific, std::ios::floatfield);
        row << (isInitial ? -1 : step) << " "
            << sigma_x << " " << sigma_y << " " << sigma_z << " "
            << emit_x  << " " << emit_y  << " " << emit_z  << " "
            << T3.x    << " " << T3.y    << " " << T3.z    << " "
            << Tnorm   << "\n";
    }

    // Physical parameters (locked to sphexa DIH reference).
    static constexpr T     ke_m            = T(2.532638e8);
    static constexpr T     boxHalf_m       = T(0.005);
    static constexpr T     beamRad_m       = T(0.001774);
    static constexpr T     smoothH_m       = T(1e-10);
    static constexpr T     focusMul_m      = T(1.5);
    static constexpr float theta_m         = 0.5f;

    Config cfg_m;

    std::vector<double> hx_m, hy_m, hz_m;

    T           focusStrength_m{T(0)};
    std::string csvPath_m;
};

}  // namespace ippl::nbody

#endif  // IPPL_BH_DISORDER_HEATING_BH_MANAGER_HPP
