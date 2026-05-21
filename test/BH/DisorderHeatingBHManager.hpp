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
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "NBody/BeamDiagnostics.hpp"
#include "NBody/NBodyManager.hpp"

namespace ippl::nbody {

// 12-component per-step beam-physics reduction, mirroring DIH's BeamStats:
//   vals[ 0] = Σ x²        vals[ 1] = Σ vₓ²       vals[ 2] = Σ x·vₓ
//   vals[ 3] = Σ y²        vals[ 4] = Σ vᵧ²       vals[ 5] = Σ y·vᵧ
//   vals[ 6] = Σ z²        vals[ 7] = Σ v_z²      vals[ 8] = Σ z·v_z
//   vals[ 9] = Σ x         vals[10] = Σ y         vals[11] = Σ z
template <class T>
struct BeamStats12 {
    T vals[12];
};

template <class P>
BeamStats12<typename P::Tc> reduceBeamStats(SphexaParticleContainer<P, 3>& pc);

template <class P>
Triple<typename P::Tc> reduceMeanAbsAccel(SphexaParticleContainer<P, 3>& pc);

template <class P>
void sampleDihIC(SphexaParticleContainer<P, 3>& pc,
                 unsigned localN,
                 unsigned long firstGlobal,
                 typename P::Tc beamRad,
                 typename P::Th smoothingH,
                 unsigned long seed);

namespace dih_detail {

// Constant linear focusing toward the box origin, applied as an additive force
// on top of the BH-computed Ex/Ey/Ez. Mirrors the reference DIH driver at
// disorder_heating.cu:117 (`a -= strength · r / beamRad`).
//
// Sign: our integrator's full kick is `P -= dt · E`, so adding
// `+ strength · R / beamRad` to E produces `dP = -dt · strength · R / beamRad`,
// pushing P toward the origin — same effect as `a -= strength · r / bRad`
// followed by `v += dt · a`.
//
// Mixed-precision: positions and `scale` are at Tc; Ex/Ey/Ez are at Ta. The
// add converts the Tc·Tc product to Ta implicitly.
template <class Tc, class Ta>
__global__ void focusingKernel(unsigned start, unsigned n,
                               Tc scale,
                               const Tc* __restrict__ Rx,
                               const Tc* __restrict__ Ry,
                               const Tc* __restrict__ Rz,
                               Ta* __restrict__ Ex,
                               Ta* __restrict__ Ey,
                               Ta* __restrict__ Ez) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Ex[i] += static_cast<Ta>(scale * Rx[i]);
    Ey[i] += static_cast<Ta>(scale * Ry[i]);
    Ez[i] += static_cast<Ta>(scale * Rz[i]);
}

template <class P>
inline void applyFocusing(SphexaParticleContainer<P, 3>& pc,
                          typename P::Tc strength,
                          typename P::Tc beamRad) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    constexpr unsigned kBlockSize = 256;
    const Tc           scale      = strength / beamRad;
    const unsigned     grid       = (n + kBlockSize - 1) / kBlockSize;
    focusingKernel<Tc, Ta><<<grid, kBlockSize>>>(
        start, n, scale,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw());
}

// Copy a std::vector<double> of positions to a device buffer typed as Tc.
// For Tc=double this is a direct memcpy; for Tc=float we narrow on the host
// into a scratch vector and then memcpy. File mode is only exercised with
// the reference 4096-particle .bin (all-double); the float fallback exists
// for symmetry with the precision policy, not because of an existing use case.
template <class Tc>
inline void copyHostPositions(Tc* dev, const std::vector<double>& src) {
    if constexpr (std::is_same_v<Tc, double>) {
        cudaMemcpy(dev, src.data(), src.size() * sizeof(double), cudaMemcpyHostToDevice);
    } else {
        std::vector<Tc> tmp(src.begin(), src.end());
        cudaMemcpy(dev, tmp.data(), tmp.size() * sizeof(Tc), cudaMemcpyHostToDevice);
    }
}

}  // namespace dih_detail

// Binary format mirrors sphexa's DIH driver:
//   uint64_t N
//   double[N] x
//   double[N] y
//   double[N] z
//
// Distributed read: rank 0 reads the file, broadcasts N, then scatters per-rank
// slices [r*N/R, (r+1)*N/R) via MPI_Scatterv. After return, x/y/z hold each
// rank's local slice. Single-rank: degenerate to a plain read (rank 0 only).
inline void loadDihPositions(const std::string& path, std::vector<double>& x,
                             std::vector<double>& y, std::vector<double>& z,
                             std::uint64_t* globalN = nullptr) {
    const int rank     = ippl::Comm->rank();
    const int numRanks = ippl::Comm->size();
    MPI_Comm  comm     = ippl::Comm->getCommunicator();

    std::vector<double> hostX, hostY, hostZ;
    std::uint64_t       N = 0;

    if (rank == 0) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open DIH initial-position file: " + path);
        }
        in.read(reinterpret_cast<char*>(&N), sizeof(N));
        hostX.resize(N);
        hostY.resize(N);
        hostZ.resize(N);
        in.read(reinterpret_cast<char*>(hostX.data()), N * sizeof(double));
        in.read(reinterpret_cast<char*>(hostY.data()), N * sizeof(double));
        in.read(reinterpret_cast<char*>(hostZ.data()), N * sizeof(double));
        if (!in) {
            throw std::runtime_error("Failed to read all particle data from " + path);
        }
    }

    MPI_Bcast(&N, 1, MPI_UINT64_T, 0, comm);
    if (globalN) { *globalN = N; }

    // Per-rank slice [first, last). Compute on every rank.
    auto firstOf = [N, numRanks](int r) -> std::uint64_t {
        return (N * static_cast<std::uint64_t>(r)) /
               static_cast<std::uint64_t>(numRanks);
    };
    const std::uint64_t firstLocal = firstOf(rank);
    const std::uint64_t lastLocal  = firstOf(rank + 1);
    const int           localN     = static_cast<int>(lastLocal - firstLocal);

    x.resize(localN);
    y.resize(localN);
    z.resize(localN);

    // Build (sendcounts, displs) on rank 0; ignored on others.
    std::vector<int> counts(numRanks), displs(numRanks);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(firstOf(r + 1) - firstOf(r));
            displs[r] = static_cast<int>(firstOf(r));
        }
    }

    auto scatter = [&](double* sendBuf, double* recvBuf) {
        MPI_Scatterv(sendBuf, counts.data(), displs.data(), MPI_DOUBLE,
                     recvBuf, localN, MPI_DOUBLE, 0, comm);
    };
    scatter(rank == 0 ? hostX.data() : nullptr, x.data());
    scatter(rank == 0 ? hostY.data() : nullptr, y.data());
    scatter(rank == 0 ? hostZ.data() : nullptr, z.data());
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
template <class P, unsigned Dim>
class DisorderHeatingBHManager : public NBodyManager<P, Dim> {
    static_assert(Dim == 3, "DisorderHeatingBHManager requires Dim == 3");

    using Base = NBodyManager<P, Dim>;
    using typename Base::Container;
    using typename Base::SolverParams;
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;

public:
    // IC sources: load from a sphexa-compatible .bin (reference 4096-particle
    // run, fixed-N) or generate a per-rank uniform-in-sphere sample at any N
    // (used by the scaling sweeps).
    enum class Mode { File, Generator };

    struct Config {
        Mode          mode = Mode::File;
        int           Nt   = 1000;
        unsigned long seed = 42;  // generator-mode only
    };

    // File mode. globalN is the total particle count across all ranks (read
    // from the .bin header on rank 0 and broadcast). hx/hy/hz hold THIS rank's
    // slice only; their .size() == localN per the per-rank split done by
    // loadDihPositions.
    DisorderHeatingBHManager(std::uint64_t       globalN,
                             std::vector<double> hx,
                             std::vector<double> hy,
                             std::vector<double> hz,
                             Config              cfg = {})
        : Base(globalN, cfg.Nt, /*dt=*/Tc(2.15623e-13))
        , cfg_m([&] { cfg.mode = Mode::File; return cfg; }())
        , beamRad_m(beamRadRef_m * scaleFactor(globalN))
        , boxHalf_m(boxHalfRef_m * scaleFactor(globalN))
        , hx_m(std::move(hx))
        , hy_m(std::move(hy))
        , hz_m(std::move(hz)) {}

    // Generator mode. globalN is the requested total particle count; each rank
    // generates the slice [firstGlobal(), lastGlobal()) directly on the device.
    // No file IO, no Scatterv — the IC is keyed on `seed` and reproducible
    // across rank counts.
    DisorderHeatingBHManager(std::uint64_t globalN,
                             Config        cfg)
        : Base(globalN, cfg.Nt, /*dt=*/Tc(2.15623e-13))
        , cfg_m([&] { cfg.mode = Mode::Generator; return cfg; }())
        , beamRad_m(beamRadRef_m * scaleFactor(globalN))
        , boxHalf_m(boxHalfRef_m * scaleFactor(globalN)) {}

protected:
    void initializeContainer() override {
        using cstone::BoundaryType;
        const unsigned bucketSizeFocus = 64u;
        const unsigned bucketSize      = std::max<unsigned>(
            bucketSizeFocus,
            static_cast<unsigned>(this->N() / (100ul * this->numRanks())));
        auto pc = std::make_unique<Container>(
            this->rank(), this->numRanks(),
            bucketSize, bucketSizeFocus, theta_m,
            std::array<Tc, 6>{-boxHalf_m, boxHalf_m,
                              -boxHalf_m, boxHalf_m,
                              -boxHalf_m, boxHalf_m},
            std::array<BoundaryType, 3>{
                BoundaryType::open, BoundaryType::open, BoundaryType::open});
        this->setContainer(std::move(pc));
    }

    void initializeParticles() override {
        if (cfg_m.mode == Mode::Generator) {
            sampleDihIC<P>(this->pc(),
                           static_cast<unsigned>(this->localN()),
                           this->firstGlobal(),
                           beamRad_m,
                           smoothH_m,
                           cfg_m.seed);
            this->pc().setUniformH(smoothH_m);
            return;
        }

        // File mode: hx/hy/hz already hold this rank's slice from loadDihPositions.
        const std::size_t localN = hx_m.size();
        dih_detail::copyHostPositions<Tc>(this->pc().getRxRaw(), hx_m);
        dih_detail::copyHostPositions<Tc>(this->pc().getRyRaw(), hy_m);
        dih_detail::copyHostPositions<Tc>(this->pc().getRzRaw(), hz_m);

        thrust::fill_n(thrust::device_ptr<Tm>(this->pc().getChargeRaw()), localN, Tm(1));
        thrust::fill_n(thrust::device_ptr<Th>(this->pc().getHRaw()),      localN, smoothH_m);
        thrust::fill_n(thrust::device_ptr<Tc>(this->pc().getPxRaw()),     localN, Tc(0));
        thrust::fill_n(thrust::device_ptr<Tc>(this->pc().getPyRaw()),     localN, Tc(0));
        thrust::fill_n(thrust::device_ptr<Tc>(this->pc().getPzRaw()),     localN, Tc(0));
        this->pc().setUniformH(smoothH_m);

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
        // reduceMeanAbsAccel now MPI_Allreduces so every rank gets the same
        // global mean — the focusing strength is identical on all ranks.
        const auto avgAcc = reduceMeanAbsAccel<P>(this->pc());
        const Tc avgForce = std::sqrt(avgAcc.x * avgAcc.x +
                                      avgAcc.y * avgAcc.y +
                                      avgAcc.z * avgAcc.z);
        focusStrength_m = focusMul_m * avgForce;
        if (this->rank() == 0) {
            std::printf("applyConstantFocusing> Focusing Force %g\n",
                        static_cast<double>(focusStrength_m));
            std::printf("Pre Run finished\n");
        }

        const std::string fnameCsv = "data/DIH_IPPL_BH.csv";
        csvPath_m = fnameCsv;
        if (this->rank() != 0) { return; }
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
        dih_detail::applyFocusing<P>(this->pc(), focusStrength_m, beamRad_m);
    }

    void dumpImpl() override {
        // it_m == 0: initial dump from pre_run(); it_m == k+1: post-step
        // dump after iteration k. Every rank must enter printDihStats — the
        // reductions inside MPI_Allreduce — but only rank 0 writes/prints.
        const int step = this->it() - 1;
        const bool isInitial = (step == -1);
        printDihStats(this->pc(), csvPath_m, step, isInitial, this->N(),
                      this->rank() == 0);

        if (!isInitial && this->rank() == 0) {
            std::printf("LeapFrogStep> Step %d Finished.\n", step);
            std::fflush(stdout);
        }
    }

private:
    // Reproduces DIH's printStats (sphexa reference) so the existing
    // plot_dih_comparison.py parses our log unchanged. Also appends a CSV row
    // with the same metrics.
    //
    // MPI: reduceBeamStats / reduceMeanVelocity / reduceTemperature each do an
    // internal MPI_Allreduce, so every rank holds the global value after the
    // call. The 1/N normalization uses the *global* N (passed in). Only rank 0
    // writes to stdout and to the CSV.
    static void printDihStats(SphexaParticleContainer<P, 3>& pc,
                              const std::string& fnameCsv,
                              int step, bool isInitial,
                              unsigned long globalN, bool isRoot) {
        auto bs    = reduceBeamStats<P>(pc);
        const Tc invN = Tc(1) / static_cast<Tc>(globalN);
        for (int i = 0; i < 12; ++i) { bs.vals[i] *= invN; }

        auto safeSqrt = [](Tc v) { return std::sqrt(std::max(Tc(0), v)); };
        const Tc sigma_x = safeSqrt(bs.vals[0] - bs.vals[9]  * bs.vals[9]);
        const Tc sigma_y = safeSqrt(bs.vals[3] - bs.vals[10] * bs.vals[10]);
        const Tc sigma_z = safeSqrt(bs.vals[6] - bs.vals[11] * bs.vals[11]);
        const Tc emit_x  = safeSqrt(bs.vals[0] * bs.vals[1] - bs.vals[2] * bs.vals[2]);
        const Tc emit_y  = safeSqrt(bs.vals[3] * bs.vals[4] - bs.vals[5] * bs.vals[5]);
        const Tc emit_z  = safeSqrt(bs.vals[6] * bs.vals[7] - bs.vals[8] * bs.vals[8]);

        auto avgV = reduceMeanVelocity<P>(pc);
        auto T3   = reduceTemperature<P>(pc, avgV);
        const Tc Tnorm = std::sqrt(T3.x * T3.x + T3.y * T3.y + T3.z * T3.z);

        if (!isRoot) { return; }

        std::printf("ComputeBeamStatistics> Beam Statistics: \n");
        std::printf("ComputeBeamStatistics> RMS Beam Size: %g , %g , %g\n",
                    static_cast<double>(sigma_x), static_cast<double>(sigma_y),
                    static_cast<double>(sigma_z));
        std::printf("ComputeBeamStatistics> RMS Emittance: %g , %g , %g\n",
                    static_cast<double>(emit_x), static_cast<double>(emit_y),
                    static_cast<double>(emit_z));
        std::printf("computeTemperature> Average Velocity: ( %g , %g , %g )\n",
                    static_cast<double>(avgV.x), static_cast<double>(avgV.y),
                    static_cast<double>(avgV.z));
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

    // Physical parameters. The (beamRad, boxHalf, ke, ...) tuple is locked to
    // the P3M comparison test case (`examples/collisions/P3MHeating.cpp`,
    // N_ref = 156055). For the scaling sweep we hold *density* constant by
    // scaling both `beamRad` and `boxHalf` with N^(1/3) — the beam-to-box
    // ratio stays fixed, so we always place a small beam inside a strictly
    // larger volume. At N = N_ref this is a no-op, so the P3M-comparison run
    // is unchanged.
    static constexpr Tc    ke_m            = Tc(2.532638e8);
    static constexpr Tc    boxHalfRef_m    = Tc(0.005);
    static constexpr Tc    beamRadRef_m    = Tc(0.001774);
    static constexpr std::uint64_t nRef_m  = 156055;
    static constexpr Th    smoothH_m       = Th(1e-10);
    static constexpr Tc    focusMul_m      = Tc(1.5);
    static constexpr float theta_m         = 0.5f;

    static Tc scaleFactor(std::uint64_t globalN) {
        return static_cast<Tc>(std::cbrt(
            static_cast<double>(globalN) / static_cast<double>(nRef_m)));
    }

    Config cfg_m;
    const Tc beamRad_m;
    const Tc boxHalf_m;

    std::vector<double> hx_m, hy_m, hz_m;

    Tc          focusStrength_m{Tc(0)};
    std::string csvPath_m;
};

}  // namespace ippl::nbody

#endif  // IPPL_BH_DISORDER_HEATING_BH_MANAGER_HPP
