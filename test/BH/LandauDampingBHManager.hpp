#ifndef IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP
#define IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "NBody/NBodyManager.hpp"

namespace ippl::nbody {

// `reduceExGridEnergyCIC` is only invoked inside an `if constexpr (kEnableCicEnergy)`
// branch below. With the flag off (production default) the branch is discarded
// during template instantiation, so no definition is required at link time even
// though the declaration must be in scope for the discarded branch to parse.
template <class P>
typename P::Tc reduceExSumSq(SphexaParticleContainer<P, 3>& pc);

template <class P>
typename P::Tc reduceExGridEnergyCIC(SphexaParticleContainer<P, 3>& pc,
                                     typename P::Tc L, int G);

template <class P>
void sampleLandauIC(SphexaParticleContainer<P, 3>& pc,
                    unsigned localN,
                    unsigned long firstGlobal,
                    typename P::Tc alpha,
                    typename P::Tc kx,
                    typename P::Tc ky,
                    typename P::Tc kz,
                    typename P::Tc sigmaV,
                    typename P::Tm qPerParticle,
                    typename P::Th smoothingH,
                    unsigned long seed);


// Landau-damping driver. Step ordering is KDK leapfrog
// (kickHalf - drift - wrap - solve - kickHalf). Output: data/FieldLandauBH_<N>.csv,
// columns:
//   time Ex_field_energy
// where Ex_field_energy is the particle-sampled MC estimator of ∫ Eₓ² dV.
//
// The CIC-scattered grid-energy diagnostic (Ex_field_energy_cic) is gated by
// kEnableCicEnergy below — it converges to the particle estimator at the N's
// we run in production, so we don't pay for the scatter every step. Flip the
// flag (and recompile) to bring the CIC column back if you want to diff the
// two estimators on a new configuration.
template <class P, unsigned Dim>
class LandauDampingBHManager : public NBodyManager<P, Dim> {
    static_assert(Dim == 3, "LandauDampingBHManager requires Dim == 3");

    using Base = NBodyManager<P, Dim>;
    using typename Base::Container;
    using typename Base::Solver;
    using typename Base::SolverParams;
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;

    // Source-level toggle for the CIC grid-energy diagnostic. OFF in production
    // because particle- and grid-sampled energies agree to within run-to-run
    // noise for the N we care about. Flip to `true` and rebuild to re-enable
    // the apples-to-apples comparison vs FFT-PIC.
    static constexpr bool kEnableCicEnergy = false;
    static constexpr int  kCicGrid         = 32;  // grid per axis when enabled

public:

    struct Config {
        unsigned long N         = 10000;
        int           Nt        = 25;
        float         theta     = 0.5f;
        unsigned long seed      = 42;
        int           numShells = 1;        // BH/Ewald image-shell count
        Th            smoothH   = Th(0.05); // BH softening; should be ≲ mean
                                            // interparticle spacing for the
                                            // resonant short-range Coulomb to be
                                            // faithful (default tuned for
                                            // N≈10⁷ in the (4π)³ box).
        bool          leafBasedH = true;    // per-particle h = leaf edge length.
                                            // Decouples softening from a global
                                            // smoothH; smoothH is still consulted
                                            // on the first step before any
                                            // syncGrav has populated the tree.
    };

    explicit LandauDampingBHManager(Config cfg)
        : Base(cfg.N, cfg.Nt, /*dt=*/Tc(0.05))
        , cfg_m(cfg) {}

protected:

    void prepareSolverInputs(bool collect) override {
        using C = SphexaParticleContainer<P, 3>;
        constexpr auto idxCharge = C::template idxOf<"charge">;
        constexpr auto idxID     = C::template idxOf<"ID">;
        constexpr auto idxPx     = C::template idxOf<"Px">;
        constexpr auto idxPy     = C::template idxOf<"Py">;
        constexpr auto idxPz     = C::template idxOf<"Pz">;
        {
            static auto t = IpplTimings::getTimer("bh.syncGrav");
            ippl::nbody::GpuTimer scope(t, collect);
            this->pc().template updateGrav<idxCharge, idxID, idxPx, idxPy, idxPz>();
        }
        {
            static auto t = IpplTimings::getTimer("bh.haloCharge");
            ippl::nbody::GpuTimer scope(t, collect);
            this->pc().template exchangeHalos<idxCharge>();
        }
    }

    void initializeContainer() override {
        using cstone::BoundaryType;
        const unsigned bucketSizeFocus = 64u;
        const unsigned bucketSize      = std::max<unsigned>(
            bucketSizeFocus,
            static_cast<unsigned>(cfg_m.N / (100ul * this->numRanks())));
        auto pc = std::make_unique<Container>(
            this->rank(), this->numRanks(),
            bucketSize, bucketSizeFocus, cfg_m.theta,
            std::array<Tc, 6>{Tc(0), L_m, Tc(0), L_m, Tc(0), L_m},
            std::array<BoundaryType, 3>{
                BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic});
        this->setContainer(std::move(pc));
    }

    void initializeParticles() override {
        sampleLandauIC<P>(this->pc(),
                          static_cast<unsigned>(this->localN()),
                          this->firstGlobal(),
                          alpha_m, k_m, k_m, k_m,
                          sigmaV_m,
                          static_cast<Tm>(qPerN_m),
                          cfg_m.smoothH,
                          cfg_m.seed);
        this->pc().setUniformH(cfg_m.smoothH);
        this->pc().setLeafBasedH(cfg_m.leafBasedH);
    }

    void initializeSolverParams(SolverParams& p) override {
        // Textbook Coulomb prefactor in plasma units (n=e=m=ε₀=1):
        //   φ = (1/(4π)) Σ q/|r-r_p|
        // Negative sign matches the q/m=-1 leapfrog kick `P -= dt·E`.
        p.G                              = -Tc(1) / (Tc(4) * pi_m);
        p.theta                          = cfg_m.theta;
        p.numShells                      = cfg_m.numShells;
        p.ewaldSettings.numReplicaShells = cfg_m.numShells;
    }

    void post_initial_solve() override {
        openCsvHeader();
    }

    void advanceImpl() override {
        this->kickHalf();
        this->drift();
        this->wrap();
        this->solve();
        this->kickHalf();
    }

    void dumpImpl() override {
        // reduceExSumSq MPI_Allreduces internally so every rank holds the
        // global sumSq; only rank 0 writes the CSV row.
        //
        // fieldEnergy: particle-sampled MC estimator of ∫ Ex² dV,
        //              (V/N) · Σ_p Ex²(x_p).
        const Tc sumSq       = reduceExSumSq<P>(this->pc());
        const Tc fieldEnergy = (volume_m / static_cast<Tc>(this->N())) * sumSq;

        // CIC-scattered grid energy diagnostic. Compile-time gated by
        // kEnableCicEnergy: when off, the scatter kernel, the device G³
        // buffers, and the 2·G³ Allreduce are all compiled out. The reduction
        // is collective, so it has to run on every rank before the rank-0
        // guard below.
        if constexpr (kEnableCicEnergy) {
            const Tc fieldEnergyCIC = reduceExGridEnergyCIC<P>(this->pc(), L_m, kCicGrid);
            if (this->rank() != 0) { return; }
            std::ofstream row(csvPath_m, std::ios::app);
            row.precision(16);
            row.setf(std::ios::scientific, std::ios::floatfield);
            row << this->time() << " " << fieldEnergy
                << " "          << fieldEnergyCIC << "\n";
        } else {
            if (this->rank() != 0) { return; }
            std::ofstream row(csvPath_m, std::ios::app);
            row.precision(16);
            row.setf(std::ios::scientific, std::ios::floatfield);
            row << this->time() << " " << fieldEnergy << "\n";
        }
    }

private:
    void openCsvHeader() {
        std::ostringstream oss;
        oss << "data/FieldLandauBH_" << this->N() << ".csv";
        csvPath_m = oss.str();
        if (this->rank() != 0) { return; }
        std::ofstream header(csvPath_m, std::ios::out);
        header << "time Ex_field_energy";
        if constexpr (kEnableCicEnergy) {
            header << " Ex_field_energy_cic";
        }
        header << "\n";
    }

    // Physical parameters (Alpine-aligned).
    const Tc pi_m      = Tc(M_PI);
    const Tc k_m       = Tc(0.5);
    const Tc L_m       = Tc(2) * pi_m / k_m;        // 4π
    const Tc alpha_m   = Tc(0.05);
    const Tc sigmaV_m  = Tc(1.0);
    const Tc volume_m  = L_m * L_m * L_m;
    const Tc qPerN_m   = -volume_m / static_cast<Tc>(this->N());

    Config      cfg_m;
    std::string csvPath_m;
};

}  // namespace ippl::nbody

#endif  // IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP
