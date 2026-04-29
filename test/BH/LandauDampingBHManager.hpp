#ifndef IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP
#define IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "NBody/BeamDiagnostics.hpp"
#include "NBody/NBodyManager.hpp"

#include "LandauDiagnostics.hpp"
#include "LandauInit.hpp"

namespace ippl::nbody {

// Landau-damping driver. Step ordering is KDK leapfrog
// (kickHalf - drift - wrap - solve - kickHalf). Output: data/FieldLandauBH_<N>.csv,
// columns:
//   time Ex_field_energy Ex_max_norm Ex_cos_amp meanVx meanVy meanVz Tx Ty Tz
//   Ex_field_energy_cic
template <class T, unsigned Dim>
class LandauDampingBHManager : public NBodyManager<T, Dim> {
    static_assert(Dim == 3, "LandauDampingBHManager requires Dim == 3");

    using Base = NBodyManager<T, Dim>;
    using typename Base::Container;
    using typename Base::Solver;
    using typename Base::SolverParams;

public:

    struct Config {
        unsigned long N         = 10000;
        int           Nt        = 25;
        float         theta     = 0.5f;
        unsigned long seed      = 42;
        int           numShells = 1;       // BH/Ewald image-shell count
        T             smoothH   = T(0.05); // BH softening; should be ≲ mean
                                           // interparticle spacing for the
                                           // resonant short-range Coulomb to be
                                           // faithful (default tuned for
                                           // N≈10⁷ in the (4π)³ box).
        int           cicGrid   = 32;      // CIC-smoothed energy grid (per axis).
                                           // Match the FFT-PIC nx for an
                                           // apples-to-apples diagnostic.
    };

    explicit LandauDampingBHManager(Config cfg)
        : Base(cfg.N, cfg.Nt, /*dt=*/T(0.05))
        , cfg_m(cfg) {}

protected:

    void initializeContainer() override {
        using cstone::BoundaryType;
        auto pc = std::make_unique<Container>(
            /*rank=*/0, /*nRanks=*/1,
            /*bucketSize=*/64, /*bucketSizeFocus=*/64, cfg_m.theta,
            std::array<T, 6>{T(0), L_m, T(0), L_m, T(0), L_m},
            std::array<BoundaryType, 3>{
                BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic});
        this->setContainer(std::move(pc));
    }

    void initializeParticles() override {
        sampleLandauIC<T>(this->pc(),
                          static_cast<unsigned>(this->N()),
                          alpha_m, k_m, k_m, k_m,
                          sigmaV_m,
                          qPerN_m,
                          cfg_m.smoothH,
                          cfg_m.seed);
    }

    void initializeSolverParams(SolverParams& p) override {
        // Textbook Coulomb prefactor in plasma units (n=e=m=ε₀=1):
        //   φ = (1/(4π)) Σ q/|r-r_p|
        // Negative sign matches the q/m=-1 leapfrog kick `P -= dt·E`.
        p.G                              = -T(1) / (T(4) * pi_m);
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
        const auto stats = reduceExStats(this->pc());
        const auto kmode = reduceExCosineMode(this->pc(), k_m);
        const auto avgV  = reduceMeanVelocity(this->pc());
        const auto Tvec  = reduceTemperature(this->pc(), avgV);

        // Particle-side Monte-Carlo estimator of ∫ Ex² dV:
        //   fieldEnergy = (V / N) · Σ_p Ex²(x_p)
        const T fieldEnergy = (volume_m / static_cast<T>(this->N())) * stats.sumSq;
        const T kAmp        = std::sqrt(kmode.cosAmp * kmode.cosAmp +
                                        kmode.sinAmp * kmode.sinAmp);

        // CIC-smoothed grid energy: drops modes above k_Nyquist = π·G/L.
        // Matches FFT-PIC's grid integration when cicGrid equals the PIC nx.
        const T fieldEnergyCIC = reduceExGridEnergyCIC<T>(this->pc(), L_m,
                                                          cfg_m.cicGrid);

        std::ofstream row(csvPath_m, std::ios::app);
        row.precision(16);
        row.setf(std::ios::scientific, std::ios::floatfield);
        row << this->time() << " " << fieldEnergy << " " << stats.maxAbs << " " << kAmp
            << " " << avgV.x  << " " << avgV.y  << " " << avgV.z
            << " " << Tvec.x  << " " << Tvec.y  << " " << Tvec.z
            << " " << fieldEnergyCIC << "\n";
    }

private:
    void openCsvHeader() {
        std::ostringstream oss;
        oss << "data/FieldLandauBH_" << this->N() << ".csv";
        csvPath_m = oss.str();
        std::ofstream header(csvPath_m, std::ios::out);
        header << "time Ex_field_energy Ex_max_norm Ex_cos_amp "
               << "meanVx meanVy meanVz Tx Ty Tz "
               << "Ex_field_energy_cic\n";
    }

    // Physical parameters (Alpine-aligned).
    const T pi_m      = T(M_PI);
    const T k_m       = T(0.5);
    const T L_m       = T(2) * pi_m / k_m;        // 4π
    const T alpha_m   = T(0.05);
    const T sigmaV_m  = T(1.0);
    const T volume_m  = L_m * L_m * L_m;
    const T qPerN_m   = -volume_m / static_cast<T>(this->N());

    Config      cfg_m;
    std::string csvPath_m;
};

}  // namespace ippl::nbody

#endif  // IPPL_BH_LANDAU_DAMPING_BH_MANAGER_HPP
