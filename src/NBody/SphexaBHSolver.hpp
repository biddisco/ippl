#ifndef IPPL_NBODY_SPHEXA_BH_SOLVER_HPP
#define IPPL_NBODY_SPHEXA_BH_SOLVER_HPP

#include <memory>

#include "ryoanji/nbody/ewald.h"

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Barnes-Hut field solver: build local BH tree, P2M + M2M upsweep, traverse
// to write Ex/Ey/Ez. When the container's box is fully periodic the canonical
// Ewald correction (real-space erfc tail + reciprocal-space sum) is appended
// after the truncated near-field lattice sum; for open BCs Ewald is skipped.
template <class T, unsigned Dim>
class SphexaBHSolver {
    static_assert(Dim == 3, "SphexaBHSolver requires Dim == 3");

public:
    using Container = SphexaParticleContainer<T, Dim>;

    struct Params {
        T     G         = T(1);   // Coulomb / gravitational prefactor
        float theta     = 0.5f;   // multipole-acceptance angle
        int   numShells = 1;      // BH near-field image-lattice extent (periodic only)
        ryoanji::EwaldSettings ewaldSettings;
    };

    SphexaBHSolver(Container& pc, Params params);
    ~SphexaBHSolver();

    SphexaBHSolver(const SphexaBHSolver&)            = delete;
    SphexaBHSolver& operator=(const SphexaBHSolver&) = delete;
    SphexaBHSolver(SphexaBHSolver&&) noexcept;
    SphexaBHSolver& operator=(SphexaBHSolver&&) noexcept;

    void runSolver();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_SPHEXA_BH_SOLVER_HPP
