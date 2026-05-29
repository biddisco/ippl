/*
 * IPPL Barnes-Hut
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 PSI, Villigen
 *
 * Please refer to the LICENSE file in the root directory
 * SPDX-License-Identifier: GPL-3.0
 */

/*! @file
 * @brief BH solver for coulomb interactions in NBody simulations
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once 

#include <memory>

#include "ryoanji/nbody/ewald.h"

#include "NBody/NBodyParticleContainer.hpp"

namespace ippl::nbody {

// Barnes-Hut field solver: build local BH tree, P2M + M2M upsweep, traverse
// to write Ex/Ey/Ez. When the container's box is fully periodic the canonical
// Ewald correction (real-space erfc tail + reciprocal-space sum) is appended
// after the truncated near-field lattice sum; for open BCs Ewald is skipped.
//
// Templated on a precision policy P (see NBody/BHPrecision.hpp). The five
// ryoanji scalar template parameters (Tc, Th, Tm, Ta, Tf) plus the multipole
// value type (Tmm) all come from P, so a single template parameter selects
// one of the three pre-compiled ryoanji traversal instantiations.
template <class P, unsigned Dim>
class NBodySolver {
    static_assert(Dim == 3, "NBodySolver requires Dim == 3");

public:
    using Container = NBodyParticleContainer<P, Dim>;
    using Tc        = typename P::Tc;

    struct Params {
        Tc    G         = Tc(1);  // Coulomb / gravitational prefactor (position units)
        float theta     = 0.5f;   // multipole-acceptance angle
        int   numShells = 1;      // BH near-field image-lattice extent (periodic only)
        ryoanji::EwaldSettings ewaldSettings;
    };

    NBodySolver(Container& pc, Params params);
    ~NBodySolver();

    NBodySolver(const NBodySolver&)            = delete;
    NBodySolver& operator=(const NBodySolver&) = delete;
    NBodySolver(NBodySolver&&) noexcept;
    NBodySolver& operator=(NBodySolver&&) noexcept;

    // warmup=true skips the per-scope bh.* IpplTimings and the per-step stats
    // printout. Used by NBodyManager::pre_run() so the initial t=0 solve does
    // not pollute the per-step BH timing accumulators (the "firstSolve" outer
    // timer in pre_run already captures init cost separately).
    void runSolver(bool warmup = false);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_SOLVER_HPP
