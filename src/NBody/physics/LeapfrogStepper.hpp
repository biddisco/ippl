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
 * @brief Leapfrog KDK integrator (kick / drift+wrap) for time integration
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 * */
#pragma once

#include "cstone/sfc/box.hpp"

#include "NBody/core/Accelerator.hpp"
#include "NBody/NBodyParticleContainer.hpp"
#include "NBody/helpers/primitives_gpu.h"

namespace ippl::nbody {

// KDK leapfrog primitives over the locally-owned range [startIndex(), endIndex()).
// Velocity (Px/Py/Pz) and the field (Ex/Ey/Ez) are container-resident. The drift
// folds the periodic wrap (cstone::putInBox) into the position update, the way
// sphexa's positionUpdate does. Each primitive dispatches at compile time via
// kHaveGpu to a GPU kernel (helpers/primitives_gpu.cu) or the inline OpenMP loops
// below.

namespace detail {

// CPU (OpenMP) bodies.
template <class P>
void leapfrogKickHalfCpu(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    const Ta* Ex = getRaw<"Ex">(pc);
    const Ta* Ey = getRaw<"Ey">(pc);
    const Ta* Ez = getRaw<"Ez">(pc);
    Tc* px = getRaw<"Px">(pc); Tc* py = getRaw<"Py">(pc); Tc* pz = getRaw<"Pz">(pc);
    const Tc halfDt = Tc(0.5) * dt;
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        px[i] -= halfDt * static_cast<Tc>(Ex[i]);
        py[i] -= halfDt * static_cast<Tc>(Ey[i]);
        pz[i] -= halfDt * static_cast<Tc>(Ez[i]);
    }
}

template <class P>
void leapfrogKickCpu(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    const Ta* Ex = getRaw<"Ex">(pc);
    const Ta* Ey = getRaw<"Ey">(pc);
    const Ta* Ez = getRaw<"Ez">(pc);
    Tc* px = getRaw<"Px">(pc); Tc* py = getRaw<"Py">(pc); Tc* pz = getRaw<"Pz">(pc);
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        px[i] -= dt * static_cast<Tc>(Ex[i]);
        py[i] -= dt * static_cast<Tc>(Ey[i]);
        pz[i] -= dt * static_cast<Tc>(Ez[i]);
    }
}

template <class P>
void leapfrogDriftCpu(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    const auto box = pc.box();
    Tc* Rx = getRaw<"Rx">(pc);
    Tc* Ry = getRaw<"Ry">(pc);
    Tc* Rz = getRaw<"Rz">(pc);
    const Tc* px = getRaw<"Px">(pc); const Tc* py = getRaw<"Py">(pc); const Tc* pz = getRaw<"Pz">(pc);
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        cstone::Vec3<Tc> X{Rx[i] + dt * px[i], Ry[i] + dt * py[i], Rz[i] + dt * pz[i]};
        X = cstone::putInBox(X, box);
        Rx[i] = X[0];
        Ry[i] = X[1];
        Rz[i] = X[2];
    }
}

}  // namespace detail

// Half-kick: P_i -= 0.5 * dt * E_i  (axpy with s = -0.5*dt).
template <class P>
void leapfrogKickHalf(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    if constexpr (kHaveGpu) {
        const unsigned start = pc.startIndex();
        const unsigned n     = pc.endIndex() - start;
        detail::axpy3Gpu<Tc, Ta>(start, n,
                                 getRaw<"Px">(pc), getRaw<"Py">(pc), getRaw<"Pz">(pc),
                                 getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc),
                                 -Tc(0.5) * dt);
    } else {
        detail::leapfrogKickHalfCpu(pc, dt);
    }
}

// Full kick: P_i -= dt * E_i  (axpy with s = -dt).
template <class P>
void leapfrogKick(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    if constexpr (kHaveGpu) {
        const unsigned start = pc.startIndex();
        const unsigned n     = pc.endIndex() - start;
        detail::axpy3Gpu<Tc, Ta>(start, n,
                                 getRaw<"Px">(pc), getRaw<"Py">(pc), getRaw<"Pz">(pc),
                                 getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc),
                                 -dt);
    } else {
        detail::leapfrogKickCpu(pc, dt);
    }
}

// Drift + periodic wrap: R_i = putInBox(R_i + dt * P_i, box). Open-BC axes pass
// through unchanged. No standalone wrap pass is needed afterwards.
template <class P>
void leapfrogDrift(NBodyParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    if constexpr (kHaveGpu) {
        const unsigned start = pc.startIndex();
        const unsigned n     = pc.endIndex() - start;
        detail::driftWrapGpu<Tc>(start, n,
                                 getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
                                 getRaw<"Px">(pc), getRaw<"Py">(pc), getRaw<"Pz">(pc),
                                 dt, pc.box());
    } else {
        detail::leapfrogDriftCpu(pc, dt);
    }
}

} // namespace ippl::nbody