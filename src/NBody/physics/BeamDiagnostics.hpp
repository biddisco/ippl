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
 * @brief Compute mean velocity and temperatures of particles in a beam
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <cstddef>

#include "NBody/core/Accelerator.hpp"
#include "NBody/NBodyParticleContainer.hpp"
#include "NBody/helpers/primitives_gpu.h"

namespace ippl::nbody {

namespace detail {

// CPU (OpenMP) reductions.
template <class P>
Triple<typename P::Tc> reduceMeanVelocityCpu(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        const Tc* px = getRaw<"Px">(pc); const Tc* py = getRaw<"Py">(pc); const Tc* pz = getRaw<"Pz">(pc);
        Tc sx = Tc(0), sy = Tc(0), sz = Tc(0);
#pragma omp parallel for reduction(+ : sx, sy, sz) schedule(static)
        for (long i = start; i < end; ++i) {
            sx += px[i];
            sy += py[i];
            sz += pz[i];
        }
        localSum = {sx, sy, sz};
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0),
                                  pc.comm());
}

template <class P>
Triple<typename P::Tc> reduceTemperatureCpu(NBodyParticleContainer<P, 3>& pc,
                                            Triple<typename P::Tc> avgV) {
    using Tc = typename P::Tc;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        const Tc* px = getRaw<"Px">(pc); const Tc* py = getRaw<"Py">(pc); const Tc* pz = getRaw<"Pz">(pc);
        const Tc mvx = avgV.x, mvy = avgV.y, mvz = avgV.z;
        Tc sx = Tc(0), sy = Tc(0), sz = Tc(0);
#pragma omp parallel for reduction(+ : sx, sy, sz) schedule(static)
        for (long i = start; i < end; ++i) {
            const Tc dx = px[i] - mvx;
            const Tc dy = py[i] - mvy;
            const Tc dz = pz[i] - mvz;
            sx += dx * dx;
            sy += dy * dy;
            sz += dz * dz;
        }
        localSum = {sx, sy, sz};
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0),
                                  pc.comm());
}

}  // namespace detail

// Mean velocity over [startIndex(), endIndex()), reading container-resident
// Px/Py/Pz. Dispatches to the thrust (GPU) or OpenMP (CPU) body at compile time.
template <class P>
Triple<typename P::Tc> reduceMeanVelocity(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    if constexpr (kHaveGpu) {
        const unsigned start = pc.startIndex();
        const unsigned end   = pc.endIndex();
        const std::size_t n  = (end > start) ? end - start : 0u;
        Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
        if (n > 0) {
            localSum = detail::sumTripleGpu<Tc>(getRaw<"Px">(pc) + start,
                                                getRaw<"Py">(pc) + start,
                                                getRaw<"Pz">(pc) + start, n);
        }
        return detail::finalizeMeanTriple<Tc>(localSum, static_cast<unsigned long>(n), pc.comm());
    } else {
        return detail::reduceMeanVelocityCpu(pc);
    }
}

// Per-axis variance of velocity around (avgVx, avgVy, avgVz).
template <class P>
Triple<typename P::Tc> reduceTemperature(NBodyParticleContainer<P, 3>& pc,
                                         Triple<typename P::Tc> avgV) {
    using Tc = typename P::Tc;
    if constexpr (kHaveGpu) {
        const unsigned start = pc.startIndex();
        const unsigned end   = pc.endIndex();
        const std::size_t n  = (end > start) ? end - start : 0u;
        Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
        if (n > 0) {
            localSum = detail::varianceTripleGpu<Tc>(getRaw<"Px">(pc) + start,
                                                     getRaw<"Py">(pc) + start,
                                                     getRaw<"Pz">(pc) + start, n,
                                                     avgV.x, avgV.y, avgV.z);
        }
        return detail::finalizeMeanTriple<Tc>(localSum, static_cast<unsigned long>(n), pc.comm());
    } else {
        return detail::reduceTemperatureCpu(pc, avgV);
    }
}

}  // namespace ippl::nbody
