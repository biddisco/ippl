#ifndef IPPL_NBODY_BEAM_DIAGNOSTICS_HPP
#define IPPL_NBODY_BEAM_DIAGNOSTICS_HPP

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Per-axis triple of the same type. Used as the result type of mean-velocity
// and per-axis variance reductions, and exposed publicly so simulation-specific
// diagnostics can build on it (e.g. DIH's reduceMeanAbsAccel).
template <class T>
struct Triple {
    T x;
    T y;
    T z;
};

// Mean velocity over [startIndex(), endIndex()). Velocity is at the container's
// coordinate type P::Tc.
template <class P>
Triple<typename P::Tc> reduceMeanVelocity(SphexaParticleContainer<P, 3>& pc);

// Per-axis variance of velocity around (avgVx, avgVy, avgVz). Caller computes
// avgV via reduceMeanVelocity first.
template <class P>
Triple<typename P::Tc> reduceTemperature(SphexaParticleContainer<P, 3>& pc,
                                         Triple<typename P::Tc> avgV);

}  // namespace ippl::nbody

#endif  // IPPL_NBODY_BEAM_DIAGNOSTICS_HPP
