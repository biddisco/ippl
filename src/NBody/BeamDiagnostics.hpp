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

// Mean velocity over [startIndex(), endIndex()).
template <class T>
Triple<T> reduceMeanVelocity(SphexaParticleContainer<T, 3>& pc);

// Per-axis variance of velocity around (avgVx, avgVy, avgVz). Caller computes
// avgV via reduceMeanVelocity first.
template <class T>
Triple<T> reduceTemperature(SphexaParticleContainer<T, 3>& pc, Triple<T> avgV);

}  // namespace ippl::nbody

#endif  // IPPL_NBODY_BEAM_DIAGNOSTICS_HPP
