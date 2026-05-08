#ifndef IPPL_BH_DIH_DIAGNOSTICS_HPP
#define IPPL_BH_DIH_DIAGNOSTICS_HPP

#include "NBody/BeamDiagnostics.hpp"  // Triple<T>
#include "NBody/SphexaParticleContainer.hpp"

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

// Compute the 12-component sum over [startIndex(), endIndex()).
template <class T>
BeamStats12<T> reduceBeamStats(SphexaParticleContainer<T, 3>& pc);

// Mean of |Eₓ|, |Eᵧ|, |E_z| over [startIndex(), endIndex()) — used by DIH's
// initial-force calibration to size the constant focusing field.
template <class T>
Triple<T> reduceMeanAbsAccel(SphexaParticleContainer<T, 3>& pc);

}  // namespace ippl::nbody

#endif  // IPPL_BH_DIH_DIAGNOSTICS_HPP
