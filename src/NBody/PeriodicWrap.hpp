#ifndef IPPL_NBODY_PERIODIC_WRAP_HPP
#define IPPL_NBODY_PERIODIC_WRAP_HPP

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Apply periodic wrap to positions on dimensions whose boundary type is
// cstone::BoundaryType::periodic. Wraps r -> lo + ((r-lo) mod L) per axis;
// the result lies in [lo, hi) (a particle at exactly hi maps to lo).
// Open-BC dimensions are untouched. No-op (no kernel launch) if no dimension
// is periodic. Iterates [startIndex(), endIndex()).
//
// Precondition: pc.update() or pc.updateGrav() has been called at least once.
template <class P>
void wrapToBox(SphexaParticleContainer<P, 3>& pc);

} // namespace ippl::nbody

#endif // IPPL_NBODY_PERIODIC_WRAP_HPP
