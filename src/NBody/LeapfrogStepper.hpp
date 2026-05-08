#ifndef IPPL_NBODY_LEAPFROG_STEPPER_HPP
#define IPPL_NBODY_LEAPFROG_STEPPER_HPP

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Half-kick: P_i -= 0.5 * dt * E_i for every locally-owned particle.
//
// Iterates [startIndex(), endIndex()) — multi-rank-correct as written.
// Caller is responsible for ensuring (Ex, Ey, Ez) holds the field for the kick
// they want — typically just-computed E for the pre-drift kick, or freshly-
// recomputed E (after solver.runSolver()) for the post-drift kick.
//
// Precondition: pc.update() or pc.updateGrav() has been called at least once,
// so startIndex()/endIndex() are valid.
template <class T>
void leapfrogKickHalf(SphexaParticleContainer<T, 3>& pc, T dt);

// Full kick: P_i -= dt * E_i. Same iteration range and preconditions as
// leapfrogKickHalf — used by Euler-style D→K integrators (e.g. DIH).
template <class T>
void leapfrogKick(SphexaParticleContainer<T, 3>& pc, T dt);

// Drift: R_i += dt * P_i for every locally-owned particle.
// Iterates [startIndex(), endIndex()).
//
// Precondition: pc.update() or pc.updateGrav() has been called at least once.
template <class T>
void leapfrogDrift(SphexaParticleContainer<T, 3>& pc, T dt);

} // namespace ippl::nbody

#endif // IPPL_NBODY_LEAPFROG_STEPPER_HPP
