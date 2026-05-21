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
//
// Velocity is stored at P::Tc; E-field at P::Ta. For mixed precision the kernel
// upcasts E_i to Tc before the multiply, so velocity accumulators stay full
// precision while the E-field memory traffic is halved.
template <class P>
void leapfrogKickHalf(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt);

// Full kick: P_i -= dt * E_i. Same iteration range and preconditions as
// leapfrogKickHalf — used by Euler-style D→K integrators (e.g. DIH).
template <class P>
void leapfrogKick(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt);

// Drift: R_i += dt * P_i for every locally-owned particle.
// Iterates [startIndex(), endIndex()).
//
// Precondition: pc.update() or pc.updateGrav() has been called at least once.
template <class P>
void leapfrogDrift(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt);

} // namespace ippl::nbody

#endif // IPPL_NBODY_LEAPFROG_STEPPER_HPP
