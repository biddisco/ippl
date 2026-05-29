#ifndef IPPL_NBODY_LEAPFROG_STEPPER_HPP
#define IPPL_NBODY_LEAPFROG_STEPPER_HPP

#include "NBody/Accelerator.hpp"
#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// KDK leapfrog primitives over the locally-owned range [startIndex(), endIndex()).
// The body is defined in LeapfrogStepper.cu (GPU kernels) or LeapfrogStepper.cpp
// (OpenMP loops); the build selects the translation unit per backend.

// Half-kick: P_i -= 0.5 * dt * E_i.
template <class P>
void leapfrogKickHalf(SphexaParticleContainer<P, 3>& pc,
                      FieldVector<typename P::Tc>& Px,
                      FieldVector<typename P::Tc>& Py,
                      FieldVector<typename P::Tc>& Pz,
                      typename P::Tc dt);

// Full kick: P_i -= dt * E_i.
template <class P>
void leapfrogKick(SphexaParticleContainer<P, 3>& pc,
                  FieldVector<typename P::Tc>& Px,
                  FieldVector<typename P::Tc>& Py,
                  FieldVector<typename P::Tc>& Pz,
                  typename P::Tc dt);

// Drift: R_i += dt * P_i.
template <class P>
void leapfrogDrift(SphexaParticleContainer<P, 3>& pc,
                   FieldVector<typename P::Tc>& Px,
                   FieldVector<typename P::Tc>& Py,
                   FieldVector<typename P::Tc>& Pz,
                   typename P::Tc dt);

} // namespace ippl::nbody

#endif // IPPL_NBODY_LEAPFROG_STEPPER_HPP
