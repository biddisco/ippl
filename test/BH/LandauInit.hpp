#ifndef IPPL_BH_LANDAU_INIT_HPP
#define IPPL_BH_LANDAU_INIT_HPP

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Sample Landau-damping initial conditions directly into the container's six
// scalar arrays (Rx/Ry/Rz, Px/Py/Pz) plus the source attribute (charge) and
// smoothing length (h).
//
// Positions: per-axis inverse transform of the CDF F(x) = x + (alpha/k)·sin(k·x)
// for the perturbed density f(x) = 1 + alpha·cos(k·x). Newton-Raphson, 6 iters
// converges for alpha <= 0.1. Caller's box is assumed to span exactly one
// wavelength per periodic axis (k·L == 2π); this matches Alpine's setup.
//
// Velocities: independent Maxwellian N(0, sigmaV) per axis via Box-Muller
// (cuRAND).
//
// Charge: uniform value `qPerParticle` for every particle. Smoothing length:
// uniform `smoothingH`.
//
// Caller is responsible for having called pc.create(N) with the same N before
// this. After this function returns, call pc.updateGrav() to SFC-sort and
// populate centers before the first BH solve. N is passed explicitly because
// the container's getLocalNum() reads from cstone::Domain, which returns 0
// before the first sync.
template <class T>
void sampleLandauIC(SphexaParticleContainer<T, 3>& pc,
                    unsigned N,
                    T alpha, T kx, T ky, T kz,
                    T sigmaV,
                    T qPerParticle,
                    T smoothingH,
                    unsigned long seed);

}  // namespace ippl::nbody

#endif  // IPPL_BH_LANDAU_INIT_HPP
