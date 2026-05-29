#include "NBody/PeriodicWrap.hpp"

#include "cstone/sfc/box.hpp"

namespace ippl::nbody {

// CPU counterpart of PeriodicWrap.cu: the same cstone::putInBox single-box-length
// wrap over the locally-owned range, as an OpenMP loop.
template <class P>
void wrapToBox(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;

    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    if (end <= start) { return; }

    const auto box = pc.box();
    const bool anyPeriodic =
        box.boundaryX() == cstone::BoundaryType::periodic ||
        box.boundaryY() == cstone::BoundaryType::periodic ||
        box.boundaryZ() == cstone::BoundaryType::periodic;
    if (!anyPeriodic) { return; }   // open-BC fast path

    Tc* Rx = getRaw<"Rx">(pc);
    Tc* Ry = getRaw<"Ry">(pc);
    Tc* Rz = getRaw<"Rz">(pc);

#pragma omp parallel for schedule(static)
    for (long i = start; i < end; ++i) {
        cstone::Vec3<Tc> X{Rx[i], Ry[i], Rz[i]};
        X = cstone::putInBox(X, box);
        Rx[i] = X[0];
        Ry[i] = X[1];
        Rz[i] = X[2];
    }
}

template void wrapToBox<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template void wrapToBox<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template void wrapToBox<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

} // namespace ippl::nbody
