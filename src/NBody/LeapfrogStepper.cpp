#include "NBody/LeapfrogStepper.hpp"

namespace ippl::nbody {

// CPU counterpart of LeapfrogStepper.cu. Same KDK primitives over the
// locally-owned range [startIndex(), endIndex()), expressed as OpenMP loops.

template <class P>
void leapfrogKickHalf(SphexaParticleContainer<P, 3>& pc,
                      FieldVector<typename P::Tc>& Px,
                      FieldVector<typename P::Tc>& Py,
                      FieldVector<typename P::Tc>& Pz,
                      typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    const Ta* Ex = getRaw<"Ex">(pc);
    const Ta* Ey = getRaw<"Ey">(pc);
    const Ta* Ez = getRaw<"Ez">(pc);
    Tc* px = Px.data(); Tc* py = Py.data(); Tc* pz = Pz.data();
    const Tc halfDt = Tc(0.5) * dt;
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        px[i] -= halfDt * static_cast<Tc>(Ex[i]);
        py[i] -= halfDt * static_cast<Tc>(Ey[i]);
        pz[i] -= halfDt * static_cast<Tc>(Ez[i]);
    }
}

template <class P>
void leapfrogKick(SphexaParticleContainer<P, 3>& pc,
                  FieldVector<typename P::Tc>& Px,
                  FieldVector<typename P::Tc>& Py,
                  FieldVector<typename P::Tc>& Pz,
                  typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    const Ta* Ex = getRaw<"Ex">(pc);
    const Ta* Ey = getRaw<"Ey">(pc);
    const Ta* Ez = getRaw<"Ez">(pc);
    Tc* px = Px.data(); Tc* py = Py.data(); Tc* pz = Pz.data();
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        px[i] -= dt * static_cast<Tc>(Ex[i]);
        py[i] -= dt * static_cast<Tc>(Ey[i]);
        pz[i] -= dt * static_cast<Tc>(Ez[i]);
    }
}

template <class P>
void leapfrogDrift(SphexaParticleContainer<P, 3>& pc,
                   FieldVector<typename P::Tc>& Px,
                   FieldVector<typename P::Tc>& Py,
                   FieldVector<typename P::Tc>& Pz,
                   typename P::Tc dt) {
    using Tc = typename P::Tc;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end <= start) { return; }
    Tc* Rx = getRaw<"Rx">(pc);
    Tc* Ry = getRaw<"Ry">(pc);
    Tc* Rz = getRaw<"Rz">(pc);
    const Tc* px = Px.data(); const Tc* py = Py.data(); const Tc* pz = Pz.data();
#pragma omp parallel for schedule(static)
    for (unsigned i = start; i < end; ++i) {
        Rx[i] += dt * px[i];
        Ry[i] += dt * py[i];
        Rz[i] += dt * pz[i];
    }
}

#define INSTANTIATE_LEAPFROG(POLICY, DT)                                                    \
    template void leapfrogKickHalf<POLICY>(SphexaParticleContainer<POLICY, 3>&,             \
                                            FieldVector<POLICY::Tc>&,                       \
                                            FieldVector<POLICY::Tc>&,                       \
                                            FieldVector<POLICY::Tc>&, DT);                  \
    template void leapfrogKick<POLICY>(SphexaParticleContainer<POLICY, 3>&,                 \
                                        FieldVector<POLICY::Tc>&,                           \
                                        FieldVector<POLICY::Tc>&,                           \
                                        FieldVector<POLICY::Tc>&, DT);                      \
    template void leapfrogDrift<POLICY>(SphexaParticleContainer<POLICY, 3>&,                \
                                         FieldVector<POLICY::Tc>&,                          \
                                         FieldVector<POLICY::Tc>&,                          \
                                         FieldVector<POLICY::Tc>&, DT);

INSTANTIATE_LEAPFROG(DoublePrecision, double)
INSTANTIATE_LEAPFROG(MixedPrecision,  double)
INSTANTIATE_LEAPFROG(FloatPrecision,  float)

#undef INSTANTIATE_LEAPFROG

} // namespace ippl::nbody
