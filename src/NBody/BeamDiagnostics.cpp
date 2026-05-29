#include "NBody/BeamDiagnostics.hpp"

#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace ippl::nbody {

namespace {

template <class T>
Triple<T> finalizeMeanTriple(Triple<T> localSum, unsigned long localN) {
    T inout[4] = {localSum.x, localSum.y, localSum.z, static_cast<T>(localN)};
    MPI_Allreduce(MPI_IN_PLACE, inout, 4, ::MpiType<T>{}, MPI_SUM, MPI_COMM_WORLD);
    const T invN = (inout[3] > T(0)) ? T(1) / inout[3] : T(0);
    return Triple<T>{inout[0] * invN, inout[1] * invN, inout[2] * invN};
}

}  // namespace

template <class P>
Triple<typename P::Tc> reduceMeanVelocity(SphexaParticleContainer<P, 3>& pc,
                                          const FieldVector<typename P::Tc>& Px,
                                          const FieldVector<typename P::Tc>& Py,
                                          const FieldVector<typename P::Tc>& Pz) {
    using Tc = typename P::Tc;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        const Tc* px = Px.data(); const Tc* py = Py.data(); const Tc* pz = Pz.data();
        Tc sx = Tc(0), sy = Tc(0), sz = Tc(0);
#pragma omp parallel for reduction(+ : sx, sy, sz) schedule(static)
        for (long i = start; i < end; ++i) {
            sx += px[i];
            sy += py[i];
            sz += pz[i];
        }
        localSum = {sx, sy, sz};
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0));
}

template <class P>
Triple<typename P::Tc> reduceTemperature(SphexaParticleContainer<P, 3>& pc,
                                         const FieldVector<typename P::Tc>& Px,
                                         const FieldVector<typename P::Tc>& Py,
                                         const FieldVector<typename P::Tc>& Pz,
                                         Triple<typename P::Tc> avgV) {
    using Tc = typename P::Tc;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        const Tc* px = Px.data(); const Tc* py = Py.data(); const Tc* pz = Pz.data();
        const Tc mvx = avgV.x, mvy = avgV.y, mvz = avgV.z;
        Tc sx = Tc(0), sy = Tc(0), sz = Tc(0);
#pragma omp parallel for reduction(+ : sx, sy, sz) schedule(static)
        for (long i = start; i < end; ++i) {
            const Tc dx = px[i] - mvx;
            const Tc dy = py[i] - mvy;
            const Tc dz = pz[i] - mvz;
            sx += dx * dx;
            sy += dy * dy;
            sz += dz * dz;
        }
        localSum = {sx, sy, sz};
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0));
}

#define INSTANTIATE_DIAG(POLICY, T)                                                  \
    template Triple<T> reduceMeanVelocity<POLICY>(                                   \
        SphexaParticleContainer<POLICY, 3>&,                                         \
        const FieldVector<POLICY::Tc>&,                                              \
        const FieldVector<POLICY::Tc>&,                                              \
        const FieldVector<POLICY::Tc>&);                                             \
    template Triple<T> reduceTemperature<POLICY>(                                    \
        SphexaParticleContainer<POLICY, 3>&,                                         \
        const FieldVector<POLICY::Tc>&,                                              \
        const FieldVector<POLICY::Tc>&,                                              \
        const FieldVector<POLICY::Tc>&,                                              \
        Triple<T>);

INSTANTIATE_DIAG(DoublePrecision, double)
INSTANTIATE_DIAG(MixedPrecision,  double)
INSTANTIATE_DIAG(FloatPrecision,  float)

#undef INSTANTIATE_DIAG

}  // namespace ippl::nbody
