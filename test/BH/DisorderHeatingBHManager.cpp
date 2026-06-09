#include "DisorderHeatingBHManager.hpp"

#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace ippl::nbody {

// CPU counterpart of DisorderHeatingBHManager.cu: OpenMP reductions + focusing.
// (The IC sampler is host-side in the header, shared by both backends.)
template <class P>
BeamStats12<typename P::Tc> reduceBeamStats(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());

    Tc local[12] = {Tc(0)};
    if (end > start) {
        const Tc* Rx = getRaw<"Rx">(pc); const Tc* Ry = getRaw<"Ry">(pc); const Tc* Rz = getRaw<"Rz">(pc);
        const Tc* px = getRaw<"Px">(pc); const Tc* py = getRaw<"Py">(pc); const Tc* pz = getRaw<"Pz">(pc);
        Tc s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0, s8=0, s9=0, s10=0, s11=0;
#pragma omp parallel for schedule(static) \
    reduction(+ : s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11)
        for (long i = start; i < end; ++i) {
            s0 += Rx[i] * Rx[i];  s1 += px[i] * px[i];  s2 += Rx[i] * px[i];
            s3 += Ry[i] * Ry[i];  s4 += py[i] * py[i];  s5 += Ry[i] * py[i];
            s6 += Rz[i] * Rz[i];  s7 += pz[i] * pz[i];  s8 += Rz[i] * pz[i];
            s9 += Rx[i];          s10 += Ry[i];         s11 += Rz[i];
        }
        local[0]=s0; local[1]=s1; local[2]=s2; local[3]=s3; local[4]=s4; local[5]=s5;
        local[6]=s6; local[7]=s7; local[8]=s8; local[9]=s9; local[10]=s10; local[11]=s11;
    }
    BeamStats12<Tc> out{};
    ::mpiAllreduce(local, out.vals, 12, MPI_SUM, pc.comm());
    return out;
}

template <class P>
Triple<typename P::Tc> reduceMeanAbsAccel(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());

    Tc local[4] = {Tc(0), Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        const Ta* Ex = getRaw<"Ex">(pc); const Ta* Ey = getRaw<"Ey">(pc); const Ta* Ez = getRaw<"Ez">(pc);
        Tc ax = 0, ay = 0, az = 0;
#pragma omp parallel for reduction(+ : ax, ay, az) schedule(static)
        for (long i = start; i < end; ++i) {
            Tc ex = static_cast<Tc>(Ex[i]);
            Tc ey = static_cast<Tc>(Ey[i]);
            Tc ez = static_cast<Tc>(Ez[i]);
            ax += ex < Tc(0) ? -ex : ex;
            ay += ey < Tc(0) ? -ey : ey;
            az += ez < Tc(0) ? -ez : ez;
        }
        local[0] = ax; local[1] = ay; local[2] = az;
        local[3] = static_cast<Tc>(end - start);
    }
    Tc global[4] = {Tc(0), Tc(0), Tc(0), Tc(0)};
    ::mpiAllreduce(local, global, 4, MPI_SUM, pc.comm());
    const Tc invN = (global[3] > Tc(0)) ? Tc(1) / global[3] : Tc(0);
    return Triple<Tc>{global[0] * invN, global[1] * invN, global[2] * invN};
}

#define INSTANTIATE_DIH_REDUCE(POLICY, T)                                  \
    template BeamStats12<T> reduceBeamStats<POLICY>(NBodyParticleContainer<POLICY, 3>&); \
    template Triple<T> reduceMeanAbsAccel<POLICY>(NBodyParticleContainer<POLICY, 3>&);

INSTANTIATE_DIH_REDUCE(DoublePrecision, double)
INSTANTIATE_DIH_REDUCE(MixedPrecision,  double)
INSTANTIATE_DIH_REDUCE(FloatPrecision,  float)

#undef INSTANTIATE_DIH_REDUCE

namespace dih_detail {

template <class P>
void applyFocusing(NBodyParticleContainer<P, 3>& pc,
                   typename P::Tc strength,
                   typename P::Tc beamRad) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());
    if (end <= start) { return; }
    const Tc  scale = strength / beamRad;
    const Tc* Rx = getRaw<"Rx">(pc); const Tc* Ry = getRaw<"Ry">(pc); const Tc* Rz = getRaw<"Rz">(pc);
    Ta* Ex = getRaw<"Ex">(pc); Ta* Ey = getRaw<"Ey">(pc); Ta* Ez = getRaw<"Ez">(pc);
#pragma omp parallel for schedule(static)
    for (long i = start; i < end; ++i) {
        Ex[i] += static_cast<Ta>(scale * Rx[i]);
        Ey[i] += static_cast<Ta>(scale * Ry[i]);
        Ez[i] += static_cast<Ta>(scale * Rz[i]);
    }
}

template void applyFocusing<DoublePrecision>(NBodyParticleContainer<DoublePrecision, 3>&, double, double);
template void applyFocusing<MixedPrecision> (NBodyParticleContainer<MixedPrecision,  3>&, double, double);
template void applyFocusing<FloatPrecision> (NBodyParticleContainer<FloatPrecision,  3>&, float,  float);

}  // namespace dih_detail

}  // namespace ippl::nbody
