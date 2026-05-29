#include "LandauDampingBHManager.hpp"

#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace ippl::nbody {

// CPU counterpart of LandauDampingBHManager.cu: Σ Ex² via OpenMP, then global
// MPI_SUM. (The IC sampler is host-side in the header, shared by both backends.)
template <class P>
typename P::Tc reduceExSumSq(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const long start = static_cast<long>(pc.startIndex());
    const long end   = static_cast<long>(pc.endIndex());

    Tc local = Tc(0);
    if (end > start) {
        const Ta* Ex = getRaw<"Ex">(pc);
        Tc s = Tc(0);
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (long i = start; i < end; ++i) {
            Tc e = static_cast<Tc>(Ex[i]);
            s += e * e;
        }
        local = s;
    }
    Tc out = Tc(0);
    ::mpiAllreduce(&local, &out, 1, MPI_SUM, pc.comm());
    return out;
}

template double reduceExSumSq<DoublePrecision>(NBodyParticleContainer<DoublePrecision, 3>&);
template double reduceExSumSq<MixedPrecision> (NBodyParticleContainer<MixedPrecision,  3>&);
template float  reduceExSumSq<FloatPrecision> (NBodyParticleContainer<FloatPrecision,  3>&);

}  // namespace ippl::nbody
