#include "LandauDampingBHManager.hpp"

#include <mpi.h>

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace ippl::nbody {

namespace {

template <class Tc, class Ta>
struct SquarePromoteOp {
    __device__ Tc operator()(Ta x) const {
        Tc xc = static_cast<Tc>(x);
        return xc * xc;
    }
};

}  // namespace

// Σ Ex² over locally-owned particles (GPU: thrust), then global MPI_SUM. The IC
// sampler lives in the header now (host-side, sphexa-style); this TU only holds
// the device reduction.
template <class P>
typename P::Tc reduceExSumSq(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    Tc local = Tc(0);
    if (end > start) {
        thrust::device_ptr<const Ta> Ex(getRaw<"Ex">(pc));
        local = thrust::transform_reduce(Ex + start, Ex + end,
                                         SquarePromoteOp<Tc, Ta>(),
                                         Tc(0), thrust::plus<Tc>());
    }
    Tc out = Tc(0);
    ::mpiAllreduce(&local, &out, 1, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

template double reduceExSumSq<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template double reduceExSumSq<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template float  reduceExSumSq<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

}  // namespace ippl::nbody
