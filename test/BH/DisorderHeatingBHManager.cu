#include "DisorderHeatingBHManager.hpp"

#include <mpi.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

template <class T>
struct BeamStatsRaw {
    T vals[12];
    __host__ __device__ BeamStatsRaw() {
        for (int i = 0; i < 12; ++i) { vals[i] = T(0); }
    }
};

template <class T>
struct BeamStatsPlus {
    __host__ __device__ BeamStatsRaw<T> operator()(const BeamStatsRaw<T>& a,
                                                   const BeamStatsRaw<T>& b) const {
        BeamStatsRaw<T> r;
        for (int i = 0; i < 12; ++i) { r.vals[i] = a.vals[i] + b.vals[i]; }
        return r;
    }
};

template <class Tc>
struct BeamStatsFunctor {
    const Tc* __restrict__ Rx;
    const Tc* __restrict__ Ry;
    const Tc* __restrict__ Rz;
    const Tc* __restrict__ Px;
    const Tc* __restrict__ Py;
    const Tc* __restrict__ Pz;

    __device__ BeamStatsRaw<Tc> operator()(unsigned i) const {
        BeamStatsRaw<Tc> s;
        s.vals[ 0] = Rx[i] * Rx[i];   s.vals[ 1] = Px[i] * Px[i];   s.vals[ 2] = Rx[i] * Px[i];
        s.vals[ 3] = Ry[i] * Ry[i];   s.vals[ 4] = Py[i] * Py[i];   s.vals[ 5] = Ry[i] * Py[i];
        s.vals[ 6] = Rz[i] * Rz[i];   s.vals[ 7] = Pz[i] * Pz[i];   s.vals[ 8] = Rz[i] * Pz[i];
        s.vals[ 9] = Rx[i];           s.vals[10] = Ry[i];           s.vals[11] = Rz[i];
        return s;
    }
};

// |Ex|/|Ey|/|Ez|: Ex read at Ta, promoted to Tc on demand so the |·|
// accumulator stays at storage precision under MixedPrecision.
template <class Tc, class Ta>
struct AbsTripleFunctor {
    const Ta* __restrict__ Ex;
    const Ta* __restrict__ Ey;
    const Ta* __restrict__ Ez;
    __device__ thrust::tuple<Tc, Tc, Tc> operator()(unsigned i) const {
        Tc ex = static_cast<Tc>(Ex[i]);
        Tc ey = static_cast<Tc>(Ey[i]);
        Tc ez = static_cast<Tc>(Ez[i]);
        return thrust::make_tuple(ex < Tc(0) ? -ex : ex,
                                  ey < Tc(0) ? -ey : ey,
                                  ez < Tc(0) ? -ez : ez);
    }
};

template <class T>
struct TuplePlus3 {
    __host__ __device__ thrust::tuple<T, T, T>
    operator()(const thrust::tuple<T, T, T>& a, const thrust::tuple<T, T, T>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b),
                                  thrust::get<2>(a) + thrust::get<2>(b));
    }
};

}  // namespace

template <class P>
BeamStats12<typename P::Tc> reduceBeamStats(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Tc local[12] = {Tc(0)};
    if (end > start) {
        BeamStatsFunctor<Tc> functor{
            getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
            getRaw<"Px">(pc), getRaw<"Py">(pc), getRaw<"Pz">(pc)};
        auto first = thrust::counting_iterator<unsigned>(start);
        auto last  = thrust::counting_iterator<unsigned>(end);
        BeamStatsRaw<Tc> sum = thrust::transform_reduce(
            first, last, functor, BeamStatsRaw<Tc>(), BeamStatsPlus<Tc>());
        for (int i = 0; i < 12; ++i) { local[i] = sum.vals[i]; }
    }
    BeamStats12<Tc> out{};
    ::mpiAllreduce(local, out.vals, 12, MPI_SUM, pc.comm());
    return out;
}

template <class P>
Triple<typename P::Tc> reduceMeanAbsAccel(NBodyParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Tc local[4] = {Tc(0), Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        AbsTripleFunctor<Tc, Ta> functor{getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc)};
        auto first = thrust::counting_iterator<unsigned>(start);
        auto last  = thrust::counting_iterator<unsigned>(end);
        auto sum = thrust::transform_reduce(
            first, last, functor,
            thrust::make_tuple(Tc(0), Tc(0), Tc(0)),
            TuplePlus3<Tc>());
        local[0] = thrust::get<0>(sum);
        local[1] = thrust::get<1>(sum);
        local[2] = thrust::get<2>(sum);
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

namespace {

template <class Tc, class Ta>
__global__ void focusingKernel(unsigned start, unsigned n,
                               Tc scale,
                               const Tc* __restrict__ Rx,
                               const Tc* __restrict__ Ry,
                               const Tc* __restrict__ Rz,
                               Ta* __restrict__ Ex,
                               Ta* __restrict__ Ey,
                               Ta* __restrict__ Ez) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Ex[i] += static_cast<Ta>(scale * Rx[i]);
    Ey[i] += static_cast<Ta>(scale * Ry[i]);
    Ez[i] += static_cast<Ta>(scale * Rz[i]);
}

}  // namespace

template <class P>
void applyFocusing(NBodyParticleContainer<P, 3>& pc,
                   typename P::Tc strength,
                   typename P::Tc beamRad) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    const Tc       scale = strength / beamRad;
    const unsigned grid  = (n + kBlockSize - 1) / kBlockSize;
    focusingKernel<Tc, Ta><<<grid, kBlockSize>>>(
        start, n, scale,
        getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
        getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc));
}

template void applyFocusing<DoublePrecision>(NBodyParticleContainer<DoublePrecision, 3>&, double, double);
template void applyFocusing<MixedPrecision> (NBodyParticleContainer<MixedPrecision,  3>&, double, double);
template void applyFocusing<FloatPrecision> (NBodyParticleContainer<FloatPrecision,  3>&, float,  float);

}  // namespace dih_detail

}  // namespace ippl::nbody
