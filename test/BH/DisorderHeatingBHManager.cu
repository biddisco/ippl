#include "DisorderHeatingBHManager.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace ippl::nbody {

namespace {

template <class T> struct MpiTypeOf;
template <> struct MpiTypeOf<float>  { static MPI_Datatype value() { return MPI_FLOAT;  } };
template <> struct MpiTypeOf<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };

constexpr unsigned kBlockSize = 256;

template <class Tc, class Th, class Tm>
__global__ void sampleKernel(unsigned start, unsigned n,
                             unsigned long firstGlobal,
                             Tc beamRad,
                             Th smoothingH,
                             Tc* __restrict__ Rx, Tc* __restrict__ Ry, Tc* __restrict__ Rz,
                             Tc* __restrict__ Px, Tc* __restrict__ Py, Tc* __restrict__ Pz,
                             Tm* __restrict__ charge,
                             Th* __restrict__ h,
                             unsigned long seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;

    // Subsequence keyed on the global particle index — bit-reproducible across
    // rank counts. Philox: 2^64-wide subsequence space, safe for any N.
    curandStatePhilox4_32_10_t state;
    curand_init(seed,
                /*subsequence=*/static_cast<unsigned long long>(firstGlobal) +
                                static_cast<unsigned long long>(tid),
                /*offset=*/0ULL, &state);

    // P3MHeating samples u first then three normals; keep the same draw order
    // so seed reuse picks structurally similar IC even across RNG families.
    Tc u  = static_cast<Tc>(curand_uniform_double(&state));
    Tc nx = static_cast<Tc>(curand_normal_double(&state));
    Tc ny = static_cast<Tc>(curand_normal_double(&state));
    Tc nz = static_cast<Tc>(curand_normal_double(&state));

    Tc normsq = nx * nx + ny * ny + nz * nz;
    // curand_uniform_double returns (0, 1] so u^(1/3) is well-defined;
    // normsq > 0 with probability 1, but guard against the measure-zero zero.
    Tc invNorm = (normsq > Tc(0)) ? (Tc(1) / ::sqrt(normsq)) : Tc(0);
    Tc r       = beamRad * ::pow(u, Tc(1) / Tc(3));

    Rx[i] = r * nx * invNorm;
    Ry[i] = r * ny * invNorm;
    Rz[i] = r * nz * invNorm;

    Px[i] = Tc(0);
    Py[i] = Tc(0);
    Pz[i] = Tc(0);

    charge[i] = Tm(1);
    h[i]      = smoothingH;
}

}  // namespace

template <class P>
void sampleDihIC(SphexaParticleContainer<P, 3>& pc,
                 unsigned localN,
                 unsigned long firstGlobal,
                 typename P::Tc beamRad,
                 typename P::Th smoothingH,
                 unsigned long seed) {
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;

    if (localN == 0) { return; }
    const unsigned grid = (localN + kBlockSize - 1) / kBlockSize;
    sampleKernel<Tc, Th, Tm><<<grid, kBlockSize>>>(
        /*start=*/0, localN, firstGlobal,
        beamRad, smoothingH,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        pc.getChargeRaw(),
        pc.getHRaw(),
        seed);
    cudaDeviceSynchronize();
}

template void sampleDihIC<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&,
                                           unsigned, unsigned long,
                                           double, double, unsigned long);
template void sampleDihIC<MixedPrecision>(SphexaParticleContainer<MixedPrecision, 3>&,
                                          unsigned, unsigned long,
                                          double, float,  unsigned long);
template void sampleDihIC<FloatPrecision>(SphexaParticleContainer<FloatPrecision, 3>&,
                                          unsigned, unsigned long,
                                          float,  float,  unsigned long);

namespace {

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
BeamStats12<typename P::Tc> reduceBeamStats(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Tc local[12] = {Tc(0)};
    if (end > start) {
        BeamStatsFunctor<Tc> functor{
            pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
            pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw()};
        auto first = thrust::counting_iterator<unsigned>(start);
        auto last  = thrust::counting_iterator<unsigned>(end);
        BeamStatsRaw<Tc> sum = thrust::transform_reduce(
            first, last, functor, BeamStatsRaw<Tc>(), BeamStatsPlus<Tc>());
        for (int i = 0; i < 12; ++i) { local[i] = sum.vals[i]; }
    }
    BeamStats12<Tc> out{};
    MPI_Allreduce(local, out.vals, 12, MpiTypeOf<Tc>::value(), MPI_SUM, MPI_COMM_WORLD);
    return out;
}

template <class P>
Triple<typename P::Tc> reduceMeanAbsAccel(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Tc local[4] = {Tc(0), Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        AbsTripleFunctor<Tc, Ta> functor{pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw()};
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
    MPI_Allreduce(local, global, 4, MpiTypeOf<Tc>::value(), MPI_SUM, MPI_COMM_WORLD);
    const Tc invN = (global[3] > Tc(0)) ? Tc(1) / global[3] : Tc(0);
    return Triple<Tc>{global[0] * invN, global[1] * invN, global[2] * invN};
}

template BeamStats12<double> reduceBeamStats<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template BeamStats12<double> reduceBeamStats<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template BeamStats12<float>  reduceBeamStats<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);
template Triple<double> reduceMeanAbsAccel<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template Triple<double> reduceMeanAbsAccel<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template Triple<float>  reduceMeanAbsAccel<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

}  // namespace ippl::nbody