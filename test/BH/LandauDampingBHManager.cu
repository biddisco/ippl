#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/BHPrecision.hpp"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>

#include "cstone/sfc/box.hpp"

namespace ippl::nbody {

namespace {

template <class T> struct MpiTypeOf;
template <> struct MpiTypeOf<float>  { static MPI_Datatype value() { return MPI_FLOAT;  } };
template <> struct MpiTypeOf<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };

template <class Tc, class Ta>
struct SquarePromoteOp {
    __device__ Tc operator()(Ta x) const {
        Tc xc = static_cast<Tc>(x);
        return xc * xc;
    }
};

constexpr unsigned kBlockSize = 256;

// Newton iteration for the inverse of F(x) = x + (alpha/k)·sin(k·x) over [0, L)
// with k·L = 2π (single-wavelength box). With |alpha| < 1, F is strictly
// monotone and Newton converges quadratically; 6 iterations are well below
// double-precision saturation for alpha up to 0.1.
template <class Tc>
__device__ inline Tc inverseLandauCdf(Tc u, Tc alpha, Tc k) {
    Tc x = u;
#pragma unroll
    for (int it = 0; it < 6; ++it) {
        // `::` prefix bypasses name lookup in namespace ippl::nbody, which
        // would otherwise resolve to ippl::sin / ippl::cos (IPPL math wrappers
        // pulled in transitively via NBodyManager.hpp -> Ippl.h) and produce
        // "no instance matches" against a plain Tc argument. We want CUDA's
        // device-side ::sin / ::cos intrinsic here.
        Tc sinkx = ::sin(k * x);
        Tc coskx = ::cos(k * x);
        Tc f     = x + (alpha / k) * sinkx - u;
        Tc fp    = Tc(1) + alpha * coskx;
        x        = x - f / fp;
    }
    return x;
}

template <class Tc, class Th, class Tm>
__global__ void sampleKernel(unsigned start, unsigned n,
                             unsigned long firstGlobal,
                             Tc xmin, Tc ymin, Tc zmin,
                             Tc Lx, Tc Ly, Tc Lz,
                             Tc alpha, Tc kx, Tc ky, Tc kz,
                             Tc sigmaV,
                             Tm qPerParticle,
                             Th smoothingH,
                             Tc* __restrict__ Rx, Tc* __restrict__ Ry, Tc* __restrict__ Rz,
                             Tc* __restrict__ Px, Tc* __restrict__ Py, Tc* __restrict__ Pz,
                             Tm* __restrict__ charge,
                             Th* __restrict__ h,
                             unsigned long seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;

    // Subsequence keyed on the global particle index — reproducible across
    // rank counts. Philox: subsequence space is 2^64, so this is safe for any N.
    curandStatePhilox4_32_10_t state;
    curand_init(seed,
                /*subsequence=*/static_cast<unsigned long long>(firstGlobal) +
                                static_cast<unsigned long long>(tid),
                /*offset=*/0ULL, &state);

    // Sample uniform/normal in double, cast to Tc on write. Keeps the IC
    // bit-reproducible across Double/Mixed/Float by using the same RNG state
    // sequence; only the final precision differs.
    Tc ux = static_cast<Tc>(curand_uniform_double(&state)) * Lx;
    Tc uy = static_cast<Tc>(curand_uniform_double(&state)) * Ly;
    Tc uz = static_cast<Tc>(curand_uniform_double(&state)) * Lz;
    Rx[i] = xmin + inverseLandauCdf(ux, alpha, kx);
    Ry[i] = ymin + inverseLandauCdf(uy, alpha, ky);
    Rz[i] = zmin + inverseLandauCdf(uz, alpha, kz);

    Px[i] = sigmaV * static_cast<Tc>(curand_normal_double(&state));
    Py[i] = sigmaV * static_cast<Tc>(curand_normal_double(&state));
    Pz[i] = sigmaV * static_cast<Tc>(curand_normal_double(&state));

    charge[i] = qPerParticle;
    h[i]      = smoothingH;
}

}

template <class P>
typename P::Tc reduceExSumSq(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    Tc local = Tc(0);
    if (end > start) {
        thrust::device_ptr<const Ta> Ex(pc.getExRaw());
        local = thrust::transform_reduce(Ex + start, Ex + end,
                                         SquarePromoteOp<Tc, Ta>(),
                                         Tc(0), thrust::plus<Tc>());
    }
    Tc out = Tc(0);
    MPI_Allreduce(&local, &out, 1, MpiTypeOf<Tc>::value(), MPI_SUM, MPI_COMM_WORLD);
    return out;
}

template double reduceExSumSq<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template double reduceExSumSq<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template float  reduceExSumSq<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

template <class P>
void sampleLandauIC(SphexaParticleContainer<P, 3>& pc,
                    unsigned localN,
                    unsigned long firstGlobal,
                    typename P::Tc alpha,
                    typename P::Tc kx,
                    typename P::Tc ky,
                    typename P::Tc kz,
                    typename P::Tc sigmaV,
                    typename P::Tm qPerParticle,
                    typename P::Th smoothingH,
                    unsigned long seed) {
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;

    const unsigned n     = localN;
    const unsigned start = 0;
    if (n == 0) { return; }

    const auto box = pc.box();
    const Tc xmin = box.xmin(), ymin = box.ymin(), zmin = box.zmin();
    const Tc Lx   = box.lx(),   Ly   = box.ly(),   Lz   = box.lz();

    const unsigned grid = (n + kBlockSize - 1) / kBlockSize;
    sampleKernel<Tc, Th, Tm><<<grid, kBlockSize>>>(
        start, n, firstGlobal,
        xmin, ymin, zmin, Lx, Ly, Lz,
        alpha, kx, ky, kz,
        sigmaV, qPerParticle, smoothingH,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        pc.getChargeRaw(),
        pc.getHRaw(),
        seed);
    cudaDeviceSynchronize();
}

template void sampleLandauIC<DoublePrecision>(
    SphexaParticleContainer<DoublePrecision, 3>&, unsigned, unsigned long,
    double, double, double, double, double, double, double, unsigned long);
template void sampleLandauIC<MixedPrecision>(
    SphexaParticleContainer<MixedPrecision, 3>&, unsigned, unsigned long,
    double, double, double, double, double, float, float, unsigned long);
template void sampleLandauIC<FloatPrecision>(
    SphexaParticleContainer<FloatPrecision, 3>&, unsigned, unsigned long,
    float, float, float, float, float, float, float, unsigned long);
    
} // namespace ippl::nbody