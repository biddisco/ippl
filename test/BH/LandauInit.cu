#include "LandauInit.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cstone/sfc/box.hpp"

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

// Newton iteration for the inverse of F(x) = x + (alpha/k)·sin(k·x) over [0, L)
// with k·L = 2π (single-wavelength box). With |alpha| < 1, F is strictly
// monotone and Newton converges quadratically; 6 iterations are well below
// double-precision saturation for alpha up to 0.1.
template <class T>
__device__ inline T inverseLandauCdf(T u, T alpha, T k) {
    T x = u;
#pragma unroll
    for (int it = 0; it < 6; ++it) {
        T sinkx = sin(k * x);
        T coskx = cos(k * x);
        T f     = x + (alpha / k) * sinkx - u;
        T fp    = T(1) + alpha * coskx;
        x       = x - f / fp;
    }
    return x;
}

template <class T>
__global__ void sampleKernel(unsigned start, unsigned n,
                             T xmin, T ymin, T zmin,
                             T Lx, T Ly, T Lz,
                             T alpha, T kx, T ky, T kz,
                             T sigmaV,
                             T qPerParticle,
                             T smoothingH,
                             T* __restrict__ Rx, T* __restrict__ Ry, T* __restrict__ Rz,
                             T* __restrict__ Px, T* __restrict__ Py, T* __restrict__ Pz,
                             T* __restrict__ charge,
                             T* __restrict__ h,
                             unsigned long seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/static_cast<unsigned long long>(tid),
                /*offset=*/0ULL, &state);

    T ux = static_cast<T>(curand_uniform_double(&state)) * Lx;
    T uy = static_cast<T>(curand_uniform_double(&state)) * Ly;
    T uz = static_cast<T>(curand_uniform_double(&state)) * Lz;
    Rx[i] = xmin + inverseLandauCdf(ux, alpha, kx);
    Ry[i] = ymin + inverseLandauCdf(uy, alpha, ky);
    Rz[i] = zmin + inverseLandauCdf(uz, alpha, kz);

    Px[i] = sigmaV * static_cast<T>(curand_normal_double(&state));
    Py[i] = sigmaV * static_cast<T>(curand_normal_double(&state));
    Pz[i] = sigmaV * static_cast<T>(curand_normal_double(&state));

    charge[i] = qPerParticle;
    h[i]      = smoothingH;
}

}  // namespace

template <class T>
void sampleLandauIC(SphexaParticleContainer<T, 3>& pc,
                    unsigned N,
                    T alpha, T kx, T ky, T kz,
                    T sigmaV,
                    T qPerParticle,
                    T smoothingH,
                    unsigned long seed) {
    const unsigned n     = N;
    const unsigned start = 0;
    if (n == 0) { return; }

    const auto box = pc.box();
    const T xmin = box.xmin(), ymin = box.ymin(), zmin = box.zmin();
    const T Lx   = box.lx(),   Ly   = box.ly(),   Lz   = box.lz();

    const unsigned grid = (n + kBlockSize - 1) / kBlockSize;
    sampleKernel<T><<<grid, kBlockSize>>>(
        start, n,
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

template void sampleLandauIC<float>(SphexaParticleContainer<float, 3>&,
                                    unsigned,
                                    float, float, float, float, float, float, float,
                                    unsigned long);
template void sampleLandauIC<double>(SphexaParticleContainer<double, 3>&,
                                     unsigned,
                                     double, double, double, double, double, double, double,
                                     unsigned long);

}  // namespace ippl::nbody
