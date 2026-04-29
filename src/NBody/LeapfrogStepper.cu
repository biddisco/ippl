#include "NBody/LeapfrogStepper.hpp"

#include <cuda_runtime.h>

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

inline unsigned gridFor(unsigned n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

template <class T>
__global__ void kickHalfKernel(unsigned start, unsigned n,
                               const T* __restrict__ Ex,
                               const T* __restrict__ Ey,
                               const T* __restrict__ Ez,
                               T* __restrict__ Px,
                               T* __restrict__ Py,
                               T* __restrict__ Pz,
                               T halfDt) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Px[i] -= halfDt * Ex[i];
    Py[i] -= halfDt * Ey[i];
    Pz[i] -= halfDt * Ez[i];
}

template <class T>
__global__ void driftKernel(unsigned start, unsigned n,
                            T* __restrict__ Rx,
                            T* __restrict__ Ry,
                            T* __restrict__ Rz,
                            const T* __restrict__ Px,
                            const T* __restrict__ Py,
                            const T* __restrict__ Pz,
                            T dt) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Rx[i] += dt * Px[i];
    Ry[i] += dt * Py[i];
    Rz[i] += dt * Pz[i];
}

} // namespace

template <class T>
void leapfrogKickHalf(SphexaParticleContainer<T, 3>& pc, T dt) {
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    kickHalfKernel<T><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        T(0.5) * dt);
}

template <class T>
void leapfrogKick(SphexaParticleContainer<T, 3>& pc, T dt) {
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    kickHalfKernel<T><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        dt);
}

template <class T>
void leapfrogDrift(SphexaParticleContainer<T, 3>& pc, T dt) {
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    driftKernel<T><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        dt);
}

template void leapfrogKickHalf<float>(SphexaParticleContainer<float, 3>&, float);
template void leapfrogKickHalf<double>(SphexaParticleContainer<double, 3>&, double);
template void leapfrogKick<float>(SphexaParticleContainer<float, 3>&, float);
template void leapfrogKick<double>(SphexaParticleContainer<double, 3>&, double);
template void leapfrogDrift<float>(SphexaParticleContainer<float, 3>&, float);
template void leapfrogDrift<double>(SphexaParticleContainer<double, 3>&, double);

} // namespace ippl::nbody
