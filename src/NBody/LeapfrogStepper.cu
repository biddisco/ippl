#include "NBody/LeapfrogStepper.hpp"

#include <cuda_runtime.h>

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

inline unsigned gridFor(unsigned n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

// Tc: velocity / position scalar; Ta: E-field scalar. May differ under
// MixedPrecision (Tc=double, Ta=float). The multiplications upcast Ta -> Tc
// implicitly, so velocity accumulators run at full Tc precision.
template <class Tc, class Ta>
__global__ void kickHalfKernel(unsigned start, unsigned n,
                               const Ta* __restrict__ Ex,
                               const Ta* __restrict__ Ey,
                               const Ta* __restrict__ Ez,
                               Tc* __restrict__ Px,
                               Tc* __restrict__ Py,
                               Tc* __restrict__ Pz,
                               Tc halfDt) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Px[i] -= halfDt * static_cast<Tc>(Ex[i]);
    Py[i] -= halfDt * static_cast<Tc>(Ey[i]);
    Pz[i] -= halfDt * static_cast<Tc>(Ez[i]);
}

template <class Tc>
__global__ void driftKernel(unsigned start, unsigned n,
                            Tc* __restrict__ Rx,
                            Tc* __restrict__ Ry,
                            Tc* __restrict__ Rz,
                            const Tc* __restrict__ Px,
                            const Tc* __restrict__ Py,
                            const Tc* __restrict__ Pz,
                            Tc dt) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Rx[i] += dt * Px[i];
    Ry[i] += dt * Py[i];
    Rz[i] += dt * Pz[i];
}

} // namespace

template <class P>
void leapfrogKickHalf(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    kickHalfKernel<Tc, Ta><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        Tc(0.5) * dt);
}

template <class P>
void leapfrogKick(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    using Ta = typename P::Ta;
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    kickHalfKernel<Tc, Ta><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        dt);
}

template <class P>
void leapfrogDrift(SphexaParticleContainer<P, 3>& pc, typename P::Tc dt) {
    using Tc = typename P::Tc;
    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }
    driftKernel<Tc><<<gridFor(n), kBlockSize>>>(
        start, n,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        dt);
}

template void leapfrogKickHalf<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&, double);
template void leapfrogKickHalf<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&, double);
template void leapfrogKickHalf<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&, float);
template void leapfrogKick<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&, double);
template void leapfrogKick<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&, double);
template void leapfrogKick<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&, float);
template void leapfrogDrift<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&, double);
template void leapfrogDrift<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&, double);
template void leapfrogDrift<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&, float);

} // namespace ippl::nbody
