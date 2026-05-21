#include "NBody/PeriodicWrap.hpp"

#include <cuda_runtime.h>

#include "cstone/sfc/box.hpp"

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

template <class T>
__device__ inline T wrapAxis(T r, T lo, T L, bool periodic) {
    if (!periodic) { return r; }
    T s = (r - lo) / L;
    s = s - floor(s);            // fractional part in [0, 1)
    return lo + s * L;
}

template <class T>
__global__ void wrapKernel(unsigned start, unsigned n,
                           T* __restrict__ Rx,
                           T* __restrict__ Ry,
                           T* __restrict__ Rz,
                           T xmin, T ymin, T zmin,
                           T lx,   T ly,   T lz,
                           bool px, bool py, bool pz) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    Rx[i] = wrapAxis(Rx[i], xmin, lx, px);
    Ry[i] = wrapAxis(Ry[i], ymin, ly, py);
    Rz[i] = wrapAxis(Rz[i], zmin, lz, pz);
}

} // namespace

template <class P>
void wrapToBox(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;

    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }

    const auto box = pc.box();
    const bool px = box.boundaryX() == cstone::BoundaryType::periodic;
    const bool py = box.boundaryY() == cstone::BoundaryType::periodic;
    const bool pz = box.boundaryZ() == cstone::BoundaryType::periodic;
    if (!px && !py && !pz) { return; }   // open-BC fast path

    const unsigned grid = (n + kBlockSize - 1) / kBlockSize;
    wrapKernel<Tc><<<grid, kBlockSize>>>(
        start, n,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        box.xmin(), box.ymin(), box.zmin(),
        box.lx(),   box.ly(),   box.lz(),
        px, py, pz);
}

template void wrapToBox<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template void wrapToBox<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template void wrapToBox<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

} // namespace ippl::nbody
