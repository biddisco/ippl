#include "NBody/PeriodicWrap.hpp"

#include "cstone/sfc/box.hpp"

namespace ippl::nbody {

namespace {

constexpr unsigned kBlockSize = 256;

// Wrap each owned particle back into the box via cstone::putInBox — the same
// single-box-length wrap sphexa applies after the drift (positions.hpp). A
// particle never moves more than one box-length per step, so a single wrap
// suffices; open-BC axes are left untouched by putInBox.
template <class T>
__global__ void wrapKernel(unsigned start, unsigned n,
                           T* __restrict__ Rx,
                           T* __restrict__ Ry,
                           T* __restrict__ Rz,
                           cstone::Box<T> box) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    cstone::Vec3<T> X{Rx[i], Ry[i], Rz[i]};
    X = cstone::putInBox(X, box);
    Rx[i] = X[0];
    Ry[i] = X[1];
    Rz[i] = X[2];
}

} // namespace

template <class P>
void wrapToBox(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;

    const unsigned start = pc.startIndex();
    const unsigned n     = pc.endIndex() - start;
    if (n == 0) { return; }

    const auto box = pc.box();
    const bool anyPeriodic =
        box.boundaryX() == cstone::BoundaryType::periodic ||
        box.boundaryY() == cstone::BoundaryType::periodic ||
        box.boundaryZ() == cstone::BoundaryType::periodic;
    if (!anyPeriodic) { return; }   // open-BC fast path

    const unsigned grid = (n + kBlockSize - 1) / kBlockSize;
    wrapKernel<Tc><<<grid, kBlockSize>>>(
        start, n,
        getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
        box);
}

template void wrapToBox<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template void wrapToBox<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template void wrapToBox<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);

} // namespace ippl::nbody
