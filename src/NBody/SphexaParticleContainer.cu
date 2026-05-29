#include "NBody/SphexaParticleContainer.hpp"

#include <cmath>
#include <tuple>
#include <vector>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"

namespace ippl::nbody {

namespace {

template <class KeyType, class Th>
__global__ void setHFromLeafKernel(const KeyType* keys,
                                   const KeyType* leaves,
                                   int            nLeafKeys,
                                   Th             cbrtVol,
                                   Th*            h,
                                   unsigned       n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    KeyType                k         = keys[i];
    cstone::TreeNodeIndex  leafIdx   = cstone::findNodeBelow(leaves, nLeafKeys, k);
    KeyType                codeRange = leaves[leafIdx + 1] - leaves[leafIdx];
    unsigned               level     = cstone::treeLevel(codeRange);
    h[i] = cbrtVol / Th(1u << level);
}

}  // namespace

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::SphexaParticleContainer(
    int rank, int nRanks,
    unsigned bucketSize, unsigned bucketSizeFocus,
    float theta,
    std::array<Tc, 6> boxLoHi,
    std::array<cstone::BoundaryType, 3> boundaries)
    : domain_(rank, nRanks, bucketSize, bucketSizeFocus, theta, MPI_COMM_WORLD,
              cstone::Box<Tc>(boxLoHi[0], boxLoHi[1],
                              boxLoHi[2], boxLoHi[3],
                              boxLoHi[4], boxLoHi[5],
                              boundaries[0], boundaries[1], boundaries[2])) {}

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::~SphexaParticleContainer() = default;

template <class P, unsigned Dim>
auto SphexaParticleContainer<P, Dim>::dataTuple() -> DataTupleT {
    return std::tie(
        Rx, Ry, Rz,
        h,
        ID,
        charge,
        Px, Py, Pz,
        Ex, Ey, Ez);
}

template <class P, unsigned Dim>
auto SphexaParticleContainer<P, Dim>::dataTuple() const -> DataTupleConstT {
    return std::tie(
        Rx, Ry, Rz,
        h,
        ID,
        charge,
        Px, Py, Pz,
        Ex, Ey, Ez);
}

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::SphexaParticleContainer(
    SphexaParticleContainer&&) noexcept = default;

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>&
SphexaParticleContainer<P, Dim>::operator=(SphexaParticleContainer&&) noexcept = default;

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::create(LocalIndex nLocal) {
    Rx.resize(nLocal);
    Ry.resize(nLocal);
    Rz.resize(nLocal);
    h.resize(nLocal);
    ID.resize(nLocal);
    charge.resize(nLocal);
    Ex.resize(nLocal);
    Ey.resize(nLocal);
    Ez.resize(nLocal);
    Px.resize(nLocal);
    Py.resize(nLocal);
    Pz.resize(nLocal);
    // particleKeys must match the size of (Rx, Ry, Rz, h, charge, ID) on entry to
    // domain.sync(Grav) — cstone's checkSizesEqual at domain.hpp:451 enforces this
    // even though sync() resizes keys internally afterwards. Scratch buffers
    // are sized lazily by sync.
    particleKeys.resize(nLocal);

    hZero.resize(nLocal);
    cstone::fillGpu(hZero.data(), hZero.data() + nLocal, Th(0));

    for (auto* a : userAttribs) a->resize(nLocal);
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::addAttribute(ParticleAttribBase& attr) {
    userAttribs.push_back(&attr);
    if (Rx.size() > 0) attr.resize(Rx.size());
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::setUniformH(Th val) {
    uniformH = val;
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::setLeafBasedH(bool on) {
    leafBasedH = on;
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::update() {
    // Px/Py/Pz are threaded through the property tuple so cstone permutes them
    // in lockstep with positions during the SFC sort. Without this, drift writes
    // mismatched (R[j], P[j]) pairs after every sync.
    domain_.sync(
        particleKeys,
        Rx, Ry, Rz, h,
        std::tie(ID, Px, Py, Pz),
        std::tie(scratch0, scratch1, scratchTh, scratchSfc));
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::updateGrav() {
    // Pass hZero (all zeros) as the h argument so cstone's
    // discoverHalos sees zero-radius bounding boxes — dominant cost reduction
    // for findHalosKernel. The real h (uniform, used by ryoanji P2P)
    // is refilled below on grow.
    domain_.syncGrav(
        particleKeys,
        Rx, Ry, Rz, hZero, charge,
        std::tie(ID, Px, Py, Pz),
        std::tie(scratch0, scratch1, scratchTh, scratchSfc));
    refreshPostSyncGrav();
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::refreshPostSyncGrav() {
    // Ex/Ey/Ez are output buffers — not transported by syncGrav. Resize to match
    // position-array extent so consumer kernels can index over [0, nWithHalos)
    // uniformly. Halo-region content is undefined.
    const LocalIndex n = domain_.nParticlesWithHalos();
    Ex.resize(n);
    Ey.resize(n);
    Ez.resize(n);

    // h is no longer threaded through syncGrav; restore it for ryoanji.
    if (leafBasedH) {
        // Per-particle h = local octree leaf edge length. Recomputed every
        // step over [0, nWithHalos) because focus-tree refinement migrates
        // particles between leaves between calls.
        h.resize(n);
        auto leaves = domain_.focusTree().treeLeavesAcc();
        const auto& bx = domain_.box();
        const Tc vol   = bx.lx() * bx.ly() * bx.lz();
        const Th cbrtVol = static_cast<Th>(std::cbrt(static_cast<double>(vol)));
        const unsigned blockSize = 256u;
        const unsigned numBlocks = (n + blockSize - 1u) / blockSize;
        setHFromLeafKernel<<<numBlocks, blockSize>>>(
            particleKeys.data(),
            leaves.data(),
            static_cast<int>(leaves.size()),
            cbrtVol,
            h.data(),
            n);
    } else {
        // Fill only newly-grown entries — assumes uniform h (the existing entries
        // already hold the correct value from create()/IC or the previous grow).
        const LocalIndex oldH = h.size();
        h.resize(n);
        if (n > oldH) {
            cstone::fillGpu(h.data() + oldH, h.data() + n, uniformH);
        }
    }
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::getLocalNum() const {
    return domain_.nParticles();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::startIndex() const {
    return domain_.startIndex();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::endIndex() const {
    return domain_.endIndex();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::nWithHalos() const {
    return domain_.nParticlesWithHalos();
}


template <class P, unsigned Dim>
cstone::Box<typename P::Tc> SphexaParticleContainer<P, Dim>::box() const {
    return domain_.box();
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::exchangeChargeHalos() {
    domain_.exchangeHalos(
        std::tie(charge),
        haloSendBuf, haloRecvBuf);
}


// Explicit instantiations — one per precision policy. Each maps to a
// pre-compiled ryoanji TRAVERSE_MPOLE row; adding a fourth combo requires
// a matching TRAVERSE row in extern/ryoanji/.../traversal_gpu.cu.
template class SphexaParticleContainer<DoublePrecision, 3>;
template class SphexaParticleContainer<MixedPrecision,  3>;
template class SphexaParticleContainer<FloatPrecision,  3>;

} // namespace ippl::nbody
