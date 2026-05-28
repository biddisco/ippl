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
class SphexaParticleContainer<P, Dim>::Impl {
public:
    using Tc      = typename P::Tc;
    using Th      = typename P::Th;
    using Tm      = typename P::Tm;
    using Ta      = typename P::Ta;
    using DomainT = cstone::Domain<KeyType, Tc, cstone::GpuTag>;

    Impl(int rank, int nRanks,
         unsigned bucketSize, unsigned bucketSizeFocus,
         float theta,
         const cstone::Box<Tc>& box)
        : domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, MPI_COMM_WORLD, box) {}

    DomainT                              domain;
    cstone::DeviceVector<KeyType>        particleKeys;
    cstone::DeviceVector<Tc>             Rx, Ry, Rz;
    cstone::DeviceVector<Th>             h;
    cstone::DeviceVector<IdType>         ID;
    // charge is a scalar source threaded through Domain::syncGrav (so cstone
    // updates focus-tree expansion centers in lockstep with positions).
    // Ex/Ey/Ez are output buffers written by SphexaBHSolver::runSolver(); they
    // never appear in any sync/scratch tuple.
    cstone::DeviceVector<Tm>             charge;
    cstone::DeviceVector<Ta>             Ex, Ey, Ez;
    // Velocity. Owned-state only (not in any sync property tuple). Resized to
    // nWithHalos by updateGrav() to align with position-array extent; halo-
    // region content is undefined.
    cstone::DeviceVector<Tc>             Px, Py, Pz;

    // Scratch buffers for domain.sync. The last is the SFC ordering buffer.
    // cstone supports heterogeneous-type conserved fields (Domain::sync/syncGrav take
    // independent template params for x/y/z, h, m, and each property in the tuple),
    // but updateLayout's per-array swap-space lookup (extern/cstone/.../domain.hpp:523)
    // uses an *exact* type match via util::Contains — the staticChecks fallback for
    // "value_type with equal or bigger size" only covers the initial assignment step,
    // not the layout step. So we need one DeviceVector per unique value_type that
    // appears in the conserved-field list.
    //
    // Layout:
    //   scratch0, scratch1: DV<Tc>  — first two slots that GlobalAssignment::distribute
    //                                  binds to as s0, s1; both must share the position
    //                                  value_type, which is Tc.
    //   scratchTh:          DV<Th>  — covers the float h (and float charge, since
    //                                  Tm == Th in all three precision policies). When
    //                                  Th == Tc (Double/Float), this scratch is the
    //                                  same type as scratch0/1 — harmless redundancy.
    //   scratchSfc:         DV<LocalIndex> — SFC permutation buffer; via discardLastElement.
    cstone::DeviceVector<Tc>                 scratch0, scratch1;
    cstone::DeviceVector<Th>                 scratchTh;
    cstone::DeviceVector<cstone::LocalIndex> scratchSfc;

    // Send/receive scratch for halo exchange of the charge array (BH source).
    // Sized lazily by Domain::exchangeHalos.
    cstone::DeviceVector<Tm>                 haloSendBuf, haloRecvBuf;

    // Dummy all-zero h passed to syncGrav in place of impl_->h, so cstone's
    // halo-discovery sees zero-radius leaf bounding boxes. Initialised once at
    // create(); cstone's permutation + halo exchange preserve all-zero values.
    cstone::DeviceVector<Th>                 hZero;
    // Uniform smoothing length used by ryoanji P2P. Captured via setUniformH()
    // from the manager; used to fill impl_->h on grow after each syncGrav.
    Th                                       uniformH{Th(0)};
    // When true, the post-syncGrav refill writes h_i = leaf_edge(particle_i)
    // instead of fillGpu(uniformH). Set via setLeafBasedH().
    bool                                     leafBasedH{false};

    std::vector<ParticleAttribBase*>         userAttribs;
};

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::SphexaParticleContainer(
    int rank, int nRanks,
    unsigned bucketSize, unsigned bucketSizeFocus,
    float theta,
    std::array<Tc, 6> boxLoHi,
    std::array<cstone::BoundaryType, 3> boundaries)
    : impl_(std::make_unique<Impl>(
          rank, nRanks, bucketSize, bucketSizeFocus, theta,
          cstone::Box<Tc>(boxLoHi[0], boxLoHi[1],
                          boxLoHi[2], boxLoHi[3],
                          boxLoHi[4], boxLoHi[5],
                          boundaries[0], boundaries[1], boundaries[2]))) {}

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::~SphexaParticleContainer() = default;

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>::SphexaParticleContainer(
    SphexaParticleContainer&&) noexcept = default;

template <class P, unsigned Dim>
SphexaParticleContainer<P, Dim>&
SphexaParticleContainer<P, Dim>::operator=(SphexaParticleContainer&&) noexcept = default;

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::create(LocalIndex nLocal) {
    impl_->Rx.resize(nLocal);
    impl_->Ry.resize(nLocal);
    impl_->Rz.resize(nLocal);
    impl_->h.resize(nLocal);
    impl_->ID.resize(nLocal);
    impl_->charge.resize(nLocal);
    impl_->Ex.resize(nLocal);
    impl_->Ey.resize(nLocal);
    impl_->Ez.resize(nLocal);
    impl_->Px.resize(nLocal);
    impl_->Py.resize(nLocal);
    impl_->Pz.resize(nLocal);
    // particleKeys must match the size of (Rx, Ry, Rz, h, charge, ID) on entry to
    // domain.sync(Grav) — cstone's checkSizesEqual at domain.hpp:451 enforces this
    // even though sync() resizes keys internally afterwards. Scratch buffers
    // are sized lazily by sync.
    impl_->particleKeys.resize(nLocal);

    impl_->hZero.resize(nLocal);
    cstone::fillGpu(impl_->hZero.data(), impl_->hZero.data() + nLocal, Th(0));

    for (auto* a : impl_->userAttribs) a->resize(nLocal);
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::addAttribute(ParticleAttribBase& attr) {
    impl_->userAttribs.push_back(&attr);
    if (impl_->Rx.size() > 0) attr.resize(impl_->Rx.size());
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::setUniformH(Th val) {
    impl_->uniformH = val;
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::setLeafBasedH(bool on) {
    impl_->leafBasedH = on;
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::update() {
    // Px/Py/Pz are threaded through the property tuple so cstone permutes them
    // in lockstep with positions during the SFC sort. Without this, drift writes
    // mismatched (R[j], P[j]) pairs after every sync.
    impl_->domain.sync(
        impl_->particleKeys,
        impl_->Rx, impl_->Ry, impl_->Rz, impl_->h,
        std::tie(impl_->ID, impl_->Px, impl_->Py, impl_->Pz),
        std::tie(impl_->scratch0, impl_->scratch1, impl_->scratchTh, impl_->scratchSfc));
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::updateGrav() {
    // Pass impl_->hZero (all zeros) as the h argument so cstone's
    // discoverHalos sees zero-radius bounding boxes — dominant cost reduction
    // for findHalosKernel. The real impl_->h (uniform, used by ryoanji P2P)
    // is refilled below on grow.
    impl_->domain.syncGrav(
        impl_->particleKeys,
        impl_->Rx, impl_->Ry, impl_->Rz, impl_->hZero, impl_->charge,
        std::tie(impl_->ID, impl_->Px, impl_->Py, impl_->Pz),
        std::tie(impl_->scratch0, impl_->scratch1, impl_->scratchTh, impl_->scratchSfc));
    // Ex/Ey/Ez are output buffers — not transported by syncGrav. Resize to match
    // position-array extent so consumer kernels can index over [0, nWithHalos)
    // uniformly. Halo-region content is undefined. (Px/Py/Pz are sized by cstone
    // via the property tuple above, so no manual resize needed for them.)
    const LocalIndex n = impl_->domain.nParticlesWithHalos();
    impl_->Ex.resize(n);
    impl_->Ey.resize(n);
    impl_->Ez.resize(n);

    // impl_->h is no longer threaded through syncGrav; restore it for ryoanji.
    if (impl_->leafBasedH) {
        // Per-particle h = local octree leaf edge length. Recomputed every
        // step over [0, nWithHalos) because focus-tree refinement migrates
        // particles between leaves between calls.
        impl_->h.resize(n);
        auto leaves = impl_->domain.focusTree().treeLeavesAcc();
        const auto& bx = impl_->domain.box();
        const Tc vol   = bx.lx() * bx.ly() * bx.lz();
        const Th cbrtVol = static_cast<Th>(std::cbrt(static_cast<double>(vol)));
        const unsigned blockSize = 256u;
        const unsigned numBlocks = (n + blockSize - 1u) / blockSize;
        setHFromLeafKernel<<<numBlocks, blockSize>>>(
            impl_->particleKeys.data(),
            leaves.data(),
            static_cast<int>(leaves.size()),
            cbrtVol,
            impl_->h.data(),
            n);
    } else {
        // Fill only newly-grown entries — assumes uniform h (the existing entries
        // already hold the correct value from create()/IC or the previous grow).
        const LocalIndex oldH = impl_->h.size();
        impl_->h.resize(n);
        if (n > oldH) {
            cstone::fillGpu(impl_->h.data() + oldH, impl_->h.data() + n, impl_->uniformH);
        }
    }
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::getLocalNum() const {
    return impl_->domain.nParticles();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::startIndex() const {
    return impl_->domain.startIndex();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::endIndex() const {
    return impl_->domain.endIndex();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::LocalIndex
SphexaParticleContainer<P, Dim>::nWithHalos() const {
    return impl_->domain.nParticlesWithHalos();
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRxRaw() { return impl_->Rx.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRyRaw() { return impl_->Ry.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRzRaw() { return impl_->Rz.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRxRaw() const { return impl_->Rx.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRyRaw() const { return impl_->Ry.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getRzRaw() const { return impl_->Rz.data(); }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Th*
SphexaParticleContainer<P, Dim>::getHRaw() { return impl_->h.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Th*
SphexaParticleContainer<P, Dim>::getHRaw() const { return impl_->h.data(); }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::IdType*
SphexaParticleContainer<P, Dim>::getIDRaw() { return impl_->ID.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::IdType*
SphexaParticleContainer<P, Dim>::getIDRaw() const { return impl_->ID.data(); }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRxView() {
    return KView<Tc>(impl_->Rx.data(), impl_->Rx.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRyView() {
    return KView<Tc>(impl_->Ry.data(), impl_->Ry.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRzView() {
    return KView<Tc>(impl_->Rz.data(), impl_->Rz.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRxView() const {
    return KViewConst<Tc>(impl_->Rx.data(), impl_->Rx.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRyView() const {
    return KViewConst<Tc>(impl_->Ry.data(), impl_->Ry.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getRzView() const {
    return KViewConst<Tc>(impl_->Rz.data(), impl_->Rz.size());
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Th>
SphexaParticleContainer<P, Dim>::getHView() {
    return KView<Th>(impl_->h.data(), impl_->h.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Th>
SphexaParticleContainer<P, Dim>::getHView() const {
    return KViewConst<Th>(impl_->h.data(), impl_->h.size());
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::IDView
SphexaParticleContainer<P, Dim>::getIDView() {
    return IDView(impl_->ID.data(), impl_->ID.size());
}

template <class P, unsigned Dim>
cstone::Box<typename P::Tc> SphexaParticleContainer<P, Dim>::box() const {
    return impl_->domain.box();
}

template <class P, unsigned Dim>
void SphexaParticleContainer<P, Dim>::exchangeChargeHalos() {
    impl_->domain.exchangeHalos(
        std::tie(impl_->charge),
        impl_->haloSendBuf, impl_->haloRecvBuf);
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::DomainT&
SphexaParticleContainer<P, Dim>::domain() { return impl_->domain; }

template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::DomainT&
SphexaParticleContainer<P, Dim>::domain() const { return impl_->domain; }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tm*
SphexaParticleContainer<P, Dim>::getChargeRaw() { return impl_->charge.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tm*
SphexaParticleContainer<P, Dim>::getChargeRaw() const { return impl_->charge.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tm>
SphexaParticleContainer<P, Dim>::getChargeView() {
    return KView<Tm>(impl_->charge.data(), impl_->charge.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tm>
SphexaParticleContainer<P, Dim>::getChargeView() const {
    return KViewConst<Tm>(impl_->charge.data(), impl_->charge.size());
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getExRaw() { return impl_->Ex.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getEyRaw() { return impl_->Ey.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getEzRaw() { return impl_->Ez.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getExRaw() const { return impl_->Ex.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getEyRaw() const { return impl_->Ey.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Ta*
SphexaParticleContainer<P, Dim>::getEzRaw() const { return impl_->Ez.data(); }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Ta>
SphexaParticleContainer<P, Dim>::getExView() {
    return KView<Ta>(impl_->Ex.data(), impl_->Ex.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Ta>
SphexaParticleContainer<P, Dim>::getEyView() {
    return KView<Ta>(impl_->Ey.data(), impl_->Ey.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Ta>
SphexaParticleContainer<P, Dim>::getEzView() {
    return KView<Ta>(impl_->Ez.data(), impl_->Ez.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Ta>
SphexaParticleContainer<P, Dim>::getExView() const {
    return KViewConst<Ta>(impl_->Ex.data(), impl_->Ex.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Ta>
SphexaParticleContainer<P, Dim>::getEyView() const {
    return KViewConst<Ta>(impl_->Ey.data(), impl_->Ey.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Ta>
SphexaParticleContainer<P, Dim>::getEzView() const {
    return KViewConst<Ta>(impl_->Ez.data(), impl_->Ez.size());
}

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPxRaw() { return impl_->Px.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPyRaw() { return impl_->Py.data(); }
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPzRaw() { return impl_->Pz.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPxRaw() const { return impl_->Px.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPyRaw() const { return impl_->Py.data(); }
template <class P, unsigned Dim>
const typename SphexaParticleContainer<P, Dim>::Tc*
SphexaParticleContainer<P, Dim>::getPzRaw() const { return impl_->Pz.data(); }

template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPxView() {
    return KView<Tc>(impl_->Px.data(), impl_->Px.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPyView() {
    return KView<Tc>(impl_->Py.data(), impl_->Py.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KView<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPzView() {
    return KView<Tc>(impl_->Pz.data(), impl_->Pz.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPxView() const {
    return KViewConst<Tc>(impl_->Px.data(), impl_->Px.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPyView() const {
    return KViewConst<Tc>(impl_->Py.data(), impl_->Py.size());
}
template <class P, unsigned Dim>
typename SphexaParticleContainer<P, Dim>::template KViewConst<typename P::Tc>
SphexaParticleContainer<P, Dim>::getPzView() const {
    return KViewConst<Tc>(impl_->Pz.data(), impl_->Pz.size());
}

// Explicit instantiations — one per precision policy. Each maps to a
// pre-compiled ryoanji TRAVERSE_MPOLE row; adding a fourth combo requires
// a matching TRAVERSE row in extern/ryoanji/.../traversal_gpu.cu.
template class SphexaParticleContainer<DoublePrecision, 3>;
template class SphexaParticleContainer<MixedPrecision,  3>;
template class SphexaParticleContainer<FloatPrecision,  3>;

} // namespace ippl::nbody
