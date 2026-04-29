#include "NBody/SphexaParticleContainer.hpp"

#include <tuple>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/primitives/primitives_acc.hpp"

namespace ippl::nbody {

template <class T, unsigned Dim>
class SphexaParticleContainer<T, Dim>::Impl {
public:
    using DomainT = cstone::Domain<KeyType, T, cstone::GpuTag>;

    Impl(int rank, int nRanks,
         unsigned bucketSize, unsigned bucketSizeFocus,
         float theta,
         const cstone::Box<T>& box)
        : domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, box) {}

    DomainT                              domain;
    cstone::DeviceVector<KeyType>        particleKeys;
    cstone::DeviceVector<T>              Rx, Ry, Rz, h;
    cstone::DeviceVector<IdType>         ID;
    // charge is a scalar source threaded through Domain::syncGrav (so cstone
    // updates focus-tree expansion centers in lockstep with positions).
    // Ex/Ey/Ez are output buffers written by SphexaBHSolver::runSolver(); they
    // never appear in any sync/scratch tuple.
    cstone::DeviceVector<T>              charge;
    cstone::DeviceVector<T>              Ex, Ey, Ez;
    // Velocity. Owned-state only (not in any sync property tuple). Resized to
    // nWithHalos by updateGrav() to align with position-array extent; halo-
    // region content is undefined.
    cstone::DeviceVector<T>              Px, Py, Pz;

    // Three scratch buffers required by domain.sync; the last is the SFC ordering buffer.
    // cstone's design assumes homogeneous-type conserved fields:
    //   - GlobalAssignment::distribute (assignment.hpp:165-174) takes the first two
    //     scratch buffers as `Vector& s0, Vector& s1` — same template type required.
    //   - domain.hpp:523 (pickType) does exact-type lookup: for every position/h array
    //     of type T in Arrays1, the scratch tuple must contain a DeviceVector<T>.
    //   - domain.hpp:433 (SmallerElementSize) requires some scratch with element size
    //     ≥ every conserved-field element size; with IdType==uint32_t (4 B) ≤ sizeof(T),
    //     a DeviceVector<T> scratch satisfies this for both T=float and T=double.
    // Two DeviceVector<T> scratches satisfy all three constraints simultaneously.
    cstone::DeviceVector<T>                  scratch0, scratch1;
    cstone::DeviceVector<cstone::LocalIndex> scratchSfc;

    // Send/receive scratch for halo exchange of the charge array (BH source).
    // Sized lazily by Domain::exchangeHalos.
    cstone::DeviceVector<T>                  haloSendBuf, haloRecvBuf;
};

template <class T, unsigned Dim>
SphexaParticleContainer<T, Dim>::SphexaParticleContainer(
    int rank, int nRanks,
    unsigned bucketSize, unsigned bucketSizeFocus,
    float theta,
    std::array<T, 6> boxLoHi,
    std::array<cstone::BoundaryType, 3> boundaries)
    : impl_(std::make_unique<Impl>(
          rank, nRanks, bucketSize, bucketSizeFocus, theta,
          cstone::Box<T>(boxLoHi[0], boxLoHi[1],
                         boxLoHi[2], boxLoHi[3],
                         boxLoHi[4], boxLoHi[5],
                         boundaries[0], boundaries[1], boundaries[2]))) {}

template <class T, unsigned Dim>
SphexaParticleContainer<T, Dim>::~SphexaParticleContainer() = default;

template <class T, unsigned Dim>
SphexaParticleContainer<T, Dim>::SphexaParticleContainer(
    SphexaParticleContainer&&) noexcept = default;

template <class T, unsigned Dim>
SphexaParticleContainer<T, Dim>&
SphexaParticleContainer<T, Dim>::operator=(SphexaParticleContainer&&) noexcept = default;

template <class T, unsigned Dim>
void SphexaParticleContainer<T, Dim>::create(LocalIndex nLocal) {
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
}

template <class T, unsigned Dim>
void SphexaParticleContainer<T, Dim>::update() {
    // Px/Py/Pz are threaded through the property tuple so cstone permutes them
    // in lockstep with positions during the SFC sort. Without this, drift writes
    // mismatched (R[j], P[j]) pairs after every sync.
    impl_->domain.sync(
        impl_->particleKeys,
        impl_->Rx, impl_->Ry, impl_->Rz, impl_->h,
        std::tie(impl_->ID, impl_->Px, impl_->Py, impl_->Pz),
        std::tie(impl_->scratch0, impl_->scratch1, impl_->scratchSfc));
}

template <class T, unsigned Dim>
void SphexaParticleContainer<T, Dim>::updateGrav() {
    impl_->domain.syncGrav(
        impl_->particleKeys,
        impl_->Rx, impl_->Ry, impl_->Rz, impl_->h, impl_->charge,
        std::tie(impl_->ID, impl_->Px, impl_->Py, impl_->Pz),
        std::tie(impl_->scratch0, impl_->scratch1, impl_->scratchSfc));
    // Ex/Ey/Ez are output buffers — not transported by syncGrav. Resize to match
    // position-array extent so consumer kernels can index over [0, nWithHalos)
    // uniformly. Halo-region content is undefined. (Px/Py/Pz are sized by cstone
    // via the property tuple above, so no manual resize needed for them.)
    const LocalIndex n = impl_->domain.nParticlesWithHalos();
    impl_->Ex.resize(n);
    impl_->Ey.resize(n);
    impl_->Ez.resize(n);
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::LocalIndex
SphexaParticleContainer<T, Dim>::getLocalNum() const {
    return impl_->domain.nParticles();
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::LocalIndex
SphexaParticleContainer<T, Dim>::startIndex() const {
    return impl_->domain.startIndex();
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::LocalIndex
SphexaParticleContainer<T, Dim>::endIndex() const {
    return impl_->domain.endIndex();
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::LocalIndex
SphexaParticleContainer<T, Dim>::nWithHalos() const {
    return impl_->domain.nParticlesWithHalos();
}

template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getRxRaw() { return impl_->Rx.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getRyRaw() { return impl_->Ry.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getRzRaw() { return impl_->Rz.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getRxRaw() const { return impl_->Rx.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getRyRaw() const { return impl_->Ry.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getRzRaw() const { return impl_->Rz.data(); }

template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getHRaw() { return impl_->h.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getHRaw() const { return impl_->h.data(); }

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::IdType*
SphexaParticleContainer<T, Dim>::getIDRaw() { return impl_->ID.data(); }
template <class T, unsigned Dim>
const typename SphexaParticleContainer<T, Dim>::IdType*
SphexaParticleContainer<T, Dim>::getIDRaw() const { return impl_->ID.data(); }

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getRxView() {
    return KView(impl_->Rx.data(), impl_->Rx.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getRyView() {
    return KView(impl_->Ry.data(), impl_->Ry.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getRzView() {
    return KView(impl_->Rz.data(), impl_->Rz.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getRxView() const {
    return KViewConst(impl_->Rx.data(), impl_->Rx.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getRyView() const {
    return KViewConst(impl_->Ry.data(), impl_->Ry.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getRzView() const {
    return KViewConst(impl_->Rz.data(), impl_->Rz.size());
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getHView() {
    return KView(impl_->h.data(), impl_->h.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getHView() const {
    return KViewConst(impl_->h.data(), impl_->h.size());
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::IDView
SphexaParticleContainer<T, Dim>::getIDView() {
    return IDView(impl_->ID.data(), impl_->ID.size());
}

template <class T, unsigned Dim>
cstone::Box<T> SphexaParticleContainer<T, Dim>::box() const {
    return impl_->domain.box();
}

template <class T, unsigned Dim>
void SphexaParticleContainer<T, Dim>::exchangeChargeHalos() {
    impl_->domain.exchangeHalos(
        std::tie(impl_->charge),
        impl_->haloSendBuf, impl_->haloRecvBuf);
}

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::DomainT&
SphexaParticleContainer<T, Dim>::domain() { return impl_->domain; }

template <class T, unsigned Dim>
const typename SphexaParticleContainer<T, Dim>::DomainT&
SphexaParticleContainer<T, Dim>::domain() const { return impl_->domain; }

template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getChargeRaw() { return impl_->charge.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getChargeRaw() const { return impl_->charge.data(); }
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getChargeView() {
    return KView(impl_->charge.data(), impl_->charge.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getChargeView() const {
    return KViewConst(impl_->charge.data(), impl_->charge.size());
}

template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getExRaw() { return impl_->Ex.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getEyRaw() { return impl_->Ey.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getEzRaw() { return impl_->Ez.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getExRaw() const { return impl_->Ex.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getEyRaw() const { return impl_->Ey.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getEzRaw() const { return impl_->Ez.data(); }

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getExView() {
    return KView(impl_->Ex.data(), impl_->Ex.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getEyView() {
    return KView(impl_->Ey.data(), impl_->Ey.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getEzView() {
    return KView(impl_->Ez.data(), impl_->Ez.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getExView() const {
    return KViewConst(impl_->Ex.data(), impl_->Ex.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getEyView() const {
    return KViewConst(impl_->Ey.data(), impl_->Ey.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getEzView() const {
    return KViewConst(impl_->Ez.data(), impl_->Ez.size());
}

template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getPxRaw() { return impl_->Px.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getPyRaw() { return impl_->Py.data(); }
template <class T, unsigned Dim>
T* SphexaParticleContainer<T, Dim>::getPzRaw() { return impl_->Pz.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getPxRaw() const { return impl_->Px.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getPyRaw() const { return impl_->Py.data(); }
template <class T, unsigned Dim>
const T* SphexaParticleContainer<T, Dim>::getPzRaw() const { return impl_->Pz.data(); }

template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getPxView() {
    return KView(impl_->Px.data(), impl_->Px.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getPyView() {
    return KView(impl_->Py.data(), impl_->Py.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KView
SphexaParticleContainer<T, Dim>::getPzView() {
    return KView(impl_->Pz.data(), impl_->Pz.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getPxView() const {
    return KViewConst(impl_->Px.data(), impl_->Px.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getPyView() const {
    return KViewConst(impl_->Py.data(), impl_->Py.size());
}
template <class T, unsigned Dim>
typename SphexaParticleContainer<T, Dim>::KViewConst
SphexaParticleContainer<T, Dim>::getPzView() const {
    return KViewConst(impl_->Pz.data(), impl_->Pz.size());
}

// Explicit instantiations. Add additional T types here as downstream phases require them.
template class SphexaParticleContainer<float, 3>;
template class SphexaParticleContainer<double, 3>;

} // namespace ippl::nbody
