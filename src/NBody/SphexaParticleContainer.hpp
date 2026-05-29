#ifndef IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
#define IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"

#include "NBody/BHPrecision.hpp"
#include "NBody/ParticleAttribBase.hpp"

namespace ippl::nbody {

template <class P, unsigned Dim>
class SphexaParticleContainer {
    static_assert(Dim == 3, "SphexaParticleContainer requires Dim == 3");

public:
    using Precision  = P;
    using Tc         = typename P::Tc;
    using Th         = typename P::Th;
    using Tm         = typename P::Tm;
    using Ta         = typename P::Ta;
    using KeyType    = std::uint64_t;
    using LocalIndex = unsigned;

    template <class U>
    using KView      = Kokkos::View<U*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    template <class U>
    using KViewConst = Kokkos::View<const U*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // ID is stored as uint32_t (4 B) so it fits the same-type scratch-buffer
    // constraint imposed by cstone's GlobalAssignment::distribute (which
    // requires the first two scratch buffers to share the value_type of T).
    // ~4e9 IDs; promote to uint64_t when needed.
    using IdType     = std::uint32_t;
    using IDView     = Kokkos::View<IdType*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    using DomainT = cstone::Domain<KeyType, Tc, cstone::GpuTag>;

    // Order must match dataTuple() below; cstone::get<"Name">(pc) maps a string to
    // the tuple slot via getFieldIndex(name, fieldNames).
    static constexpr std::array<std::string_view, 12> fieldNames{
        "Rx", "Ry", "Rz", "h", "ID", "charge",
        "Px", "Py", "Pz", "Ex", "Ey", "Ez"};

    using DataTupleT = std::tuple<
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Th>&,
        cstone::DeviceVector<IdType>&,
        cstone::DeviceVector<Tm>&,
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Tc>&,
        cstone::DeviceVector<Ta>&,
        cstone::DeviceVector<Ta>&,
        cstone::DeviceVector<Ta>&>;

    using DataTupleConstT = std::tuple<
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Th>&,
        const cstone::DeviceVector<IdType>&,
        const cstone::DeviceVector<Tm>&,
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Tc>&,
        const cstone::DeviceVector<Ta>&,
        const cstone::DeviceVector<Ta>&,
        const cstone::DeviceVector<Ta>&>;

    DataTupleT      dataTuple();
    DataTupleConstT dataTuple() const;

    SphexaParticleContainer(int rank, int nRanks,
                            unsigned bucketSize, unsigned bucketSizeFocus,
                            float theta,
                            std::array<Tc, 6> boxLoHi,
                            std::array<cstone::BoundaryType, 3> boundaries);
    ~SphexaParticleContainer();

    SphexaParticleContainer(const SphexaParticleContainer&)            = delete;
    SphexaParticleContainer& operator=(const SphexaParticleContainer&) = delete;
    SphexaParticleContainer(SphexaParticleContainer&&) noexcept;
    SphexaParticleContainer& operator=(SphexaParticleContainer&&) noexcept;

    // Allocate the built-in attribute storage to nLocal entries on this rank.
    // Caller initializes positions, h, and ID through the raw / View accessors before
    // calling update().
    void create(LocalIndex nLocal);

    // Caller owns the attribute and must keep it alive while registered.
    // Owned-only: not permuted by update() / updateGrav() and not halo-exchanged.
    void addAttribute(ParticleAttribBase& attr);

    // SFC-sort by position, redistribute particles across ranks, exchange position halos.
    // Permutes the ID attribute in lockstep with positions. After return,
    // [startIndex(), endIndex()) is this rank's owned range; [endIndex(), nWithHalos())
    // holds halo entries for positions and h (not ID — properties have no halo data).
    void update();

    // Gravity-aware sync: same as update(), but also threads `charge` through cstone's
    // gravity sync path (Domain::syncGrav), which updates focus-tree expansion centers
    // and MAC spheres in lockstep with the SFC sort. Required before any BH traversal.
    // Sizes Ex/Ey/Ez to nWithHalos on return so consumer kernels can index uniformly;
    // halo-region content of Ex/Ey/Ez is undefined (these are output buffers, not
    // transported across ranks).
    void updateGrav();

    // Propagate the `charge` array into halo entries via cstone halo exchange.
    // Required before BH traversal so source charges at halo indices are valid.
    // On nRanks=1 this is a no-op.
    void exchangeChargeHalos();

    // Compile-time index of field Name in fieldNames.
    template <util::StructuralString Name>
    static constexpr std::size_t idxOf =
        cstone::getFieldIndex(Name.value, fieldNames);

    // SFC-sort + redistribute. ConservedIdxs lists dataTuple() slots that get
    // permuted + redistributed alongside positions.
    template <std::size_t... ConservedIdxs>
    void update() {
        auto t = dataTuple();
        domain_.sync(
            particleKeys,
            Rx, Ry, Rz, h,
            std::tie(std::get<ConservedIdxs>(t)...),
            std::tie(scratch0, scratch1, scratchTh, scratchSfc));
    }

    // Gravity-aware sync. MSlotIdx is the dataTuple() slot fed into syncGrav's
    // mass argument (gravity expansion centers / MAC). ConservedIdxs lists the
    // conserved slots permuted in lockstep with positions.
    template <std::size_t MSlotIdx, std::size_t... ConservedIdxs>
    void updateGrav() {
        auto t = dataTuple();
        domain_.syncGrav(
            particleKeys,
            Rx, Ry, Rz, hZero,
            std::get<MSlotIdx>(t),
            std::tie(std::get<ConservedIdxs>(t)...),
            std::tie(scratch0, scratch1, scratchTh, scratchSfc));
        refreshPostSyncGrav();
    }

    // Halo exchange for an arbitrary set of source field slots.
    template <std::size_t... HaloIdxs>
    void exchangeHalos() {
        auto t = dataTuple();
        domain_.exchangeHalos(
            std::tie(std::get<HaloIdxs>(t)...),
            haloSendBuf, haloRecvBuf);
    }

    // Raw pointer to the storage of field Name. The field must be one of fieldNames.
    template <util::StructuralString Name>
    auto* getRaw() {
        return std::get<idxOf<Name>>(dataTuple()).data();
    }
    template <util::StructuralString Name>
    auto const* getRaw() const {
        return std::get<idxOf<Name>>(dataTuple()).data();
    }

    // Unmanaged Kokkos::View over the storage of field Name.
    template <util::StructuralString Name>
    auto getView() {
        auto& dv = std::get<idxOf<Name>>(dataTuple());
        using T  = typename std::remove_reference_t<decltype(dv)>::value_type;
        return KView<T>(dv.data(), dv.size());
    }
    template <util::StructuralString Name>
    auto getView() const {
        auto& dv = std::get<idxOf<Name>>(dataTuple());
        using T  = typename std::remove_reference_t<decltype(dv)>::value_type;
        return KViewConst<T>(dv.data(), dv.size());
    }

    LocalIndex getLocalNum() const;
    LocalIndex startIndex() const;
    LocalIndex endIndex() const;
    LocalIndex nWithHalos() const;

    // Record the uniform smoothing length used by ryoanji P2P softening.
    // updateGrav() passes a dummy h=0 buffer to cstone (so halo bounding boxes
    // are not inflated by 2*h*factor), then refills h to nWithHalos with this
    // value for ryoanji. Caller must call this once after initializing particles,
    // before the first updateGrav(). Assumes h is uniform; non-uniform unsupported.
    void setUniformH(Th val);

    // Switch the post-syncGrav h refill from uniform (fill with uniformH) to
    // leaf-derived (per-particle h = local octree leaf edge length). Off by default.
    void setLeafBasedH(bool on);

    cstone::Box<Tc> box() const;

    // Underlying cstone::Domain reference. Used by SphexaBHSolver to call
    // exchangeHalos / focusTree / globalTree / layout.
    DomainT&       domain()       { return domain_; }
    const DomainT& domain() const { return domain_; }

private:
    void refreshPostSyncGrav();

    DomainT                                  domain_;
    cstone::DeviceVector<KeyType>            particleKeys;
    cstone::DeviceVector<Tc>                 Rx, Ry, Rz;
    cstone::DeviceVector<Th>                 h;
    cstone::DeviceVector<IdType>             ID;
    cstone::DeviceVector<Tm>                 charge;
    cstone::DeviceVector<Ta>                 Ex, Ey, Ez;
    cstone::DeviceVector<Tc>                 Px, Py, Pz;

    cstone::DeviceVector<Tc>                 scratch0, scratch1;
    cstone::DeviceVector<Th>                 scratchTh;
    cstone::DeviceVector<cstone::LocalIndex> scratchSfc;

    cstone::DeviceVector<Tm>                 haloSendBuf, haloRecvBuf;

    cstone::DeviceVector<Th>                 hZero;
    Th                                       uniformH{Th(0)};
    bool                                     leafBasedH{false};

    std::vector<ParticleAttribBase*>         userAttribs;
};

// Free-function shortcuts so callers in templated contexts don't need
// `.template` to invoke the member-template accessor.
template <util::StructuralString Name, class P, unsigned Dim>
auto* getRaw(SphexaParticleContainer<P, Dim>& pc) {
    return pc.template getRaw<Name>();
}
template <util::StructuralString Name, class P, unsigned Dim>
auto const* getRaw(const SphexaParticleContainer<P, Dim>& pc) {
    return pc.template getRaw<Name>();
}

template <util::StructuralString Name, class P, unsigned Dim>
auto getView(SphexaParticleContainer<P, Dim>& pc) {
    return pc.template getView<Name>();
}
template <util::StructuralString Name, class P, unsigned Dim>
auto getView(const SphexaParticleContainer<P, Dim>& pc) {
    return pc.template getView<Name>();
}

} // namespace ippl::nbody

#endif // IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
