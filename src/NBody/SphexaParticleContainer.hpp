#ifndef IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
#define IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"

#include "NBody/Accelerator.hpp"
#include "NBody/BHPrecision.hpp"
#include "NBody/ParticleAttribBase.hpp"

namespace ippl::nbody {

namespace detail {

// Per-particle leaf-derived smoothing length: h[i] = cbrtVol / 2^level(leaf(i)).
// Backend split: GPU launches a kernel (SphexaParticleContainer.cu), CPU runs an
// OpenMP loop (SphexaParticleContainer.cpp). Both call the same host/device
// cstone helpers (findNodeBelow, treeLevel). Instantiated for (Th, KeyType).
template <class Th, class KeyType>
void setHFromLeaves(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                    Th cbrtVol, Th* h, unsigned n);

}  // namespace detail

// Gridless particle container wrapping cstone::Domain. Doubles as the sphexa
// "DataType": its public field members (x, y, z, h, m, ax, ay, az, ugrav) and
// the AcceleratorType / KeyType / RealType typedefs are exactly the surface the
// vendored gravity_wrapper.hpp MultipoleHolder{Cpu,Gpu} read. The friendly
// getRaw<"Name">/idxOf/getDV accessors map the legacy names onto these members
// (Rx->x, charge->m, Ex->ax, ...), so existing call sites are unchanged.
//
// Field storage is FieldVector<T>: cstone::DeviceVector<T> on GPU builds, plain
// std::vector<T> on the CPU build. The accelerator (NBodyAcc) is build-global,
// selected by USE_CUDA — one AccType per build, like sphexa.
template <class P, unsigned Dim>
class SphexaParticleContainer {
    static_assert(Dim == 3, "SphexaParticleContainer requires Dim == 3");

public:
    using Precision       = P;
    using Tc              = typename P::Tc;
    using Th              = typename P::Th;
    using Tm              = typename P::Tm;
    using Ta              = typename P::Ta;
    using KeyType         = std::uint64_t;
    using LocalIndex      = unsigned;
    using AcceleratorType = NBodyAcc;
    using RealType        = Tc;

    // Memory space of the consumer-facing Kokkos view accessor: the device space
    // IPPL configured (CudaSpace/HIPSpace) on GPU builds, HostSpace on CPU —
    // matching where the FieldVector storage lives.
    using ViewSpace = std::conditional_t<kHaveGpu,
                                         Kokkos::DefaultExecutionSpace::memory_space,
                                         Kokkos::HostSpace>;
    template <class U>
    using KView      = Kokkos::View<U*, ViewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    template <class U>
    using KViewConst = Kokkos::View<const U*, ViewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // ID is stored as uint32_t (4 B) so it fits the same-type scratch-buffer
    // constraint imposed by cstone's GlobalAssignment::distribute (which
    // requires the first two scratch buffers to share the value_type of T).
    // ~4e9 IDs; promote to uint64_t when needed.
    using IdType = std::uint32_t;
    using IDView = Kokkos::View<IdType*, ViewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    using DomainT = cstone::Domain<KeyType, Tc, NBodyAcc>;

    // Legacy field names kept for the getRaw<"Name"> API. The dataTuple() order
    // below maps them onto the sphexa-named storage members.
    static constexpr std::array<std::string_view, 8> fieldNames{
        "Rx", "Ry", "Rz", "h", "charge", "Ex", "Ey", "Ez"};

    using DataTupleT = std::tuple<
        FieldVector<Tc>&, FieldVector<Tc>&, FieldVector<Tc>&,
        FieldVector<Th>&, FieldVector<Tm>&,
        FieldVector<Ta>&, FieldVector<Ta>&, FieldVector<Ta>&>;
    using DataTupleConstT = std::tuple<
        const FieldVector<Tc>&, const FieldVector<Tc>&, const FieldVector<Tc>&,
        const FieldVector<Th>&, const FieldVector<Tm>&,
        const FieldVector<Ta>&, const FieldVector<Ta>&, const FieldVector<Ta>&>;

    DataTupleT      dataTuple()       { return std::tie(x, y, z, h, m, ax, ay, az); }
    DataTupleConstT dataTuple() const { return std::tie(x, y, z, h, m, ax, ay, az); }

    SphexaParticleContainer(int rank, int nRanks,
                            unsigned bucketSize, unsigned bucketSizeFocus,
                            float theta,
                            std::array<Tc, 6> boxLoHi,
                            std::array<cstone::BoundaryType, 3> boundaries)
        : domain_(rank, nRanks, bucketSize, bucketSizeFocus, theta, MPI_COMM_WORLD,
                  cstone::Box<Tc>(boxLoHi[0], boxLoHi[1], boxLoHi[2], boxLoHi[3],
                                  boxLoHi[4], boxLoHi[5],
                                  boundaries[0], boundaries[1], boundaries[2])) {}
    ~SphexaParticleContainer() = default;

    SphexaParticleContainer(const SphexaParticleContainer&)            = delete;
    SphexaParticleContainer& operator=(const SphexaParticleContainer&) = delete;
    SphexaParticleContainer(SphexaParticleContainer&&) noexcept            = default;
    SphexaParticleContainer& operator=(SphexaParticleContainer&&) noexcept = default;

    // Allocate the 8 built-in fields to nLocal entries on this rank. Caller
    // initializes positions and h through getRaw / getView before update().
    void create(LocalIndex nLocal) {
        x.resize(nLocal); y.resize(nLocal); z.resize(nLocal);
        h.resize(nLocal); m.resize(nLocal);
        ax.resize(nLocal); ay.resize(nLocal); az.resize(nLocal);
        ugrav.resize(nLocal);
        // particleKeys must match the size of positions + h + m on entry to
        // domain.sync(Grav) — cstone's checkSizesEqual enforces this even though
        // sync() resizes keys internally afterwards. Scratch buffers are sized
        // lazily by sync.
        keys.resize(nLocal);

        hZero.resize(nLocal);
        cstone::fill<kHaveGpu>(hZero.data(), hZero.data() + nLocal, Th(0));

        for (auto* a : userAttribs) a->resize(nLocal);
    }

    // Caller owns the attribute and must keep it alive while registered.
    // Owned-only: not permuted by update() / updateGrav() and not halo-exchanged.
    void addAttribute(ParticleAttribBase& attr) {
        userAttribs.push_back(&attr);
        if (x.size() > 0) attr.resize(x.size());
    }

    // Compile-time index of field Name in fieldNames.
    template <util::StructuralString Name>
    static constexpr std::size_t idxOf = cstone::getFieldIndex(Name.value, fieldNames);

    // SFC-sort + redistribute + position halo exchange. The driver passes its
    // own conserved FieldVectors as the parameter pack; cstone permutes and
    // redistributes them in lockstep with positions.
    template <class... ConservedVecs>
    void update(ConservedVecs&... conserved) {
        domain_.sync(keys, x, y, z, h,
                     std::tie(conserved...),
                     std::tie(scratch0, scratch1, scratchTh, scratchSfc));
    }

    // Gravity-aware sync. MSlotIdx names the dataTuple() slot fed into syncGrav's
    // mass argument (focus-tree expansion centers + MAC). conserved... are the
    // driver-owned fields permuted alongside positions. Halo exchange of the
    // m-slot for BH traversal is the caller's responsibility (see exchangeHalos).
    template <std::size_t MSlotIdx, class... ConservedVecs>
    void updateGrav(ConservedVecs&... conserved) {
        auto t = dataTuple();
        domain_.syncGrav(keys, x, y, z, hZero,
                         std::get<MSlotIdx>(t),
                         std::tie(conserved...),
                         std::tie(scratch0, scratch1, scratchTh, scratchSfc));
        refreshPostSyncGrav();
    }

    // Pass-through to domain.exchangeHalos. The caller supplies the field tuple
    // (e.g. std::tie(pc.getDV<"charge">())) and matching-type scratch buffers.
    template <class Arrays, class SendBuf, class RecvBuf>
    void exchangeHalos(Arrays&& arrays, SendBuf& send, RecvBuf& recv) {
        domain_.exchangeHalos(std::forward<Arrays>(arrays), send, recv);
    }

    // Reference to the underlying FieldVector of field Name. Use to build tuples
    // for exchangeHalos / driver-side composed sync calls.
    template <util::StructuralString Name>
    auto& getDV()             { return std::get<idxOf<Name>>(dataTuple()); }
    template <util::StructuralString Name>
    auto const& getDV() const { return std::get<idxOf<Name>>(dataTuple()); }

    // Tm-typed scratch buffers for halo exchange of the m-slot field. Drivers
    // halo-exchanging other types should declare their own scratch DVs.
    FieldVector<Tm>&       haloSendBuf()       { return haloSendBuf_; }
    FieldVector<Tm>&       haloRecvBuf()       { return haloRecvBuf_; }
    const FieldVector<Tm>& haloSendBuf() const { return haloSendBuf_; }
    const FieldVector<Tm>& haloRecvBuf() const { return haloRecvBuf_; }

    // Raw pointer to the storage of field Name. The field must be one of fieldNames.
    template <util::StructuralString Name>
    auto* getRaw()             { return std::get<idxOf<Name>>(dataTuple()).data(); }
    template <util::StructuralString Name>
    auto const* getRaw() const { return std::get<idxOf<Name>>(dataTuple()).data(); }

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

    LocalIndex getLocalNum() const { return domain_.nParticles(); }
    LocalIndex startIndex()  const { return domain_.startIndex(); }
    LocalIndex endIndex()    const { return domain_.endIndex(); }
    LocalIndex nWithHalos()  const { return domain_.nParticlesWithHalos(); }

    // Record the uniform smoothing length used by ryoanji P2P softening.
    // updateGrav() passes a dummy h=0 buffer to cstone (so halo bounding boxes
    // are not inflated by 2*h*factor), then refills h to nWithHalos with this
    // value for ryoanji. Caller must call this once after initializing particles,
    // before the first updateGrav(). Assumes h is uniform; non-uniform unsupported.
    void setUniformH(Th val) { uniformH = val; }

    // Switch the post-syncGrav h refill from uniform (fill with uniformH) to
    // leaf-derived (per-particle h = local octree leaf edge length). Off by default.
    void setLeafBasedH(bool on) { leafBasedH = on; }

    cstone::Box<Tc> box() const { return domain_.box(); }

    // Underlying cstone::Domain reference. Used by SphexaBHSolver to call
    // exchangeHalos / focusTree / globalTree / layout.
    DomainT&       domain()       { return domain_; }
    const DomainT& domain() const { return domain_; }

    // --- sphexa DataType surface (read by gravity_wrapper.hpp) ----------------
    FieldVector<Tc> x, y, z;       // positions   (was Rx, Ry, Rz)
    FieldVector<Th> h;             // smoothing length
    FieldVector<Tm> m;             // mass/source (was charge)
    FieldVector<Ta> ax, ay, az;    // accelerations (was Ex, Ey, Ez)
    FieldVector<Ta> ugrav;         // gravitational potential (unused output)
    Tc              g{Tc(1)};      // gravitational/Coulomb prefactor
    Tc              egrav{Tc(0)};  // total potential energy (unused output)

private:
    void refreshPostSyncGrav() {
        // ax/ay/az/ugrav are output buffers — not transported by syncGrav. Resize
        // to match position-array extent so consumer kernels can index over
        // [0, nWithHalos) uniformly. Halo-region content is undefined.
        const LocalIndex n = domain_.nParticlesWithHalos();
        ax.resize(n); ay.resize(n); az.resize(n); ugrav.resize(n);

        // h is no longer threaded through syncGrav; restore it for ryoanji.
        if (leafBasedH) {
            // Per-particle h = local octree leaf edge length. Recomputed every
            // step over [0, nWithHalos) because focus-tree refinement migrates
            // particles between leaves between calls.
            h.resize(n);
            auto       leaves  = domain_.focusTree().treeLeavesAcc();
            const auto& bx     = domain_.box();
            const Tc    vol     = bx.lx() * bx.ly() * bx.lz();
            const Th    cbrtVol = static_cast<Th>(std::cbrt(static_cast<double>(vol)));
            detail::setHFromLeaves<Th, KeyType>(
                keys.data(), leaves.data(), static_cast<int>(leaves.size()),
                cbrtVol, h.data(), n);
        } else {
            // Fill only newly-grown entries — assumes uniform h (the existing
            // entries already hold the correct value from create()/IC or the
            // previous grow).
            const LocalIndex oldH = h.size();
            h.resize(n);
            if (n > oldH) {
                cstone::fill<kHaveGpu>(h.data() + oldH, h.data() + n, uniformH);
            }
        }
    }

    DomainT                       domain_;
    FieldVector<KeyType>          keys;
    FieldVector<Tc>               scratch0, scratch1;
    FieldVector<Th>               scratchTh;
    FieldVector<cstone::LocalIndex> scratchSfc;
    FieldVector<Tm>               haloSendBuf_, haloRecvBuf_;
    FieldVector<Th>               hZero;
    Th                            uniformH{Th(0)};
    bool                          leafBasedH{false};
    std::vector<ParticleAttribBase*> userAttribs;
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
