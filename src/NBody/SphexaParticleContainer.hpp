#ifndef IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
#define IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP

#include <array>
#include <cstdint>
#include <memory>

#include <Kokkos_Core.hpp>

#include "cstone/sfc/box.hpp"

// Forward declarations of cstone types exposed by domain() accessors below.
// Consumers that only construct/use particle attributes don't need these resolved;
// callers that invoke methods on the returned reference (e.g. SphexaBHSolver) include
// cstone/domain/domain.hpp directly to obtain the full type.
namespace cstone {
struct GpuTag;
template <class KeyType, class T, class Accelerator>
class Domain;
} // namespace cstone

namespace ippl::nbody {

template <class T, unsigned Dim>
class SphexaParticleContainer {
    static_assert(Dim == 3, "SphexaParticleContainer requires Dim == 3");

public:
    using value_type = T;
    using KeyType    = std::uint64_t;
    using LocalIndex = unsigned;

    using KView      = Kokkos::View<T*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using KViewConst = Kokkos::View<const T*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // ID is stored as uint32_t (4 B) so it fits the same-type scratch-buffer
    // constraint imposed by cstone's GlobalAssignment::distribute (which
    // requires the first two scratch buffers to share the value_type of T).
    // ~4e9 IDs; promote to uint64_t when needed.
    using IdType     = std::uint32_t;
    using IDView     = Kokkos::View<IdType*, Kokkos::CudaSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SphexaParticleContainer(int rank, int nRanks,
                            unsigned bucketSize, unsigned bucketSizeFocus,
                            float theta,
                            std::array<T, 6> boxLoHi,
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
    // On nRanks=1 this is a no-op. Send/receive scratch buffers are owned by Impl.
    void exchangeChargeHalos();

    LocalIndex getLocalNum() const;
    LocalIndex startIndex() const;
    LocalIndex endIndex() const;
    LocalIndex nWithHalos() const;

    // Built-in vector attribute R (positions): three independent scalar arrays.
    T* getRxRaw();
    T* getRyRaw();
    T* getRzRaw();
    const T* getRxRaw() const;
    const T* getRyRaw() const;
    const T* getRzRaw() const;

    KView getRxView();
    KView getRyView();
    KView getRzView();
    KViewConst getRxView() const;
    KViewConst getRyView() const;
    KViewConst getRzView() const;

    // Built-in scalar attributes. h is the SPH-convention smoothing length used by
    // domain.sync's halo discovery; ID is a stable particle identifier.
    T* getHRaw();
    const T* getHRaw() const;
    KView getHView();
    KViewConst getHView() const;

    IdType* getIDRaw();
    const IdType* getIDRaw() const;
    IDView getIDView();

    // Built-in scalar source attribute (charge for plasma; mass for gravity).
    // Threaded through Domain::syncGrav by updateGrav(); read by the BH solver
    // as the monopole source for the multipole upsweep + traversal.
    T* getChargeRaw();
    const T* getChargeRaw() const;
    KView getChargeView();
    KViewConst getChargeView() const;

    // Built-in acceleration / E-field outputs. Written by SphexaBHSolver::runSolver().
    T* getExRaw();   T* getEyRaw();   T* getEzRaw();
    const T* getExRaw() const; const T* getEyRaw() const; const T* getEzRaw() const;
    KView getExView();  KView getEyView();  KView getEzView();
    KViewConst getExView() const; KViewConst getEyView() const; KViewConst getEzView() const;

    // Built-in velocity attribute. Owned-state only — never appears in any sync/scratch
    // tuple. Resized to nLocal at create() and to nWithHalos by updateGrav() so consumer
    // kernels can index uniformly over the position-array extent. Halo-region content
    // of P is undefined; the leapfrog stepper iterates [startIndex(), endIndex()) only.
    T* getPxRaw();   T* getPyRaw();   T* getPzRaw();
    const T* getPxRaw() const; const T* getPyRaw() const; const T* getPzRaw() const;
    KView getPxView();  KView getPyView();  KView getPzView();
    KViewConst getPxView() const; KViewConst getPyView() const; KViewConst getPzView() const;

    cstone::Box<T> box() const;

    // Underlying cstone::Domain reference. Used by SphexaBHSolver to call
    // exchangeHalos / focusTree / globalTree / layout. Returning the full type means
    // any caller that invokes methods on it must include cstone/domain/domain.hpp.
    using DomainT = cstone::Domain<KeyType, T, cstone::GpuTag>;
    DomainT&       domain();
    const DomainT& domain() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
