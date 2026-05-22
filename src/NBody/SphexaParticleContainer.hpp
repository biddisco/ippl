#ifndef IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
#define IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP

#include <array>
#include <cstdint>
#include <memory>

#include <Kokkos_Core.hpp>

#include "cstone/sfc/box.hpp"

#include "NBody/BHPrecision.hpp"

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
    Tc* getRxRaw();
    Tc* getRyRaw();
    Tc* getRzRaw();
    const Tc* getRxRaw() const;
    const Tc* getRyRaw() const;
    const Tc* getRzRaw() const;

    KView<Tc> getRxView();
    KView<Tc> getRyView();
    KView<Tc> getRzView();
    KViewConst<Tc> getRxView() const;
    KViewConst<Tc> getRyView() const;
    KViewConst<Tc> getRzView() const;

    // Built-in scalar attributes. h is the SPH-convention smoothing length used by
    // domain.sync's halo discovery; ID is a stable particle identifier.
    Th* getHRaw();
    const Th* getHRaw() const;
    KView<Th> getHView();
    KViewConst<Th> getHView() const;

    // Record the uniform smoothing length used by ryoanji P2P softening.
    // Required: updateGrav() passes a dummy h=0 buffer to cstone (so halo
    // bounding boxes are not inflated by 2*h*factor), then refills h to
    // nWithHalos with this value for ryoanji. Caller must call this once
    // after initializing particles, before the first updateGrav().
    // Assumes h is uniform across all particles; non-uniform h is unsupported
    // on this path.
    void setUniformH(Th val);

    // Switch the post-syncGrav h refill from uniform (fill with uniformH) to
    // leaf-derived (per-particle h = local octree leaf edge length). Off by
    // default. Orthogonal to setUniformH: when on, uniformH is only consulted
    // on the first step before any syncGrav has populated the focus tree.
    void setLeafBasedH(bool on);

    IdType* getIDRaw();
    const IdType* getIDRaw() const;
    IDView getIDView();

    // Built-in scalar source attribute (charge for plasma; mass for gravity).
    // Threaded through Domain::syncGrav by updateGrav(); read by the BH solver
    // as the monopole source for the multipole upsweep + traversal.
    Tm* getChargeRaw();
    const Tm* getChargeRaw() const;
    KView<Tm> getChargeView();
    KViewConst<Tm> getChargeView() const;

    // Built-in acceleration / E-field outputs. Written by SphexaBHSolver::runSolver().
    Ta* getExRaw();   Ta* getEyRaw();   Ta* getEzRaw();
    const Ta* getExRaw() const; const Ta* getEyRaw() const; const Ta* getEzRaw() const;
    KView<Ta> getExView();  KView<Ta> getEyView();  KView<Ta> getEzView();
    KViewConst<Ta> getExView() const; KViewConst<Ta> getEyView() const; KViewConst<Ta> getEzView() const;

    // Built-in velocity attribute. Owned-state only — never appears in any sync/scratch
    // tuple. Resized to nLocal at create() and to nWithHalos by updateGrav() so consumer
    // kernels can index uniformly over the position-array extent. Halo-region content
    // of P is undefined; the leapfrog stepper iterates [startIndex(), endIndex()) only.
    Tc* getPxRaw();   Tc* getPyRaw();   Tc* getPzRaw();
    const Tc* getPxRaw() const; const Tc* getPyRaw() const; const Tc* getPzRaw() const;
    KView<Tc> getPxView();  KView<Tc> getPyView();  KView<Tc> getPzView();
    KViewConst<Tc> getPxView() const; KViewConst<Tc> getPyView() const; KViewConst<Tc> getPzView() const;

    cstone::Box<Tc> box() const;

    // Underlying cstone::Domain reference. Used by SphexaBHSolver to call
    // exchangeHalos / focusTree / globalTree / layout. Returning the full type means
    // any caller that invokes methods on it must include cstone/domain/domain.hpp.
    using DomainT = cstone::Domain<KeyType, Tc, cstone::GpuTag>;
    DomainT&       domain();
    const DomainT& domain() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_SPHEXA_PARTICLE_CONTAINER_HPP
