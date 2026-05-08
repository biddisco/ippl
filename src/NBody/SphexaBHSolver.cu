#include "NBody/SphexaBHSolver.hpp"

#include <cstdint>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "cstone/sfc/box.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/groups_gpu.h"
#include "ryoanji/interface/ewald.cuh"
#include "ryoanji/interface/treebuilder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/upsweep_gpu.h"

namespace ippl::nbody {

template <class T, unsigned Dim>
class SphexaBHSolver<T, Dim>::Impl {
public:
    using Container     = typename SphexaBHSolver<T, Dim>::Container;
    using KeyType       = std::uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    Impl(Container& pc, Params params)
        : pc_(pc), params_(params),
          treeBuilder_(/*bucketSize=*/64) {}

    Container& pc_;
    Params     params_;

    ryoanji::TreeBuilder<KeyType>                      treeBuilder_;
    thrust::device_vector<cstone::SourceCenterType<T>> dCenters_;
    thrust::device_vector<MultipoleType>               dMultipoles_;
    thrust::device_vector<int>                         dGlobalPool_;
    cstone::GroupData<cstone::GpuTag>                  groups_;
    bool                                               groupsInitialized_ = false;
};

template <class T, unsigned Dim>
SphexaBHSolver<T, Dim>::SphexaBHSolver(Container& pc, Params params)
    : impl_(std::make_unique<Impl>(pc, params)) {}

template <class T, unsigned Dim>
SphexaBHSolver<T, Dim>::~SphexaBHSolver() = default;

template <class T, unsigned Dim>
SphexaBHSolver<T, Dim>::SphexaBHSolver(SphexaBHSolver&&) noexcept = default;

template <class T, unsigned Dim>
SphexaBHSolver<T, Dim>&
SphexaBHSolver<T, Dim>::operator=(SphexaBHSolver&&) noexcept = default;

template <class T, unsigned Dim>
void SphexaBHSolver<T, Dim>::runSolver() {
    auto& pc     = impl_->pc_;
    auto& params = impl_->params_;
    auto& s      = *impl_;

    using MultipoleType = typename Impl::MultipoleType;

    pc.updateGrav();
    pc.exchangeChargeHalos();

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    const unsigned n     = end - start;
    if (n == 0) { return; }

    const auto box = pc.box();
    const bool periodic =
        box.boundaryX() == cstone::BoundaryType::periodic &&
        box.boundaryY() == cstone::BoundaryType::periodic &&
        box.boundaryZ() == cstone::BoundaryType::periodic;
    const int numShells = periodic ? params.numShells : 0;

    int numSources = s.treeBuilder_.update(
        pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
        n, box);
    unsigned                     highestLevel = s.treeBuilder_.maxTreeLevel();
    cstone::TreeNodeIndex        numLeaves    = s.treeBuilder_.numLeafNodes();
    const cstone::TreeNodeIndex* levelRange   = s.treeBuilder_.levelRange();
    const float invTheta = 1.0f / params.theta;

    s.dCenters_.resize(numSources);
    cstone::computeLeafSourceCenterGpu(
        pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
        pc.getChargeRaw() + start,
        s.treeBuilder_.leafToInternal(), numLeaves,
        s.treeBuilder_.layout(),
        thrust::raw_pointer_cast(s.dCenters_.data()));
    cstone::upsweepCentersGpu(
        highestLevel, s.treeBuilder_.levelRange(),
        s.treeBuilder_.childOffsets(),
        thrust::raw_pointer_cast(s.dCenters_.data()));
    cstone::setMacGpu(
        s.treeBuilder_.nodeKeys(), cstone::TreeNodeIndex(numSources),
        thrust::raw_pointer_cast(s.dCenters_.data()), invTheta, box);

    s.dMultipoles_.resize(numSources);
    thrust::fill(s.dMultipoles_.begin(), s.dMultipoles_.end(), MultipoleType{});
    ryoanji::computeLeafMultipoles(
        pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
        pc.getChargeRaw() + start,
        s.treeBuilder_.leafToInternal(), numLeaves,
        s.treeBuilder_.layout(),
        thrust::raw_pointer_cast(s.dCenters_.data()),
        thrust::raw_pointer_cast(s.dMultipoles_.data()));
    // Upsweep all internal levels including the root: the Ewald reciprocal-
    // space sum reads dMultipoles_[0] as the box-wide expansion.
    for (int level = int(highestLevel) - 1; level >= 0; --level) {
        cstone::TreeNodeIndex first = levelRange[level];
        cstone::TreeNodeIndex last  = levelRange[level + 1];
        if (first < last) {
            ryoanji::upsweepMultipoles(
                first, last, s.treeBuilder_.childOffsets(),
                thrust::raw_pointer_cast(s.dCenters_.data()),
                thrust::raw_pointer_cast(s.dMultipoles_.data()));
        }
    }

    if (!s.groupsInitialized_) {
        cstone::computeFixedGroups(
            cstone::LocalIndex(0), cstone::LocalIndex(n),
            ryoanji::bhMaxTargetSize(), s.groups_);
        s.dGlobalPool_.resize(ryoanji::stackSize(s.groups_.numGroups));
        s.groupsInitialized_ = true;
    }
    cstone::TreeNodeIndex rootChildOffset;
    cudaMemcpy(&rootChildOffset, s.treeBuilder_.childOffsets(),
               sizeof(cstone::TreeNodeIndex), cudaMemcpyDeviceToHost);

    // ryoanji::traverse accumulates onto Ex/Ey/Ez — zero them first.
    {
        const unsigned extent = pc.nWithHalos();
        if (extent > 0) {
            const std::size_t bytes = static_cast<std::size_t>(extent) * sizeof(T);
            cudaMemsetAsync(pc.getExRaw(), 0, bytes);
            cudaMemsetAsync(pc.getEyRaw(), 0, bytes);
            cudaMemsetAsync(pc.getEzRaw(), 0, bytes);
        }
    }

    ryoanji::traverse(
        s.groups_.view(), rootChildOffset,
        pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
        pc.getChargeRaw() + start, pc.getHRaw() + start,
        pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
        pc.getChargeRaw() + start, pc.getHRaw() + start,
        s.treeBuilder_.childOffsets(), s.treeBuilder_.internalToLeaf(),
        s.treeBuilder_.layout(),
        thrust::raw_pointer_cast(s.dCenters_.data()),
        thrust::raw_pointer_cast(s.dMultipoles_.data()),
        T(params.G), numShells,
        ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
        (T*)nullptr,
        pc.getExRaw() + start, pc.getEyRaw() + start, pc.getEzRaw() + start,
        thrust::raw_pointer_cast(s.dGlobalPool_.data()));

    if (periodic) {
        // Long-range Ewald correction. BH traverse contributes the truncated
        // near-field lattice sum; computeGravityEwaldGpu adds the real-space
        // erfc tail + reciprocal-space sum out to hCut. Tin-foil convention
        // (k=0 mode skipped) — caller must seed a charge-neutral system.
        using SC = cstone::SourceCenterType<T>;
        SC hostCenter{};
        cudaMemcpy(&hostCenter, thrust::raw_pointer_cast(s.dCenters_.data()),
                   sizeof(SC), cudaMemcpyDeviceToHost);
        MultipoleType hostMroot{};
        cudaMemcpy(&hostMroot, thrust::raw_pointer_cast(s.dMultipoles_.data()),
                   sizeof(MultipoleType), cudaMemcpyDeviceToHost);

        T ugravTotDummy = T(0);
        ryoanji::computeGravityEwaldGpu(
            cstone::Vec3<T>{hostCenter[0], hostCenter[1], hostCenter[2]},
            hostMroot,
            s.groups_.view(),
            pc.getRxRaw() + start, pc.getRyRaw() + start, pc.getRzRaw() + start,
            pc.getChargeRaw() + start,
            box, static_cast<float>(params.G),
            /*ugrav=*/static_cast<T*>(nullptr),
            pc.getExRaw() + start, pc.getEyRaw() + start, pc.getEzRaw() + start,
            /*ugravTot=*/&ugravTotDummy,
            params.ewaldSettings);
    }

    cudaDeviceSynchronize();
}

template class SphexaBHSolver<float, 3>;
template class SphexaBHSolver<double, 3>;

} // namespace ippl::nbody
