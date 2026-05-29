#include "NBody/SphexaBHSolver.hpp"

#include "NBody/GpuTimer.hpp"

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <mpi.h>

#include "Utility/IpplTimings.h"

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/interface/ewald.cuh"
#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/types.h"

namespace ippl::nbody {

using ippl::nbody::GpuTimer;

// Distributed BH+Ewald wrapper. Mirrors sphexa main/src/propagator/gravity_wrapper.hpp
// (MultipoleHolderGpu::traverse): the MultipoleHolder owns the per-rank focus-tree
// upsweep and the BH near-field traverse, and on periodic boxes we follow up with
// computeGravityEwaldGpu using the focus-tree root center + root multipole.
template <class P, unsigned Dim>
class SphexaBHSolver<P, Dim>::Impl {
public:
    using Container     = typename SphexaBHSolver<P, Dim>::Container;
    using Tc            = typename P::Tc;
    using Th            = typename P::Th;
    using Tm            = typename P::Tm;
    using Ta            = typename P::Ta;
    using Tf            = typename P::Tf;
    using Tmm           = typename P::Tmm;
    using KeyType       = std::uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmm>;
    using Holder        = ryoanji::MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MultipoleType>;

    Impl(Container& pc, Params params) : pc_(pc), params_(params) {}

    Container& pc_;
    Params     params_;
    Holder     mHolder_;
};

template <class P, unsigned Dim>
SphexaBHSolver<P, Dim>::SphexaBHSolver(Container& pc, Params params)
    : impl_(std::make_unique<Impl>(pc, params)) {}

template <class P, unsigned Dim>
SphexaBHSolver<P, Dim>::~SphexaBHSolver() = default;

template <class P, unsigned Dim>
SphexaBHSolver<P, Dim>::SphexaBHSolver(SphexaBHSolver&&) noexcept = default;

template <class P, unsigned Dim>
SphexaBHSolver<P, Dim>&
SphexaBHSolver<P, Dim>::operator=(SphexaBHSolver&&) noexcept = default;

template <class P, unsigned Dim>
void SphexaBHSolver<P, Dim>::runSolver(bool warmup) {
    auto& pc     = impl_->pc_;
    auto& params = impl_->params_;
    auto& s      = *impl_;

    using Tc            = typename Impl::Tc;
    using Ta            = typename Impl::Ta;
    using MultipoleType = typename Impl::MultipoleType;

    static IpplTimings::TimerRef tUpswp  = IpplTimings::getTimer("bh.upsweep");
    static IpplTimings::TimerRef tGroups = IpplTimings::getTimer("bh.groups");
    static IpplTimings::TimerRef tZeroE  = IpplTimings::getTimer("bh.zeroE");
    static IpplTimings::TimerRef tBH     = IpplTimings::getTimer("bh.compute");
    static IpplTimings::TimerRef tEwald  = IpplTimings::getTimer("bh.ewald");

    const bool collect = !warmup;

    auto&         domain = pc.domain();
    const auto    box    = pc.box();
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    if (end == start) { return; }

    const bool periodic =
        box.boundaryX() == cstone::BoundaryType::periodic &&
        box.boundaryY() == cstone::BoundaryType::periodic &&
        box.boundaryZ() == cstone::BoundaryType::periodic;
    const int numShells = periodic ? params.numShells : 0;

    {
        GpuTimer t(tUpswp, collect);
        s.mHolder_.upsweep(
            getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc), getRaw<"charge">(pc),
            domain.globalTree(), domain.focusTree(), domain.layout().data());
    }

    cstone::GroupView grp;
    {
        GpuTimer t(tGroups, collect);
        grp = s.mHolder_.computeSpatialGroups(
            start, end,
            getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc), getRaw<"h">(pc),
            domain.focusTree(), domain.layout().data(), box);
    }

    {
        GpuTimer t(tZeroE, collect);
        const unsigned extent = pc.nWithHalos();
        if (extent > 0) {
            const std::size_t bytes = static_cast<std::size_t>(extent) * sizeof(Ta);
            cudaMemsetAsync(getRaw<"Ex">(pc), 0, bytes);
            cudaMemsetAsync(getRaw<"Ey">(pc), 0, bytes);
            cudaMemsetAsync(getRaw<"Ez">(pc), 0, bytes);
        }
    }

    {
        GpuTimer t(tBH, collect);
        s.mHolder_.compute(
            grp,
            getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
            getRaw<"charge">(pc), getRaw<"h">(pc),
            Tc(params.G), numShells, box,
            /*ugrav=*/static_cast<Ta*>(nullptr),
            getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc));
    }

    // sphexa gravity_wrapper.hpp guard: ryoanji signals traversal-stack
    // exhaustion by setting maxP2P = 0xFFFFFFFF. Without this check the
    // kernel returns with corrupt index data and the next syncGrav deadlocks
    // on bad SFC keys (silent hang).
    {
        auto stats = s.mHolder_.readStats();
        if (stats[1] == 0xFFFFFFFFull) {
            int mpiRank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            throw std::runtime_error(
                "Barnes-Hut traversal stack exhausted on rank " +
                std::to_string(mpiRank) +
                ". Raise theta, reduce numShells, or shrink particle count per rank.");
        }

        // Per-step BH stats — global aggregates across ranks for direct
        // comparison with sphexa's timer.logStatistics output. sphexa logs
        // stats[0]/lastStepTime and stats[2]/lastStepTime (rates); we log
        // the raw counts so the comparison is unambiguous. Skipped during the
        // pre_run warmup solve so the printed counts reflect only the timed
        // steps (mirrors the timing skip above).
        if (collect) {
            unsigned long long local[5] = {
                static_cast<unsigned long long>(stats[0]),  // sumP2P
                static_cast<unsigned long long>(stats[2]),  // sumM2P
                static_cast<unsigned long long>(stats[1]),  // maxP2P
                static_cast<unsigned long long>(stats[3]),  // maxM2P
                static_cast<unsigned long long>(stats[4])}; // maxStack
            unsigned long long sum[2] = {0, 0};
            unsigned long long mx[3]  = {0, 0, 0};
            MPI_Allreduce(&local[0], &sum[0], 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local[2], &mx[0],  3, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            int mpiRank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            if (mpiRank == 0) {
                std::printf("[bh.stats] sumP2P=%llu sumM2P=%llu maxP2P=%llu maxM2P=%llu maxStack=%llu\n",
                            sum[0], sum[1], mx[0], mx[1], mx[2]);
                std::fflush(stdout);
            }
        }
    }

    if (periodic) {
        GpuTimer t(tEwald, collect);
        ryoanji::Vec4<Tc> rootCenter{};
        cudaMemcpy(&rootCenter,
                   domain.focusTree().expansionCentersAcc().data(),
                   sizeof(rootCenter), cudaMemcpyDeviceToHost);
        MultipoleType rootM{};
        cudaMemcpy(&rootM, s.mHolder_.deviceMultipoles(),
                   sizeof(MultipoleType), cudaMemcpyDeviceToHost);

        // ugravTot accumulator type: per ryoanji's COMPUTE_GRAVITY_EWALD_GPU
        // instantiations, the Tu template param matches Tc for all three
        // pre-compiled precision combos (double/double/float for Double/Mixed/Float).
        Tc ugravTotDummy = Tc(0);
        ryoanji::computeGravityEwaldGpu(
            cstone::Vec3<Tc>{rootCenter[0], rootCenter[1], rootCenter[2]},
            rootM, grp,
            getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
            getRaw<"charge">(pc),
            box, static_cast<float>(params.G),
            /*ugrav=*/static_cast<Ta*>(nullptr),
            getRaw<"Ex">(pc), getRaw<"Ey">(pc), getRaw<"Ez">(pc),
            /*ugravTot=*/&ugravTotDummy,
            params.ewaldSettings);
    }
}

// Explicit instantiations — one per precision policy. Each maps to a
// pre-compiled ryoanji TRAVERSE_MPOLE + COMPUTE_GRAVITY_EWALD_GPU row.
template class SphexaBHSolver<DoublePrecision, 3>;
template class SphexaBHSolver<MixedPrecision,  3>;
template class SphexaBHSolver<FloatPrecision,  3>;

} // namespace ippl::nbody
