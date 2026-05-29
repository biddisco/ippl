#include "NBody/SphexaBHSolver.hpp"

#include "NBody/Accelerator.hpp"
#include "NBody/GpuTimer.hpp"
#include "NBody/GravityWrapper.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include "Utility/IpplTimings.h"

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"

namespace ippl::nbody {

// Pure host orchestration — no device kernels live here. Compiled by the CXX
// compiler on every backend; USE_CUDA (propagated PUBLIC from cstone_gpu) selects
// NBodyAcc=GpuTag and the GPU MultipoleHolder, otherwise CpuTag and the CPU one.
// Mirrors sphexa main/src/propagator/nbody.hpp::computeForces:
//   zero accel -> computeSpatialGroups -> upsweep -> traverse (BH + Ewald).
template <class P, unsigned Dim>
class SphexaBHSolver<P, Dim>::Impl {
public:
    using Container     = typename SphexaBHSolver<P, Dim>::Container;
    using Tmm           = typename P::Tmm;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmm>;
    using DomainT       = typename Container::DomainT;
    using Holder        = MultipoleHolder<MultipoleType, DomainT, Container, NBodyAcc>;

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
    auto& s      = *impl_;
    auto& pc     = s.pc_;
    auto& params = s.params_;
    using Ta     = typename P::Ta;

    static IpplTimings::TimerRef tUpswp  = IpplTimings::getTimer("bh.upsweep");
    static IpplTimings::TimerRef tGroups = IpplTimings::getTimer("bh.groups");
    static IpplTimings::TimerRef tZeroE  = IpplTimings::getTimer("bh.zeroE");
    static IpplTimings::TimerRef tBH     = IpplTimings::getTimer("bh.compute");

    const bool     collect = !warmup;
    auto&          domain  = pc.domain();
    const unsigned start   = pc.startIndex();
    const unsigned end     = pc.endIndex();
    if (end == start) { return; }

    // Gravitational prefactor read by the holder (d.g). The BH near-field
    // lattice extent and the Ewald shell count are the same number; bind both
    // to params.numShells so behavior matches the pre-portability solver
    // regardless of how a driver fills ewaldSettings.
    pc.g                            = params.G;
    auto ewald                      = params.ewaldSettings;
    ewald.numReplicaShells          = params.numShells;
    s.mHolder_.setEwaldSettings(ewald);

    {
        GpuTimer t(tZeroE, collect);
        cstone::fill<kHaveGpu>(pc.ax.data() + start, pc.ax.data() + end, Ta(0));
        cstone::fill<kHaveGpu>(pc.ay.data() + start, pc.ay.data() + end, Ta(0));
        cstone::fill<kHaveGpu>(pc.az.data() + start, pc.az.data() + end, Ta(0));
    }

    cstone::GroupView grp;
    { GpuTimer t(tGroups, collect); grp = s.mHolder_.computeSpatialGroups(pc, domain); }
    { GpuTimer t(tUpswp,  collect); s.mHolder_.upsweep(pc, domain); }
    { GpuTimer t(tBH,     collect); s.mHolder_.traverse(grp, pc, domain); }

    // sphexa gravity_wrapper guard: ryoanji signals traversal-stack exhaustion by
    // setting maxP2P = 0xFFFFFFFF. Without this check the kernel returns with
    // corrupt index data and the next syncGrav deadlocks on bad SFC keys. The CPU
    // holder returns zeroed stats, so this is a no-op there.
    auto stats = s.mHolder_.readStats();
    if (stats[1] == 0xFFFFFFFFull) {
        int mpiRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        throw std::runtime_error(
            "Barnes-Hut traversal stack exhausted on rank " + std::to_string(mpiRank) +
            ". Raise theta, reduce numShells, or shrink particle count per rank.");
    }

    // Per-step BH stats — global aggregates across ranks for direct comparison
    // with sphexa's timer.logStatistics output. Skipped during the pre_run warmup
    // solve so the printed counts reflect only the timed steps.
    if (collect) {
        unsigned long long local[5] = {
            static_cast<unsigned long long>(stats[0]),   // sumP2P
            static_cast<unsigned long long>(stats[2]),   // sumM2P
            static_cast<unsigned long long>(stats[1]),   // maxP2P
            static_cast<unsigned long long>(stats[3]),   // maxM2P
            static_cast<unsigned long long>(stats[4])};  // maxStack
        unsigned long long sum[2] = {0, 0};
        unsigned long long mx[3]  = {0, 0, 0};
        MPI_Allreduce(&local[0], &sum[0], 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local[2], &mx[0], 3, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        int mpiRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        if (mpiRank == 0) {
            std::printf("[bh.stats] sumP2P=%llu sumM2P=%llu maxP2P=%llu maxM2P=%llu maxStack=%llu\n",
                        sum[0], sum[1], mx[0], mx[1], mx[2]);
            std::fflush(stdout);
        }
    }
}

// Explicit instantiations — one per precision policy. Each maps to a
// pre-compiled ryoanji TRAVERSE_MPOLE + COMPUTE_GRAVITY_EWALD_GPU row (GPU) or
// the header-only ryoanji CPU path.
template class SphexaBHSolver<DoublePrecision, 3>;
template class SphexaBHSolver<MixedPrecision,  3>;
template class SphexaBHSolver<FloatPrecision,  3>;

}  // namespace ippl::nbody
