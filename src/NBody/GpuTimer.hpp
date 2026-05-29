#ifndef IPPL_NBODY_GPU_TIMER_HPP
#define IPPL_NBODY_GPU_TIMER_HPP

#include "cstone/cuda/cuda_utils.hpp"  // syncGpu (real on GPU builds, stub decl on CPU)

#include "Utility/IpplTimings.h"

#include "NBody/Accelerator.hpp"

namespace ippl::nbody {

// Times a scope including the trailing device synchronize. On GPU builds the
// dtor blocks on syncGpu() so the elapsed time covers async kernel completion;
// on the CPU build the syncGpu() call is compiled out (kernels are synchronous).
// collect=false skips both start/stop, leaving the timer untouched.
class GpuTimer {
public:
    explicit GpuTimer(IpplTimings::TimerRef ref, bool collect = true)
        : ref_(ref), collect_(collect) {
        if (collect_) { IpplTimings::startTimer(ref_); }
    }
    ~GpuTimer() {
        if constexpr (kHaveGpu) { syncGpu(); }
        if (collect_) { IpplTimings::stopTimer(ref_); }
    }
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

private:
    IpplTimings::TimerRef ref_;
    bool                  collect_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_GPU_TIMER_HPP
