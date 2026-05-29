#ifndef IPPL_NBODY_GPU_TIMER_HPP
#define IPPL_NBODY_GPU_TIMER_HPP

#include <cuda_runtime.h>

#include "Utility/IpplTimings.h"

namespace ippl::nbody {

// Times a scope including the trailing cudaDeviceSynchronize.
// collect=false skips both start/stop, leaving the timer untouched.
class GpuTimer {
public:
    explicit GpuTimer(IpplTimings::TimerRef ref, bool collect = true)
        : ref_(ref), collect_(collect) {
        if (collect_) { IpplTimings::startTimer(ref_); }
    }
    ~GpuTimer() {
        cudaDeviceSynchronize();
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
