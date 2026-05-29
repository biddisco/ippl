#ifndef IPPL_NBODY_ACCELERATOR_HPP
#define IPPL_NBODY_ACCELERATOR_HPP

#include <type_traits>
#include <vector>

// cuda_utils.hpp (memcpyH2D/D2H/D2D, syncGpu) must precede primitives_acc.hpp:
// cstone::copy_n there calls the global memcpyD2D, which has to be declared at
// that template's definition point. cstone's own TUs order it the same way
// (see cstone/domain/domain.hpp). Real defs on GPU builds, decls-only on CPU.
#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/primitives_acc.hpp"

namespace ippl::nbody {

// Build-global accelerator tag, chosen at configure time exactly like sphexa's
// USE_CUDA-driven AccType. The vendored cstone_gpu target defines USE_CUDA
// PUBLIC, so any GPU build (CUDA or HIP) sees it; the CPU build links only
// cstone_headers and never defines it.
#if defined(USE_CUDA)
using NBodyAcc = cstone::GpuTag;
#else
using NBodyAcc = cstone::CpuTag;
#endif

inline constexpr bool kHaveGpu = bool(cstone::HaveGpu<NBodyAcc>{});

// Per-component field storage: GPU-resident DeviceVector on device builds, plain
// std::vector on the CPU build. Mirrors sphexa ParticlesData::FieldVector.
template <class T>
using FieldVector =
    std::conditional_t<kHaveGpu, cstone::DeviceVector<T>, std::vector<T>>;

}  // namespace ippl::nbody

#endif  // IPPL_NBODY_ACCELERATOR_HPP
