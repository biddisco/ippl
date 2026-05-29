/*
 * IPPL Barnes-Hut
 * 
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 PSI, Villigen
 * 
 * Please refer to the LICENSE file in the root directory
 * SPDX-License-Identifier: GPL-3.0
 */

/*! @file
 * @brief Realize the configure-time GPU/CPU choice
 * 
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <type_traits>
#include <vector>

// cuda_utils.hpp (memcpyH2D/D2H/D2D, syncGpu) must precede primitives_acc.hpp:
// cstone::copy_n calls the global memcpyD2D, which has to be declared at
// that template's definition point.
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

// This inline expression is used to select between CPU and GPU implementations
inline constexpr bool kHaveGpu = bool(cstone::HaveGpu<NBodyAcc>{});

// Per-component field storage: GPU-resident DeviceVector on device builds, plain
// std::vector on the CPU build. Mirrors sphexa ParticlesData::FieldVector.
template <class T>
using FieldVector =
    std::conditional_t<kHaveGpu, cstone::DeviceVector<T>, std::vector<T>>;

}  // namespace ippl::nbody
