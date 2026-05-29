#ifndef IPPL_NBODY_TEST_UTIL_HPP
#define IPPL_NBODY_TEST_UTIL_HPP

// Backend-agnostic host<->storage transfer helpers for the NBody unit tests.
// On GPU builds they use cstone's portable memcpy wrappers (which dispatch to
// CUDA/HIP); on the CPU build they are plain std::copy over std::vector storage.
// This lets the container/leapfrog/wrap tests run unchanged on either backend.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"  // memcpyH2D / memcpyD2H (portable)

#include "NBody/Accelerator.hpp"  // kHaveGpu

namespace ippl::nbody::test {

template <class T>
void uploadHost(const std::vector<T>& host, T* dst) {
    if (host.empty()) { return; }
    if constexpr (kHaveGpu) { memcpyH2D(host.data(), host.size(), dst); }
    else { std::copy(host.begin(), host.end(), dst); }
}

template <class T>
void downloadDevice(const T* src, std::size_t n, std::vector<T>& host) {
    host.resize(n);
    if (n == 0) { return; }
    if constexpr (kHaveGpu) { memcpyD2H(src, n, host.data()); }
    else { std::copy(src, src + n, host.begin()); }
}

}  // namespace ippl::nbody::test

#endif  // IPPL_NBODY_TEST_UTIL_HPP
