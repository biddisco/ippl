#ifndef IPPL_BH_RANDOM_HPP
#define IPPL_BH_RANDOM_HPP

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace ippl::nbody {

// Counter-based RNG for device-parallel IC sampling. Stateless per particle
// (key on the global index) so the IC is reproducible across rank counts, and
// cheap — no engine state to seed. Replaces the per-particle std::mt19937_64,
// whose full-state seeding made the IC pathologically slow and which only ran
// serial on HIP (OpenMP is off there). Usable on CUDA/HIP/CPU via Kokkos.
KOKKOS_INLINE_FUNCTION uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform in [0, 1): top 53 bits scaled by 2^-53.
template <class T>
KOKKOS_INLINE_FUNCTION T bhUniform(uint64_t& state) {
    return static_cast<T>(static_cast<double>(splitmix64(state) >> 11) *
                          (1.0 / 9007199254740992.0));
}

// One standard normal via Box-Muller (consumes two uniforms).
template <class T>
KOKKOS_INLINE_FUNCTION T bhNormal(uint64_t& state) {
    T u1 = bhUniform<T>(state);
    T u2 = bhUniform<T>(state);
    // Guard the log against u1 == 0. bhUniform yields k*2^-53 for integer k, so the
    // only value that breaks log is k == 0; clamp it to the smallest positive uniform
    // (2^-53). A compile-time constant, so it stays device-callable (unlike host-only
    // std::numeric_limits::min) and is representable in float (unlike a 1e-300 double).
    constexpr T tiny = static_cast<T>(1.0 / 9007199254740992.0);
    if (u1 < tiny) { u1 = tiny; }
    return Kokkos::sqrt(T(-2) * Kokkos::log(u1)) *
           Kokkos::cos(T(6.283185307179586) * u2);
}

}  // namespace ippl::nbody

#endif  // IPPL_BH_RANDOM_HPP
