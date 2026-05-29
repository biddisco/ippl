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
  * @brief Wrapper for Kokkos around raw pointers exposed by the particle container
  *
  * @author Timo Schwab, <tischwab@ethz.ch>
  */
#pragma once 

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "NBody/core/Accelerator.hpp"
#include "NBody/NBodyParticleContainer.hpp"

// Consumer-side adapter: unmanaged Kokkos::View accessors over the container's
// raw field storage. Included ONLY by Kokkos consumers (the .cu kernel TUs,
// diagnostics, and the OPAL-X kicker) — never by the producer path. Keeping this
// out of NBodyParticleContainer.hpp is what lets the g++ sync TU include the
// container without dragging in Kokkos (and the -DKOKKOS_DEPENDENCE that would
// reroute it through nvcc_wrapper).

namespace ippl::nbody {

// Memory space of the consumer-facing view: the device space IPPL configured
// (CudaSpace/HIPSpace) on GPU builds, HostSpace on CPU — matching where the
// FieldVector storage lives.
using ViewSpace = std::conditional_t<kHaveGpu,
                                     Kokkos::DefaultExecutionSpace::memory_space,
                                     Kokkos::HostSpace>;
template <class U>
using KView = Kokkos::View<U*, ViewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class U>
using KViewConst = Kokkos::View<const U*, ViewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Unmanaged Kokkos::View over the storage of field Name.
template <util::StructuralString Name, class P, unsigned Dim>
auto getView(NBodyParticleContainer<P, Dim>& pc) {
    auto& dv = pc.template getDV<Name>();
    using T  = typename std::remove_reference_t<decltype(dv)>::value_type;
    return KView<T>(dv.data(), dv.size());
}
template <util::StructuralString Name, class P, unsigned Dim>
auto getView(const NBodyParticleContainer<P, Dim>& pc) {
    auto const& dv = pc.template getDV<Name>();
    using T  = typename std::remove_reference_t<decltype(dv)>::value_type;
    return KViewConst<T>(dv.data(), dv.size());
}

}  // namespace ippl::nbody
