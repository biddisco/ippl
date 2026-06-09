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
 * @brief Kokkos-free particle attribute base class for NBody
 * 
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <cstddef>

namespace ippl::nbody {

class ParticleAttribBase {
public:
    virtual ~ParticleAttribBase() = default;

    virtual void        resize(std::size_t n) = 0;
    virtual std::size_t size() const          = 0;
};

} // namespace ippl::nbody
