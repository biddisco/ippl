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
 * @brief minimal particle attribute class to mirror class hierarchy
 * 
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include "NBody/core/Accelerator.hpp"
#include "NBody/core/ParticleAttribBase.hpp"

namespace ippl::nbody {

template <class T>
class ParticleAttrib : public ParticleAttribBase {
public:
    using value_type = T;

    void        resize(std::size_t n) override { data_.resize(n); }
    std::size_t size() const override          { return data_.size(); }

    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    FieldVector<T>&       container()       { return data_; }
    const FieldVector<T>& container() const { return data_; }

private:
    FieldVector<T> data_;
};

} // namespace ippl::nbody