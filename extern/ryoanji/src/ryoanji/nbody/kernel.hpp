/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

/*! @file
 * @brief  Particle-2-particle (P2P) kernel
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#include "types.h"

namespace ryoanji
{

HOST_DEVICE_FUN HOST_DEVICE_INLINE float inverseSquareRoot(float x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN HOST_DEVICE_INLINE double inverseSquareRoot(double x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

/*! @brief interaction between two particles
 *
 * @param acc     acceleration to add to
 * @param pos_i
 * @param pos_j
 * @param m_j
 * @param h_i
 * @param h_j
 * @return        input acceleration plus contribution from this call
 */
template<class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> P2P(Vec4<Ta> acc, const Vec3<Tc>& pos_i, const Vec3<Tc>& pos_j, Tm m_j, Th h_i,
                                           Th h_j)
{
    Vec3<Tc> dX = pos_j - pos_i;
    Tc       R2 = norm2(dX);

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc R2eff = (R2 < h_ij2) ? h_ij2 : R2;

    Tc invR   = inverseSquareRoot(R2eff);
    Tc invR2  = invR * invR;
    Tc invR3m = m_j * invR * invR2;

    acc[0] = Ta(fma(-invR3m, R2, Tc(acc[0])));
    acc[1] = Ta(fma(dX[0], invR3m, Tc(acc[1])));
    acc[2] = Ta(fma(dX[1], invR3m, Tc(acc[2])));
    acc[3] = Ta(fma(dX[2], invR3m, Tc(acc[3])));

    return acc;
}

/*! @brief symmetric two-sided pair interaction (Newton's 3rd law)
 *
 * Computes the pair interaction once and applies it to both particles.
 * acc_i gets the force from j; acc_j gets the equal-and-opposite force from i (scaled by m_i).
 * The potential contributions differ by the mass ratio m_j/m_i.
 * All softening and inverse-sqrt work is shared.
 */
template<class Ta, class Tc, class Th, class Tm>
HOST_DEVICE_FUN DEVICE_INLINE void P2P_symmetric(Vec4<Ta>& acc_i, Vec4<Ta>& acc_j, const Vec3<Tc>& pos_i,
                                                  const Vec3<Tc>& pos_j, Tm m_i, Tm m_j, Th h_i, Th h_j)
{
    Vec3<Tc> dX = pos_j - pos_i;
    Tc       R2 = norm2(dX);

    Th h_ij  = h_i + h_j;
    Th h_ij2 = h_ij * h_ij;
    Tc R2eff = (R2 < h_ij2) ? h_ij2 : R2;

    Tc invR  = inverseSquareRoot(R2eff);
    Tc invR2 = invR * invR;
    Tc invR3 = invR * invR2;

    Tc invR3_mj = Tc(m_j) * invR3;
    Tc invR3_mi = Tc(m_i) * invR3;

    acc_i[0] = Ta(fma(-invR3_mj, R2, Tc(acc_i[0])));
    acc_i[1] = Ta(fma(dX[0], invR3_mj, Tc(acc_i[1])));
    acc_i[2] = Ta(fma(dX[1], invR3_mj, Tc(acc_i[2])));
    acc_i[3] = Ta(fma(dX[2], invR3_mj, Tc(acc_i[3])));

    acc_j[0] = Ta(fma(-invR3_mi, R2, Tc(acc_j[0])));
    acc_j[1] = Ta(fma(-dX[0], invR3_mi, Tc(acc_j[1])));
    acc_j[2] = Ta(fma(-dX[1], invR3_mi, Tc(acc_j[2])));
    acc_j[3] = Ta(fma(-dX[2], invR3_mi, Tc(acc_j[3])));
}

} // namespace ryoanji
