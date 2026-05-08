#ifndef IPPL_BH_LANDAU_DIAGNOSTICS_HPP
#define IPPL_BH_LANDAU_DIAGNOSTICS_HPP

#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// Single-component result: sum-of-squares and max-abs over a particle scalar.
template <class T>
struct AxisStats {
    T sumSq;
    T maxAbs;
};

// Compute (Σ Eₓ², max |Eₓ|) over the locally-owned range
// [startIndex(), endIndex()) of pc. Used to derive Landau-damping diagnostics:
//   field_energy = (V / N) · sumSq        (Monte-Carlo estimator of ∫ Eₓ² dV)
//   ex_max_norm  = maxAbs
template <class T>
AxisStats<T> reduceExStats(SphexaParticleContainer<T, 3>& pc);

// Cosine/sine spatial Fourier amplitudes of Eₓ at wavenumber kₓ along x:
//   cosAmp = Σ Eₓ[i] · cos(kₓ · x_i)
//   sinAmp = Σ Eₓ[i] · sin(kₓ · x_i)
// over [startIndex(), endIndex()). Caller computes |A_k| = √(cosAmp² + sinAmp²).
//
// Used to extract the perturbation-mode amplitude separately from the particle
// shot-noise floor — the BH field at low N is noise-dominated by inter-particle
// Coulomb fluctuations, but the projection onto the cosine mode at the
// perturbation k is essentially noise-free (noise has no preferred wavevector).
template <class T>
struct CosSinAmp {
    T cosAmp;
    T sinAmp;
};
template <class T>
CosSinAmp<T> reduceExCosineMode(SphexaParticleContainer<T, 3>& pc, T kx);

// CIC-smoothed grid energy diagnostic for the gridless BH path.
//
// Scatters per-particle Eₓ values onto an equivalent G³ grid using CIC weights,
// computes a cell-averaged Eₓ_avg(cell) = Σ_p w_p Eₓ_p / Σ_p w_p, then returns
// the grid-integrated Σ_cell Eₓ_avg² · ΔV_cell.
//
// Purpose: an apples-to-apples comparison with FFT-PIC's grid-integrated field
// energy. The native (V/N)·Σ Eₓ(R_p)² estimator is "honest" but carries the
// short-range Coulomb / discreteness contribution that PIC discards at the
// grid-deposit step. CIC-binning Eₓ here applies the same low-pass filter,
// dropping all power above k_Nyquist = π·G/L.
//
// Single-rank, periodic [0, L]³ box. G must match (or be chosen against) the
// FFT-PIC grid resolution for the comparison to be meaningful.
template <class T>
T reduceExGridEnergyCIC(SphexaParticleContainer<T, 3>& pc, T L, int G);

}  // namespace ippl::nbody

#endif  // IPPL_BH_LANDAU_DIAGNOSTICS_HPP
