# `src/NBody/` — gridless Barnes-Hut module

This is the producer-side home for IPPL's gridless N-body / Barnes-Hut backend, built on the vendored cstone (`extern/cstone/`) and Ryoanji (`extern/ryoanji/`) libraries from SPH-EXA. The module is gated by `IPPL_ENABLE_NBODY=ON` (which requires CUDA in `IPPL_PLATFORMS`).

## Architectural commitments

1. **Gridless.** No `Field<T,Dim>` / `VField<T,Dim>` is ever instantiated under `src/NBody/`. Self-fields are evaluated at particle positions directly.
2. **Three independent device arrays per vector attribute.** No `View<Vector<T,3>*>`. Storage is `cstone::DeviceVector<T>`.
3. **Kokkos-free producer source files.** Module `.cu`/`.cpp` files do not `#include <Kokkos_Core.hpp>`. Kokkos appears only in public headers, as the type of consumer-facing unmanaged-View accessors that downstream Kokkos kernels read from.

## Contents

- `SphexaParticleContainer<T, 3>` — gridless particle container composing `cstone::Domain<KeyType, T, cstone::GpuTag>`. Built-in attributes: positions (`Rx, Ry, Rz`), smoothing length `h`, particle ID, charge, velocity (`Px, Py, Pz`), and acceleration outputs (`Ex, Ey, Ez`). `update()` wraps `domain.sync(...)` to SFC-sort, MPI-redistribute, and exchange position halos; `updateGrav()` additionally exchanges charge halos and refreshes focus-tree expansion centers.
- `SphexaBHSolver<T, 3>` — Barnes-Hut field solver. Build local BH tree, P2M+M2M upsweep, traverse to write Ex/Ey/Ez. For periodic boxes the canonical Ewald correction is appended; for open boxes Ewald is skipped.
- `LeapfrogStepper.hpp` — KDK kernels: `leapfrogKickHalf`, `leapfrogKick`, `leapfrogDrift`.
- `PeriodicWrap.hpp` — wrap-into-box kernel for periodic axes.
- `NBodyManager<T, 3>` — IPPL `BaseManager` subclass that owns the container + solver and exposes the per-step pieces (kickHalf, drift, wrap, solve). Concrete simulations derive from it (see `test/BH/`).
- `BeamDiagnostics.hpp` — shared per-particle reductions (mean velocity, temperature) used by both Landau and DIH drivers.

## MPI communicator note

cstone's `Domain` does not accept an `MPI_Comm` argument; it uses `MPI_COMM_WORLD` internally for collective operations (see e.g. `domain.sync()` calls to `MPI_Allreduce` in `extern/cstone/include/cstone/domain/domain.hpp:203,251,270`). Bridging to `ippl::Comm`'s communicator is therefore deferred until cstone exposes a stream/communicator parameter.
