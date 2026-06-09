# `src/NBody/` — gridless Barnes-Hut module

Producer-side home for IPPL's gridless N-body backend, built on vendored
cstone (`extern/cstone/`) and Ryoanji (`extern/ryoanji/`). Gated by
`IPPL_ENABLE_NBODY=ON` (requires CUDA).

Reproducer: [`test/BH/README.md`](../../test/BH/README.md).

## Contents

Top level holds the major class holders; supporting code lives in subdirectories.

- `NBodyParticleContainer<P, 3>` — gridless particle container wrapping `cstone::Domain`. Attributes: `Rx,Ry,Rz` (Tc), `h` (Th), id, charge (Tm), `Px,Py,Pz` (Tc), `Ex,Ey,Ez` (Ta). `update()` syncs the domain; `updateGrav()` also exchanges charge halos and refreshes focus-tree centers.
- `NBodySolver<P, 3>` — BH field solver. Local tree build, P2M+M2M upsweep, traverse to write `Ex,Ey,Ez`. Periodic: Ewald appended. Open: skipped.
- `NBodyManager<P, 3>` — `BaseManager` subclass owning container + solver.

`core/` — compile-time policies and container internals.
- `BHPrecision.hpp` — precision policy tags (`Double|Mixed|Float`Precision); each bundles ryoanji's five scalar template parameters plus the multipole value type.
- `Accelerator.hpp` — backend tag (`NBodyAcc`), `kHaveGpu`, and the `FieldVector<T>` storage alias.
- `BHFieldLists.hpp` — `fields::StdConserved` / `fields::StdDependent` sync selections.
- `ParticleAttrib.hpp`, `ParticleAttribBase.hpp` — runtime user-attribute storage.
- `NBodySync.cpp` — Kokkos-free g++ TU defining `syncGravBH` / `updateBH`.

`physics/` — per-step operations over the owned range.
- `LeapfrogStepper.hpp` — KDK kernels (`leapfrogKickHalf`, `leapfrogKick`, `leapfrogDrift`); the drift folds the periodic wrap (`cstone::putInBox`) into the position update.
- `BeamDiagnostics.hpp` — per-particle reductions (mean velocity, temperature).

`wrappers/` — adapters over external libraries.
- `GravityWrapper.hpp` — ryoanji `MultipoleHolder` adapter.
- `NBodyKokkosView.hpp` — unmanaged `Kokkos::View` accessors for Kokkos/`.cu` consumers.

`helpers/` — low-level primitives.
- `primitives_gpu.{h,cu}` — GPU reductions and element-wise maps (axpy3, drift+wrap).
- `GpuTimer.hpp` — CUDA-event timer.
