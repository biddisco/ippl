# `src/NBody/` — gridless Barnes-Hut module

Producer-side home for IPPL's gridless N-body backend, built on vendored
cstone (`extern/cstone/`) and Ryoanji (`extern/ryoanji/`). Gated by
`IPPL_ENABLE_NBODY=ON` (requires CUDA).

Reproducer: [`test/BH/README.md`](../../test/BH/README.md).

## Contents

- `BHPrecision.hpp` — precision policy tags (`Double|Mixed|Float`Precision); each bundles ryoanji's five scalar template parameters plus the multipole value type.
- `SphexaParticleContainer<P, 3>` — gridless particle container wrapping `cstone::Domain`. Attributes: `Rx,Ry,Rz` (Tc), `h` (Th), id, charge (Tm), `Px,Py,Pz` (Tc), `Ex,Ey,Ez` (Ta). `update()` syncs the domain; `updateGrav()` also exchanges charge halos and refreshes focus-tree centers.
- `SphexaBHSolver<P, 3>` — BH field solver. Local tree build, P2M+M2M upsweep, traverse to write `Ex,Ey,Ez`. Periodic: Ewald appended. Open: skipped.
- `LeapfrogStepper.hpp` — KDK kernels (`leapfrogKickHalf`, `leapfrogKick`, `leapfrogDrift`).
- `PeriodicWrap.hpp` — wrap-into-box kernel.
- `NBodyManager<P, 3>` — `BaseManager` subclass owning container + solver.
- `BeamDiagnostics.hpp` — per-particle reductions (mean velocity, temperature).
