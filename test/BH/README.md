# `test/BH/` — Barnes-Hut driver workflows

End-to-end recipes for the gridless Barnes-Hut drivers shipped here, plus the
PIC reference runs they're benchmarked against. For the architecture of the
underlying library see [`src/NBody/README.md`](../../src/NBody/README.md);
this file is purely a how-to.

The two test cases are:

- **Landau damping** — periodic plasma; BH driver `LandauDampingBH` vs FFT-PIC
  driver `alpine/LandauDamping`.
- **Disorder-induced heating (DIH)** — open-BC cold beam; BH driver
  `DisorderHeatingBH` vs P3M driver `examples/collisions/P3MHeating`.

## Build

The NBody backend requires CUDA. The configuration below builds both BH
drivers, both PIC references, and the comparison post-processing — i.e. all
artefacts referenced in the run recipes below.

On Daint, start the user environment:
```
uenv start --view=default prgenv-gnu/25.6:v2
```

Then build and compile enabling NBody
```sh
# From repo root, inside the uenv shell. See INSTALLATION.md for env setup.
cmake -S . -B build-NBody \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIPPL_PLATFORMS="CUDA;OPENMP" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ARCH_NEOVERSE_V2=ON \
  -DCMAKE_CXX_COMPILER=g++ \
  -DIPPL_ENABLE_NBODY=ON \
  -DIPPL_ENABLE_ALPINE=ON \
  -DIPPL_ENABLE_EXAMPLES=ON \
  -DIPPL_ENABLE_TESTS=ON \
  -DIPPL_ENABLE_FFT=ON \
  -DIPPL_ENABLE_SOLVERS=ON

cd build-NBody && make -j32
```

After the build the binaries land at:

```
build-NBody/test/BH/LandauDampingBH
build-NBody/test/BH/DisorderHeatingBH
build-NBody/alpine/LandauDamping
build-NBody/examples/collisions/P3MHeating
```

## Landau damping

Linear k=0.5, α=0.05 perturbation in a [0, 4π]³ periodic box.

### Run BH

```sh
cd build-NBody/test/BH
./LandauDampingBH 1000000 200 0.5
# → data/FieldLandauBH_1000000.csv   (201 rows, t ∈ [0, 10])
```

CLI: `LandauDampingBH <N> <Nt> <theta> [seed] [numShells] [smoothH] [cicGrid]`.

### Run FFT-PIC reference

```sh
cd build-NBody/alpine
./LandauDamping 32 32 32 1000000 200 FFT 0.01 LeapFrog \
    --overallocate 2.0 --info 10
# → data/FieldLandau_1_manager.csv
```

### Compare and plot

```sh
cd build-NBody  # or anywhere — paths below are absolute when needed
python3 ../test/BH/scripts/compare_landau.py \
    --bh   test/BH/data/FieldLandauBH_1000000.csv \
    --ref  alpine/data/FieldLandau_1_manager.csv \
    --out  landau_compare
```

`--validation` defaults to the in-repo golden curve at
`alpine/validation/FieldLandau_valid_result.csv`; the analytical
`E₀ · exp(2γ_L t)` envelope is anchored to that file's `t=0` value rather
than to the PIC run's, so envelope agreement at `t=0` is no longer automatic
by construction.

Outputs (in `landau_compare/`):

- `landau_field_energy.png` — log-scale `∫ Ex² dV` vs `t` showing the raw BH
  particle estimator, the CIC-smoothed BH estimator (band-limited at the
  PIC Nyquist), the FFT-PIC trajectory, the validation reference, and the
  analytical envelope.
- `landau_summary.txt` — t=0 ratios of each curve to the validation
  reference.

## Disorder-induced heating

156055 electrons in a 1 cm cold-cloud configuration with a constant linear
focusing field. Open BCs.

### Get the initial-position file

`DisorderHeatingBH` reads positions from a binary blob exported by an
instrumented `P3MHeating` run; the schema is `uint64_t N` followed by
`double[N]` for x, y, z. To produce one, add a small `fwrite` block at the
end of `P3MHeating::initParticles` (uint64 count + three contiguous double
arrays for x, y, z). Cross-check that the leading `N` matches the hardcoded
`np = 156055` in `examples/collisions/P3MHeating.cpp`:

```sh
head -c 8 dih_initial_positions.bin | od -An -tu8
```

### Run BH

```sh
cd build-NBody/test/BH
srun -n 1 ./DisorderHeatingBH /path/to/dih_initial_positions.bin 1000
# → data/DIH_IPPL_BH.csv
```

CLI: `DisorderHeatingBH <dih_initial_positions.bin> [Nt]`. The path is
required.

### Run P3M reference

`P3MHeating` doesn't emit a CSV — it prints DIH-style records via `Inform`.
Capture them to a log:

```sh
cd build-NBody/examples/collisions
srun -n 1 ./P3MHeating 32 32 32 4 --info 10 > p3m.log 2>&1
```

CLI: `P3MHeating <nx> <ny> <nz> <factor>`. Hardcoded inside the driver:
`np = 156055`, `nt = 1000`, `dt = 2.156e-13 s`, `boxlen = 0.01 m`,
`beam_rad = 0.001774 m`, `focus_strength = 1.5`. `factor = 4` puts the
P2P cutoff at `4 · boxlen / nx`.

### Compare and plot

```sh
cd build-NBody
python3 ../test/BH/scripts/compare_dih.py \
    --bh      test/BH/data/DIH_IPPL_BH.csv \
    --ref-log examples/collisions/p3m.log \
    --out     dih_compare
```

`compare_dih.py` parses sigma / emittance / temperature / `‖T‖₂` lines from
the P3M log (regex on `Inform` output), step-aligns with the BH CSV, and
writes (in `dih_compare/`):

- `dih_sigma.png` — RMS beam size σ_x/y/z vs step
- `dih_emittance.png` — RMS emittance ε_x/y/z vs step (log axis)
- `dih_temperature.png` — per-axis temperature T_x/y/z vs step (log axis)
- `dih_tnorm.png` — `‖T‖₂` vs step (the heating diagnostic)
- `dih_summary.txt` — final-state values and the BH/P3M `‖T‖₂` ratio at
  the last common step

The expected outcome is a final-step `BH / P3M` ratio of `‖T‖₂` close to
`1.0` and overlapping σ/ε trajectories.

## Plot script dependencies

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib`. No SciPy required. See
[`scripts/README.md`](scripts/README.md) for a deeper walkthrough of the
comparison scripts and their CLI flags.
