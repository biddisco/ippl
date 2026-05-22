# `test/BH/` — Barnes-Hut reproducer

`$BUILD` below is your chosen cmake build directory (e.g. `build`,
`build-NBody`, …). For the weak-scaling sweep below, either name it
`build` or pass `BUILD_DIR=$PWD/$BUILD` to `submit_weak.sh`.

## Build

```sh
uenv start --view=default prgenv-gnu/26.3:v1
cmake -S . -B $BUILD -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIPPL_PLATFORMS="CUDA;OPENMP" -DCMAKE_CUDA_ARCHITECTURES=90 -DKokkos_ARCH_HOPPER90=ON -DKokkos_ARCH_NEOVERSE_V2=ON -DCMAKE_CXX_COMPILER=g++ -DIPPL_ENABLE_NBODY=ON -DIPPL_ENABLE_ALPINE=ON -DIPPL_ENABLE_EXAMPLES=ON -DIPPL_ENABLE_TESTS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_SOLVERS=ON -DCSTONE_WITH_GPU_AWARE_MPI=ON
cd $BUILD && make -j32
```

Runtime requires `MPICH_GPU_SUPPORT_ENABLED=1`. Slurm GPU jobs must use
`--ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=72`. Set
`ACCOUNT=<slurm account>` before any `submit_*.sh`. `conda activate <env>`
before any `python3` invocation.

## Landau damping

```sh
cd $BUILD/test/BH
srun -n 1 ./LandauDampingBH 1000000 200 0.5
cd $BUILD/alpine
srun -n 1 ./LandauDamping 32 32 32 1000000 200 FFT 0.01 LeapFrog --overallocate 2.0 --info 10
cd $BUILD
conda activate <env>
python3 ../test/BH/scripts/compare_landau.py --bh test/BH/data/FieldLandauBH_1000000.csv --ref alpine/data/FieldLandau_1_manager.csv --out landau_compare
```

Plots: `$BUILD/landau_compare/`.

## Landau weak scaling

```sh
ACCOUNT=<slurm account> BUILD_DIR=$PWD/$BUILD bash test/BH/scaling/submit_weak.sh
```

Plots: `$BUILD/scaling/data/weak_scaling.pdf`, `weak_precision_overlay_<TS>.pdf`, per-run `bh_weak_<nranks>_<NPM>M_<precision>.pdf`.

## Disorder-induced heating

```sh
cd $BUILD/test/BH
srun -n 1 ./DisorderHeatingBH gen 156055 1000 42
cd $BUILD/examples/collisions
srun -n 1 ./P3MHeating <hx> <hy> <hz> 4 --info 10 > p3m.log 2>&1
cd $BUILD
conda activate <env>
python3 ../test/BH/scripts/compare_dih.py --bh test/BH/data/DIH_IPPL_BH.csv --ref-log examples/collisions/p3m.log --out dih_compare
```

Plots: `$BUILD/dih_compare/`.
