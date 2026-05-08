# BH simulation comparison scripts

Post-processing for the gridless Barnes-Hut drivers in `test/BH/`. Each
script reads a CSV produced by the BH driver plus a reference produced by an
IPPL PIC solver, overlay-plots the diagnostic curves, fits the relevant
physical observable, and writes a one-page summary.

## Dependencies

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib`. No SciPy is required —
the Landau γ_L fit uses `numpy.polyfit` on log-transformed peaks.

## Landau damping — `compare_landau.py`

Compares `LandauDampingBH` (gridless BH) vs `alpine/LandauDamping` (FFT-PIC)
on the canonical k=0.5, α=0.05 setup. Both drivers emit the same CSV schema:

```
time Ex_field_energy Ex_max_norm Ex_cos_amp meanVx meanVy meanVz Tx Ty Tz
```

### Generate reference

From the build tree:

```sh
# FFT-PIC reference (alpine):
cd build/alpine
srun -n 2 ./LandauDamping 16 16 16 10000000 25 FFT 0.01 LeapFrog \
    --overallocate 2.0 --info 10
# → data/FieldLandau_2_manager.csv

# BH (this work):
cd build/test/BH
srun -n 1 ./LandauDampingBH 1000000 600 0.5
# → data/FieldLandauBH_1000000.csv
```

(Choose `Nt` so `Nt · dt = 0.05 · Nt` covers several wave periods. ω≈1.4156
gives a period ≈ 4.4, so Nt = 600 ⇒ t_end = 30 covers ~7 periods.)

### Run the comparison

```sh
python3 test/BH/scripts/compare_landau.py \
    --bh  build-NBody/test/BH/data/FieldLandauBH_1000000.csv \
    --ref build-NBody/alpine/data/FieldLandau_2_manager.csv \
    --out landau_compare
```

Outputs:

- `landau_field_energy.png` — log-scale ∫E_x² dV vs t.
- `landau_kAmp.png`         — log-scale `|Ex_cos_amp|` envelope vs t.
- `landau_meanV.png`        — `meanVx,Vy,Vz` drift (zero in a perfect
  neutralised solver; non-zero indicates a residual `<E>` leaking into the
  kick).
- `landau_summary.txt`      — γ_L fit by log-linear regression on the local
  peaks of `|Ex_cos_amp|`, compared to theory γ_L = -0.1533.

The fit window defaults to `t ∈ [5, 25]`; override with
`--fit-window 8,28`.

## Disorder-induced heating — `compare_dih.py`

Compares `DisorderHeatingBH` (BH) vs `examples/collisions/P3MHeating` (P3M).
The BH driver writes `data/DIH_IPPL_BH.csv` directly:

```
step sigma_x sigma_y sigma_z emit_x emit_y emit_z T_x T_y T_z T_norm
```

P3MHeating doesn't emit a CSV — it prints DIH-style stats to stdout via
`Inform`. Capture stdout to a file and pass it as `--ref-log`:

```sh
cd build/examples/collisions
srun -n 1 ./P3MHeating 32 32 32 4 --info 10 > p3m.log 2>&1
# → DIH-style records in p3m.log

cd ../../test/BH
srun -n 1 ./DisorderHeatingBH /path/to/dih_initial_positions.bin
# → data/DIH_IPPL_BH.csv

python3 ../../test/BH/scripts/compare_dih.py \
    --bh      data/DIH_IPPL_BH.csv \
    --ref-log ../../examples/collisions/p3m.log \
    --out     dih_compare
```

Alternatively, if the P3M driver has been adapted to emit a CSV with the same
schema as the BH driver, pass it via `--ref` instead of `--ref-log`.

Outputs:

- `dih_sigma.png`        — RMS beam size σ_x, σ_y, σ_z vs step.
- `dih_emittance.png`    — RMS emittance ε_x, ε_y, ε_z vs step (log scale).
- `dih_temperature.png`  — per-axis temperature T_x, T_y, T_z vs step.
- `dih_tnorm.png`        — ‖T‖₂ vs step (this is the heating diagnostic).
- `dih_summary.txt`      — final-state values and the BH/P3M temperature
  ratio at the last common step.

## Notes

- The Landau γ_L fit uses log-linear regression on **peaks** of `|kAmp|`
  rather than on `Ex_field_energy`, because kAmp isolates the perturbation
  Fourier mode and is essentially noise-free; the energy estimator carries
  shot noise that biases the fit at small amplitudes. Both curves are still
  plotted for inspection.
- `meanV*` should stay near zero for a correctly neutralised periodic
  solver. A linear drift indicates the BH-side k=0 mode of E is non-zero —
  that's the diagnostic for the missing P2P uniform-background correction
  discussed in `project_phase3_status.md`.
- The DIH log parser assumes the P3MHeating record order
  (sigma → emit → temp → T_norm) is unchanged. Verify with `head` on the
  log if records seem misaligned.
