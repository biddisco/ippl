# BH comparison scripts

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib`. `conda activate <env>`
before invoking. See [`test/BH/README.md`](../README.md) for the full
reproducer flow.

## Landau

```sh
python3 test/BH/scripts/compare_landau.py --bh <bh.csv> --ref <pic.csv> --out landau_compare
```

Plots in `landau_compare/`.

## DIH

```sh
python3 test/BH/scripts/compare_dih.py --bh <bh.csv> --ref-log <p3m.log> --out dih_compare
```

Plots in `dih_compare/`.
