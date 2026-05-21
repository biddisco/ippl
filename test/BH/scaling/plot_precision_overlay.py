#!/usr/bin/env python3
"""Precision-overlay aggregator plot for the Landau scaling sweep.

For each unique nranks value in the sweep CSV, render one subplot with:
  - BH(double), BH(mixed), BH(float) field-energy trajectories,
  - the PIC (same-Nt) reference,
  - the analytic Landau envelope E0·exp(2·gamma_L·t) anchored at the in-repo
    validation file's t=0 value (matches plot_per_run.py behaviour).

Layout: one subplot per nranks, ordered ascending, gridded into N_COLS columns.
Output: a single multi-page PDF (one page per nranks) for easy flipping.

Always overwrites the output PDF.
"""
from __future__ import annotations

import argparse
import os
from io import StringIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

GAMMA_L = -0.1533  # linear Landau rate, Maxwellian, k=0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VALIDATION = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "alpine", "validation",
                 "FieldLandau_valid_result.csv"))

PRECISIONS = ("double", "mixed", "float")
PRECISION_COLOR = {
    "double": "tab:red",
    "mixed":  "tab:orange",
    "float":  "tab:green",
}


def _load_field_csv(path: str) -> Optional[pd.DataFrame]:
    """Load an Ex_field_energy CSV (BH or PIC). Returns None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = "".join(ln.replace(",", " ") for ln in f)
    return pd.read_csv(StringIO(text), sep=r"\s+", engine="python")


def _envelope_e0(validation_path: str) -> Optional[float]:
    df = _load_field_csv(validation_path)
    if df is None or "Ex_field_energy" not in df.columns:
        return None
    try:
        return float(df["Ex_field_energy"].iloc[0])
    except (KeyError, ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="sweep CSV with leading 'precision' column")
    ap.add_argument("--data-root", required=True,
                    help="directory containing bh_<TAG>_<P>/ and pic_<TAG>/ subdirs")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--validation", default=DEFAULT_VALIDATION)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Read only what we need from the sweep CSV: distinct (nranks, N, grid)
    # tuples. We don't trust the full schema because plot_scaling.py already
    # tolerates malformed rows; here, mirror that — keep rows whose nranks
    # parses as int and whose precision is one of {double, mixed, float}.
    rows = []
    with open(args.csv) as f:
        lines = f.readlines()
    if not lines:
        print(f"empty CSV: {args.csv}")
        return 1
    header = lines[0].rstrip("\n").split(",")
    try:
        i_prec = header.index("precision")
        i_n    = header.index("nranks")
        i_N    = header.index("N")
        i_g    = header.index("grid")
    except ValueError as e:
        print(f"CSV header missing expected column: {e}")
        return 1
    for ln in lines[1:]:
        parts = ln.rstrip("\n").split(",")
        if len(parts) <= max(i_prec, i_n, i_N, i_g):
            continue
        if parts[i_prec] not in PRECISIONS:
            continue
        try:
            rows.append({
                "precision": parts[i_prec],
                "nranks":    int(parts[i_n]),
                "N":         int(parts[i_N]),
                "grid":      int(parts[i_g]),
            })
        except ValueError:
            continue
    if not rows:
        print(f"no usable rows in {args.csv}")
        return 1
    df = pd.DataFrame(rows)
    # Unique (nranks, N, grid) — PIC columns repeat across precisions, so
    # collapsing on (nranks) is what we want; assert N/grid match.
    grouped = df.groupby("nranks").agg({"N": "first", "grid": "first"}).reset_index()
    grouped = grouped.sort_values("nranks").reset_index(drop=True)

    e0 = _envelope_e0(args.validation)

    with PdfPages(args.out) as pdf:
        for _, g in grouped.iterrows():
            nranks = int(g["nranks"])
            N      = int(g["N"])
            grid   = int(g["grid"])
            npm    = f"{N/1e6:.4g}"
            tag    = f"{args.mode}_{nranks}_{npm}M"

            fig, ax = plt.subplots(figsize=(8, 5))

            t_max = 0.0
            for p in PRECISIONS:
                bh_csv = os.path.join(args.data_root, f"bh_{tag}_{p}", "bh.csv")
                bh_df  = _load_field_csv(bh_csv)
                if bh_df is None or "Ex_field_energy" not in bh_df.columns:
                    continue
                ax.semilogy(bh_df["time"], np.abs(bh_df["Ex_field_energy"]),
                            label=f"BH ({p})", color=PRECISION_COLOR[p],
                            linewidth=1.4)
                t_max = max(t_max, float(bh_df["time"].max()))

            pic_csv = os.path.join(args.data_root, f"pic_{tag}", "pic_eqsteps.csv")
            pic_df  = _load_field_csv(pic_csv)
            if pic_df is not None and "Ex_field_energy" in pic_df.columns:
                ax.semilogy(pic_df["time"], np.abs(pic_df["Ex_field_energy"]),
                            label="PIC (same Nt)", color="tab:blue",
                            linewidth=1.6)
                t_max = max(t_max, float(pic_df["time"].max()))

            if e0 is not None and t_max > 0.0:
                t_env = np.linspace(0.0, t_max, 200)
                ax.semilogy(t_env, e0 * np.exp(2 * GAMMA_L * t_env),
                            "k:", lw=1.2,
                            label=rf"$E_0\,e^{{2\gamma_L t}}$, $\gamma_L={GAMMA_L}$")

            ax.set_xlabel("simulated time")
            ax.set_ylabel(r"$\int E_x^2\,dV$")
            ax.set_title(f"{args.mode} scaling — precision overlay, "
                         f"{nranks} ranks, N={npm}M, grid={grid}^3")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
