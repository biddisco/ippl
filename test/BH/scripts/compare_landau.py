#!/usr/bin/env python3
"""Compare BH-Landau CSV against the FFT-PIC reference + golden validation curve.

Usage:
    compare_landau.py --bh <FieldLandauBH_*.csv> --ref <FieldLandau_*_manager.csv>
                      [--validation <path>] [--out <dir>]
                      [--label-bh <str>] [--label-ref <str>] [--label-val <str>]

Reads from each CSV the time and Ex_field_energy columns; from the BH CSV also
Ex_field_energy_cic (CIC-smoothed apples-to-apples curve, when present).

Overlays the linear Landau-damping envelope on the field-energy plot:
    E(t) = E_0 · exp(2·gamma_L·t)
with gamma_L = -0.1533 (linear Landau rate, Maxwellian, k=0.5). E_0 is
anchored at the **validation** curve's t=0 value (the regression-test golden
reference at alpine/validation/FieldLandau_valid_result.csv), so envelope
agreement with PIC or BH at t=0 is no longer automatic by construction.

Produces:
    <out>/landau_field_energy.png  -- log-scale Ex_field_energy vs t (raw BH,
                                       CIC-smoothed BH if available, FFT-PIC,
                                       validation reference, analytical envelope)
    <out>/landau_summary.txt       -- t=0 field-energy ratios (calibration)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Linear Landau-damping rate for a Maxwellian f_0 at k=0.5 (plasma units;
# matches alpine + the test/BH driver).
GAMMA_L_THEORY = -0.1533

# Default location of the in-repo validation reference (golden curve used by
# alpine/validation/LandauDampingCorrectness with 40% tolerance).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_VALIDATION = os.path.join(
    REPO_ROOT, "alpine", "validation", "FieldLandau_valid_result.csv"
)


@dataclass
class Curve:
    label: str
    df: pd.DataFrame


def load(path: str) -> pd.DataFrame:
    # Upstream alpine writes a comma-separated header but space-separated rows
    # (`Inform csvout << "time, Ex_field_energy, Ex_max_norm" << endl;` then
    # `csvout << t << " " << ...`). Strip stray commas before tokenising.
    with open(path, "r") as f:
        lines = [ln.replace(",", " ") for ln in f]
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)), sep=r"\s+", engine="python")
    if "time" not in df.columns or "Ex_field_energy" not in df.columns:
        raise ValueError(f"{path}: missing expected columns; got {list(df.columns)}")
    return df


def plot_curves(curves: list[Curve], col: str, ylabel: str, ylog: bool,
                outpath: str, title: str,
                extra: list[tuple[str, str, np.ndarray, np.ndarray]] | None = None) -> None:
    """Plot one column from each curve. `extra` adds (label, source-label, t, y) lines."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in curves:
        if col not in c.df.columns:
            continue
        y = c.df[col].to_numpy()
        if ylog:
            y = np.abs(y)
            y = np.where(y > 0, y, np.nan)
        ax.plot(c.df["time"], y, label=c.label, linewidth=1.4)
    if extra:
        for label, _src, t, y in extra:
            if ylog:
                y = np.abs(y)
                y = np.where(y > 0, y, np.nan)
            ax.plot(t, y, label=label, linewidth=1.4, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", required=True, help="FieldLandauBH_*.csv produced by LandauDampingBH")
    ap.add_argument("--ref", required=True, help="FieldLandau_*_manager.csv produced by alpine FFT-PIC")
    ap.add_argument("--validation", default=DEFAULT_VALIDATION,
                    help=f"Golden validation CSV (default: {DEFAULT_VALIDATION})")
    ap.add_argument("--out", default="landau_compare", help="output directory for plots/summary")
    ap.add_argument("--label-bh", default="BH (sampled from particles)")
    ap.add_argument("--label-ref", default="FFT-PIC (alpine)")
    ap.add_argument("--label-val", default="validation reference")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bh  = Curve(args.label_bh,  load(args.bh))
    ref = Curve(args.label_ref, load(args.ref))
    val = Curve(args.label_val, load(args.validation))
    curves = [ref, bh, val]

    # If the BH dump includes the CIC-smoothed energy column, overlay it as a
    # dashed line in the field-energy plot. This is the apples-to-apples curve
    # vs FFT-PIC's grid-integrated Ex_field_energy: it low-passes the BH field
    # at the same k_Nyquist that PIC's grid imposes by construction, dropping
    # the resolved short-range Coulomb / discreteness contribution that
    # otherwise puts a constant offset on the BH curve.
    extra_energy: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    if "Ex_field_energy_cic" in bh.df.columns:
        extra_energy.append(("BH (sampled on grid)", args.label_bh,
                             bh.df["time"].to_numpy(),
                             bh.df["Ex_field_energy_cic"].to_numpy()))

    # Linear-Landau envelope, anchored at the validation curve's t=0 value.
    # Using the validation rather than PIC means envelope agreement at t=0 is
    # no longer automatic by construction — both PIC and BH have to actually
    # match the seeded perturbation amplitude rather than just inheriting it.
    t_an   = bh.df["time"].to_numpy()
    e0_val = float(val.df["Ex_field_energy"].iloc[0])
    e_env  = e0_val * np.exp(2.0 * GAMMA_L_THEORY * t_an)
    extra_energy.append(
        (rf"theory: $E_0\,e^{{2\gamma_L t}}$, $\gamma_L={GAMMA_L_THEORY}$",
         "theory", t_an, e_env))

    plot_curves(curves, "Ex_field_energy", r"$\int E_x^2\,dV$ (estimator)",
                ylog=True, outpath=os.path.join(args.out, "landau_field_energy.png"),
                title="Landau damping: field energy",
                extra=extra_energy)

    summary_lines = [
        "Landau damping comparison",
        "=========================",
        f"BH:  {args.bh}         ({len(bh.df)} rows, t={bh.df['time'].iloc[0]:.3g}..{bh.df['time'].iloc[-1]:.3g})",
        f"Ref: {args.ref}        ({len(ref.df)} rows, t={ref.df['time'].iloc[0]:.3g}..{ref.df['time'].iloc[-1]:.3g})",
        f"Val: {args.validation} ({len(val.df)} rows, t={val.df['time'].iloc[0]:.3g}..{val.df['time'].iloc[-1]:.3g})",
        "",
        "Field-energy ratio at t=0 (vs validation E_0):",
    ]
    if len(bh.df) and len(val.df):
        e0_bh  = bh.df["Ex_field_energy"].iloc[0]
        e0_ref = ref.df["Ex_field_energy"].iloc[0]
        summary_lines.append(f"  raw          : {args.label_bh} / {args.label_val} = {e0_bh / e0_val:.4f}")
        summary_lines.append(f"  PIC          : {args.label_ref} / {args.label_val} = {e0_ref / e0_val:.4f}")
        if "Ex_field_energy_cic" in bh.df.columns:
            e0_cic = bh.df["Ex_field_energy_cic"].iloc[0]
            summary_lines.append(
                f"  CIC-smoothed : (BH CIC) / {args.label_val} = {e0_cic / e0_val:.4f}"
            )
        summary_lines.append("  (1.00 means agreement with the validation reference E_0; raw "
                             "carries the BH discreteness floor; CIC filters it; deviation of "
                             "PIC from 1.00 reflects FFT-PIC's grid resolution at the seeded "
                             "perturbation amplitude)")

    summary = "\n".join(summary_lines) + "\n"
    print(summary)
    with open(os.path.join(args.out, "landau_summary.txt"), "w") as f:
        f.write(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
