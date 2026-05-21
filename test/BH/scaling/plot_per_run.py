#!/usr/bin/env python3
"""Per-configuration BH vs PIC field-energy trajectory plot.

Reads three CSVs (BH, PIC same-Nt, PIC same-t) and overlays them on a single
log-scale `Ex_field_energy` vs `time` axis. The two PIC variants come from
the surrounding scaling sweep:
  - pic-eqsteps: PIC ran with the same number of timesteps as BH; covers
                 less simulated time when dt_PIC < dt_BH.
  - pic-eqtime : PIC ran with Nt scaled so the final simulated time matches
                 BH. Reflects the cost of pinning physical-time coverage.
The two coincide when dt_PIC == dt_BH (G ≤ ~126 for the Landau box). Both
PIC curves are still plotted; the dashed eqtime curve simply extends past
the eqsteps curve when distinct.

Always overwrites the output PDF (plt.savefig with no existence guard) so
re-runs of the same TAG never display stale figures.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAMMA_L = -0.1533  # linear Landau rate, Maxwellian, k=0.5

# Validation reference used to anchor the analytical envelope at t=0. Same
# golden file consumed by test/BH/scripts/compare_landau.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VALIDATION = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "alpine", "validation",
                 "FieldLandau_valid_result.csv")
)


@dataclass
class Curve:
    label: str
    df: pd.DataFrame
    color: str
    linestyle: str = "-"
    alpha: float = 1.0
    linewidth: float = 1.4


def load(path: str) -> pd.DataFrame:
    # Alpine writes comma-separated headers + space-separated rows; BH writes
    # everything space-separated. Strip stray commas and tokenize on whitespace.
    with open(path) as f:
        text = "".join(ln.replace(",", " ") for ln in f)
    return pd.read_csv(StringIO(text), sep=r"\s+", engine="python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh",          required=True)
    ap.add_argument("--pic-eqsteps", required=True,
                    help="PIC CSV from the same-Nt run")
    ap.add_argument("--pic-eqtime",  required=True,
                    help="PIC CSV from the same-simulated-time run")
    ap.add_argument("--validation",  default=DEFAULT_VALIDATION,
                    help="golden CSV for the t=0 E_0 anchor of the analytical envelope")
    ap.add_argument("--out",         required=True)
    ap.add_argument("--title",       default="")
    args = ap.parse_args()

    bh         = load(args.bh)
    pic_eqsts  = load(args.pic_eqsteps)
    pic_eqtime = load(args.pic_eqtime)

    fig, ax = plt.subplots(figsize=(8, 5))

    curves: list[Curve] = []

    # PIC same-Nt and same-t. Plot eqtime first as dashed so it sits beneath
    # the solid eqsteps line where they overlap (visually clean when they
    # coincide because dt_PIC == dt_BH).
    curves.append(Curve("PIC (same Nt)", pic_eqsts,  "tab:blue", "-", 1.0, 1.6))
    if not pic_eqtime.equals(pic_eqsts):
        curves.append(Curve("PIC (same t)",  pic_eqtime, "tab:blue", "--", 0.9, 1.4))

    # BH raw (always present) and BH CIC (only if the column was enabled).
    curves.append(Curve("BH (raw MC)", bh, "tab:orange", "-", 0.85, 1.2))
    if "Ex_field_energy_cic" in bh.columns:
        # Build a separate Curve so we can index a different column below.
        df_cic = bh[["time", "Ex_field_energy_cic"]].rename(
            columns={"Ex_field_energy_cic": "Ex_field_energy"})
        curves.append(Curve("BH (CIC-smoothed)", df_cic, "tab:red", "-", 1.0, 1.6))

    for c in curves:
        if "Ex_field_energy" not in c.df.columns:
            continue
        ax.semilogy(c.df["time"], np.abs(c.df["Ex_field_energy"]),
                    label=c.label,
                    color=c.color,
                    linestyle=c.linestyle,
                    alpha=c.alpha,
                    linewidth=c.linewidth)

    # Theory envelope. Anchor at the in-repo validation reference's t=0
    # value (NOT at PIC's t=0) so envelope agreement is a real check rather
    # than a free pass.
    try:
        val = load(args.validation)
        e0  = float(val["Ex_field_energy"].iloc[0])
        # Draw over the longer time axis so the dashed eqtime curve gets a
        # backdrop even when it extends past BH's final t (it doesn't —
        # Nt_eqtime is chosen to match BH — but be robust to rounding).
        t_max = max(float(bh["time"].max()),
                    float(pic_eqsts["time"].max()),
                    float(pic_eqtime["time"].max()))
        t_env = np.linspace(0.0, t_max, 200)
        ax.semilogy(t_env, e0 * np.exp(2 * GAMMA_L * t_env),
                    "k:", lw=1.2,
                    label=rf"$E_0\,e^{{2\gamma_L t}}$, $\gamma_L={GAMMA_L}$")
    except (FileNotFoundError, KeyError, ValueError):
        pass  # missing validation file or column → skip envelope, plot the rest

    ax.set_xlabel("simulated time")
    ax.set_ylabel(r"$\int E_x^2\,dV$")
    ax.set_title(args.title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    # Always overwrite — re-runs of the same TAG should never see a stale PDF.
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
