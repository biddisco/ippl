#!/usr/bin/env python3
"""Scaling plot — precision sweep edition.

Renders BH(double|mixed|float) + PIC(same Nt) + PIC(same t) as five
series per panel. Sweep CSV's first column is `precision`; PIC columns
repeat across the three precision rows for a given nranks (they are
properties of (mode, nranks), not (precision, nranks) — see the per-mode
worker run_bh.sbatch which reads PIC timings from pic_<TAG>/pic_timings.env).

PIC variants:
  - pic_eqsteps_*: same Nt as BH (less simulated time when dt_PIC < dt_BH).
  - pic_eqtime_* : Nt scaled so simulated time matches BH.
The two coincide when dt_PIC == dt_BH; in that case the duplicate run was
skipped at sweep time and both columns hold identical values.

Always overwrites the output PDF (plt.savefig with no existence guard).
"""
from __future__ import annotations

import argparse
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PRECISIONS = ("double", "mixed", "float")
PRECISION_COLOR = {
    "double": "tab:red",
    "mixed":  "tab:orange",
    "float":  "tab:green",
}
PRECISION_MARKER = {
    "double": "o-",
    "mixed":  "s-",
    "float":  "D-",
}


def _load_clean_csv(path: str) -> pd.DataFrame:
    """Tolerant CSV reader for the precision-sweep schema.

    First column is `precision` ∈ {double, mixed, float}; second is `nranks`.
    Older variants of run_one.sbatch leaked PIC stdout into the aggregate CSV;
    we keep any row whose first field is a recognized precision and whose
    second field parses as an integer (nranks). Pads short rows / truncates
    long ones to the header width.
    """
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].rstrip("\n")
    n_cols = header.count(",") + 1
    kept = [lines[0]]
    for ln in lines[1:]:
        parts = ln.rstrip("\n").split(",")
        if len(parts) < 2 or parts[0] not in PRECISIONS:
            continue
        try:
            int(parts[1])
        except ValueError:
            continue
        if len(parts) < n_cols:
            parts = parts + [""] * (n_cols - len(parts))
        elif len(parts) > n_cols:
            parts = parts[:n_cols]
        kept.append(",".join(parts) + "\n")
    return pd.read_csv(StringIO("".join(kept)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = _load_clean_csv(args.csv)
    numeric_cols = [c for c in df.columns
                    if c.endswith("_s") or c in ("nranks", "nodes",
                                                  "gpus_per_node", "N", "grid")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["nranks"]).copy()
    df["nranks"] = df["nranks"].astype(int)
    df = df.sort_values(["precision", "nranks"]).drop_duplicates(
        ["precision", "nranks"]).reset_index(drop=True)

    if len(df) == 0:
        print(f"no usable rows in {args.csv}")
        return 1

    # Step-timer columns; fall back to wall-clock if timing.dat parsing failed.
    def col(stem: str) -> str:
        return f"{stem}_step_s" if f"{stem}_step_s" in df.columns else f"{stem}_wall_s"

    bh_col       = col("bh")
    pic_eqsts_c  = col("pic_eqsteps")
    pic_eqtime_c = col("pic_eqtime")

    # Per-precision BH frames, indexed by nranks for clean joins.
    bh = {p: df[df["precision"] == p].sort_values("nranks").reset_index(drop=True)
          for p in PRECISIONS}
    # PIC frame: deduplicate on nranks — PIC columns are identical across the
    # three precision rows for a given nranks (set in run_bh.sbatch from
    # pic_timings.env). Picking one is enough.
    pic = df.drop_duplicates("nranks").sort_values("nranks").reset_index(drop=True)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- panel 1: step time vs rank count ----
    for p in PRECISIONS:
        d = bh[p]
        if len(d) == 0 or bh_col not in d.columns:
            continue
        ax_t.loglog(d["nranks"], d[bh_col], PRECISION_MARKER[p],
                    label=f"BH ({p})", color=PRECISION_COLOR[p], lw=1.5, ms=6)
    if pic_eqsts_c in pic.columns:
        ax_t.loglog(pic["nranks"], pic[pic_eqsts_c], "s-",
                    label="PIC (same Nt)", color="tab:blue", lw=1.5, ms=6)
    if pic_eqtime_c in pic.columns:
        ax_t.loglog(pic["nranks"], pic[pic_eqtime_c], "^--",
                    label="PIC (same t)", color="tab:blue", lw=1.5, ms=6)

    # Ideal references anchored at the BH(double) 1-rank baseline (and at
    # the PIC same-Nt 1-rank baseline). Keep the plot legible — one ideal
    # per family, not per precision.
    def _ideal(ax, d, c, color, label):
        if c not in d.columns or len(d[c].dropna()) == 0:
            return
        base = float(d[c].dropna().iloc[0])
        n_arr = d["nranks"].to_numpy(dtype=float)
        if args.mode == "strong":
            ax.loglog(n_arr, base / n_arr, "--", color=color, alpha=0.35,
                      lw=1.0, label=f"ideal {label} (1/N)")
        else:
            ax.axhline(base, ls="--", color=color, alpha=0.35, lw=1.0,
                       label=f"ideal {label} (flat)")
    if len(bh["double"]) > 0:
        _ideal(ax_t, bh["double"], bh_col, "tab:red", "BH (double)")
    _ideal(ax_t, pic, pic_eqsts_c, "tab:blue", "PIC")

    ax_t.set_xlabel("MPI ranks")
    ax_t.set_ylabel("step time [s] (advanceImpl, dumps excluded)")
    ax_t.set_title(f"{args.mode} scaling — step time — precision sweep")
    xticks = sorted(pic["nranks"].unique().tolist())
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels([str(int(x)) for x in xticks])
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(loc="best", fontsize=8)

    # ---- panel 2: parallel efficiency ----
    if args.mode == "strong":
        def eff(d: pd.DataFrame, c: str) -> pd.Series:
            base = float(d[c].dropna().iloc[0])
            return (base / d[c]) / d["nranks"]
        eff_label = "strong efficiency  T(1)/(N·T(N))"
    else:
        def eff(d: pd.DataFrame, c: str) -> pd.Series:
            base = float(d[c].dropna().iloc[0])
            return base / d[c]
        eff_label = "weak efficiency  T(1)/T(N)"

    eff_max = 1.1
    for p in PRECISIONS:
        d = bh[p]
        if len(d) == 0 or bh_col not in d.columns or d[bh_col].dropna().empty:
            continue
        e = eff(d, bh_col)
        ax_s.semilogx(d["nranks"], e, PRECISION_MARKER[p],
                      label=f"BH ({p})", color=PRECISION_COLOR[p], lw=1.5, ms=6)
        eff_max = max(eff_max, float(e.max()))
    for label, c, marker in (("PIC (same Nt)", pic_eqsts_c, "s-"),
                             ("PIC (same t)",  pic_eqtime_c, "^--")):
        if c not in pic.columns or pic[c].dropna().empty:
            continue
        e = eff(pic, c)
        ax_s.semilogx(pic["nranks"], e, marker,
                      label=label, color="tab:blue", lw=1.5, ms=6)
        eff_max = max(eff_max, float(e.max()))

    ax_s.axhline(1.0, color="k", ls=":", alpha=0.5, lw=1.0)
    ax_s.set_xlabel("MPI ranks")
    ax_s.set_ylabel(eff_label)
    ax_s.set_title(f"{args.mode} scaling — efficiency — precision sweep")
    ax_s.set_xticks(xticks)
    ax_s.set_xticklabels([str(int(x)) for x in xticks])
    ax_s.set_ylim(bottom=0.0, top=eff_max * 1.05)
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(loc="best", fontsize=8)

    fig.suptitle(f"Landau damping — {args.mode} scaling — precision sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("\nruntimes table:")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
