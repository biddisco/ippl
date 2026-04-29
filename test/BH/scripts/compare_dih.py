#!/usr/bin/env python3
"""Compare BH-DIH CSV against the IPPL P3M reference.

Two reference formats are supported:
  (1) CSV — same schema as DIH_IPPL_BH.csv:
        step sigma_x sigma_y sigma_z emit_x emit_y emit_z T_x T_y T_z T_norm
  (2) Inform/stdout log from examples/collisions/P3MHeating, parsed by regex.
      Pass the captured log file via --ref-log; the script extracts
        RMS Beam Size: a , b , c
        RMS Emittance: a , b , c
        Temperature: ( a , b , c )
        L2-Norm of Temperature: t

Usage:
    compare_dih.py --bh data/DIH_IPPL_BH.csv --ref-log p3m.log
    compare_dih.py --bh data/DIH_IPPL_BH.csv --ref data/DIH_P3M.csv

Produces:
    <out>/dih_sigma.png        - sigma_x, sigma_y, sigma_z vs step
    <out>/dih_emittance.png    - emit_x, emit_y, emit_z vs step
    <out>/dih_temperature.png  - T_x, T_y, T_z, T_norm vs step
    <out>/dih_summary.txt      - final-state values for each run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLS = [
    "step", "sigma_x", "sigma_y", "sigma_z",
    "emit_x", "emit_y", "emit_z",
    "T_x", "T_y", "T_z", "T_norm",
]


@dataclass
class Curve:
    label: str
    df: pd.DataFrame


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    return df[COLS]


_RE_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_TRIPLE = rf"({_RE_NUM})\s*[, ]\s*({_RE_NUM})\s*[, ]\s*({_RE_NUM})"

_RE_SIGMA = re.compile(rf"RMS Beam Size:\s*{_TRIPLE}")
_RE_EMIT  = re.compile(rf"RMS Emittance:\s*{_TRIPLE}")
_RE_TEMP  = re.compile(rf"Temperature:\s*\(?\s*{_TRIPLE}\s*\)?")
_RE_TNORM = re.compile(rf"L2-Norm of Temperature:\s*({_RE_NUM})")
_RE_STEP  = re.compile(r"Step\s+(\d+)\s+Finished")


def parse_log(path: str) -> pd.DataFrame:
    """Parse a P3MHeating-style stdout log into a DataFrame with the COLS schema.

    The log is structured as repeated record blocks of (sigma, emit, temp, T_norm)
    in that order. Step numbers come from "Step N Finished" lines (post-block).
    The pre-loop block (initial state) gets step = -1 if present.
    """
    sigmas: list[tuple[float, float, float]] = []
    emits: list[tuple[float, float, float]] = []
    temps: list[tuple[float, float, float]] = []
    tnorms: list[float] = []
    step_nums: list[int] = []

    with open(path, "r") as f:
        for line in f:
            m = _RE_SIGMA.search(line)
            if m:
                sigmas.append(tuple(float(x) for x in m.groups()))
                continue
            m = _RE_EMIT.search(line)
            if m:
                emits.append(tuple(float(x) for x in m.groups()))
                continue
            m = _RE_TEMP.search(line)
            if m:
                temps.append(tuple(float(x) for x in m.groups()))
                continue
            m = _RE_TNORM.search(line)
            if m:
                tnorms.append(float(m.group(1)))
                continue
            m = _RE_STEP.search(line)
            if m:
                step_nums.append(int(m.group(1)))
                continue

    n = min(len(sigmas), len(emits), len(temps), len(tnorms))
    if n == 0:
        raise ValueError(f"{path}: no DIH-style records found")

    # Records may include one initial-state pre-loop block. Step labels from
    # 'Step N Finished' lines are appended *after* their block, so we have one
    # fewer step number than blocks if the initial block was emitted. Pad with
    # -1 in that case; otherwise number 0..n-1.
    if len(step_nums) == n - 1:
        step_arr = [-1] + step_nums
    elif len(step_nums) == n:
        step_arr = step_nums
    else:
        # Fallback: 0..n-1.
        step_arr = list(range(n))

    rows = []
    for i in range(n):
        sx, sy, sz = sigmas[i]
        ex, ey, ez = emits[i]
        tx, ty, tz = temps[i]
        rows.append({
            "step":    step_arr[i],
            "sigma_x": sx, "sigma_y": sy, "sigma_z": sz,
            "emit_x":  ex, "emit_y":  ey, "emit_z":  ez,
            "T_x":     tx, "T_y":     ty, "T_z":     tz,
            "T_norm":  tnorms[i],
        })
    return pd.DataFrame(rows, columns=COLS)


def plot_triple(curves: list[Curve], cols: tuple[str, str, str],
                outpath: str, title: str, ylog: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)
    for i, c in enumerate(curves):
        for ax, col in zip(axes, cols):
            ax.plot(c.df["step"], c.df[col], label=c.label, linewidth=1.3)
    for ax, col in zip(axes, cols):
        ax.set_ylabel(col)
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[-1].set_xlabel("step")
    axes[0].set_title(title)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_tnorm(curves: list[Curve], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in curves:
        ax.plot(c.df["step"], c.df["T_norm"], label=c.label, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("‖T‖₂")
    ax.set_title("Disorder-induced heating: temperature norm")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", required=True, help="DIH_IPPL_BH.csv from DisorderHeatingBH")
    ap.add_argument("--ref", help="reference CSV (same schema as --bh)")
    ap.add_argument("--ref-log", help="reference Inform/stdout log from P3MHeating")
    ap.add_argument("--out", default="dih_compare", help="output directory")
    ap.add_argument("--label-bh", default="BH (gridless)")
    ap.add_argument("--label-ref", default="P3M (IPPL)")
    args = ap.parse_args()

    if not args.ref and not args.ref_log:
        ap.error("must pass either --ref <csv> or --ref-log <log>")

    os.makedirs(args.out, exist_ok=True)
    bh = Curve(args.label_bh, load_csv(args.bh))
    if args.ref:
        ref = Curve(args.label_ref, load_csv(args.ref))
    else:
        ref = Curve(args.label_ref, parse_log(args.ref_log))
    curves = [ref, bh]

    plot_triple(curves, ("sigma_x", "sigma_y", "sigma_z"),
                os.path.join(args.out, "dih_sigma.png"),
                "DIH: RMS beam size", ylog=False)
    plot_triple(curves, ("emit_x", "emit_y", "emit_z"),
                os.path.join(args.out, "dih_emittance.png"),
                "DIH: RMS emittance", ylog=True)
    plot_triple(curves, ("T_x", "T_y", "T_z"),
                os.path.join(args.out, "dih_temperature.png"),
                "DIH: per-axis temperature", ylog=True)
    plot_tnorm(curves, os.path.join(args.out, "dih_tnorm.png"))

    summary_lines = ["DIH comparison", "=============="]
    for c in curves:
        if len(c.df) == 0:
            continue
        last = c.df.iloc[-1]
        summary_lines.append(
            f"{c.label}: final step={int(last['step'])}, "
            f"σ=({last['sigma_x']:.3g}, {last['sigma_y']:.3g}, {last['sigma_z']:.3g}), "
            f"ε=({last['emit_x']:.3g}, {last['emit_y']:.3g}, {last['emit_z']:.3g}), "
            f"‖T‖₂={last['T_norm']:.3g}"
        )

    if len(bh.df) and len(ref.df):
        # Heating ratio at the final common step.
        final_step = min(int(bh.df["step"].iloc[-1]), int(ref.df["step"].iloc[-1]))
        ref_T = ref.df.loc[ref.df["step"] == final_step, "T_norm"]
        bh_T  = bh.df.loc[bh.df["step"] == final_step, "T_norm"]
        if len(ref_T) and len(bh_T):
            r = float(bh_T.iloc[0]) / float(ref_T.iloc[0])
            summary_lines.append(
                f"\nFinal-step ‖T‖₂ ratio (BH / P3M) at step {final_step}: {r:.4f}"
            )

    summary = "\n".join(summary_lines) + "\n"
    print(summary)
    with open(os.path.join(args.out, "dih_summary.txt"), "w") as f:
        f.write(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
