#!/usr/bin/env bash
# Final aggregator: read all per-(precision, nranks) rows from the sweep CSV
# and produce two PDFs:
#   1. ${MODE}_scaling.pdf            — step time + efficiency, 3 precision
#                                       curves per panel (plot_scaling.py).
#   2. ${MODE}_precision_overlay_<TS>.pdf — per-nranks pages overlaying the
#                                       three BH precision trajectories with
#                                       the PIC reference and analytic envelope
#                                       (plot_precision_overlay.py).
# Runs on 1 rank, no GPU required.
#SBATCH --no-requeue

set -euo pipefail

MODE=$1
CSV=$2
SRC_DIR="${SCALING_SRC_DIR:?SCALING_SRC_DIR not set — invoke via submit_weak.sh}"
BUILD=${SLURM_SUBMIT_DIR:-${SRC_DIR}/../../../build}
cd "${BUILD}"

OUT_SCALING=${BUILD}/scaling/data/${MODE}_scaling.pdf
TS=$(date +%Y%m%d_%H%M%S)
OUT_OVERLAY=${BUILD}/scaling/data/${MODE}_precision_overlay_${TS}.pdf

python3 ${SRC_DIR}/plot_scaling.py \
  --csv ${CSV} \
  --mode ${MODE} \
  --out ${OUT_SCALING}

python3 ${SRC_DIR}/plot_precision_overlay.py \
  --csv       ${CSV} \
  --data-root ${BUILD}/scaling/data \
  --mode      ${MODE} \
  --out       ${OUT_OVERLAY}

echo "wrote ${OUT_SCALING}"
echo "wrote ${OUT_OVERLAY}"
