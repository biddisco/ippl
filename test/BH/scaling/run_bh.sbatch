#!/usr/bin/env bash
# BH-only worker. Args: <mode> <nranks> <N> <grid> <Nt> <precision> <csv_path>.
# Runs LandauDampingBH once with --precision=<precision>. Reads the PIC step
# times and Nt values from scaling/data/pic_<TAG>/pic_timings.env, written by
# the upstream run_pic.sbatch job (afterok dependency). Appends one row to the
# sweep CSV (precision as the first column) and renders the per-run accuracy
# PDF (BH vs PIC same-Nt vs PIC same-t + analytic envelope).
#SBATCH --no-requeue

set -euo pipefail

MODE=$1
NRANKS=$2
N=$3
G=$4
NT=$5
PRECISION=$6
CSV=$7

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD=${SLURM_SUBMIT_DIR:-${SRC_DIR}/../../../build}
cd "${BUILD}"
# Resolve CSV to absolute path before any cd-ing.
case "${CSV}" in
  /*) ;;
  *) CSV="${BUILD}/${CSV}" ;;
esac
NPM=$(awk "BEGIN { printf \"%.4g\", ${N}/1000000 }")
TAG="${MODE}_${NRANKS}_${NPM}M"
PIC_DIR=${BUILD}/scaling/data/pic_${TAG}
OUT=${BUILD}/scaling/data/bh_${TAG}_${PRECISION}
mkdir -p ${OUT}

now()  { date +%s.%N; }
elapsed() { awk "BEGIN { printf \"%.3f\", $2 - $1 }"; }

# Parse a row "name........... ranks Wall_max Wall_min Wall_avg" from a
# timing.dat written by IpplTimings::print(filename). Picks Wall_max (col 3
# after the name column) since that captures the slowest rank.
parse_timing() {
  local dat=$1 name=$2
  if [ ! -f "${dat}" ]; then echo ""; return; fi
  awk -v n="^${name}\\\.+" '$0 ~ n { print $3; exit }' "${dat}"
}

bh_step_time() {
  local v
  v=$(parse_timing "$1" "step")
  echo "${v:-nan}"
}

# Load PIC timings + Nt values written by the upstream run_pic.sbatch job.
# If the file is missing (e.g. PIC job failed unexpectedly), fill nan/0 so we
# still record the BH timing rather than silently dropping the row.
if [ -f "${PIC_DIR}/pic_timings.env" ]; then
  # shellcheck disable=SC1091
  source "${PIC_DIR}/pic_timings.env"
else
  echo "warn: ${PIC_DIR}/pic_timings.env not found — recording BH only" >&2
  PIC_EQSTEPS_STEP=nan
  PIC_EQTIME_STEP=nan
  PIC_EQSTEPS_WALL=0
  PIC_EQTIME_WALL=0
  NT_EQSTEPS=${NT}
  NT_EQTIME=${NT}
  DT_PIC=0.05
fi
DT_PIC_FMT=$(awk -v dt="${DT_PIC}" 'BEGIN { printf "%.4g", dt }')

# ---- BH ----
WORK_BH=${OUT}/bh_run
mkdir -p ${WORK_BH}/data
cd ${WORK_BH}
t0=$(now)
srun ${BUILD}/test/BH/LandauDampingBH ${N} ${NT} 0.5 42 1 0.05 --precision=${PRECISION}
t1=$(now)
BH_WALL=$(elapsed ${t0} ${t1})
cp data/FieldLandauBH_${N}.csv ${OUT}/bh.csv
[ -f timing.dat ] && cp timing.dat ${OUT}/bh_timing.dat
BH_STEP=$(bh_step_time "${OUT}/bh_timing.dat")

# ---- record row (lock to allow concurrent jobs) ----
# CSV header is precision,nranks,nodes,gpus_per_node,N,grid,bh_step_s,
# pic_eqsteps_step_s,pic_eqtime_step_s,bh_wall_s,pic_eqsteps_wall_s,pic_eqtime_wall_s.
# PIC columns are properties of (mode, nranks) — they're identical across the
# 3 precision rows for a given nranks (read from pic_timings.env above).
(
  flock -x 200
  echo "${PRECISION},${NRANKS},${SLURM_NNODES:-?},${SLURM_NTASKS_PER_NODE:-?},${N},${G},${BH_STEP:-nan},${PIC_EQSTEPS_STEP:-nan},${PIC_EQTIME_STEP:-nan},${BH_WALL},${PIC_EQSTEPS_WALL},${PIC_EQTIME_WALL}" >> ${CSV}
) 200>"${CSV}.lock"

# ---- per-run accuracy PDF ----
# Each (precision, nranks) gets its own PDF tagged with the precision suffix.
python3 ${SRC_DIR}/plot_per_run.py \
  --bh           ${OUT}/bh.csv \
  --pic-eqsteps  ${PIC_DIR}/pic_eqsteps.csv \
  --pic-eqtime   ${PIC_DIR}/pic_eqtime.csv \
  --out          ${BUILD}/scaling/data/bh_${TAG}_${PRECISION}.pdf \
  --title        "${MODE} scaling, ${NRANKS} ranks, N=${NPM}M, precision=${PRECISION}, PIC grid=${G}^3 (dt_PIC=${DT_PIC_FMT}, Nt_eqsteps=${NT_EQSTEPS}, Nt_eqtime=${NT_EQTIME})" \
  || echo "warn: per-run plot failed for bh_${TAG}_${PRECISION} (run timing is recorded)"

echo "done: bh_${TAG}_${PRECISION}  step=${BH_STEP}s (wall=${BH_WALL}s)  PIC eqsteps step=${PIC_EQSTEPS_STEP}s  PIC eqtime step=${PIC_EQTIME_STEP}s  -> ${OUT}"
