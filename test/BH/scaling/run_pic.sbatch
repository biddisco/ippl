#!/usr/bin/env bash
# PIC-only worker. Args: <mode> <nranks> <N> <grid> <Nt>.
# Runs the alpine LandauDamping PIC reference twice under the same allocation:
#   - pic_eqsteps: same Nt as BH (less simulated time when dt_PIC < dt_BH)
#   - pic_eqtime : Nt scaled so simulated time matches BH
# Writes both PIC CSVs + timing.dat files into scaling/data/pic_<TAG>/. Does
# NOT append to a sweep CSV — the dependent run_bh.sbatch jobs do that, after
# reading the PIC timings written here. Designed to be submitted once per
# (mode, nranks) and shared by the three BH precision runs that depend on it.
#SBATCH --no-requeue

set -euo pipefail

MODE=$1
NRANKS=$2
N=$3
G=$4
NT=$5

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD=${SLURM_SUBMIT_DIR:-${SRC_DIR}/../../../build}
cd "${BUILD}"

NPM=$(awk "BEGIN { printf \"%.4g\", ${N}/1000000 }")
TAG="${MODE}_${NRANKS}_${NPM}M"
OUT=${BUILD}/scaling/data/pic_${TAG}
mkdir -p ${OUT}

now()  { date +%s.%N; }
elapsed() { awk "BEGIN { printf \"%.3f\", $2 - $1 }"; }

# Parse a row "name........... ranks Wall_max Wall_min Wall_avg" from a
# timing.dat written by IpplTimings::print(filename). Picks Wall_max (col 3
# after the name column) since that captures the slowest rank — the relevant
# number for collective-step time.
parse_timing() {
  local dat=$1 name=$2
  if [ ! -f "${dat}" ]; then echo ""; return; fi
  awk -v n="^${name}\\\.+" '$0 ~ n { print $3; exit }' "${dat}"
}

# Step time for PIC: alpine has no umbrella step timer. Derive as
# total - initialize - dumpData (captures advance + post_step minus dumps).
pic_step_time() {
  local dat=$1
  local total init_t dump_t
  total=$(parse_timing "${dat}" "total")
  init_t=$(parse_timing "${dat}" "initialize")
  dump_t=$(parse_timing "${dat}" "dumpData")
  awk -v t="${total:-0}" -v i="${init_t:-0}" -v d="${dump_t:-0}" \
    'BEGIN { if (t > 0) printf "%.3f", t - i - d; else print "nan" }'
}

# Mirror alpine/LandauDampingManager.h:99 — dt = min(0.05, 0.5·hr) where
# hr = 4π / G (Landau box is [0, 4π]³). Result is dt_PIC for this grid.
PI_4=$(awk "BEGIN { printf \"%.16g\", 4 * atan2(0, -1) }")
DT_BH=0.05
DT_PIC=$(awk -v G="${G}" -v dt_bh="${DT_BH}" -v pi4="${PI_4}" \
  'BEGIN {
     hr  = pi4 / G;
     dt  = 0.5 * hr;
     if (dt > dt_bh) dt = dt_bh;
     printf "%.16g", dt;
   }')
# Nt for the two PIC variants. ceil() via awk: int(x + 1 - 1e-9).
NT_EQSTEPS=${NT}
NT_EQTIME=$(awk -v nt="${NT}" -v dtbh="${DT_BH}" -v dtpic="${DT_PIC}" \
  'BEGIN {
     ratio = nt * dtbh / dtpic;
     n = int(ratio);
     if (ratio - n > 1e-9) n = n + 1;
     printf "%d", n;
   }')

# ---- PIC same-Nt ----
# Alpine opens FieldLandau_*.csv with Inform::APPEND (LandauDampingManager.h:377);
# wipe any stale work dir from a prior invocation so each run starts clean.
WORK_PIC_EQSTEPS=${OUT}/pic_eqsteps_run
rm -rf ${WORK_PIC_EQSTEPS}
mkdir -p ${WORK_PIC_EQSTEPS}/data
cd ${WORK_PIC_EQSTEPS}
t0=$(now)
srun ${BUILD}/alpine/LandauDamping ${G} ${G} ${G} ${N} ${NT_EQSTEPS} FFT 0.01 LeapFrog --overallocate 2.0 --info 10
t1=$(now)
PIC_EQSTEPS_WALL=$(elapsed ${t0} ${t1})
cp data/FieldLandau_${NRANKS}_manager.csv ${OUT}/pic_eqsteps.csv
[ -f timing.dat ] && cp timing.dat ${OUT}/pic_eqsteps_timing.dat
PIC_EQSTEPS_STEP=$(pic_step_time "${OUT}/pic_eqsteps_timing.dat")

# ---- PIC same-t (skip the duplicate when dt_PIC == dt_BH) ----
if [ "${NT_EQSTEPS}" = "${NT_EQTIME}" ]; then
  cp ${OUT}/pic_eqsteps.csv         ${OUT}/pic_eqtime.csv
  [ -f ${OUT}/pic_eqsteps_timing.dat ] && cp ${OUT}/pic_eqsteps_timing.dat ${OUT}/pic_eqtime_timing.dat
  PIC_EQTIME_WALL=${PIC_EQSTEPS_WALL}
  PIC_EQTIME_STEP=${PIC_EQSTEPS_STEP}
else
  WORK_PIC_EQTIME=${OUT}/pic_eqtime_run
  rm -rf ${WORK_PIC_EQTIME}
  mkdir -p ${WORK_PIC_EQTIME}/data
  cd ${WORK_PIC_EQTIME}
  t0=$(now)
  srun ${BUILD}/alpine/LandauDamping ${G} ${G} ${G} ${N} ${NT_EQTIME} FFT 0.01 LeapFrog --overallocate 2.0 --info 10
  t1=$(now)
  PIC_EQTIME_WALL=$(elapsed ${t0} ${t1})
  cp data/FieldLandau_${NRANKS}_manager.csv ${OUT}/pic_eqtime.csv
  [ -f timing.dat ] && cp timing.dat ${OUT}/pic_eqtime_timing.dat
  PIC_EQTIME_STEP=$(pic_step_time "${OUT}/pic_eqtime_timing.dat")
fi

# Persist timings into a small kv file the dependent BH jobs read back.
# Avoids re-parsing timing.dat from each precision job.
cat > ${OUT}/pic_timings.env <<EOF
PIC_EQSTEPS_STEP=${PIC_EQSTEPS_STEP:-nan}
PIC_EQTIME_STEP=${PIC_EQTIME_STEP:-nan}
PIC_EQSTEPS_WALL=${PIC_EQSTEPS_WALL}
PIC_EQTIME_WALL=${PIC_EQTIME_WALL}
NT_EQSTEPS=${NT_EQSTEPS}
NT_EQTIME=${NT_EQTIME}
DT_PIC=${DT_PIC}
EOF

echo "done: pic_${TAG}  eqsteps step=${PIC_EQSTEPS_STEP}s (wall=${PIC_EQSTEPS_WALL}s, Nt=${NT_EQSTEPS})  eqtime step=${PIC_EQTIME_STEP}s (wall=${PIC_EQTIME_WALL}s, Nt=${NT_EQTIME})  -> ${OUT}"
