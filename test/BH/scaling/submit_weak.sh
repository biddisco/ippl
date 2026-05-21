#!/usr/bin/env bash
# Weak scaling sweep for Landau damping — precision sweep edition.
# Powers of 8 in rank count so each step doubles per-axis problem size cleanly
# (G doubles, total N grows ×8). Three BH precision policies per nranks.
# For each nranks: submit one PIC job (run_pic.sbatch), then three BH jobs
# (run_bh.sbatch, one per precision) with --dependency=afterok:<pic_jid>.
#
# Per-rank load: N_per_rank = 8·256^3 = 134.2M particles (≈ G_per_rank = 256).
# Global grid size scales as cbrt(N_total/8) to maintain 8 particles per cell.
#
#   nranks   N_total            grid    N_per_rank
#   8        8·512^3   ≈  1.07B   512     134.2M (= 8·256^3)
#   64       8·1024^3  ≈  8.59B  1024     134.2M
#   512      8·2048^3  ≈ 68.72B  2048     134.2M
set -euo pipefail
# Sources (sbatch + plotters) live next to this script. Outputs (data/, logs/)
# go under ${BUILD_DIR}/scaling/. Default BUILD_DIR is <repo>/build/; override
# with BUILD_DIR=/abs/path if your build tree lives elsewhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/../../../build}
if [ ! -x "${BUILD_DIR}/test/BH/LandauDampingBH" ] || [ ! -x "${BUILD_DIR}/alpine/LandauDamping" ]; then
  echo "error: expected ${BUILD_DIR}/test/BH/LandauDampingBH and ${BUILD_DIR}/alpine/LandauDamping" >&2
  echo "       set BUILD_DIR=/abs/path to your build tree if it lives elsewhere" >&2
  exit 2
fi
cd "${BUILD_DIR}"

NRANKS_LIST=(8 64 512)
NT=50
PRECISIONS=(${PRECISIONS:-double mixed float})
declare -A NMAP GMAP
NMAP[8]=1073741824;    GMAP[8]=512
NMAP[64]=8589934592;   GMAP[64]=1024
NMAP[512]=68719476736; GMAP[512]=2048

walltime_for() {
  case "$1" in
    8)   echo "00:15:00" ;;
    64)  echo "00:15:00" ;;
    512) echo "00:15:00" ;;
    *)   echo "00:15:00" ;;
  esac
}

mkdir -p scaling/data
TS=$(date +%Y%m%d_%H%M%S)
LOG=scaling/logs_weak_${TS}
mkdir -p ${LOG}

CSV=scaling/data/weak_runtimes_${TS}.csv
echo "precision,nranks,nodes,gpus_per_node,N,grid,bh_step_s,pic_eqsteps_step_s,pic_eqtime_step_s,bh_wall_s,pic_eqsteps_wall_s,pic_eqtime_wall_s" > ${CSV}
printf "%s\n" "${CSV}" > scaling/.latest_weak_csv

BH_JOBS=()
for nranks in "${NRANKS_LIST[@]}"; do
  if [ ${nranks} -le 2 ]; then
    gpn=${nranks}; nodes=1
  elif [ ${nranks} -eq 4 ]; then
    gpn=4; nodes=1
  else
    gpn=4; nodes=$(( nranks / 4 ))
  fi
  N=${NMAP[${nranks}]}
  G=${GMAP[${nranks}]}
  TL=$(walltime_for ${nranks})

  # ---- one PIC job per (mode, nranks) ----
  pic_jid=$(sbatch --parsable \
    -A csstaff \
    -J "weak_pic_${nranks}" \
    -N ${nodes} \
    --ntasks-per-node ${gpn} \
    --gpus-per-task 1 \
    -t ${TL} \
    -o "${LOG}/pic_${nranks}_%j.out" \
    -e "${LOG}/pic_${nranks}_%j.err" \
    ${SCRIPT_DIR}/run_pic.sbatch weak ${nranks} ${N} ${G} ${NT})
  echo "submitted weak/pic/${nranks} as ${pic_jid}  (${nodes}n x ${gpn}gpu, N=${N}, G=${G}, t=${TL})"

  # ---- three BH jobs per (mode, nranks), gated on the PIC job ----
  for prec in "${PRECISIONS[@]}"; do
    bh_jid=$(sbatch --parsable \
      -A csstaff \
      -J "weak_bh_${nranks}_${prec}" \
      -N ${nodes} \
      --ntasks-per-node ${gpn} \
      --gpus-per-task 1 \
      -t ${TL} \
      --dependency=afterok:${pic_jid} \
      -o "${LOG}/bh_${nranks}_${prec}_%j.out" \
      -e "${LOG}/bh_${nranks}_${prec}_%j.err" \
      ${SCRIPT_DIR}/run_bh.sbatch weak ${nranks} ${N} ${G} ${NT} ${prec} ${CSV})
    BH_JOBS+=("${bh_jid}")
    echo "submitted weak/bh/${nranks}/${prec} as ${bh_jid} (dep afterok:${pic_jid})"
  done
done

DEPS=$(IFS=:; echo "${BH_JOBS[*]}")
jid=$(sbatch --parsable \
  -A csstaff \
  -J "weak_plot" \
  -N 1 --ntasks-per-node 1 \
  -t 00:10:00 \
  --dependency=afterany:${DEPS} \
  -o "${LOG}/plot_%j.out" \
  -e "${LOG}/plot_%j.err" \
  ${SCRIPT_DIR}/aggregate.sbatch weak ${CSV})
echo "submitted weak_plot as ${jid} (deps afterany:${DEPS})"
echo "CSV: ${CSV}"
echo "logs: ${LOG}"
