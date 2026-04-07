#!/usr/bin/env bash

set -euo pipefail
DATA="/home/unitree/remote_jensen2/robotics_pretrain_data/"
SAVE_ROOT_BASE="./robotics_pretrain_data"

DATASETS=(
  "Galaxea-Open-World-Dataset/lerobot_v3_new"
  "robochallenge_diff_robotics/ur5"
  "unitree_g1_hangzhou/unitree_401_g1_wo_stereo"
  "oxe_lerobot_v3_0"
)

EMBODIMENT_TAGs=(
  "R1_LITE"
  "ROBOCHALLENGE_SINGLE_ARM"
  "UNITREE_G1_EE"
  "OXE_WIDOWX"
  "OXE_GOOGLE"
)

DATASET_ID=2

# Root directory containing all lerobot_v3_0 task folders
DATA_ROOT="$DATA/${DATASETS[DATASET_ID]}"
SAVE_ROOT="$SAVE_ROOT_BASE/${DATASETS[DATASET_ID]}"

# Embodiment and output format are fixed as in the example
EMBODIMENT_TAG="${EMBODIMENT_TAGs[DATASET_ID]}"

OUTPUT_FORMAT="XYZ_ROTVEC"

# Optional: if set, save stats under SAVE_ROOT/<dataset_name>/ instead of inside dataset dir
SAVE_ROOT="${SAVE_ROOT:-}"

export PYTHONPATH="$(pwd)"

JOBS="${JOBS:-}"
if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="4"
  fi
fi

LOG_DIR="${LOG_DIR:-stats_logs}"
mkdir -p "${LOG_DIR}"

echo "Running stats for all datasets under: ${DATA_ROOT}"
echo "Parallel jobs: ${JOBS}"
echo "Log dir: ${LOG_DIR}"
echo "Tip: override with JOBS=8 LOG_DIR=stats_logs_custom ./galaxea_stats_paral.sh"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

shopt -s nullglob

declare -a DATASETS=()
for dataset_dir in "${DATA_ROOT}"/*/; do
  dataset_dir="${dataset_dir%/}" # Remove trailing slash
  [[ -d "${dataset_dir}" ]] || continue
  DATASETS+=("${dataset_dir}")
done

if (( ${#DATASETS[@]} == 0 )); then
  echo "No dataset folders found under: ${DATA_ROOT}" >&2
  exit 1
fi

run_one() {
  local dataset_dir="$1"
  local embodiment_tag="${EMBODIMENT_TAG}"
  
  # Override embodiment tag based on dataset directory path (fuzzy match)
  if [[ "${dataset_dir}" == *"bridge"* ]]; then
    embodiment_tag="OXE_WIDOWX"
  elif [[ "${dataset_dir}" == *"fractal"* ]]; then
    embodiment_tag="OXE_GOOGLE"
  fi
  
  local save_path_arg=()
  if [[ -n "${SAVE_ROOT}" ]]; then
    local dataset_name
    dataset_name="$(basename "${dataset_dir}")"
    save_path_arg=(--save-path "${SAVE_ROOT}/${dataset_name}")
  fi

  python gr00t/data/stats.py \
    --dataset-path "${dataset_dir}" \
    --embodiment-tag "${embodiment_tag}" \
    --output-format "${OUTPUT_FORMAT}" \
    "${save_path_arg[@]}"
}

declare -A PID_TO_DATASET=()
declare -a PIDS=()
declare -a FAILURES=()

spawn_job() {
  local dataset_dir="$1"
  local safe_name
  safe_name="$(printf '%s' "${dataset_dir}" | tr '/ ' '__')"
  local log_path="${LOG_DIR}/${safe_name}.log"

  {
    echo "======================================"
    echo "Processing dataset: ${dataset_dir}"
    echo "Start: $(date -Is)"
    echo "======================================"
    run_one "${dataset_dir}"
    echo "End: $(date -Is)"
  } >"${log_path}" 2>&1 &

  local pid="$!"
  PID_TO_DATASET["${pid}"]="${dataset_dir}"
  PIDS+=("${pid}")
  echo "[spawned pid=${pid}] ${dataset_dir} -> ${log_path}"
}

running=0
for dataset_dir in "${DATASETS[@]}"; do
  while (( running >= JOBS )); do
    pid="${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    if ! wait "${pid}"; then
      FAILURES+=("${PID_TO_DATASET[${pid}]} (exit=$?)")
    fi
    unset "PID_TO_DATASET[${pid}]"
    (( running-- ))
  done

  spawn_job "${dataset_dir}"
  (( ++running ))
done

while (( running > 0 )); do
  pid="${PIDS[0]}"
  PIDS=("${PIDS[@]:1}")
  if ! wait "${pid}"; then
    FAILURES+=("${PID_TO_DATASET[${pid}]} (exit=$?)")
  fi
  unset "PID_TO_DATASET[${pid}]"
  (( running-- ))
done

if (( ${#FAILURES[@]} > 0 )); then
  echo
  echo "Some datasets failed (${#FAILURES[@]}):" >&2
  for d in "${FAILURES[@]}"; do
    echo "  - ${d}" >&2
  done
  echo "See logs in: ${LOG_DIR}" >&2
  exit 1
fi

echo "All datasets processed successfully."

