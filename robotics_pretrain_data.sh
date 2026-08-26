#!/usr/bin/env bash
set -uo pipefail  # 移除 -e 避免并行任务失败直接中断整个脚本

export HF_HOME=/cpfs01/cpfs01/cache/huggingface
export HF_LEROBOT_HOME=/cpfs01/cpfs01/cache/huggingface/lerobot

# ================= 配置区 =================
DATA="/cpfs01/cpfs01/datas/robotics_pretrain_data"
SAVE_ROOT_BASE="/cpfs01/jensen/data/stats/robotics_pretrain_data"

DATASETS_NAME=(
  # "Galaxea-Open-World-Dataset/lerobot_v3_new"
  # "robochallenge_diff_robotics/ur5"
  # "robochallenge_diff_robotics/franka"
  # "robochallenge_diff_robotics/arx5"
  # "robochallenge_diff_robotics/aloha"
  # "unitree_g1_hangzhou/unitree_401_g1_wo_stereo"
  # "unitree_g1_hangzhou/unitree_410_g1_with_stereo"
  # "unitree_g1_hangzhou/unitree_410_g1_wo_stereo"
  # "unitree_g1_hangzhou/unitree_410_g1_with_stereo_0526"
  # "unitree_g1_jidi/unitree_g1_jidi_opensource_v1"
  # "unitree_g1_jidi/dex_hand/unitree_g1_jidi_opensource_0327"
  # "unitree_g1_jidi/dex_hand/unitree_g1_jidi_opensource_0422"
  # "unitree_g1_jidi/dex_hand/unitree_g1_jidi_opensource_0501"
  # "unitree_g1_jidi/gripper/unitree_g1_jidi_opensource_0422"
  # "unitree_g1_singapore_merged/with_stereo"
  # "unitree_g1_singapore_merged/wo_stereo"
  # "oxe_lerobot_v3_0"
  # "unitree_g1_hangzhou/unitree_410_g1_with_stereo_0415"
  # "unitree_g1_hangzhou/unitree_410_g1_with_stereo_0508"
  # "droid_lerobot_v3"
  # "agibot_world_beta_lerobot/gripper"
  # "simulation_dataset/libero_flip_width"
  # "simulation_dataset/RoboTwin"
  # "agibot_world_beta_lerobot/gripper"
  "unitree_g1_shanghai_smpl"
)

EMBODIMENT_TAGS=(
  # "R1_LITE"
  # "ROBOCHALLENGE_SINGLE_ARM"
  # "ROBOCHALLENGE_SINGLE_ARM"
  # "ROBOCHALLENGE_SINGLE_ARM"
  # "ROBOCHALLENGE_DUAL_ARM"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE_WITH_BASE"
  # "UNITREE_G1_EE_WITH_BASE"
  # "UNITREE_G1_EE_WITH_BASE"
  # "UNITREE_G1_EE_WITH_BASE"
  # "UNITREE_G1_EE_WITH_BASE"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  # "OXE_WIDOWX"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  # "OXE_DROID"
  # "UNITREE_G1_EE"
  # "LIBERO"
  # "UNITREE_G1_EE"
  # "UNITREE_G1_EE"
  "UNITREE_G1_SMPL_BASE_ROT"
)

# 校验数组长度是否一致
if (( ${#DATASETS_NAME[@]} != ${#EMBODIMENT_TAGS[@]} )); then
  echo "❌ 配置错误: DATASETS_NAME 与 EMBODIMENT_TAGS 数量不一致" >&2
  exit 1
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
LOG_DIR_BASE="${LOG_DIR:-stats_logs}"
OUTPUT_FORMAT="ROTVEC"
mkdir -p "${LOG_DIR_BASE}"
export PYTHONPATH="$(pwd)"

# ================= 核心函数 =================
run_subtask() {
  local dataset_dir="$1"
  local save_root="$2"
  local embodiment_tag="$3"
  local output_fmt="$4"

  local tag="${embodiment_tag}"
  # 根据路径模糊匹配覆盖 tag
  if [[ "${dataset_dir}" == *"bridge"* ]]; then
    tag="OXE_WIDOWX"
  elif [[ "${dataset_dir}" == *"fractal"* ]]; then
    tag="OXE_GOOGLE"
  fi

  local save_arg=()
  if [[ -n "${save_root}" ]]; then
    local ds_name
    ds_name="$(basename "${dataset_dir}")"
    save_arg=(--save-path "${save_root}/${ds_name}")
  fi

  python gr00t/data/stats.py \
    --dataset-path "${dataset_dir}" \
    --embodiment-tag "${tag}" \
    --output-format "${output_fmt}" \
    "${save_arg[@]}"
}

process_one_dataset() {
  local data_root="$1"
  local save_root="$2"
  local embodiment_tag="$3"
  local max_jobs="$4"
  local log_dir="$5"

  mkdir -p "${save_root}" "${log_dir}"

  shopt -s nullglob
  local subdirs=("${data_root}"/*/)
  shopt -u nullglob
  subdirs=("${subdirs[@]%/}") # 去除末尾斜杠

  if (( ${#subdirs[@]} == 0 )); then
    echo "  ⚠️  未找到子目录，跳过。"
    return 0
  fi

  declare -a PIDS=()
  declare -A PID_TO_DIR=()
  declare -a FAILURES=()
  local running=0

  for dir in "${subdirs[@]}"; do
    # 等待空闲 slot
    while (( running >= max_jobs )); do
      local pid="${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
      if ! wait "${pid}"; then
        FAILURES+=("$(basename "${PID_TO_DIR[${pid}]}")")
      fi
      unset "PID_TO_DIR[${pid}]"
      (( running-- )) || true
    done

    local safe_name
    safe_name="$(printf '%s' "${dir}" | tr '/ ' '__')"
    local log_path="${log_dir}/${safe_name}.log"

    {
      echo "======================================"
      echo "Processing: $(basename "${dir}")"
      echo "Start: $(date -Is)"
      echo "======================================"
      run_subtask "${dir}" "${save_root}" "${embodiment_tag}" "${OUTPUT_FORMAT}"
      echo "End: $(date -Is)"
    } > "${log_path}" 2>&1 &

    local pid="$!"
    PID_TO_DIR["${pid}"]="${dir}"
    PIDS+=("${pid}")
    (( running++ )) || true
    echo "  [spawned pid=${pid}] $(basename "${dir}") -> ${log_path}"
  done

  # 等待剩余任务
  while (( running > 0 )); do
    local pid="${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    if ! wait "${pid}"; then
      FAILURES+=("$(basename "${PID_TO_DIR[${pid}]}")")
    fi
    unset "PID_TO_DIR[${pid}]"
    (( running-- )) || true
  done

  if (( ${#FAILURES[@]} > 0 )); then
    echo "  ❌ 部分子任务失败 (${#FAILURES[@]}): ${FAILURES[*]}" >&2
    return 1
  fi
  echo "  ✅ 所有子任务成功完成。"
  return 0
}

# ================= 主循环 =================
GLOBAL_FAILURES=()

echo "🚀 开始批量统计计算 | JOBS=${JOBS} | LOG_DIR=${LOG_DIR_BASE}"
echo "=================================================================="

for DATASET_ID in "${!DATASETS_NAME[@]}"; do
  DS_NAME="${DATASETS_NAME[DATASET_ID]}"
  DS_TAG="${EMBODIMENT_TAGS[DATASET_ID]}"
  DATA_ROOT="${DATA}/${DS_NAME}"
  SAVE_ROOT="${SAVE_ROOT_BASE}/${DS_NAME}"
  DS_LOG_DIR="${LOG_DIR_BASE}/${DATASET_ID}_${DS_NAME//\//_}"

  echo ""
  echo "📦 [${DATASET_ID}] ${DS_NAME}"
  echo "   DATA:  ${DATA_ROOT}"
  echo "   SAVE:  ${SAVE_ROOT}"
  echo "   TAG:   ${DS_TAG}"
  echo "------------------------------------------------------------------"

  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "  ⚠️  路径不存在，跳过。"
    continue
  fi

  if ! process_one_dataset "${DATA_ROOT}" "${SAVE_ROOT}" "${DS_TAG}" "${JOBS}" "${DS_LOG_DIR}"; then
    GLOBAL_FAILURES+=("[${DATASET_ID}] ${DS_NAME}")
  fi
done

echo ""
echo "=================================================================="
if (( ${#GLOBAL_FAILURES[@]} > 0 )); then
  echo "❌ 以下数据集处理失败 (${#GLOBAL_FAILURES[@]}):" >&2
  printf "  - %s\n" "${GLOBAL_FAILURES[@]}" >&2
  echo "请检查对应日志目录: ${LOG_DIR_BASE}" >&2
  exit 1
else
  echo "✅ 全部数据集统计完成！"
  exit 0
fi