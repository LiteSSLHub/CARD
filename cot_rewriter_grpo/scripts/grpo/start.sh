#!/usr/bin/env bash
# Minimal GRPO training launcher.
#
# Usage:
#   bash scripts/grpo/start.sh 0,1 llama3_3b_grpo_2x80g

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

GPUS="${1:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
CONFIG_NAME="${2:-${CONFIG_NAME:-llama3_3b_grpo_2x80g}}"

export PROJECT_BASE_DIR="${PROJECT_BASE_DIR:-${PROJECT_ROOT}}"
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-${REPO_ROOT}/models}"
export DATA_BASE_DIR="${DATA_BASE_DIR:-${REPO_ROOT}/data}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/cot_rewriter}"
export VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export ORIGINAL_CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_BASE_DIR}:${PROJECT_BASE_DIR}/verl:${PYTHONPATH:-}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-$(whoami)}}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_enable_dashboard="${RAY_enable_dashboard:-0}"
export RAY_DISABLE_DASHBOARD_AGENT="${RAY_DISABLE_DASHBOARD_AGENT:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${CHECKPOINT_BASE_DIR}/swanlab}"

mkdir -p "${CHECKPOINT_BASE_DIR}" "${RAY_TMPDIR}"

if [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
fi

cd "${PROJECT_BASE_DIR}"

echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"

exec python verl/verl/trainer/main_ppo.py \
    --config-path="${PROJECT_BASE_DIR}/examples" \
    --config-name="${CONFIG_NAME}"
