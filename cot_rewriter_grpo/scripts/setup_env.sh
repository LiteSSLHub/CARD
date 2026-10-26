#!/usr/bin/env bash
# Environment setup for training and inference.
#
# Usage:
#   source cot_rewriter_grpo/scripts/setup_env.sh
#   MODEL_BASE_DIR=/path/to/models DATA_BASE_DIR=/path/to/data source cot_rewriter_grpo/scripts/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PROJECT_BASE_DIR="${PROJECT_BASE_DIR:-${PROJECT_ROOT}}"
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-${REPO_ROOT}/models}"
export DATA_BASE_DIR="${DATA_BASE_DIR:-${REPO_ROOT}/data}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
export VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${CHECKPOINT_BASE_DIR}/swanlab}"

export PYTHONPATH="${PROJECT_BASE_DIR}:${PROJECT_BASE_DIR}/verl:${PYTHONPATH:-}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-$(whoami)}}"
mkdir -p "$RAY_TMPDIR" "$CHECKPOINT_BASE_DIR"

echo "============================================================"
echo "CoT Rewriter GRPO environment"
echo "============================================================"
echo "PROJECT_BASE_DIR=${PROJECT_BASE_DIR}"
echo "MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "VENV_PATH=${VENV_PATH}"
echo "SWANLAB_MODE=${SWANLAB_MODE}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo ""

for path_name in MODEL_BASE_DIR DATA_BASE_DIR CHECKPOINT_BASE_DIR PROJECT_BASE_DIR; do
    path_value="${!path_name}"
    if [ -d "$path_value" ]; then
        echo "OK      ${path_name}: ${path_value}"
    else
        echo "MISSING ${path_name}: ${path_value}"
    fi
done

echo "============================================================"
