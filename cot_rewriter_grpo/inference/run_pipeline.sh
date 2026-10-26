#!/usr/bin/env bash
# Run CoT Rewrite Inference Pipeline (Steps 1-4)
#
# Usage:
#   bash inference/run_pipeline.sh inference/config_example.yaml
#   bash inference/run_pipeline.sh inference/config_example.yaml --step correct
#   bash inference/run_pipeline.sh inference/config_example.yaml --resume
#   bash inference/run_pipeline.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtual environment
VENV_PATH="${REPO_ROOT}/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: Virtual environment not found at ${VENV_PATH}" >&2
    echo "         Continuing with current Python environment." >&2
fi

# Add verl to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"

# Server compatibility
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1

exec python -m inference.pipeline "$@"
# nohup bash inference/run_pipeline.sh --config inference/config_example.yaml --resume > inference/logs/roll.log 2>&1 &
