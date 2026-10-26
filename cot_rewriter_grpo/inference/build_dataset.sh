#!/usr/bin/env bash
# Build ShareGPT dataset from CoT rewrite pipeline output (Step 5)
#
# Usage:
#   bash inference/build_dataset.sh --config inference/config_example.yaml
#   bash inference/build_dataset.sh --config inference/config_example.yaml \
#       --dataset-config inference/build_dataset_config_example.yaml
#   bash inference/build_dataset.sh --config inference/config_example.yaml \
#       --dataset-config inference/build_dataset_config_example.yaml \
#       --output-file /path/to/output.jsonl
#
# build_dataset_config_example.yaml controls:
#   build_mode      : "single" (rewrite_cot only) or "dual" (original_cot + rewrite_cot per question)
#   question_limit  : cap questions before building (e.g. 15000); null = all
#   filters         : fsf_positive, correct_only
#   selection_strategy : random / max_fsf / priority_correct
#   llamafactory    : auto-register to LlamaFactory dataset_info.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PATH="${REPO_ROOT}/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: Virtual environment not found at ${VENV_PATH}" >&2
fi

export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"

exec python -m inference.build_sharegpt "$@"
