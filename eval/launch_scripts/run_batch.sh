#!/usr/bin/env bash
# Batch evaluation launcher for eval.py.
#
# Examples:
#   bash launch_scripts/run_batch.sh /path/to/checkpoint_a /path/to/checkpoint_b
#   MODEL_PATH=/path/to/checkpoint bash launch_scripts/run_batch.sh
#   CONFIG_PATH=eval_configs/eval_all_sft.yaml RESUME=0 bash launch_scripts/run_batch.sh /path/to/checkpoint

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash launch_scripts/run_batch.sh [MODEL_PATH ...]

Environment variables:
  MODEL_PATH                 Single checkpoint path used when no positional path is provided.
  CONFIG_PATH                Evaluation config. Default: eval_configs/eval_all_sft.yaml
  VENV_PATH                  Python virtualenv. Default: ../.venv from repository root if present.
  CUDA_VISIBLE_DEVICES       GPUs visible to vLLM. Default: 0,1,2,3,4,5,6,7
  OUTPUT_BASE_DIR            Optional directory for evaluation outputs.
  RESUME                     1 to auto-resume, 0 for a fresh run. Default: 1
  FORCE_RESCORE              1 to rescore cached predictions. Default: 0
  WAIT_FOR_PID               Optional process id to wait for before evaluation.
  CLEAN_RAY_BETWEEN_JOBS     1 to clean current-user Ray processes between jobs. Default: 0
  WAIT_FOR_GPU_MEMORY        1 to wait for free GPU memory between jobs. Default: 0
  GPU_MEM_THRESHOLD_GIB      Free-memory threshold used when WAIT_FOR_GPU_MEMORY=1. Default: 30
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"

cd "$EVAL_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/hf}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${REPO_ROOT}/cache/modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

CONFIG_PATH="${CONFIG_PATH:-eval_configs/eval_all_sft.yaml}"
LOG_DIR="${LOG_DIR:-vllm_logs}"
RESUME="${RESUME:-1}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
CLEAN_RAY_BETWEEN_JOBS="${CLEAN_RAY_BETWEEN_JOBS:-0}"
WAIT_FOR_GPU_MEMORY="${WAIT_FOR_GPU_MEMORY:-0}"
GPU_MEM_THRESHOLD_GIB="${GPU_MEM_THRESHOLD_GIB:-30}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
else
    log "Virtualenv not found at ${VENV_PATH}; using current Python environment."
fi

if [ ! -f "$CONFIG_PATH" ]; then
    log "[Error] Config file does not exist: $CONFIG_PATH"
    exit 1
fi

if [ -n "${WAIT_FOR_PID:-}" ]; then
    log "Waiting for process PID ${WAIT_FOR_PID} to finish..."
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep 10
    done
    log "Process ${WAIT_FOR_PID} finished."
fi

declare -a MODEL_PATHS=("$@")
if [ ${#MODEL_PATHS[@]} -eq 0 ] && [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS=("$MODEL_PATH")
fi

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    usage
    log "[Error] No model checkpoint path provided."
    exit 1
fi

wait_for_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi not found; skipping GPU memory wait."
        return 0
    fi

    local threshold_gib="${1:-$GPU_MEM_THRESHOLD_GIB}"
    local threshold_mib=$(( threshold_gib * 1024 ))
    local attempt=0

    log "Waiting for GPU memory: min free >= ${threshold_gib} GiB."
    while true; do
        local min_free
        min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
                   | awk 'BEGIN{m=-1} {if(m<0||$1<m) m=$1} END{print m}')

        if [ -z "$min_free" ] || [ "$min_free" -lt 0 ] 2>/dev/null; then
            log "  Could not read GPU memory; retrying in 10s."
            sleep 10
            continue
        fi

        attempt=$(( attempt + 1 ))
        if [ "$min_free" -ge "$threshold_mib" ]; then
            log "  GPU memory is sufficient: ${min_free} MiB."
            return 0
        fi

        log "  [${attempt}] min free ${min_free} MiB < ${threshold_mib} MiB; retrying in 15s."
        sleep 15
    done
}

clean_stray_ray() {
    log "Cleaning current-user Ray processes..."
    pkill -u "$(id -u)" -f "ray::" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/dashboard" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/log_monitor" 2>/dev/null || true
    pkill -u "$(id -u)" -f "gcs_server" 2>/dev/null || true
    sleep 3
}

maybe_merge_lora() {
    local model_path="$1"

    if [ ! -f "${model_path}/adapter_config.json" ]; then
        return 0
    fi

    if [ -f "${model_path}/config.json" ]; then
        log "  LoRA adapter detected but config.json already exists; skipping merge."
        return 0
    fi

    log "  Unmerged LoRA adapter detected; merging into ${model_path}."

    LORA_MERGE_PATH="$model_path" python - <<'PYEOF'
import json
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ["LORA_MERGE_PATH"]
with open(os.path.join(model_path, "adapter_config.json"), encoding="utf-8") as f:
    adapter_config = json.load(f)

base_model_path = adapter_config["base_model_name_or_path"]
print(f"  Base model: {base_model_path}", flush=True)
print(f"  Merge target: {model_path}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("  Merge complete.", flush=True)
PYEOF
}

TOTAL=${#MODEL_PATHS[@]}
FAILED=0
declare -a SUCCEEDED_PATHS=()

log "========== Batch evaluation start: ${TOTAL} model(s), config=${CONFIG_PATH} =========="

for idx in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$idx]}"
    MODEL_NAME="$(basename "$MODEL_PATH")"
    JOB_NUM=$(( idx + 1 ))
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

    log "---------- Task ${JOB_NUM}/${TOTAL}: ${MODEL_NAME} ----------"
    log "  Model path: ${MODEL_PATH}"
    log "  Log file: ${LOG_FILE}"

    if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
        log "  [Error] Model path does not exist or is empty: ${MODEL_PATH}"
        FAILED=$((FAILED + 1))
        continue
    fi

    maybe_merge_lora "$MODEL_PATH" || {
        log "  [Error] LoRA merge failed."
        FAILED=$((FAILED + 1))
        continue
    }

    declare -a EVAL_ARGS=()
    if [ "$RESUME" = "1" ]; then
        EVAL_ARGS+=(--resume)
    fi
    if [ "$FORCE_RESCORE" = "1" ]; then
        EVAL_ARGS+=(--force-rescore)
    fi
    if [ -n "${OUTPUT_BASE_DIR:-}" ]; then
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"
        mkdir -p "$OUTPUT_DIR"
        EVAL_ARGS+=(--output-dir "$OUTPUT_DIR")
        log "  Output directory: ${OUTPUT_DIR}"
    fi

    log "  Starting eval.py..."
    if MODEL_PATH="$MODEL_PATH" python -u eval.py "$CONFIG_PATH" "${EVAL_ARGS[@]}" > "$LOG_FILE" 2>&1; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        log "  [Done] ${MODEL_NAME}"
        SUCCEEDED_PATHS+=("$MODEL_PATH")
    else
        log "  [Failed] ${MODEL_NAME}, exit code ${EXIT_CODE}. See ${LOG_FILE}"
        FAILED=$((FAILED + 1))
    fi

    if [ "$idx" -lt $(( TOTAL - 1 )) ]; then
        if [ "$CLEAN_RAY_BETWEEN_JOBS" = "1" ]; then
            clean_stray_ray
        fi
        if [ "$WAIT_FOR_GPU_MEMORY" = "1" ]; then
            wait_for_gpu_memory "$GPU_MEM_THRESHOLD_GIB"
        fi
    fi
done

log "========== Batch evaluation end: ${TOTAL} model(s), ${FAILED} failed =========="

if [ ${#SUCCEEDED_PATHS[@]} -gt 0 ]; then
    SUMMARY_XLSX="${LOG_DIR}/batch_summary_$(date '+%Y%m%d_%H%M%S').xlsx"
    log "Aggregating results to ${SUMMARY_XLSX}"
    if python collect_results.py "${SUCCEEDED_PATHS[@]}" --output "$SUMMARY_XLSX"; then
        log "Aggregation complete: ${SUMMARY_XLSX}"
    else
        log "Aggregation failed; run collect_results.py manually if needed."
    fi
fi

exit "$FAILED"
