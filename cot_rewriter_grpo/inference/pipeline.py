"""
CoT Rewrite Inference Pipeline - Steps 1-4

Steps:
  1. Model loading / FSDP→HF format conversion
  2. Rollout: batch CoT rewriting with vLLM (multi-GPU data parallel via Ray)
  3. Correct Tag: judge whether each rewrite leads to the correct solution
  4. FSF Tag: compute Failure Step Fraction for each rewrite

Usage:
    python -m inference.pipeline --config inference/config_example.yaml
    python -m inference.pipeline --config inference/config_example.yaml --step correct
    python -m inference.pipeline --config inference/config_example.yaml --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt builder (copied from reasoning_graph/build_cot_rewrite_from_deepmath.py)
# ---------------------------------------------------------------------------

def build_prompt(question: str, thought: str) -> Tuple[str, str]:
    system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the correct path and present it clearly.

Follow these guidelines:
1. **Refine and Bridge**: Retain the core logic, including essential self-reflections and verification steps. Supplement logical leaps and clarify complex steps to ensure a smooth, coherent flow.
2. **Filter Invalid Paths**: Remove derivation branches within the 'Original Reasoning' that do not lead to the final answer (e.g., failed attempts, incorrect directions, or dead ends). Keep only the effective reasoning path.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

**Output Rules:**
- Directly output the rewritten reasoning process.
- Do NOT mention that you are rewriting another reasoning process."""

    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}

## Rewritten Reasoning Process
"""
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_data(input_path: str) -> List[Dict[str, Any]]:
    """Load ShareGPT-format JSONL and extract question / original_cot / solution."""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            convs = data.get("conversations", [])
            if len(convs) < 2:
                continue
            question = convs[0].get("value", convs[0].get("content", "")).strip()
            assistant_content = convs[1].get("value", convs[1].get("content", ""))
            if "<think>" in assistant_content:
                original_cot = assistant_content.split("<think>")[-1].split("</think>")[0].strip()
                solution = assistant_content.split("</think>")[-1].strip()
            else:
                original_cot = assistant_content.strip()
                solution = ""
            records.append({
                "question": question,
                "original_cot": original_cot,
                "solution": solution,
            })
    return records


# ---------------------------------------------------------------------------
# Step 1: Model loading / format detection + conversion
# ---------------------------------------------------------------------------

def detect_checkpoint_format(checkpoint_path: str) -> str:
    """Detect the format of a model checkpoint."""
    # Check for HF format: config.json in root
    if os.path.exists(os.path.join(checkpoint_path, "config.json")):
        return "hf"
    # Check for LoRA adapter
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        return "lora"
    # Check for FSDP format in actor/ subdirectory
    actor_dir = os.path.join(checkpoint_path, "actor")
    if os.path.isdir(actor_dir):
        for fname in os.listdir(actor_dir):
            if fname.startswith("model_world_size") or fname == "fsdp_config.json":
                return "fsdp"
        # Fallback: if there are .pt files inside, likely FSDP
        pt_files = [f for f in os.listdir(actor_dir) if f.endswith(".pt") or f.endswith(".bin")]
        if pt_files:
            return "fsdp"
    raise ValueError(
        f"Cannot detect checkpoint format at: {checkpoint_path}\n"
        "Expected one of: HF (config.json), LoRA (adapter_config.json), "
        "FSDP (actor/ directory with weight shards)"
    )


def _get_base_model_path(actor_dir: str) -> Optional[str]:
    """Read base_model_name_or_path from lora_adapter/adapter_config.json if present."""
    adapter_config = os.path.join(actor_dir, "lora_adapter", "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            cfg = json.load(f)
        path = cfg.get("base_model_name_or_path", "")
        if path and os.path.isdir(path):
            return path
    return None


def _hf_model_has_weights(hf_path: str) -> bool:
    """Check that hf_path contains actual model weight files, not just tokenizer/config."""
    if not os.path.exists(hf_path):
        return False
    for fname in os.listdir(hf_path):
        if fname.endswith(".safetensors") or fname.endswith(".bin"):
            return True
    return False


def _ensure_tokenizer(hf_path: str, actor_dir: str, logger: logging.Logger) -> None:
    """Copy tokenizer files from base model if not already present in hf_path."""
    if os.path.exists(os.path.join(hf_path, "tokenizer.json")):
        return  # Already there
    base_model_path = _get_base_model_path(actor_dir)
    if not base_model_path:
        logger.warning(f"[Step1] Cannot find base model path to copy tokenizer from")
        return
    tokenizer_filenames = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "added_tokens.json",
        "chat_template.jinja",
    ]
    copied = []
    for fname in tokenizer_filenames:
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(hf_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(fname)
    if copied:
        logger.info(f"[Step1] Copied tokenizer files from {base_model_path}: {copied}")


def _merge_lora_into_base(hf_path: str, actor_dir: str, dtype: str, logger: logging.Logger) -> None:
    """
    If FSDP merge extracted a lora_adapter alongside the base model,
    merge it in-place so vLLM sees the final fine-tuned weights.
    Also ensures tokenizer files are present.
    """
    lora_adapter_path = os.path.join(hf_path, "lora_adapter")
    if not os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
        _ensure_tokenizer(hf_path, actor_dir, logger)
        return  # No LoRA adapter found, nothing to do

    logger.info(f"[Step1] LoRA adapter found at {lora_adapter_path}, merging into base model...")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    # Load to GPU to avoid mmap (safetensors uses mmap by default,
    # which causes SIGBUS on network filesystems like NFS/Lustre).
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Step1] Loading base model to {device} for LoRA merge")
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch_dtype, trust_remote_code=True,
        device_map=device,
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()
    model = model.cpu()
    model.save_pretrained(hf_path)
    del model
    torch.cuda.empty_cache()
    logger.info(f"[Step1] LoRA merged into base model at {hf_path}")

    _ensure_tokenizer(hf_path, actor_dir, logger)


def step_model_loading(
    config: Dict[str, Any],
    ckpt_mgr,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    """
    Step 1: Ensure model exists in HuggingFace format with LoRA merged.

    Returns the path to the HF model directory.
    """
    checkpoint_path = config["model"]["checkpoint_path"]
    hf_path = os.path.join(checkpoint_path, "actor", "huggingface")
    merge_lora = config["model"].get("merge_lora", True)

    # Already fully merged (config.json + model weights present, no pending lora_adapter)?
    lora_adapter_path = os.path.join(hf_path, "lora_adapter")
    lora_pending = merge_lora and os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json"))
    if _hf_model_has_weights(hf_path) and not lora_pending:
        logger.info(f"[Step1] HF model already exists at {hf_path}")
        ckpt_mgr.mark_step_complete("model", hf_path, 0)
        return hf_path

    fmt = config["model"].get("format", "auto")
    if fmt == "auto":
        fmt = detect_checkpoint_format(checkpoint_path)
    logger.info(f"[Step1] Detected checkpoint format: {fmt}")

    if dry_run:
        logger.info(f"[Step1] DRY-RUN: would convert {fmt} → HF at {hf_path}")
        return hf_path

    if fmt == "hf":
        logger.info(f"[Step1] Checkpoint is already HF format: {checkpoint_path}")
        ckpt_mgr.mark_step_complete("model", checkpoint_path, 0)
        return checkpoint_path

    elif fmt == "fsdp":
        actor_dir = os.path.join(checkpoint_path, "actor")
        if not _hf_model_has_weights(hf_path):
            logger.info(f"[Step1] Merging FSDP checkpoint: {actor_dir} → {hf_path}")
            os.makedirs(hf_path, exist_ok=True)
            # verl.model_merger needs hf_path/config.json + tokenizer files to
            # determine model architecture and save the tokenizer after merge.
            # If missing (e.g. training didn't write them), copy from base model.
            if not os.path.exists(os.path.join(hf_path, "config.json")):
                base_model_path = _get_base_model_path(actor_dir)
                if not base_model_path:
                    raise RuntimeError(
                        f"Cannot find config.json for FSDP merge.\n"
                        f"Expected at {hf_path}/config.json, or derivable from "
                        f"{actor_dir}/lora_adapter/adapter_config.json"
                    )
                tokenizer_filenames = [
                    "config.json", "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt", "special_tokens_map.json",
                    "added_tokens.json", "generation_config.json",
                ]
                copied = []
                for fname in tokenizer_filenames:
                    src = os.path.join(base_model_path, fname)
                    dst = os.path.join(hf_path, fname)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied.append(fname)
                logger.info(f"[Step1] Copied from base model {base_model_path}: {copied}")
            cmd = [
                sys.executable, "-m", "verl.model_merger", "merge",
                "--backend", "fsdp",
                "--local_dir", actor_dir,
                "--target_dir", hf_path,
            ]
            logger.info(f"[Step1] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            logger.info(f"[Step1] FSDP merge already done, base weights exist at {hf_path}")

        # FSDP merger extracts LoRA to {hf_path}/lora_adapter/ — merge it into base
        if merge_lora:
            _merge_lora_into_base(hf_path, actor_dir, config["model"].get("dtype", "bfloat16"), logger)

    elif fmt == "lora":
        logger.info(f"[Step1] LoRA checkpoint detected at {checkpoint_path}")
        # Expect base model path in adapter_config.json; use peft directly
        os.makedirs(hf_path, exist_ok=True)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        import json as _json

        with open(os.path.join(checkpoint_path, "adapter_config.json")) as f:
            adapter_cfg = _json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", "")
        if not base_model_path or not os.path.isdir(base_model_path):
            raise ValueError(
                f"Cannot find base model for LoRA adapter. "
                f"adapter_config.json base_model_name_or_path={base_model_path!r}"
            )
        torch_dtype = getattr(torch, config["model"].get("dtype", "bfloat16"), torch.bfloat16)
        logger.info(f"[Step1] Loading base model from {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        model.save_pretrained(hf_path)
        # Copy tokenizer from base model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tok.save_pretrained(hf_path)

    else:
        raise ValueError(f"Unsupported checkpoint format: {fmt}")

    logger.info(f"[Step1] Model ready at {hf_path}")
    ckpt_mgr.mark_step_complete("model", hf_path, 0)
    return hf_path


# ---------------------------------------------------------------------------
# Step 2: Rollout - vLLM multi-GPU data parallel via Ray
# ---------------------------------------------------------------------------

import ray


@ray.remote
class VLLMRolloutWorker:
    """Single vLLM worker for data-parallel rollout."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],
        tp_size: int,
        dtype: str,
        max_model_len: int,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

        from vllm import LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=False,
        )

    def generate_batch(
        self,
        prompt_texts: List[str],  # already formatted (system+user merged)
        num_rollouts: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> List[List[str]]:
        """
        Generate `num_rollouts` completions per prompt.
        Returns List[List[str]] shape [n_prompts, num_rollouts].
        """
        from vllm import SamplingParams
        sp = SamplingParams(
            n=num_rollouts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        # Repeat prompts for vLLM (n= handles it natively)
        outputs = self.llm.generate(prompt_texts, sampling_params=sp, use_tqdm=False)
        results = []
        for out in outputs:
            completions = [c.text for c in out.outputs]
            results.append(completions)
        return results


def _build_vllm_prompt(tokenizer, question: str, original_cot: str) -> str:
    """Build the full vLLM prompt string using the tokenizer's chat template."""
    system_prompt, user_prompt = build_prompt(question, original_cot)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: plain concatenation
        text = f"{system_prompt}\n\n{user_prompt}"
    return text


def _init_rollout_workers(config, hf_model_path, logger):
    """Initialize Ray vLLM workers for rollout. Returns (workers, tokenizer, num_workers)."""
    from transformers import AutoTokenizer

    rollout_cfg = config["rollout"]
    parallel_cfg = rollout_cfg["parallel"]
    tp_size = parallel_cfg.get("tensor_parallel_size", 1)

    gpu_ids_cfg = parallel_cfg.get("gpu_ids")
    if gpu_ids_cfg:
        all_gpu_ids = list(gpu_ids_cfg)
    else:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        all_gpu_ids = list(range(n))

    if not all_gpu_ids:
        raise RuntimeError("No GPUs available for rollout. Set rollout.parallel.gpu_ids in config.")

    num_workers = len(all_gpu_ids) // tp_size
    if num_workers == 0:
        raise RuntimeError(f"gpu_ids={all_gpu_ids} too few for tp_size={tp_size}")

    logger.info(f"[Step2] Workers={num_workers}, tp_size={tp_size}, gpu_ids={all_gpu_ids}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    workers = []
    for w in range(num_workers):
        w_gpus = all_gpu_ids[w * tp_size: (w + 1) * tp_size]
        worker = VLLMRolloutWorker.options(num_gpus=0, num_cpus=1).remote(
            model_path=hf_model_path,
            gpu_ids=w_gpus,
            tp_size=tp_size,
            dtype=config["model"]["dtype"],
            max_model_len=config["model"]["max_model_len"],
        )
        workers.append(worker)

    return workers, tokenizer, num_workers


def _topup_existing_rollouts(
    workers: list,
    tokenizer,
    output_file: str,
    num_rollouts: int,
    gen_cfg: Dict[str, Any],
    logger: logging.Logger,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    For each record in output_file with fewer than num_rollouts rewrite_cots,
    generate the missing rollouts and rewrite the file atomically.
    Returns the updated list of records.
    """
    from tqdm import tqdm

    existing_records = _read_jsonl(output_file)
    num_workers = len(workers)

    # Group records by how many extra rollouts they need.
    # Using actual len(rewrite_cots) so this is safe to re-run after a crash.
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, record in enumerate(existing_records):
        n_have = len(record.get("rewrite_cots", []))
        if n_have < num_rollouts:
            groups[num_rollouts - n_have].append(i)

    if not groups:
        logger.info("[Step2] Top-up: all existing records already have enough rollouts")
        return existing_records

    total_topup = sum(len(v) for v in groups.values())
    total_topup_rollouts = sum(len(v) * n for n, v in groups.items())
    logger.info(f"[Step2] Top-up: generating extra rollouts for {total_topup} records ({total_topup_rollouts} rollouts)")

    # Top-up writes atomically at the end (full rewrite), so no need for ordered
    # buffering — update existing_records in-place as results arrive.
    with tqdm(total=total_topup_rollouts, desc="Top-up", unit="rollout") as pbar:
        for n_extra, record_indices in groups.items():
            # batch_size is in rollouts; convert to samples for this group
            samples_per_batch = max(1, batch_size // n_extra)
            mini_batches = [
                record_indices[i:i + samples_per_batch]
                for i in range(0, len(record_indices), samples_per_batch)
            ]
            queue = list(mini_batches)
            pending = {}  # future -> (w_idx, chunk_idxs)

            # Initial dispatch
            for w_idx in range(num_workers):
                if queue:
                    chunk_idxs = queue.pop(0)
                    prompts = [
                        _build_vllm_prompt(tokenizer, existing_records[i]["question"], existing_records[i]["original_cot"])
                        for i in chunk_idxs
                    ]
                    fut = workers[w_idx].generate_batch.remote(
                        prompts, n_extra,
                        gen_cfg["temperature"], gen_cfg["top_p"], gen_cfg["max_new_tokens"],
                    )
                    pending[fut] = (w_idx, chunk_idxs)

            while pending:
                done_futs, _ = ray.wait(list(pending.keys()), num_returns=1)
                fut = done_futs[0]
                w_idx, chunk_idxs = pending.pop(fut)
                for rec_idx, completions in zip(chunk_idxs, ray.get(fut)):
                    existing_records[rec_idx]["rewrite_cots"].extend(
                        [{"rewrite_cot": c} for c in completions]
                    )
                    pbar.update(n_extra)

                # Dispatch next mini-batch to the freed worker
                if queue:
                    next_chunk = queue.pop(0)
                    prompts = [
                        _build_vllm_prompt(tokenizer, existing_records[i]["question"], existing_records[i]["original_cot"])
                        for i in next_chunk
                    ]
                    new_fut = workers[w_idx].generate_batch.remote(
                        prompts, n_extra,
                        gen_cfg["temperature"], gen_cfg["top_p"], gen_cfg["max_new_tokens"],
                    )
                    pending[new_fut] = (w_idx, next_chunk)

    # Atomic rewrite
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output_file), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fout:
            for record in existing_records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    logger.info(f"[Step2] Top-up complete, rewrote {len(existing_records)} records → {output_file}")
    return existing_records


def step_rollout(
    config: Dict[str, Any],
    ckpt_mgr,
    hf_model_path: str,
    records: List[Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    from tqdm import tqdm

    output_file = os.path.join(output_dir, "intermediate", "rollout_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rollout_cfg = config["rollout"]
    gen_cfg = rollout_cfg["generation"]
    num_rollouts = rollout_cfg["num_rollouts"]
    batch_size = rollout_cfg.get("batch_size", 96)  # rollouts per mini-batch (= samples × num_rollouts sent to vLLM)

    already_done = ckpt_mgr.get_step_completed_count("rollout")
    saved_num_rollouts = ckpt_mgr.get_step_param("rollout", "num_rollouts", 0)
    total = len(records)

    needs_topup = (
        already_done > 0
        and num_rollouts > saved_num_rollouts
        and os.path.exists(output_file)
    )

    if already_done >= total and os.path.exists(output_file) and not needs_topup:
        logger.info(f"[Step2] Rollout already complete ({already_done}/{total}), skipping")
        return output_file

    if needs_topup:
        logger.info(
            f"[Step2] num_rollouts increased {saved_num_rollouts} → {num_rollouts}, "
            f"will top up {already_done} existing records before continuing"
        )
    logger.info(f"[Step2] Rollout: {total} samples, resuming from {already_done}")

    if dry_run:
        logger.info("[Step2] DRY-RUN: skipping rollout generation")
        return output_file

    workers, tokenizer, num_workers = _init_rollout_workers(config, hf_model_path, logger)

    try:
        # Phase 1: Top-up existing records if num_rollouts was increased
        if needs_topup:
            _topup_existing_rollouts(workers, tokenizer, output_file, num_rollouts, gen_cfg, logger, batch_size)
            # Save the new num_rollouts immediately so a crash after top-up but
            # before Phase 2 doesn't redo the top-up on next resume.
            ckpt_mgr.set_step_param("rollout", "num_rollouts", num_rollouts)

        # Phase 2: Generate rollouts for records not yet processed
        remaining = records[already_done:]
        if remaining:
            total_rollouts = len(remaining) * num_rollouts

            # batch_size is in rollouts; convert to samples for slicing
            samples_per_batch = max(1, batch_size // num_rollouts)

            # Build mini-batches (start_idx relative to records[], not remaining[])
            mini_batches = []
            for i in range(0, len(remaining), samples_per_batch):
                mini_batches.append((already_done + i, remaining[i:i + samples_per_batch]))

            mode = "a" if already_done > 0 else "w"
            completed = already_done
            with open(output_file, mode) as fout:
                with tqdm(total=total_rollouts, desc="Rollout", unit="rollout") as pbar:
                    # Work-stealing: keep all workers busy with mini-batches
                    queue = list(mini_batches)
                    pending = {}      # future -> (w_idx, start_idx, batch)
                    buf = {}          # start_idx -> (batch, results) for in-order writing
                    next_write_idx = already_done

                    # Initial dispatch: one mini-batch per worker
                    for w_idx in range(num_workers):
                        if queue:
                            start_idx, batch = queue.pop(0)
                            prompts = [_build_vllm_prompt(tokenizer, r["question"], r["original_cot"]) for r in batch]
                            fut = workers[w_idx].generate_batch.remote(
                                prompts, num_rollouts,
                                gen_cfg["temperature"], gen_cfg["top_p"], gen_cfg["max_new_tokens"],
                            )
                            pending[fut] = (w_idx, start_idx, batch)

                    while pending:
                        done_futs, _ = ray.wait(list(pending.keys()), num_returns=1)
                        fut = done_futs[0]
                        w_idx, start_idx, batch = pending.pop(fut)
                        buf[start_idx] = (batch, ray.get(fut))

                        # Write completed results in-order
                        while next_write_idx in buf:
                            b, results = buf.pop(next_write_idx)
                            for record, completions in zip(b, results):
                                out_record = {
                                    "question": record["question"],
                                    "original_cot": record["original_cot"],
                                    "solution": record["solution"],
                                    "rewrite_cots": [{"rewrite_cot": c} for c in completions],
                                }
                                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                                completed += 1
                                pbar.update(num_rollouts)
                            next_write_idx += len(b)
                            ckpt_mgr.update_step_progress("rollout", completed, output_file)
                        fout.flush()

                        # Dispatch next mini-batch to the freed worker
                        if queue:
                            next_start, next_batch = queue.pop(0)
                            prompts = [_build_vllm_prompt(tokenizer, r["question"], r["original_cot"]) for r in next_batch]
                            new_fut = workers[w_idx].generate_batch.remote(
                                prompts, num_rollouts,
                                gen_cfg["temperature"], gen_cfg["top_p"], gen_cfg["max_new_tokens"],
                            )
                            pending[new_fut] = (w_idx, next_start, next_batch)

        ckpt_mgr.mark_step_complete("rollout", output_file, total)
        ckpt_mgr.set_step_param("rollout", "num_rollouts", num_rollouts)
        logger.info(f"[Step2] Rollout complete: {total} samples → {output_file}")

    finally:
        for w in workers:
            ray.kill(w, no_restart=True)

    return output_file


# ---------------------------------------------------------------------------
# Judge topup helpers (analogous to _topup_existing_rollouts for rollout)
# ---------------------------------------------------------------------------

def _topup_correct_judge_output(
    client,
    output_file: str,
    input_records: List[Dict],
    batch_size: int,
    logger: logging.Logger,
) -> None:
    """
    For records already in output_file with fewer rollouts judged than in input_records,
    judge only the extra rollouts and merge. Atomically rewrites output_file.
    """
    from tqdm import tqdm

    existing = _read_jsonl(output_file)

    flat_questions, flat_cots, flat_solutions = [], [], []
    flat_meta = []  # (existing_record_idx, rollout_idx_in_input)
    for i, out_rec in enumerate(existing):
        in_rec = input_records[i]
        n_have = len(out_rec.get("rewrite_cots", []))
        n_need = len(in_rec.get("rewrite_cots", []))
        for rw_idx in range(n_have, n_need):
            flat_questions.append(in_rec["question"])
            flat_cots.append(in_rec["rewrite_cots"][rw_idx]["rewrite_cot"])
            flat_solutions.append(in_rec.get("solution", ""))
            flat_meta.append((i, rw_idx))

    if not flat_meta:
        logger.info("[Step3] Correct tag topup: no new rollouts to judge")
        return

    logger.info(f"[Step3] Correct tag topup: {len(flat_meta)} new rollouts across {len(existing)} records")
    is_correct_results = [None] * len(flat_meta)
    with tqdm(total=len(flat_meta), desc="Correct judge topup", unit="rollout") as pbar:
        for i in range(0, len(flat_meta), batch_size):
            end = min(i + batch_size, len(flat_meta))
            result = client.compute_correct_judge(
                flat_questions[i:end], flat_cots[i:end], flat_solutions[i:end]
            )
            for j, is_corr in enumerate(result["is_correct"]):
                is_correct_results[i + j] = is_corr
            pbar.update(end - i)

    for (rec_idx, rw_idx), is_corr in zip(flat_meta, is_correct_results):
        in_rw = input_records[rec_idx]["rewrite_cots"][rw_idx]
        existing[rec_idx]["rewrite_cots"].append({"rewrite_cot": in_rw["rewrite_cot"], "is_correct": is_corr})

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output_file), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fout:
            for record in existing:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _topup_fsf_judge_output(
    client,
    output_file: str,
    input_records: List[Dict],
    batch_size: int,
    logger: logging.Logger,
) -> None:
    """
    For records already in output_file with fewer rollouts FSF-judged than in input_records,
    compute FSF only for the extra rollouts and merge. Atomically rewrites output_file.
    """
    from tqdm import tqdm

    existing = _read_jsonl(output_file)

    flat_cots = []
    flat_meta = []  # (existing_record_idx, rollout_idx_in_input)
    for i, out_rec in enumerate(existing):
        in_rec = input_records[i]
        n_have = len(out_rec.get("rewrite_cots", []))
        n_need = len(in_rec.get("rewrite_cots", []))
        for rw_idx in range(n_have, n_need):
            flat_cots.append(in_rec["rewrite_cots"][rw_idx]["rewrite_cot"])
            flat_meta.append((i, rw_idx))

    if not flat_meta:
        logger.info("[Step4] FSF tag topup: no new rollouts to judge")
        return

    logger.info(f"[Step4] FSF tag topup: {len(flat_meta)} new rollouts across {len(existing)} records")
    fsf_results = [None] * len(flat_meta)
    dot_results = [None] * len(flat_meta)
    with tqdm(total=len(flat_meta), desc="FSF judge topup", unit="rollout") as pbar:
        for i in range(0, len(flat_meta), batch_size):
            end = min(i + batch_size, len(flat_meta))
            result = client.compute_rewrite_fsf_only(flat_cots[i:end], return_dot_outputs=True)
            for j, (fsf_val, dot_out) in enumerate(zip(
                result["rewrite_fsfs"], result.get("rewrite_dot_outputs", [None] * (end - i))
            )):
                fsf_results[i + j] = fsf_val
                dot_results[i + j] = dot_out
            pbar.update(end - i)

    for (rec_idx, rw_idx), fsf_val, dot_out in zip(flat_meta, fsf_results, dot_results):
        in_rw = input_records[rec_idx]["rewrite_cots"][rw_idx]
        existing[rec_idx]["rewrite_cots"].append({
            "rewrite_cot": in_rw["rewrite_cot"],
            "is_correct": in_rw.get("is_correct"),
            "fsf": fsf_val,
            "fsf_model_output": dot_out,
        })
    # Recompute record-level fsf_tag for topped-up records
    topped_up_idxs = {rec_idx for rec_idx, _ in flat_meta}
    for idx in topped_up_idxs:
        existing[idx]["fsf_tag"] = any(
            (rw.get("fsf") or 0) > 0 for rw in existing[idx].get("rewrite_cots", [])
        )

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output_file), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fout:
            for record in existing:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Step 3: Correct Tag
# ---------------------------------------------------------------------------

def step_correct_tag(
    config: Dict[str, Any],
    ckpt_mgr,
    rollout_file: str,
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    output_file = os.path.join(output_dir, "intermediate", "correct_tag_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cj_cfg = config["correct_judge"]
    if not cj_cfg.get("enabled", True):
        logger.info("[Step3] Correct judge disabled, copying rollout output")
        shutil.copy2(rollout_file, output_file)
        return output_file

    # Read input
    records = _read_jsonl(rollout_file)
    total_records = len(records)
    already_done = ckpt_mgr.get_step_completed_count("correct_tag")

    # Detect topup: input has more rollouts per record than what was previously judged
    needs_topup = False
    if already_done > 0 and os.path.exists(output_file) and records:
        input_rollouts = len(records[0].get("rewrite_cots", []))
        with open(output_file) as _f:
            first_line = _f.readline()
        if first_line:
            existing_rollouts = len(json.loads(first_line).get("rewrite_cots", []))
            if existing_rollouts < input_rollouts:
                needs_topup = True
                logger.info(f"[Step3] Input rollouts increased ({existing_rollouts} → {input_rollouts}), will topup already-judged records")

    if already_done >= total_records and os.path.exists(output_file) and not needs_topup:
        logger.info(f"[Step3] Correct tag already complete ({already_done}/{total_records}), skipping")
        return output_file

    logger.info(f"[Step3] Correct tag: {total_records} records, resuming from {already_done}")

    if dry_run:
        logger.info("[Step3] DRY-RUN: skipping correct judge")
        return output_file

    model_path = cj_cfg.get("model_path")
    if not model_path:
        raise ValueError("correct_judge.model_path is required when correct_judge.enabled=true")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    import torch
    from verl.experimental.reward.cot_rewriter.sglang_correct_judge import SGLangCorrectJudgeClient

    gen = cj_cfg["generation"]
    devices = _get_judge_devices(cj_cfg)
    batch_size = cj_cfg.get("batch_size", 96)  # total items per step (same semantics as rollout batch_size)
    tp = cj_cfg.get("tensor_parallel_size", 1)
    num_gpus = len(devices) if devices is not None else torch.cuda.device_count()
    num_actors = num_gpus // tp
    micro_batch_size = max(1, batch_size // num_actors)
    client = SGLangCorrectJudgeClient(
        model_path=model_path,
        tensor_parallel_size=tp,
        mem_fraction_static=cj_cfg.get("mem_fraction_static", 0.4),
        dtype=cj_cfg.get("dtype", "float16"),
        max_model_len=cj_cfg.get("max_model_len", 8192),
        devices=devices,
        temperature=gen.get("temperature", 0.0),
        max_new_tokens=gen.get("max_new_tokens", 50),
        micro_batch_size=micro_batch_size,
    )

    try:
        if needs_topup:
            _topup_correct_judge_output(client, output_file, records, batch_size, logger)
            # already_done unchanged: topup only fills existing records, count stays the same

        mode = "a" if already_done > 0 else "w"
        completed = already_done

        pending_records = records[already_done:]

        # Flatten all rewrite_cots
        flat_questions = []
        flat_rewrite_cots = []
        flat_solutions = []

        for record in pending_records:
            for rw in record.get("rewrite_cots", []):
                flat_questions.append(record["question"])
                flat_rewrite_cots.append(rw["rewrite_cot"])
                flat_solutions.append(record["solution"])

        # Pre-compute cumulative rollout counts per record for incremental writing
        record_flat_end = []
        running = 0
        for record in pending_records:
            running += len(record.get("rewrite_cots", []))
            record_flat_end.append(running)

        # Run judge in batches, writing complete records incrementally (same as rollout)
        from tqdm import tqdm
        is_correct_results = [None] * len(flat_questions)
        next_record_to_write = 0
        with open(output_file, mode) as fout:
            with tqdm(total=len(flat_questions), desc="Correct judge", unit="rollout") as pbar:
                for i in range(0, len(flat_questions), batch_size):
                    end = min(i + batch_size, len(flat_questions))
                    result = client.compute_correct_judge(
                        flat_questions[i:end], flat_rewrite_cots[i:end], flat_solutions[i:end]
                    )
                    for j, is_corr in enumerate(result["is_correct"]):
                        is_correct_results[i + j] = is_corr
                    pbar.update(end - i)

                    # Write records whose all rollouts are covered by [0, end)
                    while (next_record_to_write < len(pending_records) and
                           record_flat_end[next_record_to_write] <= end):
                        record = pending_records[next_record_to_write]
                        flat_start = record_flat_end[next_record_to_write] - len(record.get("rewrite_cots", []))
                        for rw_idx, rw in enumerate(record.get("rewrite_cots", [])):
                            rw["is_correct"] = is_correct_results[flat_start + rw_idx]
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        completed += 1
                        next_record_to_write += 1
                    ckpt_mgr.update_step_progress("correct_tag", completed, output_file)

    finally:
        client.shutdown()

    ckpt_mgr.mark_step_complete("correct_tag", output_file, completed)
    logger.info(f"[Step3] Correct tag complete: {completed} records → {output_file}")
    return output_file




def _topup_original_fsf_in_file(
    client,
    output_file: str,
    batch_size: int,
    logger: logging.Logger,
) -> None:
    """
    Compute original_fsf and original_fsf_model_output for records that are missing them
    (e.g. produced by an older pipeline version that skipped original-CoT FSF computation).
    Records that gain an original_fsf_model_output have their rewrite GED set to 'pending'
    so _compute_ged_pass can fill them in with the now-available original DOT graph.
    Atomically rewrites output_file when done.
    """
    from tqdm import tqdm

    records = _read_jsonl(output_file)
    missing_idxs = [
        i for i, rec in enumerate(records)
        if not rec.get("original_fsf_model_output")
    ]

    if not missing_idxs:
        logger.info("[Step4-OrigFSF] No records missing original_fsf_model_output, skipping topup")
        return

    logger.info(f"[Step4-OrigFSF] Computing original_fsf for {len(missing_idxs)} records...")
    original_cots = [records[i]["original_cot"] for i in missing_idxs]
    orig_fsf_list = [None] * len(missing_idxs)
    orig_dot_list = [None] * len(missing_idxs)

    with tqdm(total=len(missing_idxs), desc="Original CoT FSF topup", unit="record") as pbar:
        for i in range(0, len(missing_idxs), batch_size):
            end = min(i + batch_size, len(missing_idxs))
            result = client.compute_rewrite_fsf_only(original_cots[i:end], return_dot_outputs=True)
            for j, (fsf_val, dot_out) in enumerate(zip(
                result["rewrite_fsfs"],
                result.get("rewrite_dot_outputs", [None] * (end - i)),
            )):
                orig_fsf_list[i + j] = fsf_val
                orig_dot_list[i + j] = dot_out
            pbar.update(end - i)

    for list_idx, rec_idx in enumerate(missing_idxs):
        dot_out = orig_dot_list[list_idx]
        records[rec_idx]["original_fsf"] = orig_fsf_list[list_idx]
        records[rec_idx]["original_fsf_model_output"] = dot_out[:32000] if dot_out else dot_out

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output_file), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fout:
            for record in records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    logger.info(f"[Step4-OrigFSF] Topup complete: updated {len(missing_idxs)} records → {output_file}")


# ---------------------------------------------------------------------------
# Step 4: FSF Tag
# ---------------------------------------------------------------------------

def step_fsf_tag(
    config: Dict[str, Any],
    ckpt_mgr,
    correct_file: str,
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    output_file = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fsf_cfg = config["fsf_judge"]
    if not fsf_cfg.get("enabled", True):
        logger.info("[Step4] FSF judge disabled, copying correct_tag output")
        shutil.copy2(correct_file, output_file)
        return output_file

    records = _read_jsonl(correct_file)
    total_records = len(records)
    already_done = ckpt_mgr.get_step_completed_count("fsf_tag")

    # Detect topup: input has more rollouts per record than what was previously judged
    needs_topup = False
    if already_done > 0 and os.path.exists(output_file) and records:
        input_rollouts = len(records[0].get("rewrite_cots", []))
        with open(output_file) as _f:
            first_line = _f.readline()
        if first_line:
            existing_rollouts = len(json.loads(first_line).get("rewrite_cots", []))
            if existing_rollouts < input_rollouts:
                needs_topup = True
                logger.info(f"[Step4] Input rollouts increased ({existing_rollouts} → {input_rollouts}), will topup already-judged records")

    model_path = fsf_cfg.get("model_path")

    if already_done >= total_records and os.path.exists(output_file) and not needs_topup:
        sample = _read_jsonl(output_file)
        # Check for records missing original_fsf (produced by older pipeline versions)
        n_missing_orig = sum(1 for rec in sample if not rec.get("original_fsf_model_output"))
        original_fsf_missing = n_missing_orig > 0

        if original_fsf_missing and not dry_run:
            logger.info(f"[Step4] {n_missing_orig} records missing original_fsf_model_output, running topup...")
            if not model_path:
                raise ValueError("fsf_judge.model_path is required for original_fsf topup")
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            import torch
            from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient
            gen = fsf_cfg["generation"]
            devices = _get_judge_devices(fsf_cfg)
            batch_size_cfg = fsf_cfg.get("batch_size", 96)
            tp = fsf_cfg.get("tensor_parallel_size", 1)
            num_gpus_cfg = len(devices) if devices is not None else torch.cuda.device_count()
            num_actors_cfg = num_gpus_cfg // tp
            micro_bs = max(1, batch_size_cfg // num_actors_cfg)
            topup_client = SGLangFSFClient(
                model_path=model_path,
                tensor_parallel_size=tp,
                mem_fraction_static=fsf_cfg.get("mem_fraction_static", 0.4),
                dtype=fsf_cfg.get("dtype", "float16"),
                max_model_len=fsf_cfg.get("max_model_len", 8192),
                devices=devices,
                temperature=gen.get("temperature", 0.6),
                max_new_tokens=gen.get("max_new_tokens", 8000),
                micro_batch_size=micro_bs,
            )
            try:
                _topup_original_fsf_in_file(topup_client, output_file, batch_size_cfg, logger)
            finally:
                topup_client.shutdown()
        elif original_fsf_missing and dry_run:
            logger.info(f"[Step4] DRY-RUN: would topup original_fsf for {n_missing_orig} records")
        else:
            logger.info(f"[Step4] FSF tag already complete ({already_done}/{total_records}), skipping")

        return output_file

    logger.info(f"[Step4] FSF tag: {total_records} records, resuming from {already_done}")

    if dry_run:
        logger.info("[Step4] DRY-RUN: skipping FSF judge")
        return output_file

    if not model_path:
        raise ValueError("fsf_judge.model_path is required when fsf_judge.enabled=true")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    import torch
    from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient

    gen = fsf_cfg["generation"]
    devices = _get_judge_devices(fsf_cfg)
    batch_size = fsf_cfg.get("batch_size", 96)  # total items per step (same semantics as rollout batch_size)
    tp = fsf_cfg.get("tensor_parallel_size", 1)
    num_gpus = len(devices) if devices is not None else torch.cuda.device_count()
    num_actors = num_gpus // tp
    micro_batch_size = max(1, batch_size // num_actors)
    client = SGLangFSFClient(
        model_path=model_path,
        tensor_parallel_size=tp,
        mem_fraction_static=fsf_cfg.get("mem_fraction_static", 0.4),
        dtype=fsf_cfg.get("dtype", "float16"),
        max_model_len=fsf_cfg.get("max_model_len", 8192),
        devices=devices,
        temperature=gen.get("temperature", 0.6),
        max_new_tokens=gen.get("max_new_tokens", 8000),
        micro_batch_size=micro_batch_size,
    )

    try:
        if needs_topup:
            _topup_fsf_judge_output(client, output_file, records, batch_size, logger)
            # already_done unchanged: topup only fills existing records, count stays the same

        mode = "a" if already_done > 0 else "w"
        completed = already_done

        pending_records = records[already_done:]

        # Compute original_cot FSF and dot graphs upfront (one per record)
        from tqdm import tqdm
        original_fsf_list = [None] * len(pending_records)
        original_dot_list = [None] * len(pending_records)
        if pending_records:
            flat_original_cots = [record["original_cot"] for record in pending_records]
            logger.info(f"[Step4] Computing original_cot FSF+dot for {len(flat_original_cots)} records...")
            with tqdm(total=len(flat_original_cots), desc="Original CoT FSF", unit="record") as pbar:
                for i in range(0, len(flat_original_cots), batch_size):
                    end = min(i + batch_size, len(flat_original_cots))
                    orig_result = client.compute_rewrite_fsf_only(flat_original_cots[i:end], return_dot_outputs=True)
                    for j, (fsf_val, dot_out) in enumerate(zip(
                        orig_result["rewrite_fsfs"],
                        orig_result.get("rewrite_dot_outputs", [None] * (end - i))
                    )):
                        original_fsf_list[i + j] = fsf_val
                        original_dot_list[i + j] = dot_out
                    pbar.update(end - i)

        # Flatten rewrite_cots
        flat_rewrite_cots = []
        for record in pending_records:
            for rw in record.get("rewrite_cots", []):
                flat_rewrite_cots.append(rw["rewrite_cot"])

        # Pre-compute cumulative rollout counts per record for incremental writing
        record_flat_end = []
        running = 0
        for record in pending_records:
            running += len(record.get("rewrite_cots", []))
            record_flat_end.append(running)

        # Run FSF in batches, writing complete records incrementally (same as rollout)
        fsf_results = [None] * len(flat_rewrite_cots)
        dot_results = [None] * len(flat_rewrite_cots)
        next_record_to_write = 0
        with open(output_file, mode) as fout:
            with tqdm(total=len(flat_rewrite_cots), desc="FSF judge", unit="rollout") as pbar:
                for i in range(0, len(flat_rewrite_cots), batch_size):
                    end = min(i + batch_size, len(flat_rewrite_cots))
                    batch_cots = flat_rewrite_cots[i:end]
                    result = client.compute_rewrite_fsf_only(batch_cots, return_dot_outputs=True)
                    for j, (fsf_val, dot_out) in enumerate(zip(
                        result["rewrite_fsfs"], result.get("rewrite_dot_outputs", [None] * len(batch_cots))
                    )):
                        fsf_results[i + j] = fsf_val
                        dot_results[i + j] = dot_out
                    pbar.update(end - i)

                    # Write records whose all rollouts are covered by [0, end)
                    while (next_record_to_write < len(pending_records) and
                           record_flat_end[next_record_to_write] <= end):
                        record = pending_records[next_record_to_write]
                        flat_start = record_flat_end[next_record_to_write] - len(record.get("rewrite_cots", []))
                        orig_dot = original_dot_list[next_record_to_write]
                        for rw_idx, rw in enumerate(record.get("rewrite_cots", [])):
                            rw_dot = dot_results[flat_start + rw_idx]
                            rw["fsf"] = fsf_results[flat_start + rw_idx]
                            # Cap DOT output size: parse failures can produce huge repetition-loop garbage
                            rw["fsf_model_output"] = rw_dot[:32000] if rw_dot else rw_dot
                        record["original_fsf"] = original_fsf_list[next_record_to_write]
                        record["original_fsf_model_output"] = orig_dot[:32000] if orig_dot else orig_dot
                        record["fsf_tag"] = any(
                            (rw.get("fsf") or 0) > 0 for rw in record.get("rewrite_cots", [])
                        )
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        completed += 1
                        next_record_to_write += 1
                    ckpt_mgr.update_step_progress("fsf_tag", completed, output_file)

    finally:
        client.shutdown()

    ckpt_mgr.mark_step_complete("fsf_tag", output_file, completed)
    logger.info(f"[Step4] FSF tag complete: {completed} records → {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_dot_stats(dot_string: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a DOT graph string and return failure stats.
    Must be module-level for multiprocessing.Pool.imap compatibility.
    Returns dict with {fsf, failed_nodes, total_nodes} or None if parsing fails.
    """
    if not dot_string:
        return None
    try:
        import pydot
        cleaned = dot_string.lstrip('\n').strip('`').strip()
        graphs = pydot.graph_from_dot_data(cleaned)
        if not graphs:
            return None
        graph = graphs[0]
        nodes = graph.get_nodes() or []
        node_count = len(nodes)
        if node_count == 0:
            return {"fsf": 0.0, "failed_nodes": 0, "total_nodes": 0}
        failed_nodes = 0
        for node in nodes:
            if node is None:
                continue
            try:
                fillcolor = node.get('fillcolor')
                if fillcolor is not None and str(fillcolor).strip('"').lower() == 'lightpink':
                    failed_nodes += 1
            except Exception:
                continue
        return {"fsf": failed_nodes / node_count, "failed_nodes": failed_nodes, "total_nodes": node_count}
    except Exception:
        return None


def _get_judge_devices(judge_cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Return device list from judge config, or None for auto-detect."""
    gpu_ids = judge_cfg.get("gpu_ids")
    if gpu_ids:
        return [f"cuda:{i}" for i in gpu_ids]
    return None


# ---------------------------------------------------------------------------
# Step 5 (optional): Distribution analysis
# ---------------------------------------------------------------------------

def step_analyze_distributions(
    output_file: str,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Analyze and report FSF and failure-node-count distributions.
    Reads fsf_tag_output.jsonl; saves report to {output_dir}/analysis/fsf_distributions.txt.
    Does not require GPU.
    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    if not os.path.exists(output_file):
        logger.warning(f"[Analyze] {output_file} not found, skipping distribution analysis")
        return

    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    report_file = os.path.join(analysis_dir, "fsf_distributions.txt")

    records = _read_jsonl(output_file)
    n_records = len(records)
    logger.info(f"[Analyze] Loaded {n_records} records for distribution analysis")

    # --- Collect stored FSF float values (no DOT parsing needed) ---
    orig_fsf_vals: List[float] = []
    rw_fsf_vals_all: List[float] = []
    rw_fsf_per_record: List[List[float]] = []
    for rec in records:
        fsf = rec.get("original_fsf")
        if fsf is not None:
            orig_fsf_vals.append(float(fsf))
        row: List[float] = []
        for rw in rec.get("rewrite_cots", []):
            f = rw.get("fsf")
            if f is not None:
                v = float(f)
                rw_fsf_vals_all.append(v)
                row.append(v)
        rw_fsf_per_record.append(row)

    # --- Parse DOT graphs in parallel for failure-node counts ---
    logger.info(f"[Analyze] Parsing DOT graphs for failure-node counts (parallel)...")
    orig_dot_strs = [rec.get("original_fsf_model_output") for rec in records]
    rw_dot_strs = [
        rw.get("fsf_model_output")
        for rec in records
        for rw in rec.get("rewrite_cots", [])
    ]

    n_workers = max(1, min(cpu_count() or 8, 16))
    with Pool(n_workers) as pool:
        cs = max(1, len(orig_dot_strs) // (n_workers * 4))
        orig_dot_stats = list(tqdm(
            pool.imap(_parse_dot_stats, orig_dot_strs, chunksize=cs),
            total=len(orig_dot_strs), desc="Parsing original DOTs", unit="rec",
        ))
    with Pool(n_workers) as pool:
        cs = max(1, len(rw_dot_strs) // (n_workers * 4))
        rw_dot_stats_flat = list(tqdm(
            pool.imap(_parse_dot_stats, rw_dot_strs, chunksize=cs),
            total=len(rw_dot_strs), desc="Parsing rewrite DOTs", unit="rollout",
        ))

    orig_failed_nodes = [s["failed_nodes"] for s in orig_dot_stats if s is not None]
    orig_total_nodes  = [s["total_nodes"]  for s in orig_dot_stats if s is not None]
    rw_failed_nodes   = [s["failed_nodes"] for s in rw_dot_stats_flat if s is not None]
    rw_total_nodes    = [s["total_nodes"]  for s in rw_dot_stats_flat if s is not None]

    # --- Helpers ---
    def _summary(vals: List, label: str) -> List[str]:
        if not vals:
            return [f"  {label}: no data"]
        n = len(vals)
        s = sorted(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        std = var ** 0.5
        pcts = {p: s[min(int(n * p / 100), n - 1)] for p in [10, 25, 50, 75, 90, 99]}
        return [
            f"  {label}:  n={n}  mean={mean:.3f}  std={std:.3f}",
            f"    P10={pcts[10]:.3f}  P25={pcts[25]:.3f}  P50={pcts[50]:.3f}"
            f"  P75={pcts[75]:.3f}  P90={pcts[90]:.3f}  P99={pcts[99]:.3f}",
        ]

    def _hist(vals: List, bins: List, label: str) -> List[str]:
        if not vals:
            return [f"  {label} histogram: no data"]
        n = len(vals)
        counts = [0] * (len(bins) - 1)
        for v in vals:
            placed = False
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
        max_cnt = max(counts) if counts else 1
        out = [f"  {label} histogram:"]
        for i, cnt in enumerate(counts):
            lo = bins[i]
            hi = bins[i + 1] if i + 1 < len(bins) else "inf"
            bar = "█" * max(1, int(36 * cnt / max(max_cnt, 1))) if cnt > 0 else ""
            pct = 100.0 * cnt / n if n > 0 else 0.0
            hi_str = f"{hi:.2f}" if isinstance(hi, float) else str(hi)
            out.append(f"    [{lo:>6.2f}, {hi_str:>6}): {cnt:8d} ({pct:5.1f}%) {bar}")
        return out

    fsf_bins  = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    node_bins = [0, 1, 2, 3, 5, 10, 20, 50, 200]

    # --- Build report ---
    lines: List[str] = []
    sep = "=" * 64
    lines += [sep, "FSF Distribution Analysis", sep]
    lines += [
        f"Total records:              {n_records}",
        f"Total rewrite rollouts:     {len(rw_fsf_vals_all)}",
        f"Records with original_fsf:  {len(orig_fsf_vals)} / {n_records}"
        f" ({100*len(orig_fsf_vals)/max(n_records,1):.1f}%)",
        f"DOT parse success (orig):   {len(orig_failed_nodes)} / {n_records}"
        f" ({100*len(orig_failed_nodes)/max(n_records,1):.1f}%)",
        f"DOT parse success (rw):     {len(rw_failed_nodes)} / {max(len(rw_dot_strs),1)}"
        f" ({100*len(rw_failed_nodes)/max(len(rw_dot_strs),1):.1f}%)",
        "",
    ]

    lines += ["--- Original CoT FSF ---"]
    lines += _summary(orig_fsf_vals, "FSF")
    lines += _hist(orig_fsf_vals, fsf_bins, "FSF")
    lines += [""]

    lines += ["--- Original CoT Failure Nodes ---"]
    lines += _summary(orig_failed_nodes, "Failed nodes")
    lines += _hist(orig_failed_nodes, node_bins, "Failed nodes")
    lines += [""]
    lines += _summary(orig_total_nodes, "Total nodes")
    lines += _hist(orig_total_nodes, node_bins, "Total nodes")
    lines += [""]

    lines += ["--- Rewrite CoT FSF (all rollouts) ---"]
    lines += _summary(rw_fsf_vals_all, "FSF")
    lines += _hist(rw_fsf_vals_all, fsf_bins, "FSF")
    lines += [""]

    lines += ["--- Rewrite CoT Failure Nodes ---"]
    lines += _summary(rw_failed_nodes, "Failed nodes")
    lines += _hist(rw_failed_nodes, node_bins, "Failed nodes")
    lines += [""]
    lines += _summary(rw_total_nodes, "Total nodes")
    lines += _hist(rw_total_nodes, node_bins, "Total nodes")
    lines += [""]

    # FSF improvement: compare original vs best rewrite per record
    delta: List[float] = []
    for orig_fsf, rw_row in zip(orig_fsf_vals, rw_fsf_per_record):
        if rw_row:
            delta.append(orig_fsf - min(rw_row))
    lines += ["--- FSF Improvement: original - min(rewrite) per record ---"]
    lines += _summary(delta, "Improvement")
    if delta:
        n_imp = sum(1 for d in delta if d > 0)
        n_zero = sum(1 for d in delta if d == 0)
        n_worse = sum(1 for d in delta if d < 0)
        nd = len(delta)
        lines += [
            f"  Improved  (delta > 0): {n_imp:7d} / {nd} ({100*n_imp/nd:.1f}%)",
            f"  No change (delta = 0): {n_zero:7d} / {nd} ({100*n_zero/nd:.1f}%)",
            f"  Worse     (delta < 0): {n_worse:7d} / {nd} ({100*n_worse/nd:.1f}%)",
        ]

    report = "\n".join(lines)
    logger.info("[Analyze]\n" + report)
    with open(report_file, "w") as f:
        f.write(report + "\n")
    logger.info(f"[Analyze] Report saved to {report_file}")


STEP_ORDER = ["model", "rollout", "correct", "fsf"]


def determine_start_step(
    args: argparse.Namespace,
    ckpt_mgr,
    config: Dict[str, Any],
) -> str:
    if args.step:
        step_map = {"model": "model", "rollout": "rollout", "correct": "correct", "fsf": "fsf", "analyze": "analyze"}
        return step_map.get(args.step, "model")

    if args.resume or config.get("resume", True):
        state = ckpt_mgr.load()
        if state:
            completed = state["progress"]["completed_steps"]
            for step in STEP_ORDER:
                if step not in completed:
                    return step
            return "done"

    return "model"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CoT Rewrite Inference Pipeline (steps 1-4)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--step",
        choices=["model", "rollout", "correct", "fsf", "analyze"],
        help="Start from this step (overrides resume logic). "
             "'analyze' re-runs distribution analysis on existing fsf_tag_output.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Force resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check config and paths without running anything",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Add verl to path
    verl_path = os.path.join(os.path.dirname(__file__), "..", "verl")
    if verl_path not in sys.path:
        sys.path.insert(0, os.path.abspath(verl_path))

    from inference.utils.config_loader import load_config
    from inference.utils.checkpoint import CheckpointManager

    config = load_config(args.config)
    exp_name = config["experiment_name"]
    output_dir = os.path.join(config["output_root"], exp_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir, "pipeline")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output dir: {output_dir}")

    # Copy config to output dir (for reference)
    config_copy = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_copy):
        shutil.copy2(args.config, config_copy)

    ckpt_mgr = CheckpointManager(output_dir)

    if args.dry_run:
        logger.info("DRY-RUN mode enabled - will check paths without executing")

    # --step analyze: read-only, just (re-)run distribution analysis and exit
    if args.step == "analyze":
        fsf_output = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")
        step_analyze_distributions(fsf_output, output_dir, logger)
        return

    start_step = determine_start_step(args, ckpt_mgr, config)

    if start_step == "done":
        logger.info("All steps already complete.")
        fsf_output = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")
        if os.path.exists(fsf_output):
            step_analyze_distributions(fsf_output, output_dir, logger)
        return

    logger.info(f"Starting from step: {start_step}")

    hf_model_path = None
    rollout_file = None
    correct_file = None

    # Step 1: Model loading
    if start_step == "model":
        hf_model_path = step_model_loading(config, ckpt_mgr, logger, dry_run=args.dry_run)
        start_step = "rollout"
    else:
        # Recover hf_model_path from checkpoint or detect.
        # Use _hf_model_has_weights to validate — a directory may exist but be
        # incomplete (e.g. a previous merge crashed before writing weights).
        saved_path = ckpt_mgr.get_step_output("model")
        cp = config["model"]["checkpoint_path"]
        candidate = os.path.join(cp, "actor", "huggingface")
        if saved_path and _hf_model_has_weights(saved_path):
            hf_model_path = saved_path
        elif _hf_model_has_weights(candidate):
            hf_model_path = candidate
        elif _hf_model_has_weights(cp):
            hf_model_path = cp
        else:
            logger.info("[Step1] HF model weights not found or incomplete, re-running model loading")
            hf_model_path = step_model_loading(config, ckpt_mgr, logger, dry_run=args.dry_run)

    # Step 2: Rollout
    if start_step == "rollout":
        records = load_input_data(config["rollout"]["input_data"])
        logger.info(f"Loaded {len(records)} input records")
        ckpt_mgr.init_state(
            {"model_path": hf_model_path, "input_data": config["rollout"]["input_data"]},
            len(records),
        )
        rollout_file = step_rollout(
            config, ckpt_mgr, hf_model_path, records, output_dir, logger, dry_run=args.dry_run
        )
        start_step = "correct"
    else:
        rollout_file = ckpt_mgr.get_step_output("rollout")
        if not rollout_file or not os.path.exists(rollout_file):
            raise FileNotFoundError(
                f"Rollout output not found. Run with --step rollout first.\n"
                f"Expected: {rollout_file}"
            )

    # Step 3: Correct Tag
    if start_step == "correct":
        correct_file = step_correct_tag(
            config, ckpt_mgr, rollout_file, output_dir, logger, dry_run=args.dry_run
        )
        start_step = "fsf"
    else:
        correct_file = ckpt_mgr.get_step_output("correct_tag")
        if not correct_file or not os.path.exists(correct_file):
            raise FileNotFoundError(
                f"Correct tag output not found. Run with --step correct first.\n"
                f"Expected: {correct_file}"
            )

    # Step 4: FSF Tag
    if start_step == "fsf":
        step_fsf_tag(
            config, ckpt_mgr, correct_file, output_dir, logger, dry_run=args.dry_run
        )

    logger.info("Pipeline steps 1-4 complete.")
    logger.info(f"Output directory: {output_dir}")

    # Distribution analysis (always runs after step 4, or when resuming a fully-done run)
    fsf_output = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")
    if not args.dry_run and os.path.exists(fsf_output):
        step_analyze_distributions(fsf_output, output_dir, logger)


if __name__ == "__main__":
    main()
