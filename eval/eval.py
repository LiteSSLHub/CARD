#!/usr/bin/env python3
"""
EvalScope evaluation script - Offline vLLM inference
Unified CLI for sequential and cross-benchmark streaming evaluation.

Uses vLLM offline mode for direct inference with TP/DP support; no HTTP server required.
- dp=1: single process, zero zombie-process risk
- dp>1: data parallelism via Ray, keeps all GPUs saturated

Modes:
- Sequential (default): run benchmarks one by one, supports multi-run evaluation.
- Cross-benchmark (--cross-bench): mix all prompts into a global queue;
  GPU runs at full load continuously; scoring is triggered on CPU threads
  immediately after each benchmark finishes inference.

Usage:
    # Sequential, single run
    python eval.py eval_configs/eval_all_sft.yaml

    # Cross-benchmark streaming
    python eval.py eval_configs/eval_all_sft.yaml --cross-bench

    # Resume a previous run
    python eval.py eval_configs/eval_all_sft.yaml --cross-bench --resume
"""

import os
import sys
import gc
import json
import copy
import time
import logging
import threading
import re
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union, Tuple, Any, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

# Add evalscope path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evalscope'))
from evalscope.config import TaskConfig
from evalscope.run import run_task
from evalscope.constants import EvalBackend, JudgeStrategy
from evalscope.utils.logger import get_logger
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.model.model import ModelAPI, Model
from evalscope.api.model.model_output import ModelOutput, as_stop_reason
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.dataset import Dataset, Sample
from evalscope.api.registry import get_benchmark

from vllm import LLM, SamplingParams

logger = get_logger()




# ============================================================
# Utility functions
# ============================================================

def _extract_report_subsets(dataset_result: dict) -> list:
    """Extract per-subset score list from serialized Report dict (used for difficulty category stats).
    Returns [{'name': 'easy', 'score': 0.5, 'num': 100}, ...] or empty list.
    """
    try:
        metrics = dataset_result.get('metrics', [])
        if not (metrics and isinstance(metrics, list)):
            return []
        categories = metrics[0].get('categories', [])
        if not (categories and isinstance(categories, list)):
            return []
        subsets = categories[0].get('subsets', [])
        if subsets and isinstance(subsets, list) and len(subsets) > 0:
            return subsets
    except (IndexError, KeyError, TypeError, AttributeError):
        pass
    return []


def run_task_with_progress(task_config: TaskConfig, benchmark_name: str):
    """Run evaluation task with a progress bar"""
    logger.info(f"[progress] {benchmark_name} starting evaluation")

    from threading import Thread, Event

    progress_bar = None
    stop_event = Event()

    try:
        progress_bar = tqdm(total=100, desc=f"[evaluating] {benchmark_name}",
                           bar_format='{l_bar}{bar} | [{elapsed}<{remaining}]{postfix}')

        def update_spinner():
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            cycle = 0
            while not stop_event.is_set():
                cycle = (cycle + 1) % len(spinner)
                progress_bar.set_description(f"[evaluating] {benchmark_name} {spinner[cycle]}")
                time.sleep(0.1)

        spinner_thread = Thread(target=update_spinner)
        spinner_thread.start()

        result = run_task(task_config)

        stop_event.set()
        spinner_thread.join()

        progress_bar.update(100)
        progress_bar.set_description(f"[done] {benchmark_name} ✓")
        progress_bar.close()

        logger.info(f"[progress] {benchmark_name} evaluation complete ✓")
        return result

    except Exception as e:
        logger.error(f"[progress] {benchmark_name} evaluation failed ✗")
        stop_event.set()
        if progress_bar:
            progress_bar.set_description(f"[failed] {benchmark_name} ✗")
            progress_bar.close()
        raise e


def _extract_model_name(model_path: str) -> str:
    """Extract short name from model path"""
    path_parts = model_path.split('/')
    if 'checkpoint' in path_parts[-1].lower() and len(path_parts) > 1:
        return path_parts[-2]
    return path_parts[-1]


def _resolve_lora(model_path: str):
    """Detect whether model_path is a LoRA adapter directory (contains adapter_config.json).

    Returns:
        (base_model_path, adapter_path, max_lora_rank)  — LoRA case
        (model_path, None, None)                        — regular model
    """
    import json
    adapter_cfg_file = Path(model_path) / "adapter_config.json"
    if not adapter_cfg_file.exists():
        return model_path, None, None
    cfg = json.loads(adapter_cfg_file.read_text())
    base_model = cfg["base_model_name_or_path"]
    max_lora_rank = cfg.get("r", 64)
    logger.info(f"[LoRA] detected adapter: {model_path}, base model: {base_model}, rank={max_lora_rank}")
    return base_model, model_path, max_lora_rank


# ============================================================
# Offline vLLM engine
# ============================================================

class VLLMWorker:
    """Ray actor: holds one vLLM LLM instance, handles one DP shard."""

    def __init__(self, model_path: str, tp: int,
                 gpu_memory_utilization: float,
                 max_model_len: Optional[int],
                 enforce_eager: bool,
                 lora_path: Optional[str] = None,
                 max_lora_rank: int = 64):
        from vllm import LLM as _LLM
        kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        if max_model_len:
            kwargs['max_model_len'] = max_model_len
        if lora_path:
            kwargs['enable_lora'] = True
            kwargs['max_lora_rank'] = max_lora_rank
        self.llm = _LLM(**kwargs)
        if lora_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, lora_path)
        else:
            self._lora_request = None

    def generate(self, prompts: List[str],
                 sampling_params_dict: dict,
                 max_n: int = 0) -> dict:
        """Batch inference with automatic prompt deduplication and KV cache sharing via n parameter.

        When a batch contains duplicate prompts (e.g. AIME 30 questions × 8 repeats),
        deduplication uses SamplingParams(n=repeat_count) to generate multiple completions,
        with vLLM sharing the prompt KV cache (copy-on-write) to save memory
        and allow higher concurrent batch sizes.

        Args:
            max_n: Maximum value for the n parameter. 0 means unlimited.
                   Smaller values (e.g. 8) increase the number of unique prompts,
                   improving GPU batch utilization for long-generation scenarios.
        """
        from vllm import SamplingParams as _SP

        # --- Dedup: detect duplicate prompts in batch ---
        seen = {}          # prompt_text -> index in unique_prompts
        unique_prompts = []
        n_per_unique = []  # repeat count per unique prompt
        for p in prompts:
            if p not in seen:
                seen[p] = len(unique_prompts)
                unique_prompts.append(p)
                n_per_unique.append(0)
            n_per_unique[seen[p]] += 1

        actual_max_n = max(n_per_unique) if n_per_unique else 1
        # Cap n: too large reduces unique prompts and GPU batch utilization
        if max_n > 0 and actual_max_n > max_n:
            actual_max_n = max_n
        dedup_ratio = len(prompts) / max(len(unique_prompts), 1)

        if actual_max_n > 1:
            # Duplicate prompts: use n parameter for batch rollout
            # But cap n at actual_max_n; excess become independent prompts
            sp_dict = dict(sampling_params_dict)
            sp_dict['n'] = actual_max_n

            # Reorganize: each unique prompt generates at most actual_max_n completions
            # Excess duplicate prompts are added as independent prompts
            expanded_prompts = []
            expanded_n = []
            prompt_to_slots = {}  # prompt_text -> [(expanded_idx, slot_start, count), ...]
            for p in unique_prompts:
                total_needed = n_per_unique[seen[p]]
                while total_needed > 0:
                    this_n = min(total_needed, actual_max_n)
                    exp_idx = len(expanded_prompts)
                    if p not in prompt_to_slots:
                        prompt_to_slots[p] = []
                    prompt_to_slots[p].append((exp_idx, 0, this_n))
                    expanded_prompts.append(p)
                    expanded_n.append(this_n)
                    total_needed -= this_n

            # Generate: all expanded prompts share the same n (max value)
            # But each prompt may need different numbers of completions
            # Simplified: use actual_max_n uniformly, excess outputs discarded
            outputs = self.llm.generate(
                expanded_prompts, _SP(**sp_dict),
                lora_request=self._lora_request, use_tqdm=False)

            # Map results back in original order
            occurrence = {}  # prompt_text -> total outputs taken so far
            for p in unique_prompts:
                occurrence[p] = 0

            results = []
            total_tokens = 0
            for p in prompts:
                global_j = occurrence[p]
                occurrence[p] += 1
                # Find which expanded prompt slot this j belongs to
                slots = prompt_to_slots[p]
                cumulative = 0
                for exp_idx, slot_start, count in slots:
                    if global_j < cumulative + count:
                        local_j = global_j - cumulative
                        out = outputs[exp_idx].outputs[local_j]
                        break
                    cumulative += count
                else:
                    # fallback: should not happen
                    out = outputs[prompt_to_slots[p][-1][0]].outputs[0]
                results.append({
                    'text': out.text,
                    'finish_reason': out.finish_reason,
                })
                total_tokens += len(out.token_ids)
        else:
            # No duplicates: regular generation
            outputs = self.llm.generate(
                prompts, _SP(**sampling_params_dict),
                lora_request=self._lora_request, use_tqdm=False)
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            results = [{'text': o.outputs[0].text,
                         'finish_reason': o.outputs[0].finish_reason}
                        for o in outputs]

        return {
            'results': results,
            'total_tokens': total_tokens,
            'dedup_ratio': round(dedup_ratio, 1),
        }


class VLLMOfflineEngine:
    """Offline vLLM inference engine with TP and Ray DP support.

    - dp=1: LLM created in current process; killing the process releases GPU immediately.
    - dp>1: Multiple Worker actors created via Ray, each occupying tp GPUs.
    """

    def __init__(self, model_path: str, tp: int = 1, dp: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 enforce_eager: bool = True,
                 worker_chunk_size: int = 0,
                 max_dedup_n: int = 0):
        self.model_path = model_path
        self.tp = tp
        self.dp = dp
        self.model_name = _extract_model_name(model_path)
        # Prompts per Worker per dispatch round (dp>1), controls load-balance granularity
        self._worker_chunk_size = worker_chunk_size
        # n-dedup cap: 0 = unlimited. Smaller n increases unique prompts and GPU batch utilization
        self._max_dedup_n = max_dedup_n

        # LoRA detection: if model_path is an adapter dir, split base model and adapter
        actual_model, lora_path, max_lora_rank = _resolve_lora(model_path)
        self._lora_request = None

        logger.info(f"Initializing VLLMOfflineEngine: model={self.model_name}, "
                    f"tp={tp}, dp={dp}, total_gpus={tp * dp}"
                    + (f", chunk_size={worker_chunk_size}" if dp > 1 else "")
                    + (f", lora={lora_path}" if lora_path else ""))

        if dp == 1:
            # ---------- Single-process mode ----------
            kwargs = dict(
                model=actual_model,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                trust_remote_code=True,
            )
            if max_model_len:
                kwargs['max_model_len'] = max_model_len
            if lora_path:
                kwargs['enable_lora'] = True
                kwargs['max_lora_rank'] = max_lora_rank
            self.llm = LLM(**kwargs)
            self.tokenizer = self.llm.get_tokenizer()
            if lora_path:
                from vllm.lora.request import LoRARequest
                self._lora_request = LoRARequest("adapter", 1, lora_path)
            self._workers = None
            logger.info("Single-process offline mode initialized")
        else:
            # ---------- Ray DP mode ----------
            import ray
            ray.init(ignore_reinit_error=True)

            # Main process loads only the tokenizer (a few MB), no GPU
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                lora_path or actual_model, trust_remote_code=True)

            Worker = ray.remote(num_gpus=tp)(VLLMWorker)
            self._workers = [
                Worker.remote(actual_model, tp, gpu_memory_utilization,
                              max_model_len, enforce_eager,
                              lora_path, max_lora_rank or 64)
                for _ in range(dp)
            ]
            self.llm = None
            logger.info(f"Ray DP mode initialized with {dp} Workers")

        # ---------- TPS statistics ----------
        self._tps_interval = 30          # print TPS every N seconds
        self._tps_tokens = 0             # tokens generated in current window
        self._tps_samples = 0            # samples completed in current window
        self._tps_total_tokens = 0       # total tokens across all windows
        self._tps_total_samples = 0      # total samples across all windows
        self._tps_window_start = time.time()
        self._tps_global_start = time.time()
        self._tps_callback = None        # Optional[Callable[[float, float], None]]

        # ---------- Request batching (thread-safe) ----------
        # evalscope calls generate() one-by-one via ThreadPoolExecutor,
        # but vLLM LLM.generate() is not thread-safe.
        # A queue + background dispatch thread is used to batch requests.
        self._batch_lock = threading.Lock()
        self._batch_queue: List[tuple] = []   # [(prompt, SamplingParams, Future), ...]
        self._batch_event = threading.Event()
        self._batch_running = True
        self._batch_thread = threading.Thread(
            target=self._batch_loop, daemon=True, name='vllm-batcher')
        self._batch_thread.start()

    # ---- Core inference ----

    def generate_single(self, prompt: str,
                        sampling_params: SamplingParams,
                        group_key: str = "") -> dict:
        """Thread-safe single-item inference. Requests are automatically batched.

        evalscope's ThreadPoolExecutor calls this method concurrently;
        requests are collected via a queue and dispatched to vLLM in batches.

        Args:
            group_key: Optional group identifier (e.g. benchmark name). _dispatch groups by
                       (sp_key, group_key) so prompts from the same group are dispatched
                       together first, achieving "finish the same benchmark first" scheduling.
        """
        future = concurrent.futures.Future()
        with self._batch_lock:
            self._batch_queue.append((prompt, sampling_params, future, group_key))
            self._batch_event.set()
        return future.result()

    def _tps_update(self, num_tokens: int, num_samples: int):
        """Accumulate token/sample counts and print TPS at configured intervals."""
        self._tps_tokens += num_tokens
        self._tps_samples += num_samples
        self._tps_total_tokens += num_tokens
        self._tps_total_samples += num_samples
        now = time.time()
        window_elapsed = now - self._tps_window_start
        if window_elapsed >= self._tps_interval:
            window_tps = self._tps_tokens / window_elapsed if window_elapsed > 0 else 0
            global_elapsed = now - self._tps_global_start
            global_tps = self._tps_total_tokens / global_elapsed if global_elapsed > 0 else 0
            logger.info(
                f"[TPS] last {window_elapsed:.0f}s: {self._tps_tokens} tokens, "
                f"{self._tps_samples} samples, {window_tps:.1f} tok/s | "
                f"global: {self._tps_total_tokens} tokens, "
                f"{self._tps_total_samples} samples, {global_tps:.1f} tok/s")
            if self._tps_callback is not None:
                try:
                    self._tps_callback(window_tps, global_tps)
                except Exception:
                    pass
            self._tps_tokens = 0
            self._tps_samples = 0
            self._tps_window_start = now

    def _tps_reset(self):
        """Reset TPS statistics (call when a new benchmark starts)."""
        self._tps_tokens = 0
        self._tps_samples = 0
        self._tps_total_tokens = 0
        self._tps_total_samples = 0
        self._tps_window_start = time.time()
        self._tps_global_start = time.time()

    def _batch_loop(self):
        """Background dispatch thread entry: selects dispatch strategy based on dp."""
        if self.dp == 1:
            self._batch_loop_single()
        else:
            self._batch_loop_dp()

    def _batch_loop_single(self):
        """dp=1: collect requests then batch-infer (vLLM does continuous batching internally).
        Automatically dedup prompts and share KV cache via n parameter.
        Supports per-request SamplingParams: groups by sp_key and infers separately."""
        from collections import defaultdict

        while self._batch_running:
            self._batch_event.wait(timeout=1.0)
            if not self._batch_running:
                break
            # Brief wait for evalscope ThreadPool to submit all requests in the same wave
            time.sleep(0.05)
            with self._batch_lock:
                if not self._batch_queue:
                    self._batch_event.clear()
                    continue
                batch = list(self._batch_queue)
                self._batch_queue.clear()
                self._batch_event.clear()

            # --- Group by sp_key ---
            sp_groups = defaultdict(list)  # sp_key -> [(prompt, sp, future), ...]
            for item in batch:
                key = self._sp_key(item[1])
                sp_groups[key].append(item)

            # --- Independent inference per group (dedup + n parameter) ---
            for sp_key, group in sp_groups.items():
                prompts = [x[0] for x in group]
                sp = group[0][1]
                futures = [x[2] for x in group]

                # --- Dedup + n-parameter optimization ---
                seen = {}
                unique_prompts = []
                n_per_unique = []
                for p in prompts:
                    if p not in seen:
                        seen[p] = len(unique_prompts)
                        unique_prompts.append(p)
                        n_per_unique.append(0)
                    n_per_unique[seen[p]] += 1

                max_n = max(n_per_unique) if n_per_unique else 1
                # Cap n to prevent too few unique prompts causing small GPU batch
                if self._max_dedup_n > 0 and max_n > self._max_dedup_n:
                    max_n = self._max_dedup_n

                logger.info(f"Batcher(dp=1): group={sp_key[:2]}... {len(prompts)} requests"
                            f"({len(unique_prompts)} unique, n={max_n}), running inference...")
                try:
                    if max_n > 1:
                        sp_dict = self._sampling_params_to_dict(sp)
                        sp_dict['n'] = max_n

                        # Build expanded prompts: repeat count per unique prompt
                        # If exceeding max_n, split into multiple independent prompt entries
                        expanded_prompts = []
                        prompt_to_slots = {}  # prompt -> [(exp_idx, slot_start, count), ...]
                        for p in unique_prompts:
                            total_needed = n_per_unique[seen[p]]
                            prompt_to_slots[p] = []
                            while total_needed > 0:
                                this_n = min(total_needed, max_n)
                                exp_idx = len(expanded_prompts)
                                prompt_to_slots[p].append((exp_idx, 0, this_n))
                                expanded_prompts.append(p)
                                total_needed -= this_n

                        new_sp = SamplingParams(**sp_dict)
                        outputs = self.llm.generate(
                            expanded_prompts, new_sp,
                            lora_request=self._lora_request, use_tqdm=False)

                        occurrence = {p: 0 for p in unique_prompts}
                        total_tokens = 0
                        for prompt_text, fut in zip(prompts, futures):
                            global_j = occurrence[prompt_text]
                            occurrence[prompt_text] += 1
                            # Find the corresponding expanded prompt and slot
                            slots = prompt_to_slots[prompt_text]
                            cumulative = 0
                            for exp_idx, slot_start, count in slots:
                                if global_j < cumulative + count:
                                    local_j = global_j - cumulative
                                    out = outputs[exp_idx].outputs[local_j]
                                    break
                                cumulative += count
                            else:
                                out = outputs[slots[-1][0]].outputs[0]
                            total_tokens += len(out.token_ids)
                            fut.set_result({
                                'text': out.text,
                                'finish_reason': out.finish_reason,
                            })
                        self._tps_update(total_tokens, len(prompts))
                    else:
                        outputs = self.llm.generate(
                            prompts, sp,
                            lora_request=self._lora_request, use_tqdm=False)
                        total_tokens = sum(
                            len(o.outputs[0].token_ids) for o in outputs)
                        self._tps_update(total_tokens, len(outputs))
                        for fut, o in zip(futures, outputs):
                            fut.set_result({
                                'text': o.outputs[0].text,
                                'finish_reason': o.outputs[0].finish_reason,
                            })
                except Exception as e:
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)

    def _batch_loop_dp(self):
        """dp>1: continuous dispatch, Workers receive new tasks immediately upon completion.

        Design principles:
        - Each Worker receives _worker_chunk_size prompts at a time (not an even split)
        - ray.wait(num_returns=1) detects the first Worker to finish and assigns new tasks
        - Never wait for all Workers to synchronize, eliminating slow-Worker bottlenecks
        - Supports per-request SamplingParams: groups by sp_key before dispatch
        """
        import ray as _ray
        from collections import defaultdict

        pending: List[tuple] = []   # (prompt, sp, client_future, group_key)
        inflight: dict = {}         # ray_ref -> (worker_idx, [client_future, ...])

        def _dispatch():
            """Group pending prompts by sp_key and round-robin distribute to idle Workers.

            Scheduling strategy:
            - Group by sp_key (merge different benchmarks with same params) to maximize batch size
            - Round-robin dispatch: each benchmark's prompts spread evenly across Workers,
              ensuring results arrive uniformly when Workers return
            - Pending queue order is controlled by caller (small benchmarks first)
            - chunk=0: dispatch all; chunk>0: limit per-round dispatch (controls result granularity)
            - Workers auto-dedup + share KV cache via n parameter
            """
            if not pending:
                return
            busy = {v[0] for v in inflight.values()}
            free = [i for i in range(self.dp) if i not in busy]
            if not free:
                return
            # Read latest value each time to support dynamic chunk_size adjustment per benchmark
            chunk = self._worker_chunk_size
            # max_n cap: prevent oversized n from reducing unique prompts in batch
            max_n_limit = self._max_dedup_n

            # --- Group by sp_key (merge benchmarks with same params) ---
            # Merging same-sp_key prompts ensures large enough Worker batches.
            # Pending order is controlled by _extract_all_prompts (small benchmarks first);
            # round-robin dispatch spreads each benchmark evenly across Workers.
            sp_groups = defaultdict(list)
            for item in pending:
                sp_k = self._sp_key(item[1])
                sp_groups[sp_k].append(item)

            # Clear pending and refill by group
            pending.clear()

            # Process sp_key groups in order; free Worker list is shared across groups
            for sp_key, group in sp_groups.items():
                if not free:
                    pending.extend(group)
                    continue

                sp_dict = self._sampling_params_to_dict(group[0][1])
                n_free = len(free)

                # chunk>0: limit dispatch volume per round (controls result granularity)
                # chunk=0: no limit, dispatch all
                if chunk > 0:
                    dispatch_group = group[:chunk * n_free]
                else:
                    dispatch_group = group

                # Round-robin dispatch: spread each benchmark's prompts evenly across Workers.
                # E.g. pending=[A0,A1,...,A879,B0,B1,...,B5275], 8 Workers:
                #   W0 gets [A0,A8,A16,...,B0,B8,...] → 110 A + 660 B
                #   W1 gets [A1,A9,A17,...,B1,B9,...] → 110 A + 660 B
                # When all Workers return, results for A and B arrive together.
                used_workers = []
                for idx in range(n_free):
                    take = dispatch_group[idx::n_free]  # stride sampling
                    if not take:
                        continue
                    w_idx = free[idx]
                    ref = self._workers[w_idx].generate.remote(
                        [x[0] for x in take], sp_dict, max_n_limit)
                    inflight[ref] = (w_idx, [x[2] for x in take])
                    used_workers.append(w_idx)
                    logger.info(f"Dispatcher: Worker-{w_idx} <- {len(take)} prompts "
                                f"(sp={str(sp_key)[:40]}..., "
                                f"pending={len(pending)}, inflight={len(inflight)})")

                # Return undispatched items to pending
                if len(dispatch_group) < len(group):
                    pending.extend(group[len(dispatch_group):])

                # Remove assigned Workers from free list so subsequent sp_groups skip them
                for w in used_workers:
                    free.remove(w)

        while True:
            # 1. Wait: block with ray.wait (detect Worker completion); if no inflight, wait for new requests
            if inflight:
                done, _ = _ray.wait(
                    list(inflight.keys()), num_returns=1, timeout=0.05)
            else:
                self._batch_event.wait(timeout=0.05)
                done = []

            # 2. Collect newly arrived requests
            with self._batch_lock:
                if self._batch_queue:
                    pending.extend(self._batch_queue)
                    self._batch_queue.clear()
                self._batch_event.clear()

            # 2b. If no inflight tasks (new benchmark just started),
            # wait until pending count stabilizes (stops growing), ensuring all prompts are
            # queued before dispatch — critical for n-dedup: more prompts → higher dedup ratio
            # → larger batch → better GPU utilization.
            if not inflight and pending:
                prev_count = 0
                while len(pending) != prev_count:
                    prev_count = len(pending)
                    time.sleep(0.15)
                    with self._batch_lock:
                        if self._batch_queue:
                            pending.extend(self._batch_queue)
                            self._batch_queue.clear()
                        self._batch_event.clear()
                logger.info(f"Dispatcher: first dispatch, collected {len(pending)} prompts")

            # 3. Process completed Workers → resolve their Futures, immediately trigger _dispatch
            for ref in done:
                w_idx, cfuts = inflight.pop(ref)
                try:
                    ret = _ray.get(ref)
                    results = ret['results']
                    dedup_ratio = ret.get('dedup_ratio', 1.0)
                    self._tps_update(ret['total_tokens'], len(results))
                    if dedup_ratio > 1.0:
                        logger.info(
                            f"Worker-{w_idx} done: {len(results)} samples, "
                            f"dedup {dedup_ratio}x, "
                            f"{ret['total_tokens']} tokens")
                    for fut, res in zip(cfuts, results):
                        if not fut.done():
                            fut.set_result(res)
                except Exception as e:
                    for fut in cfuts:
                        if not fut.done():
                            fut.set_exception(e)

            # 4. Assign pending to idle Workers
            _dispatch()

            # 5. Exit: _batch_running=False and all tasks processed
            if not self._batch_running and not pending and not inflight:
                break

    # ---- Lifecycle ----

    def shutdown(self):
        """Release all GPU resources."""
        # Stop batching thread
        self._batch_running = False
        self._batch_event.set()
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=5)

        if self._workers:
            import ray
            for w in self._workers:
                ray.kill(w)
            ray.shutdown()
            self._workers = None
            logger.info("Ray Workers shut down")

        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("LLM instance released")

    # ---- Internal utilities ----

    @staticmethod
    def _sampling_params_to_dict(sp: SamplingParams) -> dict:
        """Serialize SamplingParams to dict for Ray remote transmission."""
        d = {}
        for field in ('temperature', 'top_p', 'top_k',
                      'max_tokens', 'repetition_penalty'):
            val = getattr(sp, field, None)
            if val is not None:
                d[field] = val
        return d

    @staticmethod
    def _sp_key(sp: SamplingParams) -> tuple:
        """Generate SamplingParams key for grouping batches by parameters."""
        return (sp.temperature, sp.top_p, sp.top_k, sp.max_tokens, sp.repetition_penalty)


# ============================================================
# evalscope ModelAPI adapter
# ============================================================

class VLLMOfflineModelAPI(ModelAPI):
    """Wraps VLLMOfflineEngine as an evalscope ModelAPI interface.

    When evalscope detects that model is a ModelAPI instance in get_model(),
    it uses it directly without going through the OpenAI HTTP client.
    """

    def __init__(self, engine: VLLMOfflineEngine, model_name: str,
                 config: GenerateConfig = GenerateConfig()):
        super().__init__(model_name=model_name, config=config)
        self.engine = engine

    # ---- Message → prompt conversion ----

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        msg_dicts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else msg.text
            msg_dicts.append({"role": msg.role, "content": content})
        return self.engine.tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True)

    # ---- GenerateConfig → SamplingParams ----

    @staticmethod
    def _config_to_sampling_params(config: GenerateConfig) -> SamplingParams:
        kwargs = {}
        if config.temperature is not None:
            kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            kwargs['top_p'] = config.top_p
        if config.top_k is not None and config.top_k > 0:
            kwargs['top_k'] = config.top_k
        if config.max_tokens is not None:
            kwargs['max_tokens'] = config.max_tokens
        if getattr(config, 'repetition_penalty', None) is not None:
            kwargs['repetition_penalty'] = config.repetition_penalty
        return SamplingParams(**kwargs)

    # ---- ModelAPI interface implementation ----

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = self._messages_to_prompt(input)
        sp = self._config_to_sampling_params(config)
        # generate_single is thread-safe: requests enter queue and are batched by background thread
        r = self.engine.generate_single(prompt, sp)
        return ModelOutput.from_content(
            model=self.model_name,
            content=r['text'],
            stop_reason=as_stop_reason(r['finish_reason']),
        )


# ============================================================
# BenchmarkEvaluator
# ============================================================

class BenchmarkEvaluator:
    """Benchmark evaluator"""

    def __init__(self, config_file: str, run_id: int = None,
                 version: str = None,
                 vllm_engine: Optional[VLLMOfflineEngine] = None,
                 completed_benchmarks: Optional[set] = None,
                 repeats_override: Optional[dict] = None,
                 force_use_cache_dirs: Optional[dict] = None,
                 benchmark_filter: Optional[set] = None,
                 output_dir_override: Optional[str] = None):
        self.config_file = config_file
        self.config = self._load_config()
        self.vllm_engine = vllm_engine
        self.results = {}
        self.base_dir = Path(self.config['model_path'])
        self.version = version or datetime.now().strftime("version_%Y%m%d_%H%M%S")
        # Resume support: list of completed benchmarks (skip inference, load existing results)
        self.completed_benchmarks = completed_benchmarks or set()
        # Merged-inference options
        self._repeats_override = repeats_override or {}       # {bench: repeats} override repeats in yaml
        self._force_use_cache_dirs = force_use_cache_dirs or {}  # {bench: dir} force use_cache path
        self._benchmark_filter = benchmark_filter             # None=all; set=run only these
        self._pred_ts_dirs: dict = {}                         # {bench: Path} record latest prediction dir after inference

        if output_dir_override is not None:
            self.output_dir = Path(output_dir_override)
            self.run_id = run_id
        elif run_id is not None:
            self.output_dir = self.base_dir / "evalscope" / self.version / f"run_{run_id}"
            self.run_id = run_id
        else:
            self.output_dir = self.base_dir / "evalscope" / self.version
            self.run_id = None

        self.debug_mode = self.config.get('debug', False)
        self.model_name = _extract_model_name(self.config['model_path'])

    def _load_config(self) -> Dict:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _save_benchmark_result(self, benchmark_name: str, result):
        """Persist results immediately after each benchmark completes, for resume support."""
        result_path = self.output_dir / f"{benchmark_name}_result.json"
        serializable = self._convert_to_serializable(result)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"[resume] {benchmark_name} result saved: {result_path}")

    def _load_benchmark_result(self, benchmark_name: str):
        """Load existing benchmark results (used during resume).

        Preferentially reads our own {benchmark}_result.json;
        if absent (legacy run or pre-mechanism results), reads from evalscope native
        reports/ directory and rebuilds {dataset_name: report_dict} format.
        """
        # 1. New mechanism: read directly
        result_path = self.output_dir / f"{benchmark_name}_result.json"
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 2. Legacy fallback: read from evalscope reports/ directory
        bm_dir = self.output_dir / benchmark_name
        if not bm_dir.is_dir():
            return None
        # benchmark directory has one timestamp subdirectory (auto-generated by evalscope)
        ts_dirs = sorted(bm_dir.iterdir(), key=lambda d: d.name)
        for ts_dir in ts_dirs:
            reports_dir = ts_dir / "reports"
            if not reports_dir.is_dir():
                continue
            result = {}
            for model_dir in reports_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for report_file in model_dir.glob("*.json"):
                    with open(report_file, 'r', encoding='utf-8') as f:
                        result[report_file.stem] = json.load(f)
            if result:
                logger.info(f"[resume] {benchmark_name} loading results from evalscope reports/")
                return result
        return None

    def _find_partial_prediction_dir(self, benchmark_name: str) -> Optional[str]:
        """Find timestamp directories with existing predictions but no reports.

        Such directories indicate interrupted inference; use_cache can resume them.
        Called only for non-completed benchmarks (completed ones are skipped).
        """
        bm_dir = self.output_dir / benchmark_name
        if not bm_dir.is_dir():
            return None
        # Take the latest timestamp subdirectory (evalscope names by time)
        for ts_dir in sorted(bm_dir.iterdir(), key=lambda d: d.name, reverse=True):
            if not ts_dir.is_dir():
                continue
            pred_dir = ts_dir / "predictions"
            rep_dir = ts_dir / "reports"
            has_predictions = pred_dir.is_dir() and any(pred_dir.rglob("*.jsonl"))
            has_reports = rep_dir.is_dir() and any(rep_dir.rglob("*.json"))
            if has_predictions and not has_reports:
                return str(ts_dir)
        return None

    @staticmethod
    def _count_existing_predictions(ts_dir: str) -> int:
        """Count entries in existing predictions JSONL (for logging)."""
        total = 0
        pred_root = Path(ts_dir) / "predictions"
        if pred_root.is_dir():
            for jsonl in pred_root.rglob("*.jsonl"):
                try:
                    total += sum(1 for _ in open(jsonl, 'rb'))
                except OSError:
                    pass
        return total

    def _create_task_config(self, benchmark_name: str) -> TaskConfig:
        """Create TaskConfig"""
        benchmark_config = self.config['benchmarks'][benchmark_name]

        logger.info(f"Creating TaskConfig for benchmark {benchmark_name}")

        benchmark_judge_strategy = benchmark_config.get(
            'judge_strategy', JudgeStrategy.RULE)
        use_llm = 'llm' in str(benchmark_judge_strategy).lower()

        task_config = TaskConfig()

        # Pass offline ModelAPI; no HTTP server required
        model_api = VLLMOfflineModelAPI(
            engine=self.vllm_engine,
            model_name=self.model_name,
        )
        task_config.model = model_api
        task_config.model_id = self.model_name
        task_config.eval_backend = EvalBackend.NATIVE

        # Dataset
        task_config.datasets = [benchmark_config['dataset_name']]
        if 'limit' in benchmark_config:
            task_config.limit = benchmark_config['limit']
        task_config.repeats = self._repeats_override.get(
            benchmark_name, benchmark_config.get('repeats', 1))

        dataset_args = {
            'dataset_name': benchmark_config.get('subset', None),
            'few_shot_num': benchmark_config.get('few_shot_num', 0)
        }
        if 'extra_params' in benchmark_config:
            dataset_args['extra_params'] = benchmark_config['extra_params']
        task_config.dataset_args = {
            benchmark_config['dataset_name']: dataset_args}

        # Generation parameters
        generation_config_dict = {
            'temperature': benchmark_config.get('temperature', 0.0),
            'top_p': benchmark_config.get('top_p', 1.0),
            'top_k': benchmark_config.get('top_k', 50),
            'max_tokens': benchmark_config.get('max_tokens', 2048),
            'repetition_penalty': benchmark_config.get('repetition_penalty', 1.0)
        }
        gen_config = benchmark_config.get('gen_config', {})
        if gen_config:
            generation_config_dict.update(gen_config)
        task_config.generation_config = GenerateConfig.model_validate(
            generation_config_dict)

        # evalscope ThreadPoolExecutor concurrency: large enough to keep the pending queue full
        # for the continuous dispatcher (_batch_loop_dp) to fill all Workers
        vllm_cfg = self.config.get('vllm', {})
        _dp = vllm_cfg.get('dp', 1)
        _global_chunk = vllm_cfg.get('worker_chunk_size', 512)
        # Allow per-benchmark worker_chunk_size override (large datasets benefit from larger chunks)
        _chunk = benchmark_config.get('worker_chunk_size', _global_chunk)
        if self.vllm_engine is not None:
            self.vllm_engine._tps_reset()
            if self.vllm_engine.dp > 1:
                self.vllm_engine._worker_chunk_size = _chunk
                logger.info(f"[{benchmark_name}] worker_chunk_size={_chunk}")
        # dp=1: moderate concurrency; dp>1: at least enough to fill 2 rounds for all Workers
        eval_batch_size = _chunk if _dp == 1 else max(2048, _dp * _chunk * 2)
        task_config.eval_batch_size = eval_batch_size
        task_config.generation_config.batch_size = eval_batch_size

        # Working directory
        task_config.work_dir = str(self.output_dir / benchmark_name)

        # Prediction-level resume: detect existing predictions (no reports) → set use_cache
        # evalscope use_cache reads existing JSONL, filters predicted samples, runs only remainder
        partial_pred_dir = self._find_partial_prediction_dir(benchmark_name)
        if partial_pred_dir:
            task_config.use_cache = partial_pred_dir
            existing = self._count_existing_predictions(partial_pred_dir)
            logger.info(
                f"[prediction-resume] {benchmark_name}: detected {existing} existing predictions,"
                f"resuming remaining (dir: {partial_pred_dir})")
        # Merged-inference split mode: force use_cache path (highest priority, overrides auto-detect)
        if benchmark_name in self._force_use_cache_dirs:
            task_config.use_cache = self._force_use_cache_dirs[benchmark_name]
            logger.info(f"[{benchmark_name}] forcing use_cache → {task_config.use_cache}")

        # Sandbox configuration
        if 'use_sandbox' in benchmark_config:
            task_config.use_sandbox = benchmark_config['use_sandbox']
            sandbox_workers = benchmark_config.get('sandbox_workers', 5)
            task_config.judge_worker_num = sandbox_workers

            sandbox_config = self.config.get('sandbox', {})
            remote_url = sandbox_config.get('remote_url')
            if remote_url:
                remote_config = {'base_url': remote_url}
                if sandbox_config.get('timeout'):
                    remote_config['timeout'] = sandbox_config['timeout']
                task_config.sandbox_manager_config = remote_config
            elif sandbox_config.get('work_dir'):
                sandbox_work_dir = sandbox_config['work_dir']
                Path(sandbox_work_dir).mkdir(parents=True, exist_ok=True)
                task_config.sandbox_manager_config = {
                    'work_dir': sandbox_work_dir}

        # Judge configuration
        task_config.judge_strategy = benchmark_judge_strategy
        if use_llm:
            api_key = self.config.get('API_KEY', '')
            base_url = self.config.get('BASE_URL', '').replace(
                '/chat/completions', '').rstrip('/')
            judge_model = benchmark_config.get(
                'judge_model', 'Qwen/Qwen3-8B')
            if not api_key or not base_url:
                raise ValueError(
                    f"[{benchmark_name}] LLM judge requires API_KEY and BASE_URL")
            task_config.judge_model_args = {
                'api_key': api_key,
                'api_url': base_url,
                'model_id': judge_model,
                'generation_config': {
                    'temperature': 0.0, 'max_tokens': 1024,
                    'extra_body': {'enable_thinking': False}},
            }
            judge_parallel = benchmark_config.get('judge_parallel', 1)
            task_config.judge_worker_num = judge_parallel

        logger.info(f"[{benchmark_name}] TaskConfig created: "
                    f"model={task_config.model_id}, "
                    f"datasets={task_config.datasets}, "
                    f"eval_batch_size={eval_batch_size}")
        return task_config

    def _save_config(self):
        config_path = self.output_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Config saved to: {config_path}")

    def run_evaluation(self):
        """Run evaluation"""
        run_info = f" (run {self.run_id})" if self.run_id else ""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting evaluation{run_info} - model: {self.model_name}")
        logger.info(f"Config file: {self.config_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        manage_engine = False
        if self.vllm_engine is None:
            # Create engine internally (single-run mode)
            vllm_cfg = self.config.get('vllm', {})
            self.vllm_engine = VLLMOfflineEngine(
                model_path=self.config['model_path'],
                tp=vllm_cfg.get('tp', 1),
                dp=vllm_cfg.get('dp', 1),
                gpu_memory_utilization=vllm_cfg.get(
                    'gpu_memory_utilization', 0.9),
                max_model_len=vllm_cfg.get('max_model_len', None),
                enforce_eager=vllm_cfg.get('enforce_eager', True),
                worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
            )
            manage_engine = True

        try:
            # Check valid benchmarks (benchmark_filter limits which ones run)
            valid_benchmarks = []
            for benchmark_name in self.config['benchmarks'].keys():
                if self._benchmark_filter is not None and benchmark_name not in self._benchmark_filter:
                    continue
                try:
                    self._create_task_config(benchmark_name)
                    valid_benchmarks.append(benchmark_name)
                except ValueError as e:
                    logger.warning(
                        f"Skipping benchmark {benchmark_name}: {str(e)}")

            if not valid_benchmarks:
                logger.error("No valid benchmarks to evaluate")
                return

            logger.info(f"Valid benchmarks: {valid_benchmarks}")
            total = len(valid_benchmarks)

            for idx, benchmark_name in enumerate(valid_benchmarks):
                repeats = self._repeats_override.get(
                    benchmark_name,
                    self.config['benchmarks'][benchmark_name].get('repeats', 1))
                logger.info(f"\n{'='*50}")

                # Resume: skip completed benchmarks
                if benchmark_name in self.completed_benchmarks:
                    saved = self._load_benchmark_result(benchmark_name)
                    if saved is not None:
                        self.results[benchmark_name] = saved
                        logger.info(
                            f"[resume] Skipping {benchmark_name}, loading existing results "
                            f"({idx+1}/{total})")
                        continue
                    else:
                        logger.warning(
                            f"[resume] {benchmark_name} marked done but result file missing, re-evaluating")

                logger.info(
                    f"Evaluating {benchmark_name} "
                    f"(repeats: {repeats}) "
                    f"(remaining: {total - idx - 1})")
                logger.info(f"Model: {self.model_name}")

                task_config = self._create_task_config(benchmark_name)

                logger.info(f"Starting evaluation of {benchmark_name}...")
                result = run_task_with_progress(task_config, benchmark_name)
                self.results[benchmark_name] = result
                # Record latest prediction dir from this inference run (used in merged-inference split)
                bm_dir = self.output_dir / benchmark_name
                if bm_dir.is_dir():
                    for ts_dir in sorted(bm_dir.iterdir(),
                                         key=lambda d: d.name, reverse=True):
                        if ts_dir.is_dir() and (ts_dir / 'predictions').is_dir():
                            self._pred_ts_dirs[benchmark_name] = ts_dir
                            break
                # Save immediately to ensure resumability after interruption
                self._save_benchmark_result(benchmark_name, result)
                logger.info(
                    f"Completed {benchmark_name} ({idx+1}/{total})")

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
        finally:
            if manage_engine and self.vllm_engine:
                self.vllm_engine.shutdown()
                self.vllm_engine = None

    def _convert_to_serializable(self, obj):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v)
                    for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
        if hasattr(obj, 'to_dict'):
            return self._convert_to_serializable(obj.to_dict())
        return str(obj)

    def save_results(self):
        results_summary = self._prepare_results_summary()

        excel_path = self.output_dir / "summary.xlsx"
        results_summary.to_excel(excel_path, index=False)
        logger.info(f"Excel summary saved to: {excel_path}")

        json_path = self.output_dir / "detailed_results.json"
        serializable_results = self._convert_to_serializable(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {json_path}")

    def _prepare_results_summary(self) -> pd.DataFrame:
        summary_data = {'benchmark': [], 'dataset': [], 'score': []}
        serializable_results = self._convert_to_serializable(self.results)

        for benchmark_name, result in serializable_results.items():
            if isinstance(result, dict):
                for dataset_name, dataset_result in result.items():
                    if not isinstance(dataset_result, dict):
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(dataset_name)
                        summary_data['score'].append('N/A')
                        continue

                    overall_score = dataset_result.get(
                        'score', dataset_result.get('accuracy', 'N/A'))
                    subsets = _extract_report_subsets(dataset_result)

                    if subsets:
                        for subset in subsets:
                            summary_data['benchmark'].append(benchmark_name)
                            summary_data['dataset'].append(
                                f"{dataset_name}/{subset['name']}")
                            summary_data['score'].append(
                                round(subset.get('score', 0), 4))
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(
                            f"{dataset_name}/all")
                        summary_data['score'].append(overall_score)
                    else:
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(dataset_name)
                        summary_data['score'].append(overall_score)
            else:
                summary_data['benchmark'].append(benchmark_name)
                summary_data['dataset'].append('unknown')
                summary_data['score'].append('N/A')

        return pd.DataFrame(summary_data)


# ============================================================
# Merged-inference helper
# ============================================================

def _split_merged_predictions(
        merged_ts_dir: Path,
        run_idx: int,
        orig_repeats: int,
        num_runs: int,
        out_dir: Path) -> bool:
    """Split merged-inference (repeats = orig_repeats × num_runs) prediction JSONL by run.

    Block layout: entries [q×K_total .. q×K_total + K_total - 1] are all repeats for question q.
    Run j occupies [j×K_orig .. (j+1)×K_orig - 1] within each question block.

    Split condition: (index % K_total) // K_orig == run_idx
    New index = (index // K_total) * K_orig + (index % K_orig)
    """
    K_total = orig_repeats * num_runs
    K_orig = orig_repeats
    j = run_idx
    found_any = False

    pred_root = merged_ts_dir / 'predictions'
    if not pred_root.is_dir():
        logger.warning(f"[split] prediction directory not found: {pred_root}")
        return False

    for jsonl_file in pred_root.rglob('*.jsonl'):
        entries = []
        with open(jsonl_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        split_entries = []
        for e in entries:
            idx = e['index']
            if (idx % K_total) // K_orig == j:
                e_copy = dict(e)
                e_copy['index'] = (idx // K_total) * K_orig + (idx % K_orig)
                split_entries.append(e_copy)

        if not split_entries:
            continue

        rel = jsonl_file.relative_to(pred_root)
        out_file = out_dir / 'predictions' / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            for e in split_entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        logger.info(f"  [split] run_{j+1}/{num_runs}: {len(split_entries)} entries → {out_file.name}")
        found_any = True

    return found_any


# ============================================================
# MultiRunEvaluator
# ============================================================

class MultiRunEvaluator:
    """Multi-run evaluator for variance statistics. Supports resume (--resume)."""

    _PROGRESS_FILE = "progress.json"

    def __init__(self, config_file: str, num_runs: int = 1,
                 resume: bool = False, resume_dir: Optional[str] = None,
                 merge_inference: bool = False):
        self.config_file = config_file
        self.num_runs = num_runs
        self.resume = resume or (resume_dir is not None)
        self.merge_inference = merge_inference and num_runs > 1

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = Path(self.config['model_path'])
        self.all_results = []

        if resume_dir is not None:
            self._init_resume(Path(resume_dir))
        elif resume:
            self._init_resume(None)
        else:
            self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
            self.output_dir = self.base_dir / "evalscope" / self.version

    # ---- Resume support ----

    def _init_resume(self, specified_dir: Optional[Path]):
        """Locate the resume directory.
        - specified_dir not None: use it directly (supports legacy dirs without progress.json)
        - specified_dir is None: auto-select the latest version_* dir under evalscope/
        """
        if specified_dir is not None:
            # User specified a directory
            target = specified_dir.resolve()
            if not target.is_dir():
                logger.warning(
                    f"[resume] Specified directory does not exist: {target}, starting fresh")
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = self.base_dir / "evalscope" / self.version
                return
            self.output_dir = target
            self.version = target.name
            logger.info(f"[resume] Using specified directory: {target}")
        else:
            # Auto-detect: pick the latest version_* dir under evalscope/
            evalscope_dir = self.base_dir / "evalscope"
            candidates = []
            if evalscope_dir.exists():
                candidates = sorted(
                    [d for d in evalscope_dir.iterdir()
                     if d.is_dir() and d.name.startswith("version_")],
                    key=lambda d: d.name, reverse=True
                )
            if not candidates:
                logger.info("[resume] No previous version directory found, starting fresh")
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = self.base_dir / "evalscope" / self.version
                return
            self.output_dir = candidates[0]
            self.version = candidates[0].name
            logger.info(f"[resume] Auto-selected latest directory: {self.output_dir}")

        # Read historical num_runs from progress.json (if present)
        progress_path = self.output_dir / self._PROGRESS_FILE
        if progress_path.exists():
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logger.info(
                f"[resume] Version: {self.version},"
                f"original num_runs: {progress.get('num_runs', '?')}")

    def _save_progress(self):
        """Save progress.json (version marker for next --resume)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        progress = {
            'version': self.version,
            'num_runs': self.num_runs,
            'config_file': self.config_file,
            'created_at': datetime.now().isoformat(),
        }
        with open(self.output_dir / self._PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _get_completed_benchmarks(self, run_id: int) -> set:
        """Scan run_N directories and return the set of completed benchmarks.

        Decision rules (by priority):
        1. {benchmark}_result.json exists → complete (new mechanism)
        2. *.json found in evalscope native reports/ → complete (legacy fallback)
        predictions/ only, no reports/ → incomplete (inference interrupted)
        """
        run_dir = self.output_dir / f"run_{run_id}"
        if not run_dir.exists():
            return set()

        completed = set()

        # New mechanism: {benchmark}_result.json
        for p in run_dir.glob('*_result.json'):
            completed.add(p.stem[:-len('_result')])

        # Legacy fallback: scan evalscope native directory structure
        for bm_dir in run_dir.iterdir():
            if not bm_dir.is_dir() or bm_dir.name.endswith('_result.json'):
                continue
            bm_name = bm_dir.name
            if bm_name in completed:
                continue
            # Look for reports/*.json under timestamp subdirectory
            for ts_dir in bm_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                reports_dir = ts_dir / "reports"
                if reports_dir.is_dir() and any(reports_dir.rglob("*.json")):
                    completed.add(bm_name)
                    break

        return completed

    # ---- Main flow ----

    def run_all(self):
        if self.merge_inference:
            return self._run_all_merged()

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting multi-run evaluation - {self.num_runs} runs total")
        logger.info(f"Total output directory: {self.output_dir}")
        logger.info(f"{'='*70}\n")

        # Save progress marker (save on first run or resume, for next --resume)
        self._save_progress()

        all_benchmarks = set(self.config.get('benchmarks', {}).keys())

        # Determine which runs have pending benchmarks and whether an engine is needed
        run_plan = []  # list of (run_id, done_set, remaining_list)
        for run_id in range(1, self.num_runs + 1):
            done = self._get_completed_benchmarks(run_id)
            remaining = [b for b in self.config['benchmarks'] if b not in done]
            run_plan.append((run_id, done, remaining))

        needs_engine = any(remaining for _, _, remaining in run_plan)

        engine = None
        if needs_engine:
            vllm_cfg = self.config.get('vllm', {})
            engine = VLLMOfflineEngine(
                model_path=self.config['model_path'],
                tp=vllm_cfg.get('tp', 1),
                dp=vllm_cfg.get('dp', 1),
                gpu_memory_utilization=vllm_cfg.get(
                    'gpu_memory_utilization', 0.9),
                max_model_len=vllm_cfg.get('max_model_len', None),
                enforce_eager=vllm_cfg.get('enforce_eager', True),
                worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
            )

        try:
            for run_id, done_benchmarks, remaining in run_plan:
                logger.info(f"\n{'#'*70}")
                logger.info(f"# Run {run_id}/{self.num_runs}")
                if done_benchmarks:
                    logger.info(f"# [resume] Completed: {sorted(done_benchmarks)}")
                    logger.info(f"# [resume] Pending: {remaining}")
                logger.info(f"{'#'*70}\n")

                if not remaining:
                    # Entire run complete, load results from disk
                    logger.info(f"Run {run_id} fully complete, loading existing results")
                    run_dir = self.output_dir / f"run_{run_id}"
                    results = {}
                    for bm in all_benchmarks:
                        result_path = run_dir / f"{bm}_result.json"
                        if result_path.exists():
                            with open(result_path, 'r', encoding='utf-8') as f:
                                results[bm] = json.load(f)
                    self.all_results.append({
                        'run_id': run_id,
                        'results': results,
                        'output_dir': run_dir
                    })
                    continue

                evaluator = BenchmarkEvaluator(
                    self.config_file, run_id=run_id,
                    version=self.version, vllm_engine=engine,
                    completed_benchmarks=done_benchmarks)
                evaluator.run_evaluation()
                evaluator.save_results()

                self.all_results.append({
                    'run_id': run_id,
                    'results': evaluator.results,
                    'output_dir': evaluator.output_dir
                })
                logger.info(f"Run {run_id} complete")

                if run_id < self.num_runs and remaining:
                    logger.info("Waiting 5 seconds before next run...")
                    time.sleep(5)

            self._summarize_results()

        finally:
            if engine is not None:
                engine.shutdown()

    # ---- Merged-inference mode ----

    def _run_all_merged(self):
        """GPU-efficient mode: infer num_runs×repeats samples per benchmark in one shot, then score by group.

        Flow:
          1. Share one vLLM engine
          2. For each benchmark, multiply repeats by num_runs and infer all at once
          3. Split prediction JSONL by run (_split_merged_predictions)
          4. Call evalscope scoring per split (use_cache + zero inference) for per-run results
          5. Aggregate into multi_run_summary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[merged-runs] Merged-inference mode: {self.num_runs} runs merged into one inference pass")
        logger.info(f"Total output directory: {self.output_dir}")
        logger.info(f"{'='*70}\n")

        self._save_progress()

        vllm_cfg = self.config.get('vllm', {})
        engine = VLLMOfflineEngine(
            model_path=self.config['model_path'],
            tp=vllm_cfg.get('tp', 1),
            dp=vllm_cfg.get('dp', 1),
            gpu_memory_utilization=vllm_cfg.get('gpu_memory_utilization', 0.9),
            max_model_len=vllm_cfg.get('max_model_len', None),
            enforce_eager=vllm_cfg.get('enforce_eager', True),
            worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
        )

        all_benchmarks = list(self.config.get('benchmarks', {}).keys())
        # per_run_results[run_id (1-based)] = {bench: result}
        per_run_results: dict = {j: {} for j in range(1, self.num_runs + 1)}
        merged_output_dir = self.output_dir / 'merged_inference'

        try:
            for bench_name in all_benchmarks:
                orig_repeats = self.config['benchmarks'][bench_name].get('repeats', 1)
                merged_repeats = orig_repeats * self.num_runs

                logger.info(f"\n{'#'*60}")
                logger.info(f"[merged-runs] {bench_name}: "
                            f"repeats {orig_repeats}×{self.num_runs} = {merged_repeats}, single inference pass")

                # Step 1: merged inference
                merged_eval = BenchmarkEvaluator(
                    config_file=self.config_file,
                    run_id=None,
                    version=self.version,
                    vllm_engine=engine,
                    benchmark_filter={bench_name},
                    repeats_override={bench_name: merged_repeats},
                    output_dir_override=str(merged_output_dir),
                )
                merged_eval.run_evaluation()

                merged_ts_dir = merged_eval._pred_ts_dirs.get(bench_name)
                if merged_ts_dir is None:
                    logger.warning(f"[merged-runs] No prediction directory for {bench_name}, skipping split")
                    continue

                # Step 2 & 3: split predictions per run → score
                for j in range(self.num_runs):
                    run_id = j + 1
                    split_pred_dir = (self.output_dir / f'run_{run_id}'
                                      / bench_name / 'split_preds')
                    logger.info(f"[merged-runs] Splitting run_{run_id}/{self.num_runs} ...")
                    success = _split_merged_predictions(
                        merged_ts_dir, j, orig_repeats, self.num_runs, split_pred_dir)
                    if not success:
                        logger.warning(f"[merged-runs] run_{run_id} {bench_name}: split failed, skipping")
                        continue

                    # Score (full use_cache → evalscope skips inference, scores directly)
                    score_eval = BenchmarkEvaluator(
                        config_file=self.config_file,
                        run_id=run_id,
                        version=self.version,
                        vllm_engine=engine,   # pass engine to prevent re-creation; actual inference volume=0
                        benchmark_filter={bench_name},
                        force_use_cache_dirs={bench_name: str(split_pred_dir)},
                    )
                    score_eval.run_evaluation()

                    if bench_name in score_eval.results:
                        per_run_results[run_id][bench_name] = score_eval.results[bench_name]
                        score_eval._save_benchmark_result(
                            bench_name, score_eval.results[bench_name])
                        logger.info(f"[merged-runs] run_{run_id} {bench_name}: scoring complete")

        finally:
            engine.shutdown()

        self.all_results = [
            {'run_id': j,
             'results': per_run_results[j],
             'output_dir': self.output_dir / f'run_{j}'}
            for j in range(1, self.num_runs + 1)
        ]
        self._summarize_results()

    def _summarize_results(self):
        logger.info(f"\n{'='*70}")
        logger.info("Aggregating multi-run results")
        logger.info(f"{'='*70}\n")

        benchmark_scores = {}

        def _serialize(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if hasattr(obj, 'to_dict'):
                return _serialize(obj.to_dict())
            if hasattr(obj, '__dict__'):
                return _serialize(obj.__dict__)
            return str(obj)

        for run_data in self.all_results:
            results = _serialize(run_data['results'])
            for benchmark_name, result in results.items():
                if benchmark_name not in benchmark_scores:
                    benchmark_scores[benchmark_name] = {}
                if not isinstance(result, dict):
                    continue
                for dataset_name, dataset_result in result.items():
                    if not isinstance(dataset_result, dict):
                        continue
                    overall_score = dataset_result.get(
                        'score', dataset_result.get('accuracy'))
                    subsets = _extract_report_subsets(dataset_result)
                    if subsets:
                        score_entries = [
                            (f"{dataset_name}/{s['name']}", s.get('score'))
                            for s in subsets]
                        score_entries.append(
                            (f"{dataset_name}/all", overall_score))
                    else:
                        score_entries = [(dataset_name, overall_score)]
                    for key_suffix, score in score_entries:
                        full_key = f"{benchmark_name}/{key_suffix}"
                        if full_key not in benchmark_scores[benchmark_name]:
                            benchmark_scores[benchmark_name][full_key] = {
                                'scores': [],
                                'dataset_name': key_suffix,
                            }
                        if score is not None:
                            benchmark_scores[benchmark_name][full_key][
                                'scores'].append(score)

        import numpy as np

        summary_rows = []
        for benchmark_name, datasets in benchmark_scores.items():
            for key, data in datasets.items():
                scores = data['scores']
                dataset_name = data['dataset_name']
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = (np.std(scores, ddof=1)
                                 if len(scores) > 1 else 0.0)
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    row = {
                        'benchmark': benchmark_name,
                        'dataset': dataset_name,
                        'run_count': len(scores),
                        'mean': round(mean_score, 4),
                        'std': round(std_score, 4),
                        'std_percent': (round(std_score / mean_score * 100, 2)
                                        if mean_score > 0 else 0),
                        'min': round(min_score, 4),
                        'max': round(max_score, 4),
                        'range': round(max_score - min_score, 4)
                    }
                    for i, score in enumerate(scores, 1):
                        row[f'run_{i}'] = round(score, 4)
                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        excel_path = self.output_dir / "multi_run_summary.xlsx"
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"Multi-run summary saved to: {excel_path}")

        json_path = self.output_dir / "multi_run_summary.json"
        summary_data = {
            'num_runs': self.num_runs,
            'version': self.version,
            'config_file': self.config_file,
            'results': summary_rows
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed summary saved to: {json_path}")

        logger.info("\nMulti-run results summary:")
        logger.info("-" * 100)
        for _, row in summary_df.iterrows():
            logger.info(
                f"{row['benchmark']}/{row['dataset']}: "
                f"mean={row['mean']:.4f}, "
                f"std={row['std']:.4f} ({row['std_percent']:.2f}%), "
                f"range=[{row['min']:.4f}, {row['max']:.4f}]")
        logger.info("-" * 100)

        return summary_df

def _expand_env_vars(text: str) -> str:
    """Expand ${VAR} and ${VAR:-default} placeholders in YAML configs."""
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        return default if default is not None else ""

    return pattern.sub(replace, text)


# ============================================================
# Data structure definitions
# ============================================================

@dataclass
class PromptItem:
    """Single prompt item containing all information needed for inference"""
    bench_name: str          # e.g. "AIME25"
    subset_name: str         # e.g. "default"
    sample_id: int           # evalscope sample.id (with repeat expansion)
    group_id: int            # evalscope sample.group_id
    prompt_text: str         # final prompt after apply_chat_template
    sampling_params: SamplingParams  # generation parameters for this benchmark
    sample: Sample           # evalscope Sample reference (for writing JSONL)


@dataclass
class BenchState:
    """State of a single benchmark"""
    bench_name: str
    dataset_name: str
    subsets: Dict[str, Dataset]   # subset_name -> Dataset
    benchmark: Any                # evalscope benchmark adapter
    task_config: TaskConfig       # evalscope TaskConfig
    repeats: int
    total_prompts: int
    completed: int = 0
    # subset -> {sample_id -> result}
    results: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    status: str = "pending"       # pending / inferring / scoring / done / failed
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Streaming persistence: JSONL file handles per subset
    _jsonl_files: Dict[str, Any] = field(default_factory=dict)
    # Scoring results
    per_repeat_scores: List[float] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    error_msg: str = ""


# ============================================================
# Cross-Benchmark Evaluator
# ============================================================

class CrossBenchmarkEvaluator:
    """Cross-Benchmark streaming inference evaluator

    Workflow:
    1. Load all benchmark datasets and extract prompts
    2. Mix all prompts into a global queue and run inference with VLLMOfflineEngine
    3. Collect results per benchmark and write evalscope-compatible predictions JSONL
    4. After a benchmark finishes inference, asynchronously call run_task(use_cache=True) for scoring
    5. Compile all benchmark results into a unified table
    """

    def __init__(self, config_file: str, resume: bool = False,
                 resume_dir: Optional[str] = None,
                 output_dir_override: Optional[str] = None,
                 force_rescore: bool = False):
        self.config_file = config_file
        self.config = self._load_config()
        self.model_path = self.config['model_path']
        if not self.model_path:
            raise ValueError(
                "model_path is required. Set MODEL_PATH in the environment "
                "or provide a config file with model_path."
            )
        self.model_name = _extract_model_name(self.model_path)

        # Output directory
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
            self.version = self.output_dir.name
        elif resume_dir:
            self.output_dir = Path(resume_dir)
            self.version = self.output_dir.name
        elif resume:
            # Auto-detect latest version_* directory for resume
            evalscope_dir = Path(self.model_path) / "evalscope"
            latest = self._find_latest_version(evalscope_dir)
            if latest:
                self.output_dir = latest
                self.version = latest.name
                logger.info(f"Auto-detected latest version directory: {latest}")
            else:
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = evalscope_dir / self.version
                logger.info(f"No existing version detected, fresh run: {self.version}")
        else:
            self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
            self.output_dir = Path(self.model_path) / "evalscope" / self.version

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp (for predictions directory)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Benchmark state
        self.bench_states: Dict[str, BenchState] = {}

        # Resume support
        self.resume = resume or (resume_dir is not None)
        self.completed_benchmarks: Set[str] = set()

        # Whether to force rescoring (even if repeat count is the same)
        self.force_rescore = force_rescore

        # vLLM engine
        self.engine: Optional[VLLMOfflineEngine] = None

        # Scoring thread pool
        self._scoring_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scorer")

        # Progress display
        self._progress: Optional['ProgressDisplay'] = None

        logger.info("CrossBenchmarkEvaluator initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Version: {self.version}")
        logger.info(f"Force rescore: {force_rescore}")

    def _load_config(self) -> Dict:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            text = f.read()
        return yaml.safe_load(_expand_env_vars(text)) or {}

    @staticmethod
    def _find_latest_version(evalscope_dir: Path) -> Optional[Path]:
        """Return the latest version_* directory under evalscope_dir by name, or None if not found"""
        if not evalscope_dir.exists():
            return None
        candidates = sorted(evalscope_dir.glob("version_*"), reverse=True)
        return candidates[0] if candidates else None

    def _load_existing_predictions(self, bench: BenchState) -> int:
        """Load existing predictions to maximize reuse of completed data.

        Strategy:
        - Count complete_repeats: consecutive completed repeat rounds per subset
          (repeat r is complete ⟺ all questions in the subset have entries for q*old_repeats+r)
        - effective_repeats = max(complete_repeats, new_repeats)
          * complete_repeats > new_repeats: use all completed data for scoring without waste
          * complete_repeats < new_repeats: load partial data and fill remaining gaps with inference
        - Only load subsets present in bench.subsets (filter out obsolete subsets from old versions)
        - Reuse existing scores: skip scoring only when subsets are identical and effective_repeats == old_repeats

        Returns: number of loaded entries (filled into bench.results and bench.completed)
        """
        bench_dir = self.output_dir / bench.bench_name
        if not bench_dir.exists():
            return 0

        jsonl_pattern = f"*/predictions/{self.model_name}/{bench.dataset_name}_*.jsonl"
        jsonl_files = sorted(bench_dir.glob(jsonl_pattern))
        if not jsonl_files:
            return 0

        new_repeats = bench.repeats
        # Original question count per subset (non-repeat-expanded), based on current dataset
        num_q_per_subset: Dict[str, int] = {
            s: len(ds) // new_repeats for s, ds in bench.subsets.items()
        }

        # Get old_repeats and old subset info from _result.json
        result_path = self.output_dir / f"{bench.bench_name}_result.json"
        old_repeats = None
        old_result_data = None
        if result_path.exists():
            try:
                with open(result_path) as f:
                    old_result_data = json.load(f)
                old_repeats = old_result_data.get('repeats')
            except Exception:
                pass

        # Load all JSONL entries (for duplicate indices, keep the last occurrence)
        raw_entries: Dict[str, Dict[int, dict]] = {}
        for jsonl_path in jsonl_files:
            stem = jsonl_path.stem
            prefix = f"{bench.dataset_name}_"
            subset_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            if subset_name not in raw_entries:
                raw_entries[subset_name] = {}
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        raw_entries[subset_name][entry['index']] = entry
                    except Exception:
                        continue

        # Only keep subsets that exist in bench.subsets
        if bench.subsets:
            raw_entries = {k: v for k, v in raw_entries.items() if k in bench.subsets}

        total_raw = sum(len(v) for v in raw_entries.values())
        if total_raw == 0:
            return 0

        # Infer old_repeats (if result.json does not provide it)
        if old_repeats is None:
            for subset_name, entries in raw_entries.items():
                nq = num_q_per_subset.get(subset_name, 1)
                if entries:
                    old_repeats = max(1, (max(entries.keys()) + 1 + nq - 1) // nq)
                    break
        if old_repeats is None:
            old_repeats = new_repeats

        # Count complete repeat rounds per subset
        # repeat r is complete ⟺ all questions q have entries for q*old_repeats+r
        def _count_complete(entries: Dict[int, dict], nq: int, nr: int) -> int:
            return sum(
                1 for r in range(nr)
                if all(q * nr + r in entries for q in range(nq))
            )

        complete_per_subset = []
        for subset_name in bench.subsets:
            entries = raw_entries.get(subset_name, {})
            nq = num_q_per_subset.get(subset_name, 1)
            cr = _count_complete(entries, nq, old_repeats) if entries else 0
            complete_per_subset.append(cr)

        complete_repeats = min(complete_per_subset) if complete_per_subset else 0
        # Maximize use of completed data: if complete_repeats > new_repeats, use all
        effective_repeats = max(complete_repeats, new_repeats)

        logger.info(f"[{bench.bench_name}] Resume: found {total_raw} existing predictions, "
                    f"old_repeats={old_repeats}, complete_repeats={complete_repeats}, "
                    f"new_repeats={new_repeats} → effective_repeats={effective_repeats}")

        # Build sample_id → Sample mapping (using effective_repeats as step)
        # sample.id in bench.subsets is based on new_repeats; repeats beyond new_repeats have no Sample reference (missing metadata only, acceptable)
        sample_map: Dict[int, Sample] = {}
        for subset_name, dataset in bench.subsets.items():
            for sample in dataset:
                q = sample.id // new_repeats
                r = sample.id % new_repeats
                sample_map[q * effective_repeats + r] = sample

        # Remap old sample_ids → new sample_ids (using effective_repeats as step)
        loaded = 0
        for subset_name, entries in raw_entries.items():
            nq = num_q_per_subset.get(subset_name, 1)
            if subset_name not in bench.results:
                bench.results[subset_name] = {}

            for old_sid, entry in entries.items():
                question_id = old_sid // old_repeats
                old_repeat_idx = old_sid % old_repeats

                if old_repeat_idx >= effective_repeats:
                    continue
                if question_id >= nq:
                    continue

                new_sid = question_id * effective_repeats + old_repeat_idx

                choices = entry.get('model_output', {}).get('choices', [])
                if choices:
                    text = choices[0].get('message', {}).get('content', '')
                    finish_reason = choices[0].get('stop_reason', 'stop')
                else:
                    text = ''
                    finish_reason = 'unknown'

                bench.results[subset_name][new_sid] = {
                    'text': text,
                    'finish_reason': finish_reason,
                    'sample': sample_map.get(new_sid),
                }
                loaded += 1

        bench.completed = loaded

        # If effective_repeats differs from new_repeats, update bench repeats and total_prompts
        if effective_repeats != new_repeats:
            bench.repeats = effective_repeats
            bench.total_prompts = sum(
                num_q_per_subset[s] * effective_repeats for s in bench.subsets
            )

        if loaded >= bench.total_prompts:
            # Predictions sufficient, check if rescoring is needed
            # Default: if repeats are the same and force_rescore is not set, skip scoring
            should_rescore = self.force_rescore or (effective_repeats != old_repeats)

            if should_rescore:
                bench.status = "scoring"
                reason = "force_rescore" if self.force_rescore else f"repeats changed ({old_repeats}→{effective_repeats})"
                logger.info(f"[{bench.bench_name}] Resume: predictions sufficient ({loaded}/{bench.total_prompts}), "
                            f"will rescore (reason: {reason})")
            else:
                # Directly reuse existing scoring results
                if old_result_data and 'per_repeat_scores' in old_result_data:
                    bench.per_repeat_scores = old_result_data['per_repeat_scores']
                    bench.stats = {
                        'mean': old_result_data.get('mean', 0),
                        'std': old_result_data.get('std', 0),
                        'min': old_result_data.get('min', 0),
                        'max': old_result_data.get('max', 0),
                    }
                    bench.status = "done"
                    self.completed_benchmarks.add(bench.bench_name)
                    logger.info(f"[{bench.bench_name}] Resume: predictions sufficient and repeats match ({effective_repeats}), "
                                f"skipping scoring, reusing existing results (mean={bench.stats['mean']:.4f})")
                else:
                    # No old scoring results, need to rescore
                    bench.status = "scoring"
                    logger.info(f"[{bench.bench_name}] Resume: predictions sufficient but no old scoring results, will rescore")
        else:
            logger.info(f"[{bench.bench_name}] Resume: need to fill "
                        f"{bench.total_prompts - loaded}/{bench.total_prompts} predictions")

        return loaded

    def _save_progress(self):
        """Save progress file"""
        progress = {
            'version': self.version,
            'config_file': self.config_file,
            'timestamp': self._timestamp,
            'completed_benchmarks': list(self.completed_benchmarks),
            'updated_at': datetime.now().isoformat(),
        }
        progress_path = self.output_dir / "progress_v2.json"
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _create_task_config(self, bench_name: str) -> TaskConfig:
        """Create TaskConfig (reuse config parsing logic from eval.py)"""
        bench_cfg = self.config['benchmarks'][bench_name]

        task_config = TaskConfig()
        task_config.model_id = self.model_name
        task_config.model = self.model_name  # avoid "No model is provided" warning
        task_config.eval_backend = EvalBackend.NATIVE

        # Support specifying data source (huggingface/modelscope)
        if 'data_source' in bench_cfg:
            from evalscope.constants import HubType
            hub_type = bench_cfg['data_source']
            if hub_type == 'huggingface':
                task_config.dataset_hub = HubType.HUGGINGFACE
            elif hub_type == 'modelscope':
                task_config.dataset_hub = HubType.MODELSCOPE
            logger.info(f"[{bench_name}] Using data source: {task_config.dataset_hub}")

        # Dataset
        task_config.datasets = [bench_cfg['dataset_name']]
        if 'limit' in bench_cfg:
            task_config.limit = bench_cfg['limit']
        task_config.repeats = bench_cfg.get('repeats', 1)

        dataset_args = {
            'dataset_name': bench_cfg.get('subset', None),
            'few_shot_num': bench_cfg.get('few_shot_num', 0)
        }
        # If subset is specified, restrict evalscope to load only that subset via subset_list
        # (BenchmarkMeta._update will setattr subset_list, preventing other subsets from being counted as 0 during scoring)
        if bench_cfg.get('subset'):
            dataset_args['subset_list'] = [bench_cfg['subset']]
        if 'extra_params' in bench_cfg:
            dataset_args['extra_params'] = bench_cfg['extra_params']
        # Support specifying data source (huggingface/modelscope)
        if 'data_source' in bench_cfg:
            dataset_args['data_source'] = bench_cfg['data_source']
        task_config.dataset_args = {bench_cfg['dataset_name']: dataset_args}

        # Generation parameters
        generation_config_dict = {
            'temperature': bench_cfg.get('temperature', 0.0),
            'top_p': bench_cfg.get('top_p', 1.0),
            'top_k': bench_cfg.get('top_k', 50),
            'max_tokens': bench_cfg.get('max_tokens', 2048),
            'repetition_penalty': bench_cfg.get('repetition_penalty', 1.0)
        }
        gen_config = bench_cfg.get('gen_config', {})
        if gen_config:
            generation_config_dict.update(gen_config)
        task_config.generation_config = GenerateConfig.model_validate(generation_config_dict)

        # Working directory
        task_config.work_dir = str(self.output_dir / bench_name)

        # Sandbox configuration
        if 'use_sandbox' in bench_cfg:
            task_config.use_sandbox = bench_cfg['use_sandbox']
            sandbox_workers = bench_cfg.get('sandbox_workers', 5)
            task_config.judge_worker_num = sandbox_workers

            sandbox_config = self.config.get('sandbox', {})
            remote_url = sandbox_config.get('remote_url')
            if remote_url:
                remote_config = {'base_url': remote_url}
                if sandbox_config.get('timeout'):
                    remote_config['timeout'] = sandbox_config['timeout']
                task_config.sandbox_manager_config = remote_config
            elif sandbox_config.get('work_dir'):
                task_config.sandbox_manager_config = {'work_dir': sandbox_config['work_dir']}

        # Judge configuration
        bench_judge_strategy = bench_cfg.get('judge_strategy', JudgeStrategy.RULE)
        task_config.judge_strategy = bench_judge_strategy
        use_llm = 'llm' in str(bench_judge_strategy).lower()

        if use_llm:
            api_key = self.config.get('API_KEY', '')
            base_url = self.config.get('BASE_URL', '').replace('/chat/completions', '').rstrip('/')
            judge_model = bench_cfg.get('judge_model', 'Qwen/Qwen3-8B')
            if not api_key or not base_url:
                raise ValueError(f"[{bench_name}] LLM judge requires API_KEY and BASE_URL")
            task_config.judge_model_args = {
                'api_key': api_key,
                'api_url': base_url,
                'model_id': judge_model,
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 1024,
                    'extra_body': {'enable_thinking': False}
                },
            }
            judge_parallel = bench_cfg.get('judge_parallel', 1)
            task_config.judge_worker_num = judge_parallel

        return task_config

    @staticmethod
    def _config_to_sampling_params(config: GenerateConfig) -> SamplingParams:
        """Convert GenerateConfig to SamplingParams"""
        kwargs = {}
        if config.temperature is not None:
            kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            kwargs['top_p'] = config.top_p
        if config.top_k is not None and config.top_k > 0:
            kwargs['top_k'] = config.top_k
        if config.max_tokens is not None:
            kwargs['max_tokens'] = config.max_tokens
        if getattr(config, 'repetition_penalty', None) is not None:
            kwargs['repetition_penalty'] = config.repetition_penalty
        return SamplingParams(**kwargs)

    def _load_all_benchmarks(self):
        """Load all benchmark datasets"""
        logger.info("=" * 60)
        logger.info("Loading all benchmark data...")

        for bench_name, bench_cfg in self.config['benchmarks'].items():
            logger.info(f"[{bench_name}] Loading...")

            try:
                # 1. Build TaskConfig
                task_config = self._create_task_config(bench_name)

                # 2. Load benchmark + dataset via evalscope API
                benchmark = get_benchmark(bench_cfg['dataset_name'], task_config)
                dataset_dict = {k: v for k, v in benchmark.load_dataset().items()}

                # If config specifies subset, filter to keep only that subset
                requested_subset = bench_cfg.get('subset')
                if requested_subset:
                    if requested_subset in dataset_dict:
                        dataset_dict = {requested_subset: dataset_dict[requested_subset]}
                    else:
                        logger.warning(f"[{bench_name}] Specified subset '{requested_subset}' not in dataset "
                                       f"{list(dataset_dict.keys())}, using full data")

                # 3. Record BenchState
                total = sum(len(ds) for ds in dataset_dict.values())
                bench_state = BenchState(
                    bench_name=bench_name,
                    dataset_name=bench_cfg['dataset_name'],
                    subsets=dataset_dict,
                    benchmark=benchmark,
                    task_config=task_config,
                    repeats=task_config.repeats,
                    total_prompts=total,
                )
                self.bench_states[bench_name] = bench_state

                logger.info(f"[{bench_name}] Loading complete: {total} prompts, {task_config.repeats} repeats")

                # 4. Resume: load existing predictions, auto-handle repeat changes
                if self.resume:
                    loaded = self._load_existing_predictions(bench_state)
                    if bench_state.status == "done":
                        self.completed_benchmarks.add(bench_name)

            except Exception as e:
                logger.error(f"[{bench_name}] Loading failed: {e}")
                import traceback
                traceback.print_exc()
                self.bench_states[bench_name] = BenchState(
                    bench_name=bench_name,
                    dataset_name=bench_cfg.get('dataset_name', 'unknown'),
                    subsets={},
                    benchmark=None,
                    task_config=None,
                    repeats=0,
                    total_prompts=0,
                    status="failed",
                    error_msg=str(e),
                )

        total_all = sum(b.total_prompts for b in self.bench_states.values() if b.status != "failed")
        logger.info(f"Total: {len(self.bench_states)} benchmarks, {total_all} prompts")
        logger.info("=" * 60)

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert ChatMessage list to prompt text"""
        msg_dicts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else msg.text
            msg_dicts.append({"role": msg.role, "content": content})
        return self.engine.tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True)

    def _extract_all_prompts(self) -> List[PromptItem]:
        """Extract PromptItem list from all BenchStates.

        Ordering strategy:
        1. Group by benchmark (small benchmarks first), prompts within the same benchmark are consecutive.
           This lets small benchmarks finish inference earlier and start scoring sooner (CPU threads).
        2. Within each benchmark, sort by repeat round: complete repeat 0 (all questions) first, then repeat 1, ...
           This ensures that if interrupted, completed data consists of full repeats, facilitating resume and scoring.
        3. Shuffle within each repeat round, combined with max_dedup_n limit,
           to ensure enough unique prompts in the Worker to fill the GPU batch.
        """
        from collections import defaultdict

        # Collect by benchmark group
        bench_items: List[Tuple[int, str, List[PromptItem]]] = []

        for bench in self.bench_states.values():
            # Skip completed (done), pending scoring (scoring), and failed ones
            if bench.status in ("failed", "done", "scoring"):
                continue

            sp = self._config_to_sampling_params(bench.task_config.generation_config)
            existing_results = bench.results  # existing results loaded by resume
            repeats = bench.repeats

            # Collect by repeat round group
            repeat_groups: Dict[int, List[PromptItem]] = defaultdict(list)
            skipped = 0
            for subset_name, dataset in bench.subsets.items():
                for sample in dataset:
                    # Skip samples already loaded by resume
                    if existing_results.get(subset_name, {}).get(sample.id) is not None:
                        skipped += 1
                        continue
                    prompt_text = self._messages_to_prompt(sample.input)
                    item = PromptItem(
                        bench_name=bench.bench_name,
                        subset_name=subset_name,
                        sample_id=sample.id,
                        group_id=sample.group_id,
                        prompt_text=prompt_text,
                        sampling_params=sp,
                        sample=sample,
                    )
                    repeat_idx = sample.id % repeats
                    repeat_groups[repeat_idx].append(item)

            if skipped > 0:
                logger.info(f"  {bench.bench_name}: skipped {skipped} existing, need to fill "
                            f"{sum(len(g) for g in repeat_groups.values())}")

            # Sort by repeat round: repeat 0 → repeat 1 → ...
            # Shuffle within each repeat round to break locality
            items = []
            for r in sorted(repeat_groups.keys()):
                group = repeat_groups[r]
                random.shuffle(group)
                items.extend(group)

            bench_items.append((len(items), bench.bench_name, items))

        # Sort by benchmark size: smaller first, finish and score earlier
        bench_items.sort(key=lambda x: x[0])

        all_items = []
        for size, name, items in bench_items:
            all_items.extend(items)
            logger.info(f"  {name}: {size} prompts")

        logger.info(f"Prompt extraction complete: {len(all_items)} prompts, "
                    f"from {len(bench_items)} benchmarks, small benchmarks first, repeat rounds first")
        return all_items

    def _run_global_inference(self, all_prompts: List[PromptItem]):
        """Submit all prompts to VLLMOfflineEngine and collect results

        Use ThreadPoolExecutor to submit all prompts concurrently,
        engine's _batch_loop handles batching and deduplication automatically.
        """
        total = len(all_prompts)
        logger.info(f"Starting global inference: {total} prompts")

        # Calculate baseline count already loaded by resume (not counted for speed estimation)
        resumed_count = sum(
            b.completed for b in self.bench_states.values() if b.status != "failed"
        )

        # Initialize StatusPrinter, taking over logger's StreamHandler
        self._progress = StatusPrinter(self.bench_states, resumed_done=resumed_count)
        _hijacked: List[logging.Handler] = []
        _status_handler: Optional[_StatusLogHandler] = None
        root_log = logging.getLogger()
        for h in list(root_log.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                orig_fmt = h.formatter
                root_log.removeHandler(h)
                _hijacked.append(h)
        if _hijacked:
            _status_handler = _StatusLogHandler(self._progress, _hijacked[0].formatter)
            _status_handler.setLevel(logging.DEBUG)
            root_log.addHandler(_status_handler)

        # Register TPS callback (sync status block update when engine triggers TPS print)
        if self.engine:
            self.engine._tps_callback = self._progress.record_tps

        # Submit all prompts
        vllm_cfg = self.config.get('vllm', {})
        eval_batch_size = vllm_cfg.get('eval_batch_size', 512)

        with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
            futures = {}
            for item in all_prompts:
                fut = executor.submit(
                    self.engine.generate_single,
                    item.prompt_text,
                    item.sampling_params,
                    item.bench_name,  # group_key: same benchmark priority dispatch
                )
                futures[fut] = item

            logger.info(f"Submitted {len(futures)} inference tasks")

            # Collect results
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    result = fut.result()
                    self._on_result(item, result)
                except Exception as e:
                    logger.error(f"Inference failed [{item.bench_name}/{item.sample_id}]: {e}")
                    # Mark failed but continue
                    self._on_result(item, {'text': '', 'finish_reason': 'error'})

        # Restore logger handlers, clean up TPS callback
        if self.engine:
            self.engine._tps_callback = None
        if _status_handler:
            root_log.removeHandler(_status_handler)
        for h in _hijacked:
            root_log.addHandler(h)
        if self._progress:
            self._progress.close()
            self._progress = None

        logger.info("Global inference complete")

    def _get_jsonl_writer(self, bench: BenchState, subset_name: str):
        """Get or create JSONL file handle for subset (for streaming append writes)"""
        if subset_name not in bench._jsonl_files:
            work_dir = self.output_dir / bench.bench_name / self._timestamp
            pred_dir = work_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"
            bench._jsonl_files[subset_name] = open(jsonl_path, 'a', encoding='utf-8')
        return bench._jsonl_files[subset_name]

    def _close_jsonl_writers(self, bench: BenchState):
        """Close all JSONL file handles"""
        for f in bench._jsonl_files.values():
            f.close()
        bench._jsonl_files.clear()

    def _on_result(self, item: PromptItem, result: dict):
        """Receive inference result, route to corresponding BenchState, and immediately append to JSONL"""
        bench = self.bench_states.get(item.bench_name)
        if bench is None:
            return

        # Debug: check if returned result is empty
        if not result.get('text'):
            logger.warning(f"[{item.bench_name}] sample_id={item.sample_id} returned empty result! finish_reason={result.get('finish_reason')}")

        with bench.lock:
            if item.subset_name not in bench.results:
                bench.results[item.subset_name] = {}

            bench.results[item.subset_name][item.sample_id] = {
                'text': result['text'],
                'finish_reason': result['finish_reason'],
                'sample': item.sample,
            }
            bench.completed += 1

            # Streaming persistence: immediately append to predictions JSONL (recoverable after crash)
            try:
                f = self._get_jsonl_writer(bench, item.subset_name)
                sample = item.sample
                model_result = {
                    "index": item.sample_id,
                    "model": self.model_name,
                    "model_output": {
                        "model": self.model_name,
                        "choices": [{
                            "message": {"role": "assistant", "content": result['text']},
                            "stop_reason": result['finish_reason'] or "stop",
                        }],
                    },
                    "messages": [{"role": "assistant", "content": result['text']}],
                    "metadata": getattr(sample, 'metadata', {}),
                }
                f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
                f.flush()
            except Exception as e:
                logger.warning(f"[{item.bench_name}] Streaming write failed: {e}")

            # Real-time progress display refresh
            if self._progress is not None:
                self._progress.refresh()

            # Check if complete
            if bench.completed >= bench.total_prompts:
                if bench.status != "scoring":
                    bench.status = "scoring"
                    # Close JSONL file handles
                    self._close_jsonl_writers(bench)
                    # Asynchronously trigger scoring
                    self._scoring_pool.submit(self._score_benchmark, bench)

    def _write_predictions(self, bench: BenchState):
        """Write evalscope-compatible predictions JSONL"""
        work_dir = self.output_dir / bench.bench_name / self._timestamp

        for subset_name, results in bench.results.items():
            pred_dir = work_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"

            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample_id, res in sorted(results.items()):
                    sample = res['sample']
                    # ModelOutput format: needs choices array
                    model_result = {
                        "index": sample_id,
                        "model": self.model_name,
                        "model_output": {
                            "model": self.model_name,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": res['text'],
                                    },
                                    "stop_reason": res['finish_reason'] or "stop",
                                }
                            ],
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": res['text'],
                            }
                        ],
                        "metadata": getattr(sample, 'metadata', {}),
                    }
                    f.write(json.dumps(model_result, ensure_ascii=False) + '\n')

        logger.info(f"[{bench.bench_name}] Predictions written to: {work_dir}")

    def _score_benchmark(self, bench: BenchState):
        """Split repeats → score each repeat independently → aggregate statistics"""
        logger.info(f"[{bench.bench_name}] Starting scoring...")

        work_dir = self.output_dir / bench.bench_name / self._timestamp
        scores = []

        try:
            for repeat_idx in range(bench.repeats):
                # 1. Split: only take predictions belonging to this repeat
                repeat_work_dir = self._write_predictions_for_repeat(bench, repeat_idx)

                # 2. Call run_task(use_cache=True) for scoring only
                task_cfg = copy.deepcopy(bench.task_config)
                task_cfg.repeats = 1  # Score only 1 repeat each time
                task_cfg.use_cache = str(repeat_work_dir)
                task_cfg.work_dir = str(repeat_work_dir)  # Ensure scoring results are output to repeat directory

                # Use model_id instead of model (to avoid loading model)
                task_cfg.model = self.model_name

                logger.info(f"[{bench.bench_name}] Scoring repeat {repeat_idx + 1}/{bench.repeats}...")
                result = run_task(task_cfg)

                # Debug: check result type
                logger.info(f"  result type: {type(result)}")
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                    logger.info("  converted to dict")
                elif hasattr(result, 'model_dump'):
                    result = result.model_dump()
                    logger.info("  model_dump to dict")

                # 3. Extract score
                score = self._extract_score(result, bench)
                scores.append(score)
                logger.info(f"[{bench.bench_name}] repeat {repeat_idx + 1}: score={score:.4f}")

            # 4. Aggregate statistics
            bench.per_repeat_scores = scores
            bench.stats = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
            bench.status = "done"

            # Save results
            self._save_benchmark_result(bench)
            self.completed_benchmarks.add(bench.bench_name)
            self._save_progress()

            logger.info(f"[{bench.bench_name}] Scoring complete: mean={bench.stats['mean']:.4f}, std={bench.stats['std']:.4f}")

        except Exception as e:
            logger.error(f"[{bench.bench_name}] Scoring failed: {e}")
            bench.status = "failed"
            bench.error_msg = str(e)

    def _write_predictions_for_repeat(self, bench: BenchState, repeat_idx: int) -> Path:
        """Write predictions JSONL for the specified repeat

        After evalscope repeat expansion, sample_id = original_id * repeats + repeat_idx
        Each repeat contains one answer for all original questions.
        """
        work_dir = self.output_dir / bench.bench_name / self._timestamp
        repeat_dir = work_dir / f"repeat_{repeat_idx}"

        total_written = 0
        empty_count = 0

        for subset_name, results in bench.results.items():
            pred_dir = repeat_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"

            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample_id, res in sorted(results.items()):
                    # Only take samples belonging to this repeat
                    if sample_id % bench.repeats != repeat_idx:
                        continue

                    # Remap index to continuous id without repeats
                    new_index = sample_id // bench.repeats
                    sample = res.get('sample')

                    # Count empty results
                    if not res['text']:
                        empty_count += 1

                    # ModelOutput format: needs choices array
                    model_result = {
                        "index": new_index,
                        "model": self.model_name,
                        "model_output": {
                            "model": self.model_name,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": res['text'],
                                    },
                                    "stop_reason": res['finish_reason'] or "stop",
                                }
                            ],
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": res['text'],
                            }
                        ],
                        "metadata": getattr(sample, 'metadata', {}) if sample else {},
                    }
                    f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
                    total_written += 1

        logger.info(f"[{bench.bench_name}] repeat_{repeat_idx}: wrote {total_written} predictions, {empty_count} empty results")
        return repeat_dir

    def _extract_score(self, result, bench: Optional[BenchState] = None) -> float:
        """Extract score from evalscope result

        result could be:
        - dict: {dataset_name: report_dict, ...}
        - Report object
        - List[Report]

        When bench specifies a subset, only take that subset's score (to avoid being pulled down by 0 scores from other subsets)
        """
        # Get the actual subset set used by current benchmark
        valid_subsets = set(bench.subsets.keys()) if bench and bench.subsets else None

        # 1. Handle Report object
        if hasattr(result, 'score'):
            score = result.score
            logger.info(f"  Report.score = {score}")
            return float(score) if score is not None else 0.0

        # 2. Handle List[Report]
        if isinstance(result, list):
            scores = []
            for r in result:
                if hasattr(r, 'score') and r.score is not None:
                    # If Report has name/dataset_name attribute, filter by valid_subsets
                    r_name = getattr(r, 'name', None) or getattr(r, 'dataset_name', None)
                    if valid_subsets and r_name and r_name not in valid_subsets:
                        continue
                    scores.append(float(r.score))
            final_score = np.mean(scores) if scores else 0.0
            logger.info(f"  List[Report] scores = {scores}, final = {final_score}")
            return final_score

        # 3. Handle dict
        if not isinstance(result, dict):
            logger.warning(f"_extract_score: unknown type {type(result)}")
            return 0.0

        # run_task returns {dataset_name: report_dict} format
        # First check if it is a single dataset result (most common case)
        if len(result) == 1:
            dataset_result = list(result.values())[0]
            if isinstance(dataset_result, dict):
                score = dataset_result.get('score', dataset_result.get('accuracy', 0))
                if score is not None:
                    final_score = float(score)
                    logger.info(f"  Single dataset result: score = {final_score}")
                    return final_score
            elif hasattr(dataset_result, 'score') and dataset_result.score is not None:
                final_score = float(dataset_result.score)
                logger.info(f"  Single dataset Report result: score = {final_score}")
                return final_score

        # Iterate dataset results (multi-dataset case)
        # Note: dict key is dataset_name (e.g. 'aime25'), not subset_name
        # So valid_subsets should not be used for filtering; valid_subsets are subset names (e.g. 'AIME2025-I')
        scores = []
        for dataset_name, dataset_result in result.items():
            if isinstance(dataset_result, dict):
                score = dataset_result.get('score', dataset_result.get('accuracy', 0))
                if score is not None:
                    scores.append(float(score))
            elif hasattr(dataset_result, 'score'):
                if dataset_result.score is not None:
                    scores.append(float(dataset_result.score))

        final_score = np.mean(scores) if scores else 0.0
        logger.info(f"  Dict result: scores = {scores}, final = {final_score}")
        return final_score

    def _save_benchmark_result(self, bench: BenchState):
        """Save benchmark result to JSON"""
        result = {
            'benchmark': bench.bench_name,
            'dataset': bench.dataset_name,
            'subsets': list(bench.subsets.keys()),
            'repeats': bench.repeats,
            'per_repeat_scores': bench.per_repeat_scores,
            'mean': bench.stats.get('mean', 0),
            'std': bench.stats.get('std', 0),
            'min': bench.stats.get('min', 0),
            'max': bench.stats.get('max', 0),
        }

        result_path = self.output_dir / f"{bench.bench_name}_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"[{bench.bench_name}] Result saved: {result_path}")

    def _compile_summary(self):
        """Compile all benchmark results into a unified table"""
        rows = []

        for bench in self.bench_states.values():
            if bench.status != "done":
                continue

            row = {
                'benchmark': bench.bench_name,
                'dataset': bench.dataset_name,
                'repeats': bench.repeats,
                'mean': bench.stats.get('mean', 0),
                'std': bench.stats.get('std', 0),
                'min': bench.stats.get('min', 0),
                'max': bench.stats.get('max', 0),
            }

            # Add score for each repeat
            for i, score in enumerate(bench.per_repeat_scores):
                row[f'run_{i + 1}'] = round(score, 4)

            rows.append(row)

        if not rows:
            logger.warning("No completed benchmarks to summarize")
            return

        df = pd.DataFrame(rows)

        # Save Excel
        excel_path = self.output_dir / "unified_summary.xlsx"
        df.to_excel(excel_path, index=False)
        logger.info(f"Summary saved: {excel_path}")

        # Save JSON
        json_path = self.output_dir / "unified_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved: {json_path}")

        # Print summary
        logger.info("\\n" + "=" * 80)
        logger.info("Evaluation result summary:")
        logger.info("-" * 80)
        for row in rows:
            logger.info(f"{row['benchmark']:20s} | mean={row['mean']:.4f}, std={row['std']:.4f}, "
                       f"range=[{row['min']:.4f}, {row['max']:.4f}]")
        logger.info("=" * 80)

    def run(self):
        """Main workflow"""
        logger.info("\\n" + "=" * 80)
        logger.info("CrossBenchmarkEvaluator V2 started")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Config file: {self.config_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80 + "\n")

        # 1. Load all benchmarks
        self._load_all_benchmarks()

        # Filter out failed and completed ones
        active_benches = {k: v for k, v in self.bench_states.items()
                         if v.status not in ("failed", "done")}
        # Keep done ones for summary
        done_benches = {k: v for k, v in self.bench_states.items() if v.status == "done"}

        if not active_benches and not done_benches:
            logger.error("No valid benchmarks to evaluate")
            return

        # First synchronously complete benchmarks that need rescoring (to avoid hang from concurrent access to inference engine / StatusPrinter)
        scoring_benches = {k: v for k, v in active_benches.items() if v.status == "scoring"}
        if scoring_benches:
            logger.info(f"First complete scoring for {len(scoring_benches)} benchmarks (synchronous)...")
            scoring_futures = []
            for bench in scoring_benches.values():
                logger.info(f"[{bench.bench_name}] Triggering rescoring...")
                scoring_futures.append(
                    self._scoring_pool.submit(self._score_benchmark, bench))
            # Wait for all scoring to complete before starting inference engine
            for fut in scoring_futures:
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Scoring exception: {e}")
            logger.info("All pending scoring completed")

        # Benchmarks needing inference
        need_inference = {k: v for k, v in active_benches.items()
                         if v.status == "pending"}

        # Merge all benches participating in this round (need inference + scoring + completed)
        self.bench_states = {**done_benches, **active_benches}

        try:
            if need_inference:
                # 2. Initialize vLLM engine (only when inference is needed)
                vllm_cfg = self.config.get('vllm', {})
                dp = vllm_cfg.get('dp', 1)
                chunk_size = vllm_cfg.get('worker_chunk_size', 0)
                # worker_chunk_size=0 means full dispatch, but progress may not update for a long time
                # When dp>1, auto-set a reasonable chunk_size to ensure Workers return results more frequently
                if chunk_size == 0 and dp > 1:
                    chunk_size = 64
                    logger.info(f"worker_chunk_size not set, dp={dp}, auto-adjusted to {chunk_size} for higher progress update frequency")
                self.engine = VLLMOfflineEngine(
                    model_path=self.model_path,
                    tp=vllm_cfg.get('tp', 1),
                    dp=dp,
                    gpu_memory_utilization=vllm_cfg.get('gpu_memory_utilization', 0.9),
                    max_model_len=vllm_cfg.get('max_model_len', None),
                    enforce_eager=vllm_cfg.get('enforce_eager', True),
                    worker_chunk_size=chunk_size,
                    max_dedup_n=vllm_cfg.get('max_dedup_n', 8),
                )

                tps_interval = vllm_cfg.get('tps_interval', 30)
                self.engine._tps_interval = tps_interval
                self.engine._tps_reset()
                logger.info(f"TPS statistics started, printing every {tps_interval} seconds")

                # 3. Extract prompts needing inference (skip those already loaded by resume)
                all_prompts = self._extract_all_prompts()
                logger.info(f"Extracted {len(all_prompts)} prompts needing inference")

                if all_prompts:
                    # 4. Global inference
                    self._run_global_inference(all_prompts)
                else:
                    logger.info("All prompts already have predictions, skipping inference")
            else:
                logger.info("All benchmarks have sufficient predictions, skipping inference engine initialization")

            # 5. Wait for all scoring to complete
            pending_scoring = {k for k, v in self.bench_states.items()
                              if v.status not in ("done", "failed")}
            if pending_scoring:
                logger.info(f"Waiting for {len(pending_scoring)} benchmarks to finish scoring...")
                self._wait_for_scoring()

            # 6. Compile results
            self._compile_summary()

            # 7. Save config
            self._save_config()

        finally:
            if self.engine:
                self.engine.shutdown()
                self.engine = None
            self._scoring_pool.shutdown(wait=True)

        logger.info("\\nCrossBenchmarkEvaluator V2 finished!")

    def _wait_for_scoring(self):
        """Wait for all benchmark scoring to complete"""
        pending = set(self.bench_states.keys())
        last_log = time.time()
        while pending:
            done = set()
            for name in list(pending):
                if self.bench_states[name].status in ("done", "failed"):
                    done.add(name)
            for name in done:
                pending.remove(name)
                bench = self.bench_states[name]
                logger.info(f"[{name}] scoring {'completed' if bench.status == 'done' else 'failed'}")
            if pending and time.time() - last_log >= 30:
                names = ', '.join(pending)
                logger.info(f"Waiting for scoring to complete: {names}")
                last_log = time.time()
            if pending:
                time.sleep(1)

    def _save_config(self):
        """Save config file copy"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Config saved: {config_path}")


# ============================================================
# Progress display
# ============================================================

class StatusPrinter:
    """
    Use ANSI escape codes to keep a "status block fixed at the bottom" when using tail -f.

    A fixed-height status block is always kept at the end of the file. Whenever there is a log message or status update:
      1. Move up N lines + clear to end of screen (erase old status block)
      2. Write new content (log line / empty)
      3. Redraw status block

    Status block layout (total _block_height lines):
      ──────────────────────────────────────────────────────────
       Total [████████░░]  2304/ 8640  26.7%  bench 2/9  ⏱ 00:12:34
      ──────────────────────────────────────────────────────────
       AIME25          [████████████████████]  1280/ 1280 100.0% ✓
       AIME24          [████████████████████]   960/  960 100.0% ✓
       AMC             [██████░░░░░░░░░░░░░░]   368/  736  50.0% →
       ...
      ──────────────────────────────────────────────────────────
       TPS:  1234.5 tok/s (10:41:05)  →   987.3 tok/s (10:41:35)  →  1102.8 tok/s (10:42:05)
      ──────────────────────────────────────────────────────────
    """

    BAR_W        = 20    # Progress bar width (chars)
    LINE_W       = 80    # Line width limit to avoid cursor offset from terminal line wrapping
    MAX_TPS_HIST = 3     # Keep last N TPS records
    REDRAW_SEC   = 5     # Background periodic refresh interval (seconds)

    EMA_ALPHA    = 0.1   # EMA smoothing factor (smaller = smoother)

    def __init__(self, bench_states: Dict[str, 'BenchState'], resumed_done: int = 0):
        self._bench_states  = bench_states
        self._lock          = threading.Lock()
        self._tps_history: List[Tuple[float, str]] = []  # (ema_tps, time_str)
        self._ema_tps: Optional[float] = None            # TPS EMA smoothed value
        self._start_time    = time.time()
        self._initialized   = False

        # Resume baseline: count completed before inference starts, not counted for speed estimation
        self._resumed_done  = resumed_done
        self._last_done     = resumed_done   # done count at last sample time
        self._last_sample_t = self._start_time
        self._ema_speed: Optional[float] = None  # EMA smoothed inference speed (samples/sec)

        # Fixed line count: separators×4 + total progress×1 + bench×N + TPS×1
        self._block_height  = 4 + 1 + len(bench_states) + 1

        # Start background periodic refresh thread
        self._stop = threading.Event()
        self._bg   = threading.Thread(target=self._bg_loop, daemon=True, name="StatusPrinter")
        self._bg.start()

    # ── External API ──────────────────────────────────────────

    def write_log(self, msg: str):
        """Log message routing entry, called by _StatusLogHandler."""
        with self._lock:
            self._render(msg)

    def refresh(self):
        """Called by _on_result, throttled redraw (at most once per 0.5s to avoid excessive screen refresh)."""
        now = time.time()
        if now - getattr(self, '_last_refresh', 0) < 0.5:
            return
        self._last_refresh = now
        with self._lock:
            self._render(None)

    def record_tps(self, window_tps: float, _global_tps: float):
        """Record a TPS reading and trigger redraw. Called by VLLMOfflineEngine._tps_callback."""
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            # Apply EMA smoothing to window_tps to avoid spikes when large batches complete
            if self._ema_tps is None:
                self._ema_tps = window_tps
            else:
                self._ema_tps = self.EMA_ALPHA * window_tps + (1 - self.EMA_ALPHA) * self._ema_tps
            self._tps_history.append((self._ema_tps, ts))
            if len(self._tps_history) > self.MAX_TPS_HIST:
                self._tps_history.pop(0)
            self._render(None)

    def close(self):
        """Stop background thread; do one final redraw so the status block stays at the final position."""
        self._stop.set()
        with self._lock:
            self._render(None)

    # ── Internal implementation ──────────────────────────────

    def _render(self, log_line: Optional[str]):
        """Erase old status block → write log line (if any) → redraw status block. Must be called within _lock."""
        out = sys.stderr
        if self._initialized:
            out.write(f'\033[{self._block_height}A\033[J')
        if log_line is not None:
            out.write(log_line + '\n')
        for line in self._build_block():
            out.write(line + '\n')
        out.flush()
        self._initialized = True

    def _bg_loop(self):
        while not self._stop.wait(self.REDRAW_SEC):
            with self._lock:
                self._render(None)

    def _build_block(self) -> List[str]:
        """Build current status block, returning a list of exactly _block_height lines."""
        W   = self.LINE_W
        sep = '─' * W

        # ── Total progress ──────────────────────────────────
        total = sum(b.total_prompts for b in self._bench_states.values()
                    if b.status != "failed")
        done  = sum(b.completed     for b in self._bench_states.values()
                    if b.status != "failed")
        pct   = done / total * 100 if total > 0 else 0.0
        bw    = self.BAR_W
        bar   = '█' * int(bw * done / total) + '░' * (bw - int(bw * done / total)) \
                if total > 0 else '░' * bw
        elap  = time.time() - self._start_time
        etime = f"{int(elap//3600):02d}:{int(elap%3600//60):02d}:{int(elap%60):02d}"

        # Calculate ETA (based on EMA smoothed speed, excluding resume data)
        now = time.time()
        new_done = done - self._resumed_done  # actual completed count in this inference round
        eta_str = "calculating..."
        if new_done > 0 and done < total:
            # Update EMA speed: calculate instantaneous speed using delta / time difference
            dt = now - self._last_sample_t
            d_done = done - self._last_done
            if dt > 0 and d_done > 0:
                instant_speed = d_done / dt
                if self._ema_speed is None:
                    self._ema_speed = instant_speed
                else:
                    self._ema_speed = self.EMA_ALPHA * instant_speed + (1 - self.EMA_ALPHA) * self._ema_speed
                self._last_done = done
                self._last_sample_t = now

            speed = self._ema_speed
            if speed is None:
                # No delta sampled yet, use global average as fallback
                speed = new_done / elap if elap > 0 else 0

            remain = total - done
            eta_sec = remain / speed if speed > 0 else 0
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            eta_s = int(eta_sec % 60)
            eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        elif done >= total:
            eta_str = "completed"

        n_done   = sum(1 for b in self._bench_states.values() if b.status == "done")
        n_total  = len(self._bench_states)
        total_line = (
            f" Total [{bar}] {done:6d}/{total:<6d} {pct:5.1f}%"
            f"  bench {n_done}/{n_total}  ⏱ {etime}  ETA: {eta_str}"
        )

        # ── per-bench ────────────────────────────────────────
        bench_lines = []
        for name, bench in self._bench_states.items():
            bd = bench.completed
            bt = max(bench.total_prompts, 1)
            pb = bd / bt
            bbar = '█' * int(bw * pb) + '░' * (bw - int(bw * pb))
            st = bench.status
            if   st == "done":    icon, c = "✓", "\033[32m"
            elif st == "scoring": icon, c = "⏳", "\033[33m"
            elif st == "failed":  icon, c = "✗", "\033[31m"
            elif st == "pending": icon, c = " ", "\033[90m"
            else:                 icon, c = "→", ""
            rst  = "\033[0m"
            nm   = name[:16].ljust(16)
            bench_lines.append(
                f" {c}{nm}{rst} [{bbar}] {bd:5d}/{bt:5d} {pb*100:5.1f}% {icon}"
            )

        # ── TPS ──────────────────────────────────────────────
        if self._tps_history:
            parts = [f"\033[36m{wt:>8.1f}\033[0m tok/s ({ts})"
                     for wt, ts in self._tps_history]
            tps_line = " TPS: " + "  →  ".join(parts)
        else:
            tps_line = " TPS: waiting for data..."

        return [sep, total_line, sep] + bench_lines + [sep, tps_line, sep]


class _StatusLogHandler(logging.Handler):
    """Route all StreamHandler output from logger to StatusPrinter, ensuring the status block is always at the bottom."""

    def __init__(self, printer: StatusPrinter, orig_formatter: Optional[logging.Formatter] = None):
        super().__init__()
        self._printer   = printer
        self._formatter = orig_formatter or logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    def emit(self, record: logging.LogRecord):
        try:
            msg = self._formatter.format(record)
            self._printer.write_log(msg)
        except Exception:
            self.handleError(record)


# ============================================================
# CLI entry
# ============================================================

# ============================================================
# Unified CLI entry
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="EvalScope offline evaluation script - unified sequential / cross-benchmark mode",
        formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    parser.add_argument('config_file', help="Evaluation config file path")
    parser.add_argument('num_runs', nargs='?', type=int, default=1,
                        help="Number of runs (default 1, >1 calculates mean/variance)")

    # Mode selection
    parser.add_argument('--cross-bench', action='store_true',
                        help="Cross-benchmark streaming mode: mix all prompts into a global queue,\n"
                             "GPU runs continuously; scoring triggers immediately after each benchmark finishes.")

    # Shared options
    parser.add_argument('--resume', action='store_true',
                        help="Resume: auto-select latest version_* directory, skip completed benchmarks")
    parser.add_argument('--resume-dir', metavar='DIR',
                        help="Resume: directly specify last version directory path\n"
                             "Example: --resume-dir /data/.../evalscope/version_20260310_043148")
    parser.add_argument('--output-dir', metavar='DIR',
                        help="Force specify output directory (cross-bench mode only)")
    parser.add_argument('--force-rescore', action='store_true',
                        help="Force rescore: rescore even if repeat count matches (cross-bench mode only)")

    # Sequential-only options
    parser.add_argument('--merged-runs', action='store_true',
                        help="Merged inference mode (valid when num_runs>1): each benchmark infers in one shot\n"
                             "num_runs*repeats samples, then group by run for statistics, improving GPU utilization")

    args = parser.parse_args()

    if args.num_runs < 1:
        print("Number of runs must be >= 1")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        print(f"Config file does not exist: {args.config_file}")
        sys.exit(1)

    # Cross-benchmark mode
    if args.cross_bench:
        evaluator = CrossBenchmarkEvaluator(
            config_file=args.config_file,
            resume=args.resume,
            resume_dir=args.resume_dir,
            output_dir_override=args.output_dir,
            force_rescore=args.force_rescore,
        )
        evaluator.run()
        return

    # Sequential mode (default)
    is_resume = args.resume or bool(args.resume_dir)
    merge_hint = ", merged inference mode" if args.merged_runs and args.num_runs > 1 else ""
    resume_hint = (f", resume mode (specified directory: {args.resume_dir})" if args.resume_dir
                   else ", resume mode (auto-detect)" if args.resume else "")
    logger.info(
        f"Starting evaluation, config file: {args.config_file}, "
        f"num_runs: {args.num_runs}{merge_hint}{resume_hint}")

    # num_runs=1 and no resume -> single-run path; otherwise use MultiRunEvaluator
    if args.num_runs == 1 and not is_resume:
        evaluator = BenchmarkEvaluator(args.config_file)
        evaluator.run_evaluation()
        evaluator.save_results()
    else:
        multi_evaluator = MultiRunEvaluator(
            args.config_file, num_runs=args.num_runs,
            resume=args.resume, resume_dir=args.resume_dir,
            merge_inference=args.merged_runs)
        multi_evaluator.run_all()

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
