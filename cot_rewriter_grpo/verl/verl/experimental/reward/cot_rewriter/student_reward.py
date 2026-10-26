"""
Student Reward (Cognitive Reward) Module - vLLM Version

Computes R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
            = PPL(original) - PPL(rewrite)
using vLLM for fast log prob computation.

where PPL = exp(-mean(token_log_probs))

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
(lower perplexity) compared to the original CoT.

Uses Ray actors for multi-GPU parallelism. Supports two modes:
1. TP=1: Each GPU runs a separate vLLM instance (Data Parallelism)
2. TP>1: Multiple GPUs share one vLLM instance (Tensor Parallelism)
"""

from typing import Any, Dict, List, Optional
import os

# IMPORTANT: Set PYTORCH_NVML_BASED_CUDA_CHECK=1 BEFORE importing torch.
# Prevents torch.cuda.is_available() from caching False when Ray sets
# CUDA_VISIBLE_DEVICES="" for actors with num_gpus=0.
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger
from vllm import LLM
from transformers import AutoTokenizer
import torch


@ray.remote
class VLLMWorkerActor:
    """
    Ray actor that runs vLLM on one or more GPUs.

    Supports Tensor Parallelism (TP):
    - TP=1: Single GPU, model weights fully on one GPU
    - TP>1: Multiple GPUs, model weights sharded across GPUs
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],           # Physical GPU IDs (e.g., [0, 1] for TP=2)
        logical_gpu_ids: List[int],   # Logical GPU IDs within CUDA_VISIBLE_DEVICES
        tensor_parallel_size: int = 1,  # TP size (must equal len(gpu_ids))
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,
        max_num_seqs: int = 1,
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "logprob", "ppl_ratio", "cot_effect_ratio", or "cot_effect_delta_sigmoid"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
    ):
        import os
        import subprocess

        # Validate TP configuration
        assert len(gpu_ids) == tensor_parallel_size, \
            f"gpu_ids length {len(gpu_ids)} must equal tensor_parallel_size {tensor_parallel_size}"
        assert len(logical_gpu_ids) == tensor_parallel_size, \
            f"logical_gpu_ids length {len(logical_gpu_ids)} must equal tensor_parallel_size {tensor_parallel_size}"

        self.gpu_ids = gpu_ids
        self.tensor_parallel_size = tensor_parallel_size
        worker_name = f"VLLMWorker-{gpu_ids}"

        # Ray actor may have CUDA_VISIBLE_DEVICES="" when created with num_gpus=0.
        # Set CUDA_VISIBLE_DEVICES to the physical GPU IDs for this actor.
        cuda_devices_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Force vLLM V0 engine to avoid V1 serialization bugs
        os.environ["VLLM_USE_V1"] = "0"


        # Log GPU memory status BEFORE loading model
        def get_nvidia_smi_memory_for_gpu(gpu_idx):
            """Get GPU memory from nvidia-smi for a specific GPU"""
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3 and int(parts[0].strip()) == gpu_idx:
                        used_mb = int(parts[1].strip())
                        total_mb = int(parts[2].strip())
                        used_gb = used_mb / 1024
                        total_gb = total_mb / 1024
                        return f"GPU {gpu_idx}: {used_gb:.1f}GB / {total_gb:.1f}GB ({used_mb}MB used)"
                return f"GPU {gpu_idx}: not found"
            except Exception as e:
                return f"nvidia-smi error: {e}"

        def get_all_gpus_memory():
            """Get memory for all GPUs in this actor"""
            return ", ".join(get_nvidia_smi_memory_for_gpu(gpu) for gpu in gpu_ids)

        # GPU memory check (for debugging if needed)
        pass

        self._worker_name = worker_name
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu
        self._get_all_gpus_memory = get_all_gpus_memory

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        import time
        llm_start = time.time()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,  # Use actual TP size
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            max_num_seqs=max_num_seqs,
            max_logprobs=1,
        )
        llm_elapsed = time.time() - llm_start
        self.micro_batch_size = micro_batch_size
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_temperature

    def compute_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a batch of prompts.

        Uses micro-batching to avoid OOM with long sequences.

        Returns:
            List of dicts, each containing:
                - mean_log_prob: float
                - min_log_prob: float
                - token_log_probs: List[float]
        """
        from vllm import SamplingParams
        import time

        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0,
        )

        # Micro-batch processing to avoid OOM
        all_outputs = []
        total_time = 0
        micro_batch_size = self.micro_batch_size

        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]

            start_time = time.time()
            batch_outputs = self.llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - start_time
            total_time += elapsed

            all_outputs.extend(batch_outputs)

        speed = len(prompts) / total_time if total_time > 0 else 0

        # Extract log probs with mean and min
        results = []
        for output, ans_start in zip(all_outputs, answer_starts):
            logprobs = self._extract_answer_logprobs(output.prompt_logprobs, ans_start)
            if logprobs:
                results.append({
                    "mean_log_prob": sum(logprobs) / len(logprobs),
                    "min_log_prob": min(logprobs),
                    "token_log_probs": logprobs,
                })
            else:
                results.append({
                    "mean_log_prob": -100.0,
                    "min_log_prob": -100.0,
                    "token_log_probs": [],
                })

        return results

    def _extract_answer_logprobs(self, prompt_logprobs, answer_start_idx: int) -> List[float]:
        """Extract log probabilities for the answer portion."""
        if not prompt_logprobs:
            return []

        log_probs = []
        for i in range(answer_start_idx, len(prompt_logprobs)):
            token_logprob = prompt_logprobs[i]
            if token_logprob is not None:
                # Get the log prob of the actual token
                if hasattr(token_logprob, 'values'):
                    # vLLM 0.8.x format - values() returns Logprob objects
                    value = list(token_logprob.values())[0]
                    # Extract float from Logprob object if needed
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))
                elif isinstance(token_logprob, dict):
                    value = list(token_logprob.values())[0]
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))
                elif isinstance(token_logprob, (list, tuple)) and len(token_logprob) > 0:
                    value = token_logprob[0]
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))

        return log_probs if log_probs else [-100.0]


class VLLMStudentRewardClient:
    """
    Multi-GPU vLLM-based Student Reward Client using Ray actors.

    Supports two modes:
    1. TP=1 (Data Parallelism): Each GPU runs a separate vLLM instance
       - num_actors = num_gpus
       - Each actor processes a portion of the batch independently

    2. TP>1 (Tensor Parallelism): Multiple GPUs share one vLLM instance
       - num_actors = num_gpus / tensor_parallel_size
       - Model weights are sharded across GPUs, reducing per-GPU memory
       - Good for large models that don't fit on single GPU
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,  # TP size per vLLM instance
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,
        max_num_seqs: int = 1,
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "logprob", "ppl_ratio", "cot_effect_ratio", or "cot_effect_delta_sigmoid"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        use_reward_delta: bool = True,  # If True (default), R_cog = rewrite - original (delta).
                                      # If False, R_cog = rewrite only (absolute value).
        reward_only_use_logprob: bool = None,  # Deprecated: use use_reward_delta instead
    ):
        """
        Initialize Multi-GPU vLLM Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            tensor_parallel_size: Tensor parallelism size per vLLM instance.
                - TP=1: Each GPU runs independent vLLM (Data Parallelism)
                - TP=2: Every 2 GPUs share one vLLM (model sharded)
                - TP=4: Every 4 GPUs share one vLLM
            gpu_memory_utilization: Fraction of GPU memory to use per GPU
            dtype: Data type for the model
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
            max_num_seqs: Maximum concurrent sequences in vLLM scheduler
            reward_mode: Reward calculation mode - "ppl", "logprob", "ppl_ratio", or "cot_effect_ratio"
                - "ppl": R_cog = avg_prob(rewrite) (absolute quality)
                - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
                - "ppl_ratio": R_cog = min(ppl_original / ppl_rewrite, ceiling)
                - "cot_effect_ratio": R_cog = min(cot_effect(rewrite) / cot_effect(original), ceiling)
                  where cot_effect(cot) = 1/ppl(Q,CoT,A) - 1/ppl(Q,A) = avg_prob(Q,CoT,A) - avg_prob(Q,A)
            use_reward_delta: If True (default), R_cog = value(rewrite) - value(original) (delta).
                If False, R_cog = value(rewrite) only (absolute value).
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead.
            ppl_ratio_ceiling: Ceiling for ppl_ratio mode (default: 0.95).
                When ppl_ratio mode: R_cog = min(ppl_original / ppl_rewrite, ppl_ratio_ceiling).
            cot_effect_ratio_ceiling: Ceiling for cot_effect_ratio mode (default: 0.95).
                When cot_effect_ratio mode: R_cog = min(cot_effect(rewrite) / cot_effect(original), cot_effect_ratio_ceiling).
        """

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.max_num_seqs = max_num_seqs
        self.reward_mode = reward_mode
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.ppl_delta_temperature = ppl_delta_temperature

        # Handle reward_only_use_logprob deprecation and map to use_reward_delta
        # reward_only_use_logprob has INVERTED logic compared to use_reward_delta
        # - reward_only_use_logprob=True means absolute value (use_reward_delta=False)
        # - reward_only_use_logprob=False means delta (use_reward_delta=True)
        if reward_only_use_logprob is not None:
            import warnings
            warnings.warn(
                "[DEPRECATION] reward_only_use_logprob is deprecated. Use use_reward_delta instead. "
                "Note: reward_only_use_logprob has inverted logic (True=absolute, False=delta). "
                "Setting use_reward_delta accordingly.",
                DeprecationWarning,
                stacklevel=2
            )
            self.use_reward_delta = not reward_only_use_logprob  # Invert the logic!
        else:
            self.use_reward_delta = use_reward_delta
        self.reward_only_use_logprob = reward_only_use_logprob  # Keep for logging

        # Auto-detect devices if not specified
        if devices is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
        self.devices = devices
        self.num_gpus = len(devices)
        self.gpu_ids = [int(d.split(":")[-1]) for d in devices]

        # Calculate number of actors based on TP size
        # num_actors = num_gpus / tensor_parallel_size
        assert self.num_gpus >= tensor_parallel_size, \
            f"num_gpus ({self.num_gpus}) must be >= tensor_parallel_size ({tensor_parallel_size})"
        assert self.num_gpus % tensor_parallel_size == 0, \
            f"num_gpus ({self.num_gpus}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
        self.num_actors = self.num_gpus // tensor_parallel_size

        self.workers: List[ray.actor.ActorHandle] = []
        self.tokenizer = None
        self._initialized = False

    def cleanup(self):
        """Cleanup Ray vLLM actors to release GPU memory."""
        import subprocess
        import time

        if not self._initialized:
            return

        def get_nvidia_smi():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                output = []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = parts[0].strip()
                        used_mb = int(parts[1].strip())
                        used_gb = used_mb / 1024
                        output.append(f"GPU {gpu_idx}: {used_gb:.1f}GB")
                return ", ".join(output)
            except:
                return "nvidia-smi unavailable"

        # Kill all Ray actors to release GPU memory
        for i, worker in enumerate(self.workers):
            try:
                ray.kill(worker)
            except Exception as e:
                logger.warning(f"[VLLMStudentReward] Error killing worker {i}: {e}")

        self.workers = []
        self._initialized = False

        torch.cuda.empty_cache()

    def initialize(self):
        """Lazy initialization of Ray vLLM actors."""
        if self._initialized:
            return

        # Initialize tokenizer (shared, for prompt building)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Compute logical GPU IDs from CUDA_VISIBLE_DEVICES
        
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            # Parse physical GPU IDs from CUDA_VISIBLE_DEVICES
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]

            # Map physical GPU IDs to logical indices
            physical_to_logical = {gpu: idx for idx, gpu in enumerate(available_gpus)}
        else:
            # No CUDA_VISIBLE_DEVICES set, assume physical == logical
            physical_to_logical = {gpu: gpu for gpu in self.gpu_ids}
        # Group GPUs by tensor_parallel_size
        # e.g., gpu_ids=[0,1,2,3], TP=2 → [[0,1], [2,3]]
        gpu_groups = []
        for i in range(0, self.num_gpus, self.tensor_parallel_size):
            group = self.gpu_ids[i : i + self.tensor_parallel_size]
            gpu_groups.append(group)

        # Create Ray actors, one per GPU group
        for group_idx, gpu_group in enumerate(gpu_groups):
            # gpu_group contains logical device indices. Convert to physical GPU IDs.
            if cuda_visible_devices:
                physical_gpu_ids = [available_gpus[g] for g in gpu_group]
            else:
                physical_gpu_ids = list(gpu_group)
            # Actor-local logical IDs are always 0, 1, ..., tp_size-1
            logical_gpu_ids = list(range(len(gpu_group)))
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = VLLMWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_ids=physical_gpu_ids,
                logical_gpu_ids=logical_gpu_ids,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
                max_num_seqs=self.max_num_seqs,
                reward_mode=self.reward_mode,
                ppl_ratio_ceiling=self.ppl_ratio_ceiling,
                cot_effect_ratio_ceiling=self.cot_effect_ratio_ceiling,
                ppl_delta_temperature=self.ppl_delta_temperature,
            )
            self.workers.append(worker)

        init_futures = [w.compute_logprobs.remote(["test"], [0]) for w in self.workers]
        for i, future in enumerate(init_futures):
            try:
                result = ray.get(future, timeout=60)  # 60 second timeout
            except Exception as e:
                logger.error(f"[VLLMStudentReward] Worker {i} init FAILED: {e}")
                raise

        self._initialized = True

    def _build_prompt(self, question: str, cot: str, answer: str) -> tuple[str, int]:
        """
        Build the full prompt and compute the answer start position.

        Returns:
            tuple: (full_prompt, answer_start_token_idx)
        """
        # Build full prompt using chat template
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer, 'reasoning_content': cot}
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Build prefix to find where answer starts
        prefix_messages = [{'role': 'user', 'content': question}]
        prefix_base = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_with_think = prefix_base + f"<think>\n{cot}\n</think>\n\n"

        # Tokenize to get positions
        prefix_token_ids = self.tokenizer.encode(prefix_with_think, add_special_tokens=False)
        answer_start_idx = len(prefix_token_ids)

        return full_text, answer_start_idx

    def _build_prompt_without_cot(self, question: str, answer: str) -> tuple[str, int]:
        """
        Build prompt without CoT for computing R_useful baseline.

        This builds a prompt of the form:
        Question + Answer (without any CoT/reasoning)

        Returns:
            tuple: (full_prompt, answer_start_token_idx)
        """
        # Build prompt WITHOUT reasoning_content
        # This represents the baseline: how well does the model predict Answer given only Question
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}  # No reasoning_content
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Find where answer starts (after the question and assistant header)
        prefix_messages = [{'role': 'user', 'content': question}]
        prefix_base = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize to get answer start position
        prefix_token_ids = self.tokenizer.encode(prefix_base, add_special_tokens=False)
        answer_start_idx = len(prefix_token_ids)

        return full_text, answer_start_idx

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
        alpha_cog: float = 0.01,
        use_reward_delta: Optional[bool] = None,
        reward_only_use_logprob: Optional[bool] = None,
        **kwargs,  # Accept extra args for compatibility
    ) -> Dict[str, Any]:
        """
        Compute cognitive rewards for a batch using multi-GPU vLLM Ray actors.

        Each Ray actor runs vLLM on a separate GPU. We split the batch across
        all actors for true data parallelism.

        Computes two reward components:
        1. R_cog = value(rewrite) - value(original) (if use_reward_delta=True)
                  OR R_cog = value(rewrite) only (if use_reward_delta=False)
        2. R_useful = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, A)

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers
            original_cots: Optional list of original CoTs
            alpha_cog: Weight for min log prob term (only used in logprob mode)
            use_reward_delta: If True, compute delta (rewrite - original).
                If False, use absolute value. If None, use self.use_reward_delta.
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead

        Returns:
            Dict containing rewards and log probs including R_cog and R_useful
        """
        # Use instance value if not provided in call
        # Handle both new and old parameter names
        if use_reward_delta is None and reward_only_use_logprob is None:
            use_reward_delta = self.use_reward_delta
        elif reward_only_use_logprob is not None:
            # Old parameter provided: invert logic (True=absolute, False=delta)
            import warnings
            warnings.warn(
                "[DEPRECATION] reward_only_use_logprob is deprecated. Use use_reward_delta instead. "
                "Note: reward_only_use_logprob has inverted logic (True=absolute, False=delta).",
                DeprecationWarning,
                stacklevel=2
            )
            use_reward_delta = not reward_only_use_logprob
        else:
            use_reward_delta = use_reward_delta if use_reward_delta is not None else self.use_reward_delta
        import time

        self.initialize()

        batch_size = len(questions)
        start_time = time.time()

        # Phase 1: Build all prompts (CPU)

        rewrite_prompts = []
        rewrite_ans_starts = []
        original_prompts = []
        original_ans_starts = []
        original_indices = []  # Track which indices have original_cot
        # New: Build prompts WITHOUT CoT for R_useful baseline
        no_cot_prompts = []
        no_cot_ans_starts = []

        for i in range(batch_size):
            # Rewrite prompt
            rewrite_prompt, rewrite_ans_start = self._build_prompt(questions[i], rewrite_cots[i], answers[i])
            rewrite_prompts.append(rewrite_prompt)
            rewrite_ans_starts.append(rewrite_ans_start)

            # Original prompt (if provided)
            if original_cots and i < len(original_cots) and original_cots[i]:
                orig_prompt, orig_ans_start = self._build_prompt(questions[i], original_cots[i], answers[i])
                original_prompts.append(orig_prompt)
                original_ans_starts.append(orig_ans_start)
                original_indices.append(i)

            # No-CoT prompt for R_useful baseline (Question + Answer only, no CoT)
            no_cot_prompt, no_cot_ans_start = self._build_prompt_without_cot(questions[i], answers[i])
            no_cot_prompts.append(no_cot_prompt)
            no_cot_ans_starts.append(no_cot_ans_start)

        prep_time = time.time() - start_time

        # Phase 2: Distribute work to Ray actors
        inference_start = time.time()

        # Split rewrite prompts across actors (not GPUs, since TP>1 means multiple GPUs per actor)
        rewrite_chunks = self._split_into_chunks(rewrite_prompts, self.num_actors)
        rewrite_ans_start_chunks = self._split_into_chunks(rewrite_ans_starts, self.num_actors)

        # Submit rewrite tasks to all workers in parallel
        rewrite_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(rewrite_chunks, rewrite_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                rewrite_futures.append((worker_idx, len(prompts_chunk), future))

        # Split original prompts across actors (if any)
        original_futures = []
        if original_prompts:
            original_chunks = self._split_into_chunks(original_prompts, self.num_actors)
            original_ans_start_chunks = self._split_into_chunks(original_ans_starts, self.num_actors)

            for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(original_chunks, original_ans_start_chunks)):
                if prompts_chunk:
                    future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                    original_futures.append((worker_idx, len(prompts_chunk), future))

        # Split no_cot prompts across actors for R_useful baseline
        no_cot_chunks = self._split_into_chunks(no_cot_prompts, self.num_actors)
        no_cot_ans_start_chunks = self._split_into_chunks(no_cot_ans_starts, self.num_actors)

        no_cot_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(no_cot_chunks, no_cot_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                no_cot_futures.append((worker_idx, len(prompts_chunk), future))

        # Gather rewrite results
        rewrite_results = [None] * batch_size
        for worker_idx, chunk_size, future in rewrite_futures:
            get_start = time.time()
            chunk_results = ray.get(future)
            get_elapsed = time.time() - get_start
            # Place results back in original order
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_results[start_idx + i] = result

        # Gather original results
        original_results_map = {}
        if original_futures:
            for worker_idx, chunk_size, future in original_futures:
                get_start = time.time()
                chunk_results = ray.get(future)
                get_elapsed = time.time() - get_start
                start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
                for i, result in enumerate(chunk_results):
                    original_idx = original_indices[start_idx + i]
                    original_results_map[original_idx] = result

        # Gather no_cot results (R_useful baseline)
        no_cot_results = [None] * batch_size
        for worker_idx, chunk_size, future in no_cot_futures:
            get_start = time.time()
            chunk_results = ray.get(future)
            get_elapsed = time.time() - get_start
            start_idx = sum(len(no_cot_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                no_cot_results[start_idx + i] = result

        # Phase 3: Compute rewards
        rewards = []
        useful_rewards = []  # R_useful: how much does CoT help vs baseline
        rewrite_mean_log_probs = []
        original_mean_log_probs = []
        no_cot_mean_log_probs = []  # Baseline: Answer|Question only (no CoT)
        all_token_log_probs = []
        all_token_ids = []

        # Store PPL values for ppl mode (for logging/debugging)
        rewrite_ppls = []
        original_ppls = []
        # Store relative improvement rate (formula 1) as observation value
        relative_improvement_rates = []  # R_cog_formula1 = avg_prob(rewrite)/avg_prob(original) - 1

        for i in range(batch_size):
            # Get rewrite result
            rewrite_result = rewrite_results[i]
            rewrite_mean = rewrite_result["mean_log_prob"]
            rewrite_min = rewrite_result["min_log_prob"]

            # Get no_cot result (baseline without CoT)
            no_cot_result = no_cot_results[i]
            no_cot_mean = no_cot_result["mean_log_prob"]
            no_cot_min = no_cot_result["min_log_prob"]

            # Get original result (if available)
            if i in original_results_map:
                original_result = original_results_map[i]
                original_mean = original_result["mean_log_prob"]
                original_min = original_result["min_log_prob"]
            else:
                original_mean = -100.0
                original_min = -100.0

            # ===== R_cog calculation formula =====
            # R_cog = value(rewrite) - value(original) (if use_reward_delta=True)
            # OR R_cog = value(rewrite) only (if use_reward_delta=False)
            # Select calculation method based on reward_mode
            if self.reward_mode == "ppl_ratio":
                # PPL Ratio mode: R_cog = min(ppl_original / ppl_rewrite, ceiling)
                # where ppl = 1 / avg_prob = exp(-mean_log_prob)
                import math
                rewrite_ppl = math.exp(-rewrite_mean)  # PPL(rewrite) = exp(-mean_log_prob)

                # Compute formula 1 as observation value (for logging)
                if original_mean > -100:
                    original_ppl = math.exp(-original_mean)  # PPL(original)
                    if original_ppl < 1e-10:
                        ppl_ratio = 0.0
                    else:
                        ppl_ratio = original_ppl / rewrite_ppl
                    # Apply ceiling
                    reward = min(ppl_ratio, self.ppl_ratio_ceiling)
                    relative_improvement_rate = ppl_ratio - 1.0  # for logging
                    original_ppls.append(original_ppl)
                else:
                    # If no original, reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(rewrite_ppl)
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "ppl":
                # PPL mode: use formula 2 (absolute quality)
                # R_cog = avg_prob(rewrite) = 1 / PPL(rewrite)
                # Also compute formula 1 (relative improvement rate) as observation
                # R_cog_formula1 = avg_prob(rewrite) / avg_prob(original) - 1
                #               = PPL(original) / PPL(rewrite) - 1
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob for rewrite

                # Formula 2: absolute quality metric (actually used as GRPO reward)
                reward = rewrite_avg_prob

                # Compute formula 1 as observation value
                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)
                    if original_avg_prob < 1e-10:
                        relative_improvement_rate = 0.0
                    else:
                        relative_improvement_rate = (rewrite_avg_prob / original_avg_prob) - 1.0
                else:
                    relative_improvement_rate = 0.0
                relative_improvement_rates.append(relative_improvement_rate)

                # Store PPL values for logging (PPL = 1/avg_prob)
                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    original_ppls.append(float('nan'))
            elif self.reward_mode == "ppl_delta":
                # PPL Delta mode (negative value clamp only)
                # delta = avg_prob(rewrite) - avg_prob(original)
                # R_cog = max(0, delta)
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(rewrite) = 1/ppl(rewrite)

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(original) = 1/ppl(original)
                    delta = rewrite_avg_prob - original_avg_prob
                    reward = max(0.0, delta)  # negative clamp
                    relative_improvement_rate = delta  # for logging
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # No original, reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "ppl_delta_norm_sigmoid":
                # PPL Delta mode (with group normalization + sigmoid)
                # Raw delta = avg_prob(rewrite) - avg_prob(original)
                # Normalization: delta_norm = (delta - mean(delta)) / std(delta)
                # Sigmoid: reward = sigmoid(delta_norm / temperature)
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(rewrite) = 1/ppl(rewrite)

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(original) = 1/ppl(original)
                    delta = rewrite_avg_prob - original_avg_prob
                    # Save raw delta first, group normalization will be applied later
                    reward = delta  # temporary value, will be replaced later
                    relative_improvement_rate = delta  # for logging
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # No original, reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "cot_effect_ratio":
                # CoT Effect Ratio mode
                # cot_effect(cot) = 1/ppl(Q,CoT,A) - 1/ppl(Q,A) = avg_prob(Q,CoT,A) - avg_prob(Q,A)
                # R_cog = min(cot_effect(rewrite) / cot_effect(original), ceiling)
                import math

                # avg_prob = exp(mean_log_prob)
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(Q, rewrite_CoT, A)
                no_cot_avg_prob = math.exp(no_cot_mean)     # avg_prob(Q, A) - baseline without CoT

                # CoT effect for rewrite: how much does CoT help vs baseline
                cot_effect_rewrite = rewrite_avg_prob - no_cot_avg_prob

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(Q, original_CoT, A)

                    # CoT effect for original: how much did original CoT help vs baseline
                    cot_effect_original = original_avg_prob - no_cot_avg_prob

                    # Handle edge case: if cot_effect_original is too small or non-positive
                    if cot_effect_original < 1e-10:
                        # Original CoT had little to no effect, return ceiling value (max reward)
                        cot_effect_ratio = self.cot_effect_ratio_ceiling
                        relative_improvement_rate = self.cot_effect_ratio_ceiling - 1.0
                    else:
                        # Normal case: compute the ratio
                        # cot_effect_ratio > 1 means rewrite is more helpful than original
                        cot_effect_ratio = cot_effect_rewrite / cot_effect_original
                        # Apply ceiling
                        cot_effect_ratio = min(cot_effect_ratio, self.cot_effect_ratio_ceiling)
                        relative_improvement_rate = cot_effect_ratio - 1.0

                    reward = cot_effect_ratio
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # No original CoT available, reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "cot_effect_delta_sigmoid":
                # CoT Effect Delta Sigmoid mode
                #
                # Math formula:
                #   coe_orig = avg_prob(Q, original, A) - avg_prob(Q, A)
                #   coe_rew  = avg_prob(Q, rewrite, A)  - avg_prob(Q, A)
                #   delta    = avg_prob(rewrite) - avg_prob(original)
                #   delta_clamped = max(0, delta)  # treat as 0 when delta <= 0
                #   Δ_norm = delta_clamped / (|coe_orig| + ε)
                #   R_cog = sigmoid(Δ_norm / τ)
                #
                # where ε = cot_effect_epsilon (default 0.01), τ = cot_effect_temperature (default 1.0)
                # Reward range: [0.5, 1), same magnitude as other penalties
                import math

                # avg_prob = exp(mean_log_prob)
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(Q, rewrite_CoT, A)
                no_cot_avg_prob = math.exp(no_cot_mean)     # avg_prob(Q, A) - baseline

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(Q, original_CoT, A)

                    # CoT effect for original
                    cot_effect_original = original_avg_prob - no_cot_avg_prob

                    # Improvement amount (treat as 0 when delta <= 0)
                    delta = rewrite_avg_prob - original_avg_prob
                    delta_clamped = max(0.0, delta)

                    epsilon = self.cot_effect_epsilon
                    temperature = self.cot_effect_temperature

                    normalized_delta = delta_clamped / (abs(cot_effect_original) + epsilon)

                    # Sigmoid maps to [0.5, 1)
                    reward = 1.0 / (1.0 + math.exp(-normalized_delta / temperature))
                    relative_improvement_rate = delta  # for logging (raw delta)

                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # No original, reward = 0.5
                    reward = 0.5
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            else:  # "logprob" or "kl" mode
                # Logprob mode: R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
                # where answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
                rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min
                if not use_reward_delta:
                    # When use_reward_delta=False, use only rewrite logprob (no delta)
                    reward = rewrite_answer_log_prob
                else:
                    # Default: compute delta from original
                    original_answer_log_prob = original_mean + alpha_cog * original_min
                    reward = rewrite_answer_log_prob - original_answer_log_prob

                # Populate PPL and improvement rate with NaN (not applicable for logprob/kl mode)
                rewrite_ppls.append(float('nan'))
                original_ppls.append(float('nan'))
                relative_improvement_rates.append(float('nan'))

            # ===== R_useful calculation formula =====
            # R_useful = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, A)
            # This represents the usefulness of the rewritten CoT: the improvement in answer prediction
            # compared to the case without CoT
            # Note: use logprob mode here, not PPL mode
            # Because we compare logprob differences, not PPL differences
            rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min
            no_cot_answer_log_prob = no_cot_mean + alpha_cog * no_cot_min
            useful_reward = rewrite_answer_log_prob - no_cot_answer_log_prob

            rewards.append(reward)
            useful_rewards.append(useful_reward)
            rewrite_mean_log_probs.append(rewrite_mean)
            original_mean_log_probs.append(original_mean)
            no_cot_mean_log_probs.append(no_cot_mean)
            all_token_log_probs.append(rewrite_result.get("token_log_probs", []))
            all_token_ids.append([])  # Token IDs not needed for reward

        return {
            "rewards": rewards,  # R_cog (formula 2: avg_prob(rewrite))
            "useful_rewards": useful_rewards,  # R_useful
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "no_cot_mean_log_probs": no_cot_mean_log_probs,  # Baseline (no CoT)
            "rewrite_ppls": rewrite_ppls,  # PPL values for rewrite (ppl mode only)
            "original_ppls": original_ppls,  # PPL values for original (ppl mode only)
            "relative_improvement_rates": relative_improvement_rates,  # Formula 1: avg_prob(rewrite)/avg_prob(original) - 1
            "rewrite_token_log_probs": all_token_log_probs,
            "rewrite_token_ids": all_token_ids,
            "log_probs": rewrite_mean_log_probs,
            "token_log_probs": all_token_log_probs,
            "token_ids": all_token_ids,
        }

        # ===== Post-processing for ppl_delta_norm_sigmoid mode: Group normalization + Sigmoid =====
        if self.reward_mode == "ppl_delta_norm_sigmoid":
            import math
            # Extract raw deltas from rewards (which currently hold delta values)
            raw_deltas = rewards[:]  # Copy

            # Find valid deltas (from samples with original_cot)
            # First apply max(0, delta) to clamp negative values
            valid_deltas_raw = [raw_deltas[i] for i in range(batch_size) if i in original_results_map]
            valid_deltas = [max(0.0, d) for d in valid_deltas_raw]
            valid_indices = [i for i in range(batch_size) if i in original_results_map]

            if valid_deltas:
                # Compute mean and std of clamped deltas
                mean_delta = sum(valid_deltas) / len(valid_deltas)
                variance = sum((d - mean_delta) ** 2 for d in valid_deltas) / len(valid_deltas)
                std_delta = math.sqrt(variance)

                # Avoid division by zero (when all deltas are 0 or very similar)
                if std_delta < 1e-8:
                    std_delta = 1.0

                temperature = self.ppl_delta_temperature

                # Apply group normalization + sigmoid to valid samples
                for i, idx in enumerate(valid_indices):
                    clamped_delta = valid_deltas[i]
                    normalized_delta = (clamped_delta - mean_delta) / std_delta
                    # Sigmoid: reward = 1 / (1 + exp(-x))
                    rewards[idx] = 1.0 / (1.0 + math.exp(-normalized_delta / temperature))

                # For samples without original, set reward to 0.5 (sigmoid of 0)
                for i in range(batch_size):
                    if i not in valid_indices:
                        rewards[i] = 0.5

    def _split_into_chunks(self, items: List, num_chunks: int) -> List[List]:
        """Split items into roughly equal chunks."""
        if num_chunks <= 0:
            return [items]
        chunk_size = max(1, len(items) // num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            if i == num_chunks - 1:
                end = len(items)
            else:
                end = min(start + chunk_size, len(items))
            chunks.append(items[start:end])
        return chunks
