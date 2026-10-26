"""
Student Reward (Cognitive Reward) Module - SGLang Version

Computes R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
            = PPL(original) - PPL(rewrite)
using SGLang for fast log prob computation.

where PPL = exp(-mean(token_log_probs))

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
(lower perplexity) compared to the original CoT.

Uses Ray actors for multi-GPU parallelism. Supports two modes:
1. TP=1: Each GPU runs a separate SGLang instance (Data Parallelism)
2. TP>1: Multiple GPUs share one SGLang instance (Tensor Parallelism)

SGLang-specific optimizations:
- enable_memory_saver: enable memory saver, supports release/resume_memory_occupation
- logprob_start_len: only compute logprobs for the last N tokens, saves memory
"""

import asyncio
from typing import Any, Dict, List, Optional
import os

# IMPORTANT: Set PYTORCH_NVML_BASED_CUDA_CHECK=1 BEFORE importing torch.
# This prevents torch.cuda.is_available() from caching False when
# CUDA_VISIBLE_DEVICES is temporarily empty (e.g., when Ray spawns actors
# with num_gpus=0). With this flag, is_available() uses NVML-based detection
# which correctly re-evaluates device availability after CUDA_VISIBLE_DEVICES changes.
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger
from transformers import AutoTokenizer
import torch


@ray.remote
class SGLangWorkerActor:
    """
    Ray actor that runs SGLang on one or more GPUs.

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
        mem_fraction_static: float = 0.85,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,
        logprob_last_n_tokens: Optional[int] = None,  # only compute logprobs for the last N tokens
        base_port: int = 30000,  # Base port for this worker (different per worker to avoid conflicts)
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
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
        self.logprob_last_n_tokens = logprob_last_n_tokens
        self.ppl_on_full_response = ppl_on_full_response
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_temperature
        self._base_port = base_port  # Store base port for SGLang Engine
        worker_name = f"SGLangWorker-{gpu_ids}"

        # Ray actor may have CUDA_VISIBLE_DEVICES="" when created with num_gpus=0.
        # Set CUDA_VISIBLE_DEVICES to the physical GPU IDs for this actor.
        # gpu_ids contains physical GPU IDs (e.g., [2, 3]), so the actor can
        # directly access the correct physical GPUs regardless of Ray's env setup.
        cuda_devices_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Set unique MASTER_PORT to avoid torch.distributed TCP rendezvous port conflicts
        # when multiple workers initialize in parallel on the same machine.
        # Dynamically find an available port to avoid EADDRINUSE errors.
        import socket
        master_port = base_port + 500
        max_master_port = min(base_port + 999, 65535)
        found_port = False
        for offset in range(100):
            candidate = base_port + 500 + offset
            if candidate > max_master_port:
                break
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", candidate))
                    master_port = candidate
                    found_port = True
                    break
            except OSError:
                continue
        if not found_port:
            logger.warning(f"[{worker_name}] Could not find available MASTER_PORT in [{base_port + 500}, {max_master_port}], using {master_port}")
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        logger.info(f"[{worker_name}] Using MASTER_PORT={master_port}")

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

        # Create event loop for SGLang (required when running in Ray actor)
        # Reference: verl-0.5.0/verl/workers/reward_manager/prime.py
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info(f"[SGLangWorkerActor-{gpu_ids}] Created new event loop")

        # Save initialization parameters for lazy engine creation
        self._model_path = model_path
        self._tensor_parallel_size = tensor_parallel_size
        self._mem_fraction_static = mem_fraction_static
        self._dtype = dtype
        self._micro_batch_size = micro_batch_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Engine will be created lazily in _init_engine() to avoid port conflicts
        self.engine = None
        self._engine_initialized = False

        # Track memory state for release/resume
        self._memory_released = False

        logger.info(f"[SGLangWorkerActor-{gpu_ids}] Actor created, engine will be initialized on first use")

    def _init_engine(self):
        """
        Initialize SGLang Engine on demand.

        This is called separately from __init__ to avoid port conflicts when
        multiple workers are created simultaneously. The engine should be
        initialized serially (one worker at a time) to prevent race conditions.
        """
        if self._engine_initialized:
            return

        import time
        import sglang as sgl

        llm_start = time.time()

        # Initialize SGLang Engine
        # Using enable_memory_saver=True to support memory release/resume mechanism
        # This avoids port conflicts from cleanup/recreate cycles
        #
        # Each worker uses a different base_port (assigned from non-overlapping 2000-port ranges)
        # to avoid port conflicts during parallel initialization.
        self.engine = sgl.Engine(
            model_path=self._model_path,
            tp_size=self._tensor_parallel_size,
            mem_fraction_static=self._mem_fraction_static,
            dtype=self._dtype,
            trust_remote_code=True,
            enable_memory_saver=True,
            port=self._base_port,  # Use unique base port per worker
        )

        # CRITICAL: Re-create event loop after Engine creation
        # because sglang forces uvloop policy which clears the existing event loop
        # This must be done before any generate() calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Event loop re-created after Engine init")

        llm_elapsed = time.time() - llm_start
        self.micro_batch_size = self._micro_batch_size
        self._engine_initialized = True

        logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Engine created in {llm_elapsed:.2f}s")

    def initialize_and_verify(self) -> bool:
        """
        Initialize the SGLang Engine and verify it can return valid logprobs.
        
        This method is called during worker initialization to ensure:
        1. Engine is fully ready (CUDA graph capturing completed)
        2. Can successfully return logprobs for valid prompts
        
        Returns:
            True if initialization and verification succeeded
            
        Raises:
            RuntimeError: If Engine cannot return valid logprobs
        """
        import time
        
        # Initialize engine (this triggers CUDA graph capturing)
        self._init_engine()
        
        # Wait a bit for any background initialization
        time.sleep(0.5)
        
        # Ensure event loop exists
        self._ensure_event_loop()
        
        # Build a valid chat-formatted prompt for verification
        # Use a simple math question that the model can handle
        test_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "What is 2+2?\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "4<|im_end|>"
        )
        
        # Try to get logprobs with retries
        max_retries = 5
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Verification attempt {attempt + 1}/{max_retries}...")
                
                outputs = self.engine.generate(
                    prompt=test_prompt,
                    sampling_params={"max_new_tokens": 1, "temperature": 0},
                    return_logprob=True,
                    logprob_start_len=0,  # Get logprobs for all input tokens
                    top_logprobs_num=1,
                )
                
                # Verify output format
                if not isinstance(outputs, dict):
                    raise RuntimeError(f"Expected dict output, got {type(outputs)}")
                
                meta_info = outputs.get("meta_info", {})
                input_token_logprobs = meta_info.get("input_token_logprobs", [])
                
                if not input_token_logprobs:
                    raise RuntimeError(f"No input_token_logprobs in meta_info. Keys: {list(meta_info.keys())}")
                
                if len(input_token_logprobs) == 0:
                    raise RuntimeError("input_token_logprobs is empty list")
                
                # Successfully got logprobs
                logger.info(
                    f"[SGLangWorkerActor-{self.gpu_ids}] Verification SUCCESS: "
                    f"got {len(input_token_logprobs)} input_token_logprobs"
                )
                return True
                
            except Exception as e:
                logger.warning(
                    f"[SGLangWorkerActor-{self.gpu_ids}] Verification attempt {attempt + 1} failed: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"[SGLangWorkerActor-{self.gpu_ids}] Failed to verify Engine after {max_retries} attempts: {e}"
                    )
        
        return False  # Should not reach here

    def compute_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a batch of prompts.

        Uses micro-batching to avoid OOM with long sequences.
        
        SGLang-specific optimizations:
        - Use logprob_start_len to only compute logprobs for the answer portion
        - This greatly reduces returned data size and saves memory

        Returns:
            List of dicts, each containing:
                - mean_log_prob: float
                - min_log_prob: float
                - token_log_probs: List[float]
        """
        import time

        # Initialize engine on first use (lazy initialization to avoid port conflicts)
        self._init_engine()

        # Ensure event loop exists (required when running in Ray actor)
        self._ensure_event_loop()

        # Micro-batch processing to avoid OOM
        all_outputs = []
        total_time = 0
        micro_batch_size = self.micro_batch_size

        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]

            start_time = time.time()
            
            # SGLang generation parameters
            # Note: we use max_tokens=1 because we only need logprobs, not new generation
            # But SGLang needs to generate at least 1 token to get prompt_logprobs
            batch_outputs = []
            for prompt, ans_start in zip(batch_prompts, batch_ans_starts):
                # IMPORTANT: Set logprob_start_len based on ppl_on_full_response
                # - If ppl_on_full_response=False (default): logprob_start_len=ans_start
                #   SGLang's input_token_logprobs only includes tokens from logprob_start_len onwards
                #   We only get answer tokens, not question and CoT tokens
                # - If ppl_on_full_response=True: logprob_start_len=0
                #   We get logprobs for the entire response (Q+CoT+A)
                logprob_start_len = 0 if self.ppl_on_full_response else ans_start

                # Call SGLang engine
                # Note: SGLang's generate returns a dict (single prompt) or list
                outputs = self.engine.generate(
                    prompt=prompt,
                    sampling_params={
                        "max_new_tokens": 1,  # only need 1 token to trigger prompt_logprobs
                        "temperature": 0,
                    },
                    return_logprob=True,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=1,
                )
                batch_outputs.append(outputs)
            
            elapsed = time.time() - start_time
            total_time += elapsed

            all_outputs.extend(batch_outputs)

        speed = len(prompts) / total_time if total_time > 0 else 0
        logger.debug(f"[SGLangWorkerActor] Logprob computation speed: {speed:.2f} samples/s")

        # Extract log probs with mean and min
        results = []
        for outputs, ans_start in zip(all_outputs, answer_starts):
            logprobs = self._extract_answer_logprobs(outputs, ans_start)
            # _extract_answer_logprobs will raise RuntimeError if extraction fails
            results.append({
                "mean_log_prob": sum(logprobs) / len(logprobs),
                "min_log_prob": min(logprobs),
                "token_log_probs": logprobs,
            })

        return results

    def _extract_answer_logprobs(self, outputs, answer_start_idx: int) -> List[float]:
        """Extract log probabilities for the answer portion from SGLang output.
        
        SGLang output format:
        - outputs is a dict (or list containing a dict)
        - meta_info contains 'input_token_logprobs' which is a list of tuples:
          [(logprob, token_id, token_text), ...]
        - First token's logprob may be None
        
        Raises:
            RuntimeError: If logprobs cannot be extracted (for strict validation during KL computation)
        """
        try:
            # Handle both dict and list outputs
            if isinstance(outputs, dict):
                output = outputs
            elif isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[0]
            else:
                error_msg = f"[SGLangWorkerActor] Empty or invalid outputs: {type(outputs)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if not isinstance(output, dict):
                error_msg = f"[SGLangWorkerActor] output is not a dict: {type(output)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            meta_info = output.get("meta_info", {})
            
            # Get input_token_logprobs (this is the correct field name in SGLang)
            input_token_logprobs = meta_info.get("input_token_logprobs", [])
            
            if not input_token_logprobs:
                error_msg = f"[SGLangWorkerActor] No input_token_logprobs found. meta_info keys: {list(meta_info.keys())}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Extract logprobs from tuples (logprob, token_id, token_text)
            # Note: Since we set logprob_start_len=ans_start in generate(),
            # input_token_logprobs already contains ONLY answer portion logprobs
            log_probs = []
            for token_tuple in input_token_logprobs:
                if isinstance(token_tuple, tuple) and len(token_tuple) >= 2:
                    logprob = token_tuple[0]
                    if logprob is not None:
                        log_probs.append(float(logprob))
                elif isinstance(token_tuple, (int, float)):
                    # Direct value (shouldn't happen with SGLang, but handle anyway)
                    log_probs.append(float(token_tuple))

            if log_probs:
                logger.info(f"[SGLangWorkerActor] Extracted {len(log_probs)} log_probs, mean={sum(log_probs)/len(log_probs):.4f}")
                return log_probs
            else:
                error_msg = f"[SGLangWorkerActor] No valid logprobs extracted from SGLang output"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"[SGLangWorkerActor] Failed to extract logprobs: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(f"[SGLangWorkerActor] Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

    def _ensure_event_loop(self):
        """Ensure an event loop exists for the current thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def release_memory(self):
        """
        Completely destroy the SGLang Engine and release all GPU memory.

        Do not use release_memory_occupation() (has a CUDA error bug).
        Instead, directly shutdown the engine to release all resources.
        Recreate the engine via _init_engine() on next use.
        """
        # Fully shutdown the engine to release all memory, instead of trying to release KV Cache
        # Because release_memory_occupation() has a bug: RuntimeError: CUDA error: invalid argument
        # Recreate the engine via _init_engine() on next use
        try:
            if self.engine is not None:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
                del self.engine
                self.engine = None
                self._engine_initialized = False
                self._memory_released = False
                logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Engine destroyed, all GPU memory released")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Error during engine shutdown: {e}")
            self.engine = None
            self._engine_initialized = False
            self._memory_released = False


    def resume_memory(self):
        """
        Resume SGLang GPU memory occupation (restore KV Cache).
        Use resume_memory_occupation() to restore KV Cache.

        Call this before the next reward computation to restore full SGLang functionality.

        Note: only call resume if memory has been released,
        otherwise SGLang will error "Cannot resume allocation that is not paused"
        """
        if not self._memory_released:
            # First call or memory not released, no need to resume
            logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Memory not released, skipping resume")
            return

        try:
            if hasattr(self.engine, "resume_memory_occupation"):
                self.engine.resume_memory_occupation()
                self._memory_released = False
                logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Resumed KV Cache memory")
            else:
                logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Engine does not support resume_memory_occupation")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Failed to resume memory: {e}")

    def shutdown(self):
        """Shutdown the SGLang engine and release all resources."""
        try:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            del self.engine
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Engine shutdown complete")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Error during shutdown: {e}")


class SGLangStudentRewardClient:
    """
    Multi-GPU SGLang-based Student Reward Client using Ray actors.

    Supports two modes:
    1. TP=1 (Data Parallelism): Each GPU runs a separate SGLang instance
       - num_actors = num_gpus
       - Each actor processes a portion of the batch independently

    2. TP>1 (Tensor Parallelism): Multiple GPUs share one SGLang instance
       - num_actors = num_gpus / tensor_parallel_size
       - Model weights are sharded across GPUs, reducing per-GPU memory
       - Good for large models that don't fit on single GPU
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,  # TP size per SGLang instance
        mem_fraction_static: float = 0.85,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,
        logprob_last_n_tokens: Optional[int] = None,  # only compute logprobs for the last N tokens
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        use_reward_delta: bool = True,  # If True (default), R_cog = rewrite - original (delta).
                                      # If False, R_cog = rewrite only (absolute value).
        reward_only_use_logprob: bool = None,  # Deprecated: use use_reward_delta instead
        ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
    ):
        """
        Initialize Multi-GPU SGLang Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            tensor_parallel_size: Tensor parallelism size per SGLang instance.
                - TP=1: Each GPU runs independent SGLang (Data Parallelism)
                - TP=2: Every 2 GPUs share one SGLang (model sharded)
                - TP=4: Every 4 GPUs share one SGLang
            mem_fraction_static: Fraction of GPU memory for SGLang static allocation
            dtype: Data type for the model
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
            logprob_last_n_tokens: only compute logprobs for the last N tokens of the answer, saves memory
            reward_mode: Reward calculation mode - "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
                - "ppl": R_cog = avg_prob(rewrite) = 1 / PPL(rewrite) (absolute quality)
                - "ppl_delta": R_cog = max(0, avg_prob(rewrite) - avg_prob(original)) (delta with negative clamp only)
                - "ppl_delta_norm_sigmoid": R_cog = sigmoid(normalized_delta / temperature) with group normalization
                - "ppl_ratio": R_cog = min(ppl_original / ppl_rewrite, ppl_ratio_ceiling)
                - "cot_effect_ratio": R_cog = min(cot_effect(rewrite) / cot_effect(original), cot_effect_ratio_ceiling)
                - "cot_effect_delta_sigmoid": R_cog = sigmoid((avg_prob(rewrite) - avg_prob(original)) / (|cot_effect(original)| + epsilon) / temperature)
                - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
            ppl_ratio_ceiling: Ceiling for ppl_ratio mode (default: 0.95).
                When ppl_ratio mode: R_cog = min(ppl_original / ppl_rewrite, ppl_ratio_ceiling).
            cot_effect_ratio_ceiling: Ceiling for cot_effect_ratio mode (default: 0.95).
                When cot_effect_ratio mode: R_cog = min(cot_effect(rewrite) / cot_effect(original), cot_effect_ratio_ceiling).
                where cot_effect(cot) = 1/ppl(Q,CoT,A) - 1/ppl(Q,A) = avg_prob(Q,CoT,A) - avg_prob(Q,A)
            use_reward_delta: If True (default), R_cog = value(rewrite) - value(original) (delta).
                If False, R_cog = value(rewrite) only (absolute value).
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead
            ppl_on_full_response: If True, compute PPL on full response (Q+CoT+A).
                If False (default), compute PPL on answer portion only.
        """

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.logprob_last_n_tokens = logprob_last_n_tokens
        self.reward_mode = reward_mode
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_temperature
        self.ppl_on_full_response = ppl_on_full_response

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
        """
        Release KV Cache for all workers, keeping weights and Engine instances.

        Use release_memory_occupation() to release KV Cache,
        avoiding port conflicts caused by shutdown/recreate.
        """
        import subprocess
        import gc

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

        # Call release_memory() on each worker
        # This calls engine.release_memory_occupation() to release KV Cache
        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info(f"[SGLangStudentReward] Released KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to release memory: {e}")

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[SGLangStudentReward] Cleanup complete. GPU memory: {get_nvidia_smi()}")

    def initialize(self):
        """Lazy initialization of Ray SGLang actors."""
        if self._initialized:
            return

        # Clean up any leftover workers from a previous failed initialization attempt
        # This prevents worker accumulation if initialize() is retried after a failure
        if self.workers:
            logger.warning(f"[SGLangStudentReward] Cleaning up {len(self.workers)} leftover workers from previous failed init")
            for w in self.workers:
                try:
                    ray.kill(w)
                except Exception:
                    pass
            self.workers = []

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
        # Port allocation strategy:
        # - Each worker needs ~1000 ports: [base_port, base_port + 1000)
        #   - base_port: SGLang Engine port
        #   - base_port + 500: MASTER_PORT for torch.distributed
        #   - base_port + random[100, 1000): SGLang internal communication
        # - port_spacing = 1500 gives each worker a non-overlapping 1500-port range
        #   with 500 ports of headroom for fallback search within its own range
        # - Worker N's exclusive range: [base_port_start + N*1500, base_port_start + (N+1)*1500)
        #   Worker N can try base_port from range_start to range_start + 499
        #   (each base_port occupies [base_port, base_port + 1000))
        # - Port ranges across services (8 workers max):
        #   Student: [10000, 22000), FSF: [22000, 34000), CorrectJudge: [34000, 46000)
        base_port_start = 10000
        port_spacing = 1500  # Each worker gets a 1500-port range (needs ~1000, leaving 500 for fallback)
        sglang_port_range = 1000  # SGLang uses [base_port, base_port + 1000)

        def is_port_available(port: int) -> bool:
            """Check if a port is available for binding."""
            if port < 1 or port > 65535:
                return False
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                    return True
            except OSError:
                return False

        def is_port_range_available(base_port: int) -> bool:
            """Check if the full port range [base_port, base_port + 1000) is available.

            Each worker uses:
            - base_port: SGLang Engine port
            - base_port + 500: MASTER_PORT for torch.distributed
            - base_port + [100, 1000): SGLang internal communication ports

            We check a representative set of ports in this range to detect conflicts.
            """
            if not is_port_available(base_port):
                return False
            if not is_port_available(base_port + 500):
                return False
            # Sample-check the SGLang internal range [base_port+100, base_port+1000)
            for offset in range(100, 1000, 50):
                if not is_port_available(base_port + offset):
                    return False
            return True

        # Find available ports for all workers, each strictly within its own range
        assigned_ports = []
        for group_idx in range(len(gpu_groups)):
            range_start = base_port_start + group_idx * port_spacing
            range_end = range_start + port_spacing  # Exclusive upper bound of this worker's range
            # base_port can be [range_start, range_end - sglang_port_range)
            # so that [base_port, base_port + sglang_port_range) stays within the range
            max_base_port = range_end - sglang_port_range

            port = None
            for candidate in range(range_start, max_base_port):
                if is_port_range_available(candidate):
                    port = candidate
                    break

            if port is None:
                raise RuntimeError(
                    f"Cannot find available port range for worker {group_idx} "
                    f"in range [{range_start}, {max_base_port})"
                )

            assigned_ports.append(port)
            logger.info(f"[SGLangStudentReward] Worker {group_idx}: using port {port} (range: [{range_start}, {range_end}))")

        for group_idx, gpu_group in enumerate(gpu_groups):
            # gpu_group contains logical device indices (e.g., [0, 1]).
            # Convert to physical GPU IDs using available_gpus mapping.
            if cuda_visible_devices:
                physical_gpu_ids = [available_gpus[g] for g in gpu_group]
            else:
                physical_gpu_ids = list(gpu_group)
            # Actor-local logical IDs are always 0, 1, ..., tp_size-1
            # (the actor sets CUDA_VISIBLE_DEVICES to physical_gpu_ids,
            # so within the actor, these GPUs appear as logical 0, 1, ...)
            logical_gpu_ids = list(range(len(gpu_group)))
            # Assign unique base_port for each worker
            base_port = assigned_ports[group_idx]
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = SGLangWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_ids=physical_gpu_ids,
                logical_gpu_ids=logical_gpu_ids,
                tensor_parallel_size=self.tensor_parallel_size,
                mem_fraction_static=self.mem_fraction_static,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
                logprob_last_n_tokens=self.logprob_last_n_tokens,
                base_port=base_port,  # Unique base port per worker
                reward_mode=self.reward_mode,
                ppl_ratio_ceiling=self.ppl_ratio_ceiling,
                cot_effect_ratio_ceiling=self.cot_effect_ratio_ceiling,
                cot_effect_epsilon=self.cot_effect_epsilon,
                cot_effect_temperature=self.cot_effect_temperature,
                ppl_delta_temperature=self.ppl_delta_temperature,
                ppl_on_full_response=self.ppl_on_full_response,
            )
            self.workers.append(worker)
            logger.info(f"[SGLangStudentReward] Created worker {group_idx} on physical GPUs {physical_gpu_ids} with base_port={base_port}")

        # Initialize workers IN PARALLEL - each worker has a non-overlapping 2000-port range
        # so parallel initialization cannot cause port conflicts between workers.
        logger.info(f"[SGLangStudentReward] Initializing {len(self.workers)} workers IN PARALLEL with unique ports...")
        logger.info(f"[SGLangStudentReward] Each worker will verify Engine readiness by testing logprob generation...")
        
        init_futures = []
        for i, worker in enumerate(self.workers):
            logger.info(f"[SGLangStudentReward] Submitting init for worker {i}/{len(self.workers)} (base_port={assigned_ports[i]})...")
            # Use initialize_and_verify to ensure Engine is fully ready before proceeding
            init_futures.append(worker.initialize_and_verify.remote())

        # Wait for all workers to complete initialization in parallel
        try:
            results = ray.get(init_futures, timeout=300)  # 5 min timeout for all workers (including retries)
            logger.info(f"[SGLangStudentReward] All {len(self.workers)} workers initialized and verified successfully in parallel")
        except Exception as e:
            logger.error(f"[SGLangStudentReward] Parallel init FAILED: {e}")
            raise

        self._initialized = True
        logger.info(f"[SGLangStudentReward] All {len(self.workers)} workers initialized")

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

        # Compute answer start position using two apply_chat_template calls
        # Method: tokenize twice with question+cot+dummy_answer and question+cot+answer
        # answer_start_idx is the length after the first tokenization (with dummy_answer)
        #
        # Principle: regardless of the answer, the chat template format after assistant is fixed
        # So dummy_answer and answer token sequences only differ in the answer content
        # The length of the first tokenization is where the answer starts

        # First time: use dummy_answer (a single char, e.g. "X")
        dummy_answer = "X"
        messages_with_dummy = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': dummy_answer, 'reasoning_content': cot}
        ]
        text_with_dummy = self.tokenizer.apply_chat_template(
            messages_with_dummy, tokenize=False, add_generation_prompt=False
        )
        tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

        # Find the position of dummy_answer ("X") in tokens
        # This position is where the answer starts
        dummy_token_id = self.tokenizer.encode(dummy_answer, add_special_tokens=False)[0]
        # Search dummy_token_id from back to front (because there may be duplicates)
        for i in range(len(tokens_with_dummy) - 1, -1, -1):
            if tokens_with_dummy[i] == dummy_token_id:
                answer_start_idx = i
                break

        return full_text, answer_start_idx

    def _build_prompt_for_kl(self, question: str, cot: str) -> tuple[str, int, int]:
        """
        Build prompt for KL reward mode: Question + CoT only (no answer).

        The student model sees only Q + CoT and computes log probs on the CoT tokens.
        This measures how "surprising" the CoT is to a model that only knows the question.

        Returns:
            tuple: (full_prompt, cot_start_token_idx, cot_end_token_idx)
            - cot_start_token_idx: index of first CoT content token (after <think>\n)
            - cot_end_token_idx: index of first token after CoT content (before \n</think>)
        """
        # Build full prompt: Q + CoT (no answer content)
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': f'<think>\n{cot}\n</think>'}
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Find where CoT tokens start and end using the dummy technique
        dummy_cot = "X"
        messages_with_dummy = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': f'<think>\n{dummy_cot}\n</think>'}
        ]
        text_with_dummy = self.tokenizer.apply_chat_template(
            messages_with_dummy, tokenize=False, add_generation_prompt=False
        )
        tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

        dummy_token_id = self.tokenizer.encode(dummy_cot, add_special_tokens=False)[0]
        
        # Search from back to front to find dummy token position
        cot_start_idx = 0
        for i in range(len(tokens_with_dummy) - 1, -1, -1):
            if tokens_with_dummy[i] == dummy_token_id:
                cot_start_idx = i
                break
        
        # Calculate cot_end_idx: cot_start_idx + 1 (dummy is 1 token)
        # In actual CoT, this will be cot_start_idx + len(cot_tokens)
        # We'll compute this after tokenizing the actual CoT
        cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
        cot_end_idx = cot_start_idx + len(cot_tokens)

        return full_text, cot_start_idx, cot_end_idx

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
        # Use the same dummy_answer technique
        dummy_answer = "X"
        messages_with_dummy = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': dummy_answer}  # No reasoning_content
        ]
        text_with_dummy = self.tokenizer.apply_chat_template(
            messages_with_dummy, tokenize=False, add_generation_prompt=False
        )
        tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

        dummy_token_id = self.tokenizer.encode(dummy_answer, add_special_tokens=False)[0]
        for i in range(len(tokens_with_dummy) - 1, -1, -1):
            if tokens_with_dummy[i] == dummy_token_id:
                answer_start_idx = i
                break

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
        save_prompts: bool = False,  # Whether to save prompts for debugging
        **kwargs,  # Accept extra args for compatibility
    ) -> Dict[str, Any]:
        """
        Compute cognitive rewards for a batch using multi-GPU SGLang Ray actors.

        Each Ray actor runs SGLang on a separate GPU. We split the batch across
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
            alpha_cog: Weight for min log prob term
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

        # Resume KV Cache (if previously released by release_memory)
        self.resume_memory()

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

        # Save prompts for debugging if requested
        # Store prompts in dict for later return
        saved_prompts = {}
        if save_prompts:
            saved_prompts["rewrite_prompts"] = rewrite_prompts
            saved_prompts["no_cot_prompts"] = no_cot_prompts
            # For original prompts, map them back to original indices
            original_prompts_full = [None] * batch_size
            for idx, orig_prompt in zip(original_indices, original_prompts):
                original_prompts_full[idx] = orig_prompt
            saved_prompts["original_prompts"] = original_prompts_full

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

                    epsilon = getattr(self, 'cot_effect_epsilon', 0.01)
                    temperature = getattr(self, 'cot_effect_temperature', 2.0)

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

        result_dict = {
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

        # Add saved prompts if requested
        if save_prompts and saved_prompts:
            result_dict["saved_prompts"] = saved_prompts

        return result_dict

    def compute_kl_logprobs_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        response_length: int,
    ) -> torch.Tensor:
        """
        Compute per-token student log probs for KL reward mode.

        The student model sees only Q + CoT (no answer, no original_cot) and computes
        log probs on the CoT tokens. This is used to measure how "surprising" the CoT
        is to the student — higher KL means the actor is using information the student
        cannot predict from the question alone.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs (extracted content, no tags)
            response_length: Length of actor response tokens (for output tensor shape)

        Returns:
            torch.Tensor of shape [batch_size, response_length] with student log probs.
            Each row has student log probs at the first N positions (N = number of CoT tokens),
            and 0.0 at the remaining positions (handled by response_mask in KL computation).
        """
        import time

        self.initialize()
        self.resume_memory()

        batch_size = len(questions)
        start_time = time.time()

        # Phase 1: Build all KL prompts (CPU)
        kl_prompts = []
        kl_cot_starts = []
        kl_cot_ends = []
        for i in range(batch_size):
            prompt, cot_start, cot_end = self._build_prompt_for_kl(questions[i], rewrite_cots[i])
            kl_prompts.append(prompt)
            kl_cot_starts.append(cot_start)
            kl_cot_ends.append(cot_end)

        prep_time = time.time() - start_time

        # Phase 2: Distribute to workers in parallel
        inference_start = time.time()
        kl_chunks = self._split_into_chunks(kl_prompts, self.num_actors)
        kl_cot_start_chunks = self._split_into_chunks(kl_cot_starts, self.num_actors)

        kl_futures = []
        for worker_idx, (prompts_chunk, starts_chunk) in enumerate(
            zip(kl_chunks, kl_cot_start_chunks)
        ):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(
                    prompts_chunk, starts_chunk
                )
                kl_futures.append((worker_idx, len(prompts_chunk), future))

        # Gather results
        kl_results = [None] * batch_size
        for worker_idx, chunk_size, future in kl_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(kl_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                kl_results[start_idx + i] = result

        inference_time = time.time() - inference_start

        # Phase 3: Build output tensor [batch_size, response_length]
        # Extract only CoT portion (from cot_start to cot_end) for each sample
        student_log_probs = torch.zeros(batch_size, response_length)
        for i in range(batch_size):
            if kl_results[i] is not None:
                token_lps = kl_results[i].get("token_log_probs", [])
                # logprob_start_len=cot_start, so token_lps[0] corresponds to cot_start
                # We need token_lps[0:cot_end-cot_start] which is the pure CoT content
                cot_length = kl_cot_ends[i] - kl_cot_starts[i]
                cot_logprobs = token_lps[:cot_length] if len(token_lps) >= cot_length else token_lps
                n = min(len(cot_logprobs), response_length)
                if n > 0:
                    student_log_probs[i, :n] = torch.tensor(cot_logprobs[:n])

        total_time = time.time() - start_time
        logger.info(
            f"[KL] Student log probs computed for {batch_size} samples in {total_time:.2f}s "
            f"(prep={prep_time:.2f}s, inference={inference_time:.2f}s)"
        )

        return student_log_probs

    def compute_kl_answer_divergence_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        original_cots: List[str],
        answers: List[str],
        alpha_kl: float = 0.01,
    ) -> List[float]:
        """
        Compute per-sample KL divergence on answer tokens:
        D_KL(student(answer|q,rewrite_cot) || student(answer|q,original_cot))

        Uses k3 estimator: k3_t = exp(log_r_t) - 1 - log_r_t
        where log_r_t = log P(a_t|...,rewrite_cot) - log Q(a_t|...,original_cot)

        Returns:
            List of per-sample raw KL divergence values (non-negative).
        """
        import time

        self.initialize()
        self.resume_memory()

        batch_size = len(questions)
        start_time = time.time()

        # Phase 1: Build prompts for both rewrite and original conditions
        rewrite_prompts = []
        rewrite_ans_starts = []
        original_prompts = []
        original_ans_starts = []

        for i in range(batch_size):
            rw_prompt, rw_ans_start = self._build_prompt(questions[i], rewrite_cots[i], answers[i])
            rewrite_prompts.append(rw_prompt)
            rewrite_ans_starts.append(rw_ans_start)

            orig_prompt, orig_ans_start = self._build_prompt(questions[i], original_cots[i], answers[i])
            original_prompts.append(orig_prompt)
            original_ans_starts.append(orig_ans_start)

        prep_time = time.time() - start_time

        # Phase 2: Distribute to workers in parallel
        inference_start = time.time()

        # Submit rewrite prompts
        rw_chunks = self._split_into_chunks(rewrite_prompts, self.num_actors)
        rw_ans_chunks = self._split_into_chunks(rewrite_ans_starts, self.num_actors)
        rw_futures = []
        for w_idx, (p_chunk, a_chunk) in enumerate(zip(rw_chunks, rw_ans_chunks)):
            if p_chunk:
                future = self.workers[w_idx].compute_logprobs.remote(p_chunk, a_chunk)
                rw_futures.append((w_idx, len(p_chunk), future))

        # Submit original prompts
        orig_chunks = self._split_into_chunks(original_prompts, self.num_actors)
        orig_ans_chunks = self._split_into_chunks(original_ans_starts, self.num_actors)
        orig_futures = []
        for w_idx, (p_chunk, a_chunk) in enumerate(zip(orig_chunks, orig_ans_chunks)):
            if p_chunk:
                future = self.workers[w_idx].compute_logprobs.remote(p_chunk, a_chunk)
                orig_futures.append((w_idx, len(p_chunk), future))

        # Gather rewrite results
        rw_results = [None] * batch_size
        for w_idx, chunk_size, future in rw_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(rw_chunks[i]) for i in range(w_idx))
            for i, result in enumerate(chunk_results):
                rw_results[start_idx + i] = result

        # Gather original results
        orig_results = [None] * batch_size
        for w_idx, chunk_size, future in orig_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(orig_chunks[i]) for i in range(w_idx))
            for i, result in enumerate(chunk_results):
                orig_results[start_idx + i] = result

        inference_time = time.time() - inference_start

        # Phase 3: Compute KL divergence per sample using k3 estimator
        import torch
        kl_values = []
        for i in range(batch_size):
            rw_lps = rw_results[i]["token_log_probs"] if rw_results[i] else []
            orig_lps = orig_results[i]["token_log_probs"] if orig_results[i] else []

            # Align to same length (use min length)
            n = min(len(rw_lps), len(orig_lps))
            if n == 0:
                kl_values.append(0.0)
                continue

            rw_t = torch.tensor(rw_lps[:n])
            orig_t = torch.tensor(orig_lps[:n])

            # k3 estimator: D_KL(P || Q) where P=rewrite, Q=original
            log_r = rw_t - orig_t
            log_r_clamped = torch.clamp(log_r, min=-20.0, max=20.0)
            r = torch.exp(log_r_clamped)
            k3 = r - 1.0 - log_r_clamped
            k3 = torch.clamp(k3, min=-10.0, max=10.0)

            avg_kl = k3.mean()
            min_kl = k3.min()
            kl_val = (avg_kl + alpha_kl * min_kl).item()
            kl_values.append(kl_val)

        total_time = time.time() - start_time
        logger.info(
            f"[KL-Answer] Computed answer KL divergence for {batch_size} samples in {total_time:.2f}s "
            f"(prep={prep_time:.2f}s, inference={inference_time:.2f}s), "
            f"mean={sum(kl_values)/len(kl_values):.4f}"
        )

        return kl_values

    def release_memory(self):
        """
        Release KV Cache for all workers, keeping weights.
        Used to free GPU memory for other models before reward computation.
        """
        if not self._initialized:
            return
        
        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangStudentReward] Released memory for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to release memory: {e}")

    def resume_memory(self):
        """
        Resume KV Cache for all workers.
        Used to restore full SGLang functionality after reward computation.
        """
        if not self._initialized:
            return
        
        try:
            futures = [w.resume_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangStudentReward] Resumed memory for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to resume memory: {e}")

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
