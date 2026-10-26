"""
Student Reward (Cognitive Reward) Module - Transformers Version

Computes R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
using HuggingFace Transformers for log prob computation (memory-efficient alternative to vLLM).

where answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
compared to the original CoT.

Uses Ray actors for multi-GPU parallelism. Each GPU runs a separate
Transformers model instance (Data Parallelism).

Advantages over vLLM:
- Lower memory footprint (no kv-cache overhead)
- No need for tensor parallelism (each GPU is independent)
- Better for batch processing of fixed-length sequences
"""

from typing import Any, Dict, List, Optional
import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger


@ray.remote
class TransformersWorkerActor:
    """
    Ray actor that runs HuggingFace Transformers on a single GPU.
    
    Each actor loads its own model copy and processes batches independently.
    This is more memory-efficient than vLLM for computing logprobs only.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,           # Physical GPU ID
        logical_gpu_id: int,   # Logical GPU ID within CUDA_VISIBLE_DEVICES
        dtype: str = "float16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,
    ):
        import os
        import torch

        self.gpu_id = gpu_id
        self.logical_gpu_id = logical_gpu_id
        worker_name = f"TransformersWorker-{gpu_id}"

        # Set CUDA_VISIBLE_DEVICES to the logical GPU for this actor
        os.environ["CUDA_VISIBLE_DEVICES"] = str(logical_gpu_id)
        
        # Clear cache before loading model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._worker_name = worker_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Determine torch dtype
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32
        
        # Load model with memory-efficient settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=f"cuda:{logical_gpu_id}",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.micro_batch_size = micro_batch_size
        self.max_model_len = max_model_len
        
        logger.info(f"[{worker_name}] Transformers model loaded on GPU {gpu_id} (logical: {logical_gpu_id})")

    def compute_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a batch of prompts using Transformers.

        Uses micro-batching to avoid OOM with long sequences.

        Args:
            prompts: List of full prompts (question + CoT + answer)
            answer_starts: List of token indices where answer portion starts

        Returns:
            List of dicts, each containing:
                - mean_log_prob: float
                - min_log_prob: float
                - token_log_probs: List[float]
        """
        import torch
        import time
        import gc

        worker_name = self._worker_name
        
        # Compute token lengths for logging
        prompt_token_lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        print(f"[{worker_name}] compute_logprobs START: {len(prompts)} prompts, token lengths: min={min(prompt_token_lengths)}, max={max(prompt_token_lengths)}, avg={sum(prompt_token_lengths)//len(prompt_token_lengths)}", flush=True)

        # Micro-batch processing to avoid OOM
        results = []
        micro_batch_size = self.micro_batch_size

        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]

            batch_results = self._compute_batch_logprobs(batch_prompts, batch_ans_starts)
            results.extend(batch_results)

            # Release memory
            del batch_results
            gc.collect()
            torch.cuda.empty_cache()

        return results

    def _compute_batch_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a single micro-batch.
        """
        import torch
        import torch.nn.functional as F

        # Tokenize all prompts in the batch
        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
            add_special_tokens=False,
        )
        
        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)

        with torch.no_grad():
            # Forward pass to get logits
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

        # Compute log probs for each sequence
        results = []
        for i, (input_id, ans_start) in enumerate(zip(input_ids, answer_starts)):
            # Get the actual length (non-padding)
            seq_len = attention_mask[i].sum().item()
            
            # Extract logits and input_ids for this sequence
            seq_logits = logits[i, :seq_len-1, :]  # Exclude last position (no next token)
            seq_input_ids = input_id[1:seq_len]    # Target tokens (shifted by 1)
            
            # Compute log softmax
            log_probs_all = F.log_softmax(seq_logits, dim=-1)
            
            # Gather log probs for actual tokens
            token_log_probs = []
            for pos in range(len(seq_input_ids)):
                token_id = seq_input_ids[pos].item()
                token_log_prob = log_probs_all[pos, token_id].item()
                token_log_probs.append(token_log_prob)
            
            # Extract answer portion log probs
            # ans_start is the index where answer starts in the full sequence
            # We need to account for the shift by 1 in logits
            answer_log_probs = []
            if ans_start < len(token_log_probs):
                answer_log_probs = token_log_probs[ans_start:]
            
            if answer_log_probs:
                results.append({
                    "mean_log_prob": sum(answer_log_probs) / len(answer_log_probs),
                    "min_log_prob": min(answer_log_probs),
                    "token_log_probs": answer_log_probs,
                })
            else:
                results.append({
                    "mean_log_prob": -100.0,
                    "min_log_prob": -100.0,
                    "token_log_probs": [],
                })

        return results

    def cleanup(self):
        """Cleanup model and release GPU memory."""
        import torch
        import gc
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TransformersStudentRewardClient:
    """
    Multi-GPU Transformers-based Student Reward Client using Ray actors.

    Each GPU runs an independent Transformers model instance (Data Parallelism).
    This is more memory-efficient than vLLM for computing logprobs only,
    as there's no kv-cache overhead.
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.4,  # Not used but kept for API compatibility
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,
        min_free_memory_gb: float = 5.0,  # Min free memory required before init
        memory_wait_timeout: int = 300,    # Max seconds to wait for memory release
    ):
        """
        Initialize Multi-GPU Transformers Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            gpu_memory_utilization: Ignored (kept for API compatibility)
            dtype: Data type for the model ("float16", "bfloat16", "float32")
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
            min_free_memory_gb: Minimum free GPU memory (GB) required before initializing
            memory_wait_timeout: Maximum time (seconds) to wait for memory release
        """
        import torch

        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.min_free_memory_gb = min_free_memory_gb
        self.memory_wait_timeout = memory_wait_timeout

        # Auto-detect devices if not specified
        if devices is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
        self.devices = devices
        self.num_gpus = len(devices)
        self.gpu_ids = [int(d.split(":")[-1]) for d in devices]

        self.num_actors = self.num_gpus  # One actor per GPU for Transformers
        self.workers: List[ray.actor.ActorHandle] = []
        self.tokenizer = None
        self._initialized = False

        logger.info(f"[TransformersStudentReward] GPUs:{self.num_gpus} Actors:{self.num_actors}")
        logger.info(f"[TransformersStudentReward] Dtype: {dtype}, MaxLen: {max_model_len}")

    def cleanup(self):
        """Cleanup Ray actors to release GPU memory."""
        import time
        import gc
        import torch

        if not self._initialized:
            return

        # Kill all Ray actors to release GPU memory
        for worker in self.workers:
            try:
                ray.get(worker.cleanup.remote())
                ray.kill(worker)
            except Exception:
                pass

        self.workers = []
        self._initialized = False

        # Force garbage collection and CUDA cache cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[TransformersStudentReward] Cleanup completed")

    def _get_gpu_memory_info(self) -> dict:
        """Get current GPU memory usage for all target GPUs using pynvml."""
        import pynvml
        
        gpu_memory = {}
        try:
            pynvml.nvmlInit()
            for gpu_idx in self.gpu_ids:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    total_mb = info.total / (1024 * 1024)
                    used_mb = info.used / (1024 * 1024)
                    
                    gpu_memory[gpu_idx] = {
                        'used_mb': int(used_mb),
                        'total_mb': int(total_mb),
                        'used_gb': used_mb / 1024,
                        'total_gb': total_mb / 1024,
                        'free_gb': (total_mb - used_mb) / 1024,
                    }
                except Exception:
                    continue
        except Exception:
            pass
        return gpu_memory

    def _wait_for_gpu_memory_release(self, min_free_gb: float = 5.0, max_wait_time: int = 300):
        """
        Wait for GPU memory to be released.
        
        Args:
            min_free_gb: Minimum free memory required per GPU (in GB)
            max_wait_time: Maximum time to wait in seconds
        """
        import time
        
        start_time = time.time()
        
        while True:
            current_memory = self._get_gpu_memory_info()
            
            # Check each target GPU
            all_gpus_ready = True
            for gpu_id in self.gpu_ids:
                if gpu_id not in current_memory:
                    all_gpus_ready = False
                    continue
                if current_memory[gpu_id]['free_gb'] < min_free_gb:
                    all_gpus_ready = False
            
            elapsed = time.time() - start_time
            
            if all_gpus_ready:
                return True
            
            # Check timeout
            if elapsed > max_wait_time:
                logger.warning(f"[TransformersStudentReward] GPU memory wait timeout after {max_wait_time}s")
                return False
            
            time.sleep(1)

    def initialize(self):
        """Lazy initialization of Ray Transformers actors."""
        if self._initialized:
            return

        from transformers import AutoTokenizer

        # Wait for GPU memory release after other models
        self._wait_for_gpu_memory_release(
            min_free_gb=self.min_free_memory_gb, 
            max_wait_time=self.memory_wait_timeout
        )

        # Initialize tokenizer (shared, for prompt building)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Compute logical GPU IDs from CUDA_VISIBLE_DEVICES
        import os
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            physical_to_logical = {gpu: idx for idx, gpu in enumerate(available_gpus)}
        else:
            physical_to_logical = {gpu: gpu for gpu in self.gpu_ids}

        # Create Ray actors, one per GPU
        for gpu_id in self.gpu_ids:
            logical_gpu_id = physical_to_logical.get(gpu_id, 0)

            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = TransformersWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_id=gpu_id,
                logical_gpu_id=logical_gpu_id,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
            )
            self.workers.append(worker)

        # Wait for all actors to initialize
        init_timeout = 180
        logger.info(f"[TransformersStudentReward] Waiting for {self.num_actors} actors to initialize (timeout={init_timeout}s)...")
        init_futures = [w.compute_logprobs.remote(["test"], [0]) for w in self.workers]
        
        for i, future in enumerate(init_futures):
            logger.info(f"[TransformersStudentReward] Waiting for worker {i} init...")
            try:
                result = ray.get(future, timeout=init_timeout)
                logger.info(f"[TransformersStudentReward] Worker {i} init done")
            except Exception as e:
                logger.error(f"[TransformersStudentReward] Worker {i} init FAILED: {e}")
                # Clean up all created actors to prevent GPU memory leak
                logger.warning(f"[TransformersStudentReward] Cleaning up {len(self.workers)} actors after init failure...")
                self.cleanup()
                raise
        
        self._initialized = True
        logger.info(f"[TransformersStudentReward] All {self.num_actors} workers initialized successfully")

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

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
        alpha_cog: float = 0.01,
        **kwargs,  # Accept extra args for compatibility
    ) -> Dict[str, Any]:
        """
        Compute cognitive rewards for a batch using multi-GPU Transformers Ray actors.

        Each Ray actor runs Transformers on a separate GPU. We split the batch across
        all actors for true data parallelism.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers
            original_cots: Optional list of original CoTs
            alpha_cog: Weight for min log prob term

        Returns:
            Dict containing rewards and log probs
        """
        import time

        self.initialize()

        batch_size = len(questions)

        # Build all prompts (CPU)
        rewrite_prompts = []
        rewrite_ans_starts = []
        original_prompts = []
        original_ans_starts = []
        original_indices = []

        for i in range(batch_size):
            rewrite_prompt, rewrite_ans_start = self._build_prompt(questions[i], rewrite_cots[i], answers[i])
            rewrite_prompts.append(rewrite_prompt)
            rewrite_ans_starts.append(rewrite_ans_start)

            if original_cots and i < len(original_cots) and original_cots[i]:
                orig_prompt, orig_ans_start = self._build_prompt(questions[i], original_cots[i], answers[i])
                original_prompts.append(orig_prompt)
                original_ans_starts.append(orig_ans_start)
                original_indices.append(i)

        # Split rewrite prompts across actors
        rewrite_chunks = self._split_into_chunks(rewrite_prompts, self.num_actors)
        rewrite_ans_start_chunks = self._split_into_chunks(rewrite_ans_starts, self.num_actors)

        # Submit rewrite tasks to all workers in parallel
        rewrite_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(rewrite_chunks, rewrite_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                rewrite_futures.append((worker_idx, future))

        # Split original prompts across actors (if any)
        original_futures = []
        if original_prompts:
            original_chunks = self._split_into_chunks(original_prompts, self.num_actors)
            original_ans_start_chunks = self._split_into_chunks(original_ans_starts, self.num_actors)

            for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(original_chunks, original_ans_start_chunks)):
                if prompts_chunk:
                    future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                    original_futures.append((worker_idx, future))

        # Gather rewrite results
        rewrite_results = [None] * batch_size
        for worker_idx, future in rewrite_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_results[start_idx + i] = result

        # Gather original results
        original_results_map = {}
        if original_futures:
            for worker_idx, future in original_futures:
                chunk_results = ray.get(future)
                start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
                for i, result in enumerate(chunk_results):
                    original_idx = original_indices[start_idx + i]
                    original_results_map[original_idx] = result

        rewards = []
        rewrite_mean_log_probs = []
        original_mean_log_probs = []
        all_token_log_probs = []
        all_token_ids = []

        for i in range(batch_size):
            # Get rewrite result
            rewrite_result = rewrite_results[i]
            rewrite_mean = rewrite_result["mean_log_prob"]
            rewrite_min = rewrite_result["min_log_prob"]
            rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min

            # Get original result (if available)
            if i in original_results_map:
                original_result = original_results_map[i]
                original_mean = original_result["mean_log_prob"]
                original_min = original_result["min_log_prob"]
                original_answer_log_prob = original_mean + alpha_cog * original_min
            else:
                original_mean = -100.0
                original_answer_log_prob = -100.0

            # ===== R_cog calculation formula =====
            # R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
            # where answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            reward = rewrite_answer_log_prob - original_answer_log_prob

            rewards.append(reward)
            rewrite_mean_log_probs.append(rewrite_mean)
            original_mean_log_probs.append(original_mean)
            all_token_log_probs.append(rewrite_result.get("token_log_probs", []))
            all_token_ids.append([])  # Token IDs not needed for reward

        logger.info(f"[TransformersStudentReward] All {batch_size} rewards computed")

        return {
            "rewards": rewards,
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "rewrite_token_log_probs": all_token_log_probs,
            "rewrite_token_ids": all_token_ids,
            "log_probs": rewrite_mean_log_probs,
            "token_log_probs": all_token_log_probs,
            "token_ids": all_token_ids,
        }

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
