"""
Correct Judge - SGLang Version

Determines if the reasoning process leads to the correct solution.
Uses Ray actors for multi-GPU parallelism.

The judge evaluates whether the rewrite_cot logically leads to the solution.
If the answer is "No", it indicates a correctness issue.
"""

import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# IMPORTANT: Set PYTORCH_NVML_BASED_CUDA_CHECK=1 BEFORE importing torch.
# Prevents torch.cuda.is_available() from caching False when Ray sets
# CUDA_VISIBLE_DEVICES="" for actors with num_gpus=0.
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import ray
import torch

from verl.experimental.reward.cot_rewriter.training_logger import get_logger
from transformers import AutoTokenizer

# Logger for SGLang Correct Judge
logger = get_logger("sglang_correct_judge")


@ray.remote
class SGLangCorrectJudgeActor:
    """
    Ray actor that runs Correct Judge SGLang on one or more GPUs.

    Determines if reasoning process leads to the correct solution.
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],           # Physical GPU IDs (e.g., [0, 1] for TP=2)
        logical_gpu_ids: List[int],   # Logical GPU IDs within CUDA_VISIBLE_DEVICES
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        temperature: float = 0.0,     # Use 0 for deterministic judgment
        top_k: int = 1,
        top_p: float = 1.0,
        max_new_tokens: int = 50,     # Short output: <judge>Yes</judge> or <judge>No</judge>
        base_port: int = 50000,  # Base port for this worker (different from FSF)
    ):
        import subprocess

        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self._base_port = base_port
        worker_name = f"SGLangCorrectJudgeActor-{gpu_ids}"

        # Set CUDA_VISIBLE_DEVICES to physical GPU IDs for this actor
        cuda_devices_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Set unique MASTER_PORT to avoid torch.distributed TCP rendezvous port conflicts
        # when multiple CorrectJudge workers initialize in parallel on the same machine.
        # Dynamically find an available port instead of using a fixed offset,
        # to avoid EADDRINUSE errors when ports are still in TIME_WAIT state.
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

        # Log GPU memory status
        def get_nvidia_smi_memory_for_gpu(gpu_idx):
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
                        return f"GPU {gpu_idx}: {used_gb:.1f}GB / {total_gb:.1f}GB"
                return f"GPU {gpu_idx}: not found"
            except Exception as e:
                return f"nvidia-smi error: {e}"

        memory_info = ", ".join([get_nvidia_smi_memory_for_gpu(gpu) for gpu in gpu_ids])
        logger.info(f"[{worker_name}] GPU Memory: {memory_info}")

        # Create event loop for SGLang
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Engine will be created lazily
        self.engine = None
        self._engine_initialized = False
        self._memory_released = False

        logger.info(f"[{worker_name}] Actor created, engine will be initialized on first use")

    def _init_engine(self):
        """Initialize SGLang Engine on demand."""
        if self._engine_initialized:
            return

        import sglang as sgl

        logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Creating SGLang Engine...")
        start_time = time.time()

        self.engine = sgl.Engine(
            model_path=self.model_path,
            tp_size=self.tensor_parallel_size,
            mem_fraction_static=self.mem_fraction_static,
            dtype=self.dtype,
            trust_remote_code=True,
            enable_memory_saver=True,
            port=self._base_port,
        )

        # Re-create event loop after Engine creation
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        elapsed = time.time() - start_time
        self._engine_initialized = True
        logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Engine created in {elapsed:.2f}s")

    def _ensure_event_loop(self):
        """Ensure an event loop exists for the current thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _build_correct_judge_prompt(
        self,
        question: str,
        rewrite_cot: str,
        solution: str,
    ) -> str:
        """
        Build Correct Judge prompt.

        Determines if the reasoning process leads to the given solution.
        """
        JUDGE_PROMPT = """Evaluate the relationship between the [Reasoning Process] and the [Solution]. 

## Evaluation Criteria
1. Conclusion Consistency: Does the reasoning process derive the exact same final result as the [Solution]?
2. Semantic Coherence: Is there a natural, logically sound, and coherent semantic transition between the [Reasoning Process] and the [Solution]? They must not feel disconnected or logically contradictory.

## Question
{question}

## Reasoning Process
{rewrite_cot}

## Solution
{solution}

## Output Format
First, state your conclusion. Output <judge>Yes</judge> if BOTH criteria are completely satisfied. Output <judge>No</judge> if ANY criterion fails.
Then, output your analysis.
"""

        # Handle <rewrite_reasoning> tags if present
        if '<rewrite_reasoning>' in rewrite_cot:
            rewrite_cot = rewrite_cot.split('<rewrite_reasoning>')[-1]
        if '</rewrite_reasoning>' in rewrite_cot:
            rewrite_cot = rewrite_cot.split('</rewrite_reasoning>')[0]

        rewrite_cot = rewrite_cot.strip()

        prompt = JUDGE_PROMPT.format(
            question=question.strip(),
            rewrite_cot=rewrite_cot,
            solution=solution.strip(),
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt_text

    def _parse_judge_result(self, output: str) -> Optional[bool]:
        """
        Parse judge result from model output.

        Returns:
            True if reasoning leads to solution (Yes)
            False if reasoning does not lead to solution (No)
            None if parsing fails
        """
        output = output.strip().lower()

        # Try to extract <judge>...</judge> tag
        judge_match = re.search(r'<judge>\s*(yes|no)\s*</judge>', output, re.IGNORECASE)
        if judge_match:
            result = judge_match.group(1).strip()
            is_correct = result == 'yes'
            logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Parsed judge result: <judge>{result}</judge> -> is_correct={is_correct}")
            return is_correct

        # Fallback: look for "yes" or "no" in the output
        if 'yes' in output and 'no' not in output:
            logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Fallback parse: found 'yes' in output -> is_correct=True")
            return True
        elif 'no' in output and 'yes' not in output:
            logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Fallback parse: found 'no' in output -> is_correct=False")
            return False

        logger.warning(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Failed to parse judge result from output: '{output[:100]}...'")
        return None

    def compute_correct_judge_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        solutions: List[str],
        micro_batch_size: int = 8,
        save_prompts: bool = False,  # Whether to save request prompts and responses
    ) -> Tuple[List[Optional[bool]], Optional[Dict[str, List[str]]]]:
        """
        Batch computation of correct judge results.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoT reasonings
            solutions: List of ground truth solutions
            micro_batch_size: Max number of prompts per SGLang call
            save_prompts: If True, also return saved prompts and responses dict

        Returns:
            Tuple of:
            - List of boolean results (True=correct, False=incorrect, None=parse failure)
            - Saved data dict (if save_prompts=True) or None, containing:
                - "correct_judge_prompts": List of request prompts
                - "correct_judge_responses": List of raw judge responses
        """
        # Initialize engine on first use
        self._init_engine()
        self._ensure_event_loop()

        # Build prompts
        prompts = []
        for question, rewrite_cot, solution in zip(questions, rewrite_cots, solutions):
            # Truncate long reasoning to avoid OOM
            rewrite_cot_truncated = rewrite_cot[:4000] if len(rewrite_cot) > 4000 else rewrite_cot
            prompt = self._build_correct_judge_prompt(question, rewrite_cot_truncated, solution)
            prompts.append(prompt)

        # Save prompts for debugging if requested
        saved_data = {}
        if save_prompts:
            saved_data["correct_judge_prompts"] = prompts

        # Process in micro-batches to avoid OOM
        results = []
        responses = []  # Store raw responses if save_prompts=True
        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]

            try:
                outputs = self.engine.generate(
                    prompt=batch_prompts,
                    sampling_params={
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                    },
                )

                # Parse judge result from each output
                for output in outputs:
                    if isinstance(output, dict):
                        generated_text = output.get("text", "")
                    else:
                        generated_text = str(output)
                    results.append(self._parse_judge_result(generated_text))
                    if save_prompts:
                        responses.append(generated_text)

            except Exception as e:
                logger.error(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Micro-batch compute error at batch {i}: {e}")
                # Fill remaining with None for this batch
                remaining = len(batch_prompts)
                results.extend([None] * remaining)
                if save_prompts:
                    responses.extend([""] * remaining)  # Empty string for errors

        if save_prompts:
            saved_data["correct_judge_responses"] = responses
            return results, saved_data
        return results, None

    def release_memory(self):
        """
        Completely destroy SGLang Engine to release all GPU memory.
        """
        if not self._engine_initialized:
            return

        try:
            if self.engine is not None:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
                del self.engine
                self.engine = None
                self._engine_initialized = False
                self._memory_released = False
                logger.debug(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Engine destroyed, all GPU memory released")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Error during engine shutdown: {e}")
            self.engine = None
            self._engine_initialized = False
            self._memory_released = False

    def resume_memory(self):
        """Resume KV Cache memory."""
        if not self._memory_released or not self._engine_initialized:
            return

        try:
            if hasattr(self.engine, "resume_memory_occupation"):
                self.engine.resume_memory_occupation()
                self._memory_released = False
                logger.debug(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Resumed KV Cache")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Failed to resume memory: {e}")

    def shutdown(self):
        """Shutdown the engine and release all resources."""
        try:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            del self.engine
            self._engine_initialized = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Shutdown complete")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeActor-{self.gpu_ids}] Shutdown error: {e}")


class SGLangCorrectJudgeClient:
    """
    Multi-GPU SGLang-based Correct Judge Client using Ray actors.

    Determines if the reasoning process leads to the correct solution.

    If is_correct=False (reasoning doesn't lead to solution),
    the R_cog reward is forced to a preset value.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        micro_batch_size: int = 8,  # Micro batch size to avoid OOM
    ):
        """
        Initialize Multi-GPU SGLang Correct Judge Client.

        Args:
            model_path: Path to Correct Judge Model
            tensor_parallel_size: TP size per SGLang instance
            mem_fraction_static: GPU memory fraction
            dtype: Data type (float16 or bfloat16)
            max_model_len: Maximum context length
            devices: List of CUDA devices (auto-detected if None)
            temperature: Sampling temperature (0.0 for deterministic)
            top_k: Sampling top-k
            top_p: Sampling top-p
            max_new_tokens: Max tokens to generate (short output)
            micro_batch_size: Micro batch size per worker to avoid OOM
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.micro_batch_size = micro_batch_size

        # Auto-detect devices
        if devices is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
        self.devices = devices
        self.num_gpus = len(devices)
        self.gpu_ids = [int(d.split(":")[-1]) for d in devices]

        # Calculate number of actors
        assert self.num_gpus >= tensor_parallel_size, \
            f"num_gpus ({self.num_gpus}) must be >= tensor_parallel_size ({tensor_parallel_size})"
        assert self.num_gpus % tensor_parallel_size == 0, \
            f"num_gpus ({self.num_gpus}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
        self.num_actors = self.num_gpus // tensor_parallel_size

        self.workers: List[ray.actor.ActorHandle] = []
        self.tokenizer = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of Ray SGLang actors."""
        if self._initialized:
            return

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Compute logical GPU IDs
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            physical_to_logical = {gpu: idx for idx, gpu in enumerate(available_gpus)}
        else:
            physical_to_logical = {gpu: gpu for gpu in self.gpu_ids}

        # Group GPUs by tensor_parallel_size
        gpu_groups = []
        for i in range(0, self.num_gpus, self.tensor_parallel_size):
            group = self.gpu_ids[i : i + self.tensor_parallel_size]
            gpu_groups.append(group)

        # Port ranges across services (8 workers max):
        #   Student: [10000, 22000), FSF: [22000, 34000), CorrectJudge: [34000, 46000)
        base_port_start = 34000
        port_spacing = 1500
        sglang_port_range = 1000

        def is_port_available(port: int) -> bool:
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
            """Check that the full port range [base_port, base_port + 1000) is available."""
            if not is_port_available(base_port):
                return False
            if not is_port_available(base_port + 500):
                return False
            for offset in range(100, 1000, 50):
                if not is_port_available(base_port + offset):
                    return False
            return True

        assigned_ports = []
        for group_idx in range(len(gpu_groups)):
            range_start = base_port_start + group_idx * port_spacing
            range_end = range_start + port_spacing
            max_base_port = range_end - sglang_port_range

            port = None
            for candidate in range(range_start, max_base_port + 1):
                if is_port_range_available(candidate):
                    port = candidate
                    break

            if port is None:
                raise RuntimeError(
                    f"Cannot find available CorrectJudge port range for worker {group_idx} "
                    f"in range [{range_start}, {max_base_port}]"
                )

            assigned_ports.append(port)
            logger.info(
                f"[SGLangCorrectJudgeClient] Worker {group_idx}: using port {port} "
                f"(range: [{range_start}, {range_end}))"
            )

        # Create Ray actors
        for group_idx, gpu_group in enumerate(gpu_groups):
            # gpu_group contains logical device indices. Convert to physical GPU IDs.
            if cuda_visible_devices:
                physical_gpu_ids = [available_gpus[g] for g in gpu_group]
            else:
                physical_gpu_ids = list(gpu_group)
            # Actor-local logical IDs are always 0, 1, ..., tp_size-1
            logical_gpu_ids = list(range(len(gpu_group)))
            base_port = assigned_ports[group_idx]

            worker = SGLangCorrectJudgeActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_ids=physical_gpu_ids,
                logical_gpu_ids=logical_gpu_ids,
                tensor_parallel_size=self.tensor_parallel_size,
                mem_fraction_static=self.mem_fraction_static,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                base_port=base_port,
            )
            self.workers.append(worker)
            logger.info(f"[SGLangCorrectJudgeClient] Created worker {group_idx} on GPUs {gpu_group} with port={base_port}")

        # Initialize workers in parallel
        logger.info(f"[SGLangCorrectJudgeClient] Initializing {len(self.workers)} workers in parallel...")
        init_futures = []
        for i, worker in enumerate(self.workers):
            # Trigger lazy engine initialization
            init_futures.append(worker.compute_correct_judge_batch.remote(
                ["test question"], ["test reasoning"], ["test solution"], self.micro_batch_size
            ))

        try:
            results = ray.get(init_futures, timeout=600)
            logger.info(f"[SGLangCorrectJudgeClient] All {len(self.workers)} workers initialized")
        except Exception as e:
            logger.error(f"[SGLangCorrectJudgeClient] Parallel init failed: {e}")
            raise

        self._initialized = True

    def cleanup(self):
        """Release KV Cache for all workers."""
        if not self._initialized:
            return

        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangCorrectJudgeClient] Released KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeClient] Failed to release memory: {e}")

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def resume_memory(self):
        """Resume KV Cache for all workers."""
        if not self._initialized:
            return

        try:
            futures = [w.resume_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangCorrectJudgeClient] Resumed KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeClient] Failed to resume memory: {e}")

    def shutdown(self):
        """Shutdown all workers."""
        if not self._initialized:
            return

        try:
            futures = [w.shutdown.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            self.workers = []
            self._initialized = False
            logger.info("[SGLangCorrectJudgeClient] All workers shutdown")
        except Exception as e:
            logger.warning(f"[SGLangCorrectJudgeClient] Shutdown error: {e}")

    def compute_correct_judge(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        solutions: List[str],
        save_prompts: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute correct judge results for rewrite CoTs.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoT reasonings
            solutions: List of ground truth solutions
            save_prompts: If True, include saved prompts in return

        Returns:
            Dict with:
                - is_correct: List of boolean results (True=correct, False=incorrect, None=parse failure)
                - saved_prompts: (optional) Dict with saved prompts if requested
        """
        self.initialize()
        self.resume_memory()

        batch_size = len(questions)

        # Split data across workers
        question_chunks = self._split_into_chunks(questions, self.num_actors)
        rewrite_cot_chunks = self._split_into_chunks(rewrite_cots, self.num_actors)
        solution_chunks = self._split_into_chunks(solutions, self.num_actors)

        # Compute judge results
        logger.info(f"[SGLangCorrectJudgeClient] Computing correct judge for {len(questions)} samples "
                    f"(micro_batch_size={self.micro_batch_size}, save_prompts={save_prompts})...")
        futures = []
        for worker_idx, (question_chunk, rewrite_cot_chunk, solution_chunk) in enumerate(zip(question_chunks, rewrite_cot_chunks, solution_chunks)):
            if question_chunk:
                future = self.workers[worker_idx].compute_correct_judge_batch.remote(
                    question_chunk, rewrite_cot_chunk, solution_chunk, self.micro_batch_size, save_prompts
                )
                futures.append((worker_idx, len(question_chunk), future))

        # Aggregate results
        is_correct_results = [None] * batch_size
        saved_prompts_out = None if save_prompts else None  # Aggregated saved prompts
        for worker_idx, chunk_size, future in futures:
            chunk_results = ray.get(future)
            if save_prompts:
                chunk_is_correct, chunk_prompts = chunk_results
                # Merge saved prompts from each worker
                if saved_prompts_out is None:
                    saved_prompts_out = chunk_prompts
                elif chunk_prompts is not None:
                    for key, values in chunk_prompts.items():
                        if key not in saved_prompts_out:
                            saved_prompts_out[key] = []
                        saved_prompts_out[key].extend(values)
            else:
                chunk_is_correct, _ = chunk_results

            start_idx = sum(len(question_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_is_correct):
                is_correct_results[start_idx + i] = result

        result = {
            "is_correct": is_correct_results,
        }
        if save_prompts and saved_prompts_out is not None:
            result["saved_prompts"] = saved_prompts_out
        return result

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


# Global singleton
_correct_judge_client: Optional[SGLangCorrectJudgeClient] = None
_correct_judge_client_kwargs: Optional[dict] = None


def get_correct_judge_client(
    model_path: str = os.environ.get("CORRECT_JUDGE_MODEL_PATH", ""),
    **kwargs,
) -> SGLangCorrectJudgeClient:
    """Get or create SGLang Correct Judge Client singleton."""
    if not model_path:
        raise ValueError("Correct Judge model path is required.")
    global _correct_judge_client, _correct_judge_client_kwargs
    if _correct_judge_client is None:
        _correct_judge_client_kwargs = kwargs.copy()
        _correct_judge_client = SGLangCorrectJudgeClient(model_path=model_path, **kwargs)
    return _correct_judge_client


def cleanup_correct_judge_client():
    """Cleanup the global Correct Judge client."""
    global _correct_judge_client, _correct_judge_client_kwargs
    if _correct_judge_client is not None:
        _correct_judge_client.shutdown()
        _correct_judge_client = None
        _correct_judge_client_kwargs = None

        import gc
        import time
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        logger.info("[CORRECT_JUDGE] Client cleanup completed")
