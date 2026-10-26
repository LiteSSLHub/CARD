"""
FSF (Failure Step Fraction) Judge - SGLang Version

Computes FSF = failed_nodes / total_nodes from DOT graph format.
Failed nodes are those with fillcolor=lightpink.

Uses Ray actors for multi-GPU parallelism. Supports two modes:
1. TP=1: Each GPU runs a separate SGLang instance (Data Parallelism)
2. TP>1: Multiple GPUs share one SGLang instance (Tensor Parallelism)
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

# Logger for SGLang FSF reward
logger = get_logger("sglang_fsf_reward")


@ray.remote
class SGLangFSFActor:
    """
    Ray actor that runs FSF Judge SGLang on one or more GPUs.
    
    Generates DOT format graph and parses FSF from it.
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
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        max_new_tokens: int = 8000,
        base_port: int = 40000,  # Base port for this worker
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
        worker_name = f"SGLangFSFActor-{gpu_ids}"

        # Set CUDA_VISIBLE_DEVICES to physical GPU IDs for this actor
        cuda_devices_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Set unique MASTER_PORT to avoid torch.distributed TCP rendezvous port conflicts
        # when multiple FSF workers initialize in parallel on the same machine.
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

        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Creating SGLang Engine...")
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
        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Engine created in {elapsed:.2f}s")

    def _ensure_event_loop(self):
        """Ensure an event loop exists for the current thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)


    def _build_fsf_prompt(self, reasoning: str) -> str:
        """
        Build FSF prompt with proper format.

        Prompt Template (from build_fsf_judge_graph.py):
        - System: "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."
        - User: GRAPH_GENERATION_PROMPT + Reasoning trace

        The FSF model will generate a DOT format graph representing the reasoning process.
        """
        GRAPH_GENERATION_PROMPT = """Parse the reasoning trace into a Graphviz diagram. Focus on these essentials:

Node Rules:
- One node per distinct reasoning step
- 'fillcolor=lightblue': Successful reasoning steps
- 'fillcolor=lightpink': Failed attempts

Edge Rules:
- Connect node A → node B if the information or insight from A is actually used to construct the reasoning in B; branch new attempts from their starting ancestor, not from dead ends.

Requirements:
- Use 'rankdir=TB'
- Include ALL attempts (including failures), do not miss any steps in the reasoning.
- ALWAYS start with a "problem statement" node
- ALWAYS end with a "final answer" node
- Do NOT reorder or reorganize the reasoning flow

Generate complete Graphviz DOT code in dot blocks."""
        # Process <rewrite_reasoning> tag
        if '<rewrite_reasoning>' in reasoning:
            reasoning = reasoning.split('<rewrite_reasoning>')[-1]
        if '</rewrite_reasoning>' in reasoning:
            reasoning = reasoning.split('</rewrite_reasoning>')[0]
            
        reasoning = reasoning.strip()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams.",
            },
            {
                "role": "user",
                "content": f"{GRAPH_GENERATION_PROMPT}\\n\\nReasoning trace:\\n{reasoning}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt
    def _parse_fsf_from_dot(self, dot_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse FSF value from DOT graph format output.
        
        FSF = failed_nodes / total_nodes
        Failed nodes have fillcolor=lightpink
        
        Args:
            dot_output: Raw model output text (DOT format)
            
        Returns:
            Dict with keys 'fsf', 'failed_nodes', 'total_nodes', or None if parsing fails
        """
        try:
            import pydot

            # Strip leading newlines and markdown code blocks
            # Like analyze_rg_fsf.py, directly parse DOT without regex
            cleaned_output = dot_output.lstrip('\n\n').strip('```')

            # Directly parse with pydot (no regex matching needed)
            graphs = pydot.graph_from_dot_data(cleaned_output)

            if not graphs or len(graphs) == 0:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Failed to parse DOT graph")
                return None

            graph = graphs[0]
            nodes = graph.get_nodes() or []
            
            node_count = len(nodes)
            if node_count == 0:
                return {"fsf": 0.0, "failed_nodes": 0, "total_nodes": 0}

            # Count failed nodes (fillcolor=lightpink)
            failed_nodes = 0
            for node in nodes:
                if node is None:
                    continue
                try:
                    fillcolor = node.get('fillcolor')
                    if fillcolor is not None:
                        fillcolor_str = str(fillcolor).strip('"').lower()
                        if fillcolor_str == 'lightpink':
                            failed_nodes += 1
                except:
                    continue

            fsf = failed_nodes / node_count
            # Log FSF parsing details at INFO level for visibility
            logger.info(f"[SGLangFSFActor-{self.gpu_ids}] FSF={fsf:.4f} ({failed_nodes}/{node_count} nodes, {node_count-failed_nodes} successful)")
            return {"fsf": fsf, "failed_nodes": failed_nodes, "total_nodes": node_count}

        except ImportError:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] pydot not installed")
            return None
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] DOT parsing error: {e}")
            return None

    def compute_fsf_batch(
        self,
        reasonings: List[str],
        micro_batch_size: int = 4,
        return_dot_outputs: bool = False,  # Whether to return raw DOT outputs
        save_prompts: bool = False,  # Whether to save request prompts
    ) -> List[Optional[Dict[str, Any]]] | Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]], Dict[str, List[str]]]:
        """
        Batch computation of FSF values with micro-batching to avoid OOM.

        Args:
            reasonings: List of reasoning texts to evaluate
            micro_batch_size: Max number of prompts per SGLang call (to control memory)
            return_dot_outputs: If True, return (fsf_values, dot_outputs) tuple
            save_prompts: If True, also return saved prompts dict

        Returns:
            If save_prompts=False and return_dot_outputs=False: List of dicts with keys 'fsf', 'failed_nodes', 'total_nodes' (None for unparseable outputs)
            If save_prompts=False and return_dot_outputs=True: Tuple of (fsf_values, dot_outputs)
            If save_prompts=True: Tuple of (fsf_values, dot_outputs or None, saved_prompts dict)
        """
        # Initialize engine on first use
        self._init_engine()
        self._ensure_event_loop()

        # Build prompts
        prompts = []
        for reasoning in reasonings:
            # Truncate long reasoning to avoid OOM
            reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning
            prompt = self._build_fsf_prompt(reasoning_truncated)
            prompts.append(prompt)

        # Save prompts for debugging if requested
        saved_prompts = {}
        if save_prompts:
            saved_prompts["fsf_prompts"] = prompts

        # Process in micro-batches to avoid OOM
        fsf_values = []
        dot_outputs = [] if return_dot_outputs else None
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
                        "presence_penalty": 1.0,
                    },
                )

                # Parse FSF from each output
                for output in outputs:
                    if isinstance(output, dict):
                        generated_text = output.get("text", "")
                    else:
                        generated_text = str(output)
                    fsf_values.append(self._parse_fsf_from_dot(generated_text))
                    if return_dot_outputs:
                        dot_outputs.append(generated_text)

            except Exception as e:
                logger.error(f"[SGLangFSFActor-{self.gpu_ids}] Micro-batch compute error at batch {i}: {e}")
                # Fill remaining with None for this batch
                remaining = len(batch_prompts)
                fsf_values.extend([None] * remaining)
                if return_dot_outputs:
                    dot_outputs.extend([None] * remaining)

        if save_prompts:
            if return_dot_outputs:
                return fsf_values, dot_outputs, saved_prompts
            return fsf_values, None, saved_prompts
        elif return_dot_outputs:
            return fsf_values, dot_outputs
        return fsf_values

    def release_memory(self):
        """
        Completely destroy the SGLang Engine and release all GPU memory.

        Do not use release_memory_occupation() (has a CUDA error bug).
        Instead, directly shutdown the engine to release all resources.
        Recreate the engine via _init_engine() on next use.
        """
        if not self._engine_initialized:
            return

        # Fully shutdown the engine to release all memory, instead of trying to release KV Cache
        # Because release_memory_occupation() has a bug: RuntimeError: CUDA error: invalid argument
        try:
            if self.engine is not None:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
                del self.engine
                self.engine = None
                self._engine_initialized = False
                self._memory_released = False
                logger.debug(f"[SGLangFSFActor-{self.gpu_ids}] Engine destroyed, all GPU memory released")
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Error during engine shutdown: {e}")
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
                logger.debug(f"[SGLangFSFActor-{self.gpu_ids}] Resumed KV Cache")
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Failed to resume memory: {e}")

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
            logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Shutdown complete")
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Shutdown error: {e}")


class SGLangFSFClient:
    """
    Multi-GPU SGLang-based FSF Judge Client using Ray actors.
    
    Similar to SGLangStudentRewardClient architecture.
    Computes FSF (Failure Step Fraction) from DOT graph format.
    
    FSF = failed_nodes / total_nodes
    R_logic = 1 if FSF(rewrite) < FSF(original) else 0
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        max_new_tokens: int = 8000,
        micro_batch_size: int = 4,  # Micro batch size to avoid OOM
    ):
        """
        Initialize Multi-GPU SGLang FSF Judge Client.
        
        Args:
            model_path: Path to FSF Judge Model
            tensor_parallel_size: TP size per SGLang instance
            mem_fraction_static: GPU memory fraction
            dtype: Data type (float16 or bfloat16)
            max_model_len: Maximum context length
            devices: List of CUDA devices (auto-detected if None)
            temperature: Sampling temperature
            top_k: Sampling top-k
            top_p: Sampling top-p
            max_new_tokens: Max tokens to generate
            micro_batch_size: Micro batch size per worker to avoid OOM (default: 4)
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
        base_port_start = 22000
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
                    f"Cannot find available FSF port range for worker {group_idx} "
                    f"in range [{range_start}, {max_base_port}]"
                )

            assigned_ports.append(port)
            logger.info(
                f"[SGLangFSFClient] Worker {group_idx}: using port {port} "
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

            worker = SGLangFSFActor.options(num_gpus=0, num_cpus=1).remote(
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
            logger.info(f"[SGLangFSFClient] Created worker {group_idx} on GPUs {gpu_group} with port={base_port}")

        # Initialize workers in parallel
        logger.info(f"[SGLangFSFClient] Initializing {len(self.workers)} workers in parallel...")
        init_futures = []
        for i, worker in enumerate(self.workers):
            # Trigger lazy engine initialization
            init_futures.append(worker.compute_fsf_batch.remote(["test reasoning"], self.micro_batch_size))

        try:
            results = ray.get(init_futures, timeout=600)
            logger.info(f"[SGLangFSFClient] All {len(self.workers)} workers initialized")
        except Exception as e:
            logger.error(f"[SGLangFSFClient] Parallel init failed: {e}")
            raise

        self._initialized = True

    def cleanup(self):
        """Release KV Cache for all workers."""
        if not self._initialized:
            return

        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangFSFClient] Released KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangFSFClient] Failed to release memory: {e}")

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
            logger.info("[SGLangFSFClient] Resumed KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangFSFClient] Failed to resume memory: {e}")

    def shutdown(self):
        """Shutdown all workers and kill Ray actors to fully release ports."""
        if not self._initialized:
            return

        try:
            futures = [w.shutdown.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangFSFClient] All workers shutdown")
        except Exception as e:
            logger.warning(f"[SGLangFSFClient] Shutdown error: {e}")

        # Kill Ray actors to ensure ports are fully released
        for w in self.workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception:
                pass
        self.workers = []
        self._initialized = False

        # Wait briefly for ports to be released from TIME_WAIT
        import time
        time.sleep(2)

    def compute_rewrite_fsf_only(
        self,
        rewrite_cots: List[str],
        return_dot_outputs: bool = False,  # Whether to return raw DOT outputs
        save_prompts: bool = False,  # Whether to save request prompts
    ) -> Dict[str, Any]:
        """
        Compute FSF for rewrite CoTs only (no original CoT computation).

        This is optimized for cases where original_fsf is pre-computed in the dataset.

        Args:
            rewrite_cots: List of rewritten CoT reasonings
            return_dot_outputs: If True, include raw DOT outputs in return
            save_prompts: If True, include saved prompts in return

        Returns:
            Dict with:
                - rewrite_fsfs: FSF values for rewrite CoTs
                - rewrite_failed_nodes: Failed node counts for rewrite CoTs
                - rewrite_total_nodes: Total node counts for rewrite CoTs
                - rewrite_dot_outputs: (optional) Raw DOT outputs for rewrite CoTs
                - saved_prompts: (optional) Dict with saved prompts if requested
        """
        self.initialize()
        self.resume_memory()

        batch_size = len(rewrite_cots)

        # Split data across workers
        rewrite_chunks = self._split_into_chunks(rewrite_cots, self.num_actors)

        # Compute rewrite FSFs
        logger.info(f"[SGLangFSFClient] Computing FSF for {len(rewrite_cots)} rewrite CoTs only "
                    f"(micro_batch_size={self.micro_batch_size}, return_dot_outputs={return_dot_outputs}, save_prompts={save_prompts})...")
        rewrite_futures = []
        for worker_idx, chunk_cots in enumerate(rewrite_chunks):
            if chunk_cots:
                future = self.workers[worker_idx].compute_fsf_batch.remote(
                    chunk_cots, self.micro_batch_size, return_dot_outputs, save_prompts
                )
                rewrite_futures.append((worker_idx, len(chunk_cots), future))

        # Aggregate rewrite results
        rewrite_fsfs = [None] * batch_size
        rewrite_failed_nodes = [None] * batch_size
        rewrite_total_nodes = [None] * batch_size
        rewrite_dot_outputs = [None] * batch_size if return_dot_outputs else None
        saved_prompts_out = None if save_prompts else None  # Aggregated saved prompts
        for worker_idx, chunk_size, future in rewrite_futures:
            chunk_results = ray.get(future)
            if save_prompts:
                if return_dot_outputs:
                    chunk_fsfs, chunk_dots, chunk_prompts = chunk_results
                else:
                    chunk_fsfs, _, chunk_prompts = chunk_results
                    chunk_dots = None
                # Merge saved prompts from each worker
                if saved_prompts_out is None:
                    saved_prompts_out = chunk_prompts
                elif chunk_prompts is not None:
                    for key, values in chunk_prompts.items():
                        if key not in saved_prompts_out:
                            saved_prompts_out[key] = []
                        saved_prompts_out[key].extend(values)
            elif return_dot_outputs:
                chunk_fsfs, chunk_dots = chunk_results
            else:
                chunk_fsfs = chunk_results
                chunk_dots = None
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_fsfs):
                if result is not None:
                    rewrite_fsfs[start_idx + i] = result.get("fsf")
                    rewrite_failed_nodes[start_idx + i] = result.get("failed_nodes")
                    rewrite_total_nodes[start_idx + i] = result.get("total_nodes")
            if return_dot_outputs and chunk_dots is not None:
                for i, dot in enumerate(chunk_dots):
                    rewrite_dot_outputs[start_idx + i] = dot

        result = {
            "rewrite_fsfs": rewrite_fsfs,
            "rewrite_failed_nodes": rewrite_failed_nodes,
            "rewrite_total_nodes": rewrite_total_nodes,
        }
        if return_dot_outputs:
            result["rewrite_dot_outputs"] = rewrite_dot_outputs
        if save_prompts and saved_prompts_out is not None:
            result["saved_prompts"] = saved_prompts_out
        return result

    def compute_logic_reward_batch(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        return_dot_outputs: bool = False,  # Whether to return raw DOT outputs
    ) -> Dict[str, Any]:
        """
        Compute logic reward based on FSF comparison.

        R_logic = 1 if FSF(rewrite) < FSF(original) else 0

        Args:
            questions: List of problem texts
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings
            return_dot_outputs: If True, include raw DOT outputs in return

        Returns:
            Dict with:
                - original_fsfs: FSF values for original CoTs
                - original_failed_nodes: Failed node counts for original CoTs
                - original_total_nodes: Total node counts for original CoTs
                - rewrite_fsfs: FSF values for rewrite CoTs
                - rewrite_failed_nodes: Failed node counts for rewrite CoTs
                - rewrite_total_nodes: Total node counts for rewrite CoTs
                - logic_rewards: Binary rewards (1 if improved, 0 otherwise, None if unparseable)
                - original_dot_outputs: (optional) Raw DOT outputs for original CoTs
                - rewrite_dot_outputs: (optional) Raw DOT outputs for rewrite CoTs
        """
        self.initialize()
        self.resume_memory()

        batch_size = len(questions)

        # Split data across workers
        original_chunks = self._split_into_chunks(original_cots, self.num_actors)
        rewrite_chunks = self._split_into_chunks(rewrite_cots, self.num_actors)

        # Compute original FSFs
        logger.info(f"[SGLangFSFClient] Computing FSF for {len(original_cots)} original CoTs "
                    f"(micro_batch_size={self.micro_batch_size}, return_dot_outputs={return_dot_outputs})...")
        original_futures = []
        for worker_idx, chunk_cots in enumerate(original_chunks):
            if chunk_cots:
                future = self.workers[worker_idx].compute_fsf_batch.remote(
                    chunk_cots, self.micro_batch_size, return_dot_outputs
                )
                original_futures.append((worker_idx, len(chunk_cots), future))

        # Aggregate original results
        original_fsfs = [None] * batch_size
        original_failed_nodes = [None] * batch_size
        original_total_nodes = [None] * batch_size
        original_dot_outputs = [None] * batch_size if return_dot_outputs else None
        for worker_idx, chunk_size, future in original_futures:
            chunk_results = ray.get(future)
            if return_dot_outputs:
                chunk_fsfs, chunk_dots = chunk_results
            else:
                chunk_fsfs = chunk_results
                chunk_dots = None
            start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_fsfs):
                if result is not None:
                    original_fsfs[start_idx + i] = result.get("fsf")
                    original_failed_nodes[start_idx + i] = result.get("failed_nodes")
                    original_total_nodes[start_idx + i] = result.get("total_nodes")
            if return_dot_outputs and chunk_dots is not None:
                for i, dot in enumerate(chunk_dots):
                    original_dot_outputs[start_idx + i] = dot

        # Compute rewrite FSFs
        logger.info(f"[SGLangFSFClient] Computing FSF for {len(rewrite_cots)} rewrite CoTs "
                    f"(micro_batch_size={self.micro_batch_size}, return_dot_outputs={return_dot_outputs})...")
        rewrite_futures = []
        for worker_idx, chunk_cots in enumerate(rewrite_chunks):
            if chunk_cots:
                future = self.workers[worker_idx].compute_fsf_batch.remote(
                    chunk_cots, self.micro_batch_size, return_dot_outputs
                )
                rewrite_futures.append((worker_idx, len(chunk_cots), future))

        # Aggregate rewrite results
        rewrite_fsfs = [None] * batch_size
        rewrite_failed_nodes = [None] * batch_size
        rewrite_total_nodes = [None] * batch_size
        rewrite_dot_outputs = [None] * batch_size if return_dot_outputs else None
        for worker_idx, chunk_size, future in rewrite_futures:
            chunk_results = ray.get(future)
            if return_dot_outputs:
                chunk_fsfs, chunk_dots = chunk_results
            else:
                chunk_fsfs = chunk_results
                chunk_dots = None
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_fsfs):
                if result is not None:
                    rewrite_fsfs[start_idx + i] = result.get("fsf")
                    rewrite_failed_nodes[start_idx + i] = result.get("failed_nodes")
                    rewrite_total_nodes[start_idx + i] = result.get("total_nodes")
            if return_dot_outputs and chunk_dots is not None:
                for i, dot in enumerate(chunk_dots):
                    rewrite_dot_outputs[start_idx + i] = dot

        # Compute logic rewards as FSF difference
        # R_logic = FSF(original) - FSF(rewrite)
        # Positive means rewrite has lower FSF (fewer failed steps) -> better
        # This will be scaled by fsf_reward_scale in combined_reward.py
        logic_rewards = []
        for o_fsf, r_fsf in zip(original_fsfs, rewrite_fsfs):
            if o_fsf is None or r_fsf is None:
                # Unparseable, return 0 so it doesn't affect reward
                logic_rewards.append(0.0)
            else:
                # FSF difference: positive means improvement
                logic_rewards.append(o_fsf - r_fsf)

        result = {
            "original_fsfs": original_fsfs,
            "original_failed_nodes": original_failed_nodes,
            "original_total_nodes": original_total_nodes,
            "rewrite_fsfs": rewrite_fsfs,
            "rewrite_failed_nodes": rewrite_failed_nodes,
            "rewrite_total_nodes": rewrite_total_nodes,
            "logic_rewards": logic_rewards,
        }
        if return_dot_outputs:
            result["original_dot_outputs"] = original_dot_outputs
            result["rewrite_dot_outputs"] = rewrite_dot_outputs
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


# Backward compatibility: keep SGLangFSFJudge as alias
SGLangFSFJudge = SGLangFSFClient

# Global singleton for backward compatibility
_fsf_client: Optional[SGLangFSFClient] = None
_fsf_client_kwargs: Optional[dict] = None


def get_fsf_client(
    model_path: str = "./models/Qwen3-4B",
    **kwargs,
) -> SGLangFSFClient:
    """Get or create SGLang FSF Client singleton."""
    global _fsf_client, _fsf_client_kwargs
    if _fsf_client is None:
        _fsf_client_kwargs = kwargs.copy()
        _fsf_client = SGLangFSFClient(model_path=model_path, **kwargs)
    return _fsf_client


def cleanup_fsf_client():
    """Cleanup the global FSF client."""
    global _fsf_client, _fsf_client_kwargs
    if _fsf_client is not None:
        _fsf_client.shutdown()
        _fsf_client = None
        _fsf_client_kwargs = None
        
        import gc
        import time
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)
        
        logger.info("[FSF] Client cleanup completed")
