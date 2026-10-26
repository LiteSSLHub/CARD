"""
Training Logger for CoT Rewriter GRPO

Provides structured logging for training process using standard logging library.
Logs are only printed at specific key points:
- Rollout start/end: GPU memory, prompt token stats, sample prompts/responses
- Reward start/end: GPU memory, data stats
- Backward start/end: GPU memory

Usage:
    from verl.experimental.reward.cot_rewriter.training_logger import get_logger
    
    logger = get_logger("trainer")
    logger.log_gpu_memory("[ROLLOUT_START]")
    logger.log_prompt_stats(prompts)
"""

import logging
import os
import random
import subprocess
import sys
from typing import List, Optional, Any

import torch

# Global logger instances
_loggers = {}


def get_logger(name: str = "cot_rewriter") -> "TrainingLogger":
    """Get or create a TrainingLogger instance."""
    if name not in _loggers:
        _loggers[name] = TrainingLogger(name)
    return _loggers[name]


class TrainingLogger:
    """Structured logger for training process."""
    
    def __init__(self, name: str, level: Optional[str] = None):
        """Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (default from env VERL_LOGGING_LEVEL or INFO)
        """
        self.logger = logging.getLogger(name)
        
        # Set level
        if level is None:
            level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Add handler if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(self.logger.level)
            
            # Format: timestamp | level | name | message
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal log method."""
        self.logger.log(level, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def log_gpu_memory(self, prefix: str = ""):
        """Log GPU memory usage and utilization for all available GPUs using nvidia-smi.

        Uses ORIGINAL_CUDA_VISIBLE_DEVICES to get physical GPU indices.
        """
        import subprocess
        import os

        try:
            # Try to get original CUDA_VISIBLE_DEVICES (set by launch script)
            cuda_visible = os.environ.get("ORIGINAL_CUDA_VISIBLE_DEVICES", "")
            if not cuda_visible:
                # Fallback to regular CUDA_VISIBLE_DEVICES
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

            # Parse CUDA_VISIBLE_DEVICES to get physical GPU IDs
            if cuda_visible:
                # Handle formats like "0,1" or "0,1,2,3"
                physical_gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            else:
                # If not set, query all GPUs
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                gpu_count = int(result.stdout.strip())
                physical_gpu_ids = list(range(gpu_count))

            # Query nvidia-smi for all physical GPUs
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')

            # Build a dict of physical GPU info
            gpu_info_dict = {}
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 4:
                    phys_idx = int(parts[0].strip())
                    used_mb = int(parts[1].strip())
                    total_mb = int(parts[2].strip())
                    util_gpu = int(parts[3].strip())
                    gpu_info_dict[phys_idx] = {
                        'used_mb': used_mb,
                        'total_mb': total_mb,
                        'util': util_gpu
                    }

            # Format output for visible GPUs
            gpu_info = []
            for i, phys_id in enumerate(physical_gpu_ids):
                if phys_id in gpu_info_dict:
                    info = gpu_info_dict[phys_id]
                    used_gb = info['used_mb'] / 1024
                    total_gb = info['total_mb'] / 1024
                    free_gb = total_gb - used_gb
                    gpu_info.append(
                        f"GPU{i}(phys{phys_id}):{used_gb:.1f}G/{total_gb:.1f}G ({info['util']}%) "
                        f"free:{free_gb:.1f}G"
                    )

            if gpu_info:
                self.info(f"{prefix} GPU_MEMORY {' | '.join(gpu_info)}")
            else:
                self.info(f"{prefix} GPU not available")
        except Exception as e:
            self.info(f"{prefix} GPU not available (failed to query: {e})")
    
    def log_prompt_stats(
        self, 
        prompts: List[torch.Tensor], 
        prefix: str = "[ROLLOUT]",
        sample_count: int = 2,
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None
    ):
        """Log prompt token statistics and sample prompts.
        
        Args:
            prompts: List of prompt token tensors
            prefix: Prefix for log messages
            sample_count: Number of random samples to print
            max_tokens_to_print: Max number of tokens to print per sample
            tokenizer: Tokenizer to decode prompts (optional)
        """
        if not prompts:
            self.info(f"{prefix} No prompts to log")
            return
        
        # Compute token length stats
        lengths = [p.shape[0] if len(p.shape) > 0 else len(p) for p in prompts]
        max_len = max(lengths)
        min_len = min(lengths)
        mean_len = sum(lengths) / len(lengths)
        
        self.info(
            f"{prefix} PROMPT_TOKENS max={max_len} min={min_len} mean={mean_len:.1f} "
            f"count={len(prompts)}"
        )
        
        # Randomly sample and print prompts
        if sample_count > 0 and tokenizer is not None:
            sample_indices = random.sample(
                range(len(prompts)),
                min(sample_count, len(prompts))
            )
            for idx in sample_indices:
                prompt_tokens = prompts[idx]

                # Get pad_token_id for skipping left padding
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                # Ensure tensor is on CPU for processing
                if hasattr(prompt_tokens, 'device') and prompt_tokens.device.type != 'cpu':
                    prompt_tokens = prompt_tokens.cpu()

                # Convert to list for easier manipulation
                token_list = prompt_tokens.tolist()

                # Find first non-pad token (for left-padded sequences)
                first_non_pad = 0
                for tok_idx, tok in enumerate(token_list):
                    if tok != pad_token_id:
                        first_non_pad = tok_idx
                        break

                # Take actual content tokens (skip left padding)
                tokens_to_decode = token_list[first_non_pad:first_non_pad + max_tokens_to_print]

                # Try to decode
                try:
                    display_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)
                    # If still empty, show token count info
                    if not display_text.strip():
                        display_text = f"<empty decode, {len(tokens_to_decode)} tokens: {tokens_to_decode[:10]}...>"
                except Exception as e:
                    display_text = f"<decode error: {e}, tokens: {tokens_to_decode[:10]}>"

                # Truncate long prompts for display
                display_text = display_text[:300] if len(display_text) > 300 else display_text
                self.info(f"{prefix} SAMPLE_PROMPT[{idx}]: {display_text}")
    
    def log_response_samples(
        self,
        responses: List[torch.Tensor],
        prefix: str = "[ROLLOUT]",
        sample_count: int = 2,
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None
    ):
        """Log random response samples.
        
        Args:
            responses: List of response token tensors
            prefix: Prefix for log messages
            sample_count: Number of random samples to print
            max_tokens_to_print: Max number of tokens to print per sample
            tokenizer: Tokenizer to decode responses
        """
        if not responses or tokenizer is None:
            return
        
        sample_indices = random.sample(
            range(len(responses)),
            min(sample_count, len(responses))
        )
        for idx in sample_indices:
            response_tokens = responses[idx]
            tokens_to_decode = response_tokens[:max_tokens_to_print]
            response_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
            self.info(f"{prefix} SAMPLE_RESPONSE[{idx}]: {response_text[:200]}...")
    
    def log_data_stats(
        self,
        data: Any,
        prefix: str = "[REWARD]",
        tokenizer: Optional[Any] = None,
        field: str = "prompts"
    ):
        """Log data statistics (for reward computation).
        
        Args:
            data: Data object containing tensors
            prefix: Prefix for log messages
            tokenizer: Tokenizer (optional)
            field: Field name to compute stats for
        """
        try:
            if hasattr(data, 'batch') and field in data.batch:
                tensor = data.batch[field]
                # Compute sequence lengths (non-padding tokens)
                if tensor.dim() >= 2:
                    # [batch_size, seq_len] - count non-zero tokens
                    lengths = (tensor != 0).sum(dim=-1).tolist()
                else:
                    lengths = [len(tensor)]
                
                max_len = max(lengths)
                min_len = min(lengths)
                mean_len = sum(lengths) / len(lengths)
                
                self.info(
                    f"{prefix} DATA_TOKENS field={field} max={max_len} min={min_len} "
                    f"mean={mean_len:.1f} count={len(lengths)}"
                )
            else:
                self.debug(f"{prefix} Cannot compute stats for field={field}")
        except Exception as e:
            self.debug(f"{prefix} Failed to compute data stats: {e}")
    
    def log_tensor_stats(
        self,
        tensor: torch.Tensor,
        name: str,
        prefix: str = "[ROLLOUT]"
    ):
        """Log tensor statistics (min, max, mean, std).
        
        Args:
            tensor: Input tensor
            name: Name of the tensor for logging
            prefix: Prefix for log messages
        """
        try:
            if tensor is None:
                self.debug(f"{prefix} {name} is None")
                return
            
            # Handle different tensor dimensions
            if tensor.numel() == 0:
                self.info(f"{prefix} {name} stats: empty tensor")
                return
            
            # Convert to float for stats calculation
            tensor_float = tensor.float()
            
            min_val = tensor_float.min().item()
            max_val = tensor_float.max().item()
            mean_val = tensor_float.mean().item()
            std_val = tensor_float.std().item()
            
            self.info(
                f"{prefix} {name}_STATS min={min_val:.4f} max={max_val:.4f} "
                f"mean={mean_val:.4f} std={std_val:.4f} shape={list(tensor.shape)}"
            )
        except Exception as e:
            self.debug(f"{prefix} Failed to compute {name} stats: {e}")
    
    def log_sequence_lengths(
        self,
        sequences: torch.Tensor,
        pad_token_id: int = 0,
        name: str = "sequences",
        prefix: str = "[ROLLOUT]"
    ):
        """Log sequence length statistics (vLLM-style).
        
        Args:
            sequences: Token sequences [batch_size, seq_len]
            pad_token_id: Padding token ID
            name: Name for logging
            prefix: Prefix for log messages
        """
        try:
            if sequences is None:
                return
            
            batch_size = sequences.size(0)
            # Count non-padding tokens per sequence
            lengths = []
            for i in range(batch_size):
                seq = sequences[i]
                # Find first pad token from the end
                non_pad_mask = (seq != pad_token_id)
                length = non_pad_mask.sum().item()
                lengths.append(length)
            
            max_len = max(lengths)
            min_len = min(lengths)
            mean_len = sum(lengths) / len(lengths)
            
            self.info(
                f"{prefix} {name}_LENGTHS max={max_len} min={min_len} "
                f"mean={mean_len:.1f} count={len(lengths)}"
            )
        except Exception as e:
            self.debug(f"{prefix} Failed to compute {name} lengths: {e}")
    
    def log_rewards(
        self,
        rewards: torch.Tensor,
        prefix: str = "[REWARD]"
    ):
        """Log reward statistics (vLLM-style).
        
        Args:
            rewards: Reward tensor
            prefix: Prefix for log messages
        """
        try:
            if rewards is None:
                return
            
            rewards_float = rewards.float()
            min_reward = rewards_float.min().item()
            max_reward = rewards_float.max().item()
            mean_reward = rewards_float.mean().item()
            std_reward = rewards_float.std().item()
            
            self.info(
                f"{prefix} REWARD_STATS min={min_reward:.4f} max={max_reward:.4f} "
                f"mean={mean_reward:.4f} std={std_reward:.4f} count={rewards.numel()}"
            )
        except Exception as e:
            self.debug(f"{prefix} Failed to compute reward stats: {e}")
    
    def log_response_preview(
        self,
        responses: torch.Tensor,
        prefix: str = "[REWARD]",
        tokenizer: Optional[Any] = None,
        num_tokens: int = 10
    ):
        """Log first few tokens of responses (excluding padding).
        
        Args:
            responses: Response token tensor [batch_size, seq_len]
            prefix: Prefix for log messages
            tokenizer: Tokenizer to decode tokens
            num_tokens: Number of tokens to print
        """
        if responses is None or tokenizer is None:
            return
        
        try:
            batch_size, seq_len = responses.shape
            
            # Get pad_token_id from tokenizer, default to 0
            pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
            if pad_token_id is None:
                pad_token_id = 0
            
            self.info(f"{prefix} RESPONSE_PREVIEW shape={responses.shape}, pad_id={pad_token_id}:")
            
            for i in range(min(3, batch_size)):  # Show first 3 samples
                response_ids = responses[i].tolist()
                
                # Find first non-padding token position
                first_non_pad_pos = 0
                while first_non_pad_pos < len(response_ids) and response_ids[first_non_pad_pos] == pad_token_id:
                    first_non_pad_pos += 1
                
                # Find last non-padding token position
                last_non_pad_pos = len(response_ids) - 1
                while last_non_pad_pos >= 0 and response_ids[last_non_pad_pos] == pad_token_id:
                    last_non_pad_pos -= 1
                
                actual_len = max(0, last_non_pad_pos - first_non_pad_pos + 1)
                
                # Extract actual content (skip left padding)
                content_ids = response_ids[first_non_pad_pos:first_non_pad_pos + num_tokens]
                
                if not content_ids:
                    self.info(f"  [{i}] (empty response, all padding)")
                    continue
                
                # Try to decode
                try:
                    text = tokenizer.decode(content_ids, skip_special_tokens=True)
                    text_compact = text[:60]
                    self.info(f"  [{i}] text: {text_compact}...")
                except Exception as decode_err:
                    self.info(f"  [{i}] ids={content_ids} (decode error: {decode_err})")
        except Exception as e:
            self.debug(f"{prefix} Failed to log response preview: {e}")
    
    def log_per_sample_prompts(
        self,
        prompts: List[torch.Tensor],
        prefix: str = "[ROLLOUT]",
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None,
        current_idx: int = 0,
        total: int = 0
    ):
        """Log each prompt with its length and first N tokens.
        
        Args:
            prompts: List of prompt token tensors
            prefix: Prefix for log messages
            max_tokens_to_print: Max number of tokens to decode and print
            tokenizer: Tokenizer to decode prompts
            current_idx: Current batch starting index (for progress)
            total: Total number of samples (for progress)
        """
        if not prompts:
            self.info(f"{prefix} No prompts to log")
            return
        
        # Progress header
        if total > 0:
            progress_pct = 100.0 * current_idx / total if total > 0 else 0
            self.info(f"{prefix} PROGRESS [{current_idx}/{total}] ({progress_pct:.1f}%) - Logging {len(prompts)} prompts")
        else:
            self.info(f"{prefix} PROGRESS - Logging {len(prompts)} prompts")
        
        # Log each prompt
        for i, prompt_tokens in enumerate(prompts):
            global_idx = current_idx + i

            # Get prompt length
            prompt_len = prompt_tokens.shape[0] if len(prompt_tokens.shape) > 0 else len(prompt_tokens)

            # Decode first max_tokens_to_print tokens
            if tokenizer is not None:
                # Get pad_token_id for skipping left padding
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                # Skip left padding: find first non-pad token
                tokens_for_decode = prompt_tokens
                if hasattr(tokens_for_decode, 'device') and tokens_for_decode.device.type != 'cpu':
                    tokens_for_decode = tokens_for_decode.cpu()

                # Convert to list for easier manipulation
                token_list = tokens_for_decode.tolist()

                # Find first non-pad token (for left-padded sequences)
                first_non_pad = 0
                for idx, tok in enumerate(token_list):
                    if tok != pad_token_id:
                        first_non_pad = idx
                        break

                # Start from first non-pad token, take up to max_tokens_to_print
                actual_content_tokens = token_list[first_non_pad:first_non_pad + max_tokens_to_print]

                try:
                    prompt_text = tokenizer.decode(actual_content_tokens, skip_special_tokens=False)
                    # Truncate long text
                    if len(prompt_text) > 300:
                        prompt_text = prompt_text[:300] + "..."
                    self.info(f"{prefix} PROMPT[{global_idx}] len={prompt_len}: {prompt_text}")
                except Exception as e:
                    # Fallback to token IDs
                    token_ids = actual_content_tokens[:20]
                    self.info(f"{prefix} PROMPT[{global_idx}] len={prompt_len} tokens={token_ids}... (decode error: {e})")
            else:
                # No tokenizer, just log length
                self.info(f"{prefix} PROMPT[{global_idx}] len={prompt_len}")
    
    def log_per_sample_responses(
        self,
        responses: torch.Tensor,
        prefix: str = "[ROLLOUT]",
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None,
        current_idx: int = 0,
        total: int = 0,
        pad_token_id: int = 0,
        max_samples: int = 2
    ):
        """Log each response with its length and first N tokens.
        
        Args:
            responses: Response token tensor [batch_size, seq_len]
            prefix: Prefix for log messages
            max_tokens_to_print: Max number of tokens to decode and print
            tokenizer: Tokenizer to decode responses
            current_idx: Current batch starting index (for progress)
            total: Total number of samples (for progress)
            pad_token_id: Padding token ID
            max_samples: Maximum number of samples to print (default: 2)
        """
        if responses is None:
            self.info(f"{prefix} No responses to log")
            return
        
        batch_size = responses.shape[0]
        
        # Progress header
        if total > 0:
            progress_pct = 100.0 * current_idx / total if total > 0 else 0
            self.info(f"{prefix} PROGRESS [{current_idx}/{total}] ({progress_pct:.1f}%) - Logging {min(max_samples, batch_size)}/{batch_size} responses")
        else:
            self.info(f"{prefix} PROGRESS - Logging {min(max_samples, batch_size)}/{batch_size} responses")
        
        # Log each response (limit to max_samples)
        for i in range(min(batch_size, max_samples)):
            global_idx = current_idx + i
            response_ids = responses[i]
            
            # Calculate actual length (excluding padding)
            if hasattr(response_ids, 'tolist'):
                ids_list = response_ids.tolist()
            else:
                ids_list = list(response_ids)
            
            # Find actual length (first non-pad from start)
            actual_len = len(ids_list)
            for j, token_id in enumerate(ids_list):
                if token_id == pad_token_id:
                    actual_len = j
                    break
            
            # Decode first max_tokens_to_print tokens
            if tokenizer is not None:
                tokens_to_decode = response_ids[:max_tokens_to_print]
                if hasattr(tokens_to_decode, 'device') and tokens_to_decode.device.type != 'cpu':
                    tokens_to_decode = tokens_to_decode.cpu()
                
                try:
                    response_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
                    # Truncate long text
                    if len(response_text) > 300:
                        response_text = response_text[:300] + "..."
                    self.info(f"{prefix} RESPONSE[{global_idx}] len={actual_len}: {response_text}")
                except Exception as e:
                    # Fallback to token IDs
                    token_ids = tokens_to_decode.tolist()[:20]
                    self.info(f"{prefix} RESPONSE[{global_idx}] len={actual_len} tokens={token_ids}... (decode error: {e})")
            else:
                # No tokenizer, just log length
                self.info(f"{prefix} RESPONSE[{global_idx}] len={actual_len}")
    
    def log_per_sample_rewards(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        rewards: List[float],
        prefix: str = "[REWARD]",
        current_idx: int = 0,
        total: int = 0,
        max_text_len: int = 100,
        max_samples: int = 2
    ):
        """Log each sample's reward information including prompt and reward value.
        
        Args:
            questions: List of questions (prompts)
            original_cots: List of original CoTs
            rewrite_cots: List of rewritten CoTs
            rewards: List of reward values
            prefix: Prefix for log messages
            current_idx: Current batch starting index (for progress)
            total: Total number of samples (for progress)
            max_text_len: Max length of text to display
            max_samples: Maximum number of samples to print (default: 2)
        """
        if not questions:
            self.info(f"{prefix} No reward data to log")
            return
        
        batch_size = len(questions)
        
        # Progress header
        if total > 0:
            progress_pct = 100.0 * current_idx / total if total > 0 else 0
            self.info(f"{prefix} PROGRESS [{current_idx}/{total}] ({progress_pct:.1f}%) - Logging {min(max_samples, batch_size)}/{batch_size} rewards")
        else:
            self.info(f"{prefix} PROGRESS - Logging {min(max_samples, batch_size)}/{batch_size} rewards")
        
        # Log each sample (limit to max_samples)
        for i in range(min(batch_size, max_samples)):
            global_idx = current_idx + i
            
            question = questions[i] if i < len(questions) else ""
            original_cot = original_cots[i] if i < len(original_cots) else ""
            rewrite_cot = rewrite_cots[i] if i < len(rewrite_cots) else ""
            reward = rewards[i] if i < len(rewards) else 0.0
            
            # Truncate texts for display
            question_display = question[:max_text_len] if question else "<empty>"
            original_display = original_cot[:max_text_len] if original_cot else "<empty>"
            rewrite_display = rewrite_cot[:max_text_len] if rewrite_cot else "<empty>"
            
            self.info(f"{prefix} SAMPLE[{global_idx}] reward={reward:.4f}")
            self.info(f"{prefix}   QUESTION: {question_display}{'...' if len(question) > max_text_len else ''}")
            self.info(f"{prefix}   ORIGINAL_CoT: {original_display}{'...' if len(original_cot) > max_text_len else ''}")
            self.info(f"{prefix}   REWRITE_CoT: {rewrite_display}{'...' if len(rewrite_cot) > max_text_len else ''}")


# Convenience function for quick logging
def log_gpu_memory(prefix: str = ""):
    """Quick GPU memory logging."""
    get_logger().log_gpu_memory(prefix)


def log_prompt_stats(prompts: List[torch.Tensor], **kwargs):
    """Quick prompt stats logging."""
    get_logger().log_prompt_stats(prompts, **kwargs)
