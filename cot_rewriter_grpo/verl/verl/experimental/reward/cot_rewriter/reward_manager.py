"""
CoT Rewriter Reward Manager for VERL

This reward manager properly interfaces with compute_cot_rewriter_reward function
which expects (data, tokenizer, ...) arguments instead of the standard
(data_source, solution_str, ground_truth, extra_info) interface.
"""

from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from verl.experimental.reward.cot_rewriter.training_logger import get_logger

# Training logger
logger = get_logger("reward_manager")


@register("cot_rewriter")
class CoTRewriterRewardManager:
    """Reward manager specifically designed for CoT Rewriter GRPO training.

    This manager directly calls compute_cot_rewriter_reward with the full DataProto
    and tokenizer, instead of the sample-by-sample interface used by NaiveRewardManager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "reward_model",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize CoT Rewriter Reward Manager.

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of samples to examine for debugging
            compute_score: The compute_cot_rewriter_reward function (wrapped with kwargs)
            reward_fn_key: Key for reward function in data
            max_resp_len: Maximum response length (required for overlong_buffer)
            overlong_buffer_cfg: Config for overlong response penalty
            **kwargs: Additional keyword arguments (passed to compute_cot_rewriter_reward)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.compute_score_kwargs = kwargs  # Store additional kwargs for compute_cot_rewriter_reward

        if self.overlong_buffer_cfg is not None:
            if self.max_resp_len is None:
                raise ValueError(
                    f"max_resp_len must be provided when overlong_buffer_cfg is set, but got None"
                )
            if self.max_resp_len < self.overlong_buffer_cfg.len:
                raise ValueError(
                    f"max_resp_len ({self.max_resp_len}) must be >= overlong_buffer_cfg.len "
                    f"({self.overlong_buffer_cfg.len})"
                )

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch of data.

        This method directly calls compute_cot_rewriter_reward with the full batch,
        which is more efficient for models that benefit from batch processing.

        Args:
            data: DataProto object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False: torch.Tensor of rewards
        """
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return data.batch["rm_scores"]

        if self.compute_score is None:
            raise ValueError(
                "compute_score (compute_cot_rewriter_reward) must be provided. "
                "Please configure custom_reward_function in your config."
            )

        # Call compute_cot_rewriter_reward directly with the full batch
        # The function is wrapped by _call_with_kwargs which merges reward_kwargs
        # We pass data, tokenizer, and additional kwargs (e.g., save_fsf_dot_outputs)
        result = self.compute_score(data=data, tokenizer=self.tokenizer, **self.compute_score_kwargs)

        # Process the result
        if isinstance(result, dict):
            # Result from compute_cot_rewriter_reward contains:
            # - reward_tensor: [batch_size, seq_len] tensor
            # - reward_extra_info: dict with additional metrics
            reward_tensor = result.get("reward_tensor")
            reward_extra_info = result.get("reward_extra_info", {})

            # Apply overlong_buffer penalty if configured
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                reward_tensor = self._apply_overlong_penalty(
                    data, reward_tensor, reward_extra_info
                )

            if reward_tensor is None:
                # Fallback: create zero tensor
                batch_size = len(data)
                response_length = data.batch["responses"].shape[1]
                device = data.batch["responses"].device
                reward_tensor = torch.zeros(
                    batch_size, response_length, device=device, dtype=torch.float32
                )

            # Print some samples for debugging
            if self.num_examine > 0:
                self._print_samples(data, result, self.num_examine)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
        else:
            # Scalar result - should not happen with compute_cot_rewriter_reward
            # but handle it for robustness
            batch_size = len(data)
            response_length = data.batch["responses"].shape[1]
            device = data.batch["responses"].device
            reward_tensor = torch.zeros(
                batch_size, response_length, device=device, dtype=torch.float32
            )

            # Assign the scalar reward to all samples at the last position
            reward_tensor[:, -1] = float(result)

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            else:
                return reward_tensor

    def _print_samples(self, data: DataProto, result: dict, num_examine: int):
        """Print sample outputs for debugging."""
        batch_size = len(data)
        num_to_print = min(num_examine, batch_size)

        extra_info = result.get("reward_extra_info", {})
        total_rewards = extra_info.get("total_rewards", [])
        cognitive_rewards = extra_info.get("cognitive_rewards", [])
        rewrite_mean_log_probs = extra_info.get("rewrite_mean_log_probs", [])
        original_mean_log_probs = extra_info.get("original_mean_log_probs", [])
        
        # Get FSF-related info
        original_fsfs = extra_info.get("original_fsfs", [])
        rewrite_fsfs = extra_info.get("rewrite_fsfs", [])
        logic_rewards = extra_info.get("logic_rewards", [])
        ged_values = extra_info.get("ged", [])

        for i in range(num_to_print):
            # Decode prompt and response
            prompt_ids = data.batch["prompts"][i]
            response_ids = data.batch["responses"][i]

            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Extract original_cot from non_tensor_batch
            original_cot = ""
            reward_model_data = data.non_tensor_batch.get("reward_model", None)
            if reward_model_data is not None:
                if hasattr(reward_model_data, '__len__') and i < len(reward_model_data):
                    rm_item = reward_model_data[i]
                    if isinstance(rm_item, dict):
                        original_cot = rm_item.get("original_cot", "")

            logger.info(f"[Sample {i}] prompt={prompt_str[:100]}... response={response_str[:100]}...")

            # Print reward info
            if i < len(total_rewards):
                logger.info(f"[Sample {i}] total_reward={total_rewards[i]}")
            if i < len(cognitive_rewards):
                logger.info(f"[Sample {i}] cognitive_reward={cognitive_rewards[i]}")
            
            # Print FSF info
            if i < len(original_fsfs) and i < len(rewrite_fsfs):
                ged_str = ged_values[i] if i < len(ged_values) else "N/A"
                logger.info(
                    f"[Sample {i}] FSF: original={original_fsfs[i]:.4f}, rewrite={rewrite_fsfs[i]:.4f}, "
                    f"ged={ged_str}, logic_reward={logic_rewards[i] if i < len(logic_rewards) else 'N/A'}"
                )

    def _apply_overlong_penalty(
        self,
        data: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
    ) -> torch.Tensor:
        """Apply overlong response penalty based on overlong_buffer_cfg.

        Zero-reward penalty: when a response exceeds expected_len = max_resp_len - buffer_len,
        the reward is set to 0 instead of subtracting a gradual penalty.

        Args:
            data: DataProto object containing the batch data
            reward_tensor: Current reward tensor [batch_size, seq_len]
            reward_extra_info: Dictionary to store extra info for logging

        Returns:
            Modified reward tensor with overlong penalties applied
        """
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        device = reward_tensor.device

        # Initialize overlong tracking in reward_extra_info
        if "overlong_rewards" not in reward_extra_info:
            reward_extra_info["overlong_rewards"] = []
        if "overlong" not in reward_extra_info:
            reward_extra_info["overlong"] = []

        overlong_buffer_len = self.overlong_buffer_cfg.len
        expected_len = self.max_resp_len - overlong_buffer_len

        for i in range(batch_size):
            # Get the valid response length (excluding padding)
            prompt_length = data.batch["prompts"].shape[1]
            attention_mask = data.batch["attention_mask"]
            valid_response_length = attention_mask[i, prompt_length:].sum().item()

            # Check if response exceeds expected length
            exceed_len = valid_response_length - expected_len
            is_overlong = exceed_len > 0

            if is_overlong:
                # Zero-reward penalty: set the reward to 0
                if valid_response_length > 0 and valid_response_length <= response_length:
                    reward_pos = valid_response_length - 1
                    reward_tensor[i, reward_pos] = 0.0

            # Track for logging (1.0 if penalized, 0.0 if not)
            reward_extra_info["overlong_rewards"].append(1.0 if is_overlong else 0.0)
            reward_extra_info["overlong"].append(is_overlong)

            if self.overlong_buffer_cfg.log and is_overlong:
                logger.info(
                    f"[OVERLONG] Sample {i}: len={valid_response_length}, "
                    f"expected={expected_len}, exceed={exceed_len}, reward set to 0"
                )

        return reward_tensor
