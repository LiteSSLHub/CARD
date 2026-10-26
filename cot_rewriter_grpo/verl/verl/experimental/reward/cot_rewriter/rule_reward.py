"""
Simple Rule-based Reward Function for debugging SGLang rollout.

This is a minimal reward function that doesn't require loading any models.
Used for testing the training pipeline before adding complex reward computation.
"""

from typing import Any, Dict, Optional
import torch
from verl import DataProto
from verl.experimental.reward.cot_rewriter.logging_config import logger


def compute_rule_reward(
    data: DataProto,
    tokenizer: Any,
    length_penalty: float = -0.001,  # penalty for overly long outputs
    min_length_bonus: float = 0.1,   # bonus for reaching minimum length
    min_length: int = 100,           # minimum expected length
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple rule-based reward for debugging.

    Rewards:
    - Give positive reward when output length is within a reasonable range
    - Give negative reward when too short or too long

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        length_penalty: Per-token penalty for length
        min_length_bonus: Bonus for reaching minimum length
        min_length: Minimum expected response length in tokens

    Returns:
        Dictionary with reward_tensor and reward_extra_info
    """
    batch_size = data.batch["responses"].shape[0]
    response_length = data.batch["responses"].shape[1]
    device = data.batch["responses"].device

    logger.info(f"[RULE_REWARD] Computing rule reward for batch_size={batch_size}")

    # Get actual length of each sample
    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long()
    else:
        # Compute length via pad token
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        seq_lengths = (data.batch["responses"] != pad_token_id).sum(dim=1).long()

    # Compute rewards
    rewards = []
    for i in range(batch_size):
        length = seq_lengths[i].item()

        # Base reward: reached minimum length
        if length >= min_length:
            reward = min_length_bonus
        else:
            # Length insufficient, reward proportionally
            reward = min_length_bonus * (length / min_length)

        # Length penalty (prevent overly long)
        if length > min_length * 3:
            reward += length_penalty * (length - min_length * 3)

        rewards.append(reward)

    # Print statistics
    avg_length = seq_lengths.float().mean().item()
    avg_reward = sum(rewards) / len(rewards)
    logger.info(f"[RULE_REWARD] avg_length={avg_length:.1f}, avg_reward={avg_reward:.4f}")

    # Convert to tensor
    total_rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

    # Create token-level rewards (only give reward at the last token)
    token_level_rewards = torch.zeros(
        batch_size, response_length, device=device, dtype=torch.float32
    )

    # Determine the last valid position of each sequence
    if response_mask is not None:
        last_positions = (seq_lengths - 1).clamp(min=0, max=response_length - 1)
    else:
        last_positions = torch.full((batch_size,), response_length - 1, device=device, dtype=torch.long)

    # Place reward at the last token position
    for i in range(batch_size):
        token_level_rewards[i, last_positions[i]] = total_rewards[i]

    logger.info(f"[RULE_REWARD] Rule reward computation completed")

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "total_rewards": rewards,
            "seq_lengths": seq_lengths.tolist(),
            "avg_length": avg_length,
            "avg_reward": avg_reward,
        },
    }
