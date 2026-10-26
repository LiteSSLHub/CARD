"""CoT Rewriter Reward Module for GRPO Training."""

from verl.experimental.reward.cot_rewriter.combined_reward import (
    compute_cot_rewriter_reward,
    CombinedRewardComputer,
)
from verl.experimental.reward.cot_rewriter.reward_manager import CoTRewriterRewardManager

# Optional imports - these depend on specific backends (vllm/sglang)
# and are imported lazily to avoid hard dependencies
try:
    from verl.experimental.reward.cot_rewriter.student_reward import VLLMStudentRewardClient
except ImportError:
    VLLMStudentRewardClient = None  # vLLM not installed

try:
    from verl.experimental.reward.cot_rewriter.sglang_student_reward import SGLangStudentRewardClient
except ImportError:
    SGLangStudentRewardClient = None  # SGLang not installed

try:
    from verl.experimental.reward.cot_rewriter.fsf_reward import HFFSFJudge
except ImportError:
    HFFSFJudge = None

__all__ = [
    "compute_cot_rewriter_reward",
    "CombinedRewardComputer",
    "CoTRewriterRewardManager",
    "VLLMStudentRewardClient",
    "SGLangStudentRewardClient",
    "HFFSFJudge",
]
