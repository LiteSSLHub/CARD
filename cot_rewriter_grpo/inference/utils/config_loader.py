"""
YAML config loader with default value filling and validation.
"""

import os
import re
from typing import Any, Dict


def expand_env_vars(text: str) -> str:
    """Expand ${VAR} and ${VAR:-default} placeholders in YAML configs."""
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        return default if default is not None else ""

    return pattern.sub(replace, text)


def load_yaml_with_env(config_path: str) -> Dict[str, Any]:
    """Load a YAML file after expanding environment-variable placeholders."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        text = expand_env_vars(f.read())
    return yaml.safe_load(text) or {}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and fill in defaults."""
    config = load_yaml_with_env(config_path)

    _fill_defaults(config)
    _validate(config)
    return config


def _fill_defaults(config: Dict[str, Any]) -> None:
    config.setdefault("experiment_name", "cot_rewrite_exp")
    config.setdefault("output_root", "./output")
    config.setdefault("resume", True)
    config.setdefault("seed", 42)

    # Model defaults
    model = config.setdefault("model", {})
    model.setdefault("format", "auto")
    model.setdefault("merge_lora", True)
    model.setdefault("dtype", "bfloat16")
    model.setdefault("max_model_len", 8192)

    # Rollout defaults
    rollout = config.setdefault("rollout", {})
    rollout.setdefault("num_rollouts", 4)
    rollout.setdefault("batch_size", 32)
    gen = rollout.setdefault("generation", {})
    gen.setdefault("temperature", 0.7)
    gen.setdefault("top_p", 0.9)
    gen.setdefault("max_new_tokens", 4096)
    parallel = rollout.setdefault("parallel", {})
    parallel.setdefault("tensor_parallel_size", 1)
    # gpu_ids defaults to None (auto-detect)

    # Correct judge defaults
    cj = config.setdefault("correct_judge", {})
    cj.setdefault("enabled", True)
    cj.setdefault("batch_size", 8)
    cj.setdefault("tensor_parallel_size", 1)
    cj.setdefault("mem_fraction_static", 0.4)
    cj.setdefault("dtype", "float16")
    cj.setdefault("max_model_len", 8192)
    cj_gen = cj.setdefault("generation", {})
    cj_gen.setdefault("temperature", 0.0)
    cj_gen.setdefault("max_new_tokens", 50)

    # FSF judge defaults
    fsf = config.setdefault("fsf_judge", {})
    fsf.setdefault("enabled", True)
    fsf.setdefault("batch_size", 4)
    fsf.setdefault("tensor_parallel_size", 1)
    fsf.setdefault("mem_fraction_static", 0.4)
    fsf.setdefault("dtype", "float16")
    fsf.setdefault("max_model_len", 8192)
    fsf_gen = fsf.setdefault("generation", {})
    fsf_gen.setdefault("temperature", 0.6)
    fsf_gen.setdefault("max_new_tokens", 8000)

    # Dataset build defaults
    db = config.setdefault("dataset_build", {})
    filters = db.setdefault("filters", {})
    filters.setdefault("fsf_positive", True)
    filters.setdefault("correct_only", True)
    db.setdefault("selection_strategy", "priority_correct")
    db.setdefault("output_filename", "train_sharegpt.jsonl")


def _validate(config: Dict[str, Any]) -> None:
    """Validate required fields."""
    errors = []

    model = config.get("model", {})
    if not model.get("checkpoint_path"):
        errors.append("model.checkpoint_path is required")

    rollout = config.get("rollout", {})
    if not rollout.get("input_data"):
        errors.append("rollout.input_data is required")

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
