"""
Custom RL Dataset with JSONL Support for CoT Rewriter GRPO

This module extends VERL's RLHFDataset to support JSONL file format.
It also handles plain text prompts by converting them to chat message format.

Supported data formats:
1. New format: system_prompt + user_prompt (separate fields)
2. Chat format: prompt_key points to a list of messages
3. Plain text: prompt_key points to a single string
"""

import logging
from typing import Optional

import torch
import datasets
import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

# Import verl's RLHFDataset as base class
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

# Flag to track if we've logged the first sample debug info
_first_sample_logged = False


class RLHFDatasetJSONL(RLHFDataset):
    """
    Extended RLHFDataset that supports JSONL file format.

    This class inherits from VERL's RLHFDataset and overrides the
    _read_files_and_tokenize method to add support for .jsonl files.

    Supported formats:
        - .parquet: Parquet files (original)
        - .json: JSON files (original)
        - .jsonl: JSON Lines files (NEW)

    Args:
        data_files (str or list): Path(s) to data file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # Initialize additional attributes before calling parent init
        # These are used in _read_files_and_tokenize and __getitem__
        self.max_samples = config.get("max_samples", -1)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # New format: support system_prompt and user_prompt as separate fields
        self.system_prompt_key = config.get("system_prompt_key", "system_prompt")
        self.user_prompt_key = config.get("user_prompt_key", "user_prompt")
        self.use_new_format = config.get("use_new_format", False)  # Auto-detect if not specified

        # Call parent init (this will call _read_files_and_tokenize)
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
        )

    def _read_files_and_tokenize(self):
        """
        Read data files and tokenize.

        Overrides parent method to add support for .jsonl files.
        """
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".jsonl"):
                # NEW: Support for JSONL files
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _build_messages(self, example: dict):
        """
        Build chat messages from example.

        Supports three data formats:
        1. New format: system_prompt + user_prompt (separate fields)
           - Messages: [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}]
        2. Chat format: prompt_key points to a list of messages
           - Use parent's method to handle existing chat format
        3. Plain text format: prompt_key points to a single string
           - Convert to: [{"role": "user", "content": prompt_content}]

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages in chat format.
        """
        # === Format 1: New format with system_prompt + user_prompt ===
        # Check if both fields exist and are non-empty
        system_prompt = example.get(self.system_prompt_key, "")
        user_prompt = example.get(self.user_prompt_key, "")

        # Auto-detect new format: if both fields exist and are non-empty strings
        is_new_format = (
            self.use_new_format or
            (isinstance(system_prompt, str) and system_prompt.strip() and
             isinstance(user_prompt, str) and user_prompt.strip())
        )

        if is_new_format:
            messages = []
            # Add system message if present
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            # Add user message
            messages.append({"role": "user", "content": user_prompt.strip()})
            return messages

        # === Format 2: Chat format (list of messages) ===
        prompt_content = example.get(self.prompt_key)
        if isinstance(prompt_content, list):
            return super()._build_messages(example)

        # === Format 3: Plain text format ===
        if isinstance(prompt_content, str):
            messages = [{"role": "user", "content": prompt_content}]
            return messages

        raise ValueError(
            f"Unsupported prompt format: {type(prompt_content)}. "
            f"Expected one of: "
            f"1) system_prompt + user_prompt fields, "
            f"2) list of messages (chat format), "
            f"3) plain string (plain text format)"
        )

    def __getitem__(self, item):
        """
        Get a single item from the dataset.

        Overrides parent method to ensure proper handling of our data format
        and pass through all necessary fields for reward computation.

        IMPORTANT: This method must generate input_ids, attention_mask, and position_ids
        tensors, which are required by the VERL trainer.
        """
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask

        row_dict: dict = dict(self.dataframe[item])  # Make a copy

        # Build messages (handles plain text -> chat format conversion)
        messages = self._build_messages(row_dict)

        # ========== Tokenization (CRITICAL for VERL trainer) ==========
        # Apply chat template and tokenize
        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
            )
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        # Post-process: padding and truncation
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Set required tensor fields
        row_dict["input_ids"] = input_ids[0]

        # Debug log: print first sample's decoded input_ids
        global _first_sample_logged
        if not _first_sample_logged:
            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            logger.info(
                f"[RLHFDatasetJSONL] First sample debug:\n"
                f"  input_ids length: {len(input_ids[0])}\n"
                f"  decoded text (full):\n{decoded_text}\n"
                f"  messages format: {messages}"
            )
            _first_sample_logged = True
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Raw prompt IDs (for compatibility)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Set raw_prompt if return_raw_chat is enabled
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # ========== End of Tokenization ==========

        # Ensure extra_info exists for metadata
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        # Set index if not present
        if "index" not in row_dict["extra_info"]:
            row_dict["extra_info"]["index"] = item
        row_dict["index"] = row_dict["extra_info"].get("index", item)

        # tools_kwargs and interaction_kwargs (required by parent class)
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})

        # Build reward_model field from our data fields for reward function
        # This maps our field names to VERL's expected structure
        if "reward_model" not in row_dict or row_dict["reward_model"] is None:
            row_dict["reward_model"] = {}

        # Pass through important fields for reward computation
        # These will be accessible in the reward function via data.non_tensor_batch
        reward_model = row_dict["reward_model"]

        # Hardcoded field mapping (no if checks)
        reward_model["ground_truth"] = row_dict.get("solution", "")
        reward_model["original_cot"] = row_dict.get("original_thought", "")
        reward_model["question"] = row_dict.get("question", "")
        reward_model["fsf"] = row_dict.get("fsf", 0.0)
        reward_model["failed_nodes"] = row_dict.get("failed_nodes", None)
        reward_model["total_nodes"] = row_dict.get("total_nodes", None)

        # Set data_source if not present (required by VERL)
        if "data_source" not in row_dict:
            row_dict["data_source"] = "cot_rewrite"

        return row_dict

    def maybe_filter_out_long_prompts(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        No-op method: filtering disabled for performance.

        Previously filtered out prompts that are too long, but this is too slow.
        Now we rely on truncation instead.

        Args:
            dataset: HuggingFace dataset to filter

        Returns:
            The dataset unmodified (no filtering applied)
        """
        # Skip filtering for performance - let truncation handle long prompts
        return dataset
