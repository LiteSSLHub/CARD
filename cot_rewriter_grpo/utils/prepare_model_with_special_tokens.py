#!/usr/bin/env python3
"""
Preprocess model: add special tokens and save.
This ensures vLLM loads a model with the correct vocab_size during training.

Key: ensure tokenizer.vocab_size == model.config.vocab_size == embedding layer size
so that vLLM loads correctly.
"""

import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed model")
    parser.add_argument("--special_tokens", nargs="+", default=["<rewrite_reasoning>", "</rewrite_reasoning>"])
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Special tokens to add: {args.special_tokens}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu"  # Operate on CPU to avoid GPU memory issues
    )

    original_vocab_size = tokenizer.vocab_size
    original_model_vocab_size = model.config.vocab_size
    print(f"Original tokenizer vocab_size: {original_vocab_size}")
    print(f"Original model vocab_size: {original_model_vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")

    # Key: use the model's original vocab_size as the target size
    # This keeps pad_token_id (151643) within valid range
    target_vocab_size = original_model_vocab_size  # 151936

    # Add special tokens to tokenizer first
    special_tokens_dict = {"additional_special_tokens": args.special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to tokenizer")

    # Add dummy tokens to expand vocab_size to target_vocab_size
    num_to_add = target_vocab_size - len(tokenizer)
    print(f"Adding {num_to_add} dummy tokens to reach vocab_size {target_vocab_size}")
    dummy_tokens = [f"<dummy_{i}>" for i in range(num_to_add)]
    tokenizer.add_tokens(dummy_tokens)

    print(f"After adding tokens - len(tokenizer): {len(tokenizer)}, tokenizer.vocab_size: {tokenizer.vocab_size}")

    # Resize model embeddings to target_vocab_size
    model.resize_token_embeddings(target_vocab_size)
    print(f"Resized model embeddings to: {target_vocab_size}")

    # Ensure vocab_size in config is correct
    model.config.vocab_size = target_vocab_size

    # Save processed model
    print(f"Saving model to: {args.output_path}")
    tokenizer.save_pretrained(args.output_path)
    model.save_pretrained(args.output_path)

    # Key: directly modify vocab_size in tokenizer_config.json
    tokenizer_config_path = os.path.join(args.output_path, "tokenizer_config.json")
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["vocab_size"] = target_vocab_size
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Updated vocab_size in tokenizer_config.json to {target_vocab_size}")

    # Verify saved tokenizer
    saved_tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
    print(f"After save - tokenizer.vocab_size: {saved_tokenizer.vocab_size}")
    print(f"After save - len(tokenizer): {len(saved_tokenizer)}")

    print("Done!")
    print(f"Final vocab_size: {target_vocab_size}")
    print(f"model.config.vocab_size: {model.config.vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id} (should be < {target_vocab_size})")


if __name__ == "__main__":
    main()
