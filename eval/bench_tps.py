#!/usr/bin/env python3
"""Single-GPU vLLM throughput benchmark script, supports LoRA online loading comparison.

Usage:
    # Pure base model
    python bench_tps.py /path/to/base_model

    # LoRA adapter (auto-detect adapter_config.json)
    python bench_tps.py /path/to/lora_adapter
"""
import argparse
import os
import json
import time
from pathlib import Path
from vllm import LLM, SamplingParams


def resolve_lora(model_path: str):
    """Detect if path is a LoRA adapter directory."""
    cfg_file = Path(model_path) / "adapter_config.json"
    if not cfg_file.exists():
        return model_path, None, None
    cfg = json.loads(cfg_file.read_text())
    base_model = cfg["base_model_name_or_path"]
    rank = cfg.get("r", 64)
    print(f"[LoRA] adapter: {model_path}, base: {base_model}, rank={rank}")
    return base_model, model_path, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model path (base model or LoRA adapter)",default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--num-prompts", type=int, default=64,
                        help="Number of concurrent prompts")
    parser.add_argument("--input-len", type=int, default=512,
                        help="Input token count per prompt")
    parser.add_argument("--output-len", type=int, default=16000,
                        help="Max generated tokens per prompt")
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    # Detect LoRA
    actual_model, lora_path, max_lora_rank = resolve_lora(args.model)

    mode = "LoRA online" if lora_path else "Base model"
    print(f"=== {mode} ===")
    print(f"Model: {actual_model}")
    if lora_path:
        print(f"LoRA: {lora_path} (rank={max_lora_rank})")
    print(f"max_model_len={args.max_model_len}, gpu_mem={args.gpu_mem}, "
          f"enforce_eager={args.enforce_eager}")

    kwargs = dict(
        model=actual_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    if lora_path:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, lora_path)

    # Construct fixed-length prompt
    filler = "Hello world. This is a benchmark prompt for throughput testing. "
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    repeat_times = max(1, args.input_len // len(filler_ids))
    prompt_text = filler * repeat_times
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[:args.input_len]
    prompt_text = tokenizer.decode(prompt_ids)

    prompts = [prompt_text] * args.num_prompts
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.8)

    # warmup
    print("Warmup...")
    _ = llm.generate(prompts[:2], sp, lora_request=lora_request, use_tqdm=False)

    # Formal benchmark
    print(f"\n{'batch':>5s} | {'avg_out':>7s} | {'output_tps':>12s} | {'total_tps':>12s} | {'time':>6s}")
    print("-" * 65)
    for batch_size in [1, 4, 16, 32, 64]:
        if batch_size > args.num_prompts:
            break
        batch = prompts[:batch_size]
        t0 = time.time()
        outputs = llm.generate(batch, sp, lora_request=lora_request, use_tqdm=False)
        elapsed = time.time() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_in = args.input_len * batch_size
        out_tps = total_out / elapsed
        total_tps = (total_in + total_out) / elapsed
        avg_out = total_out / batch_size
        print(f"{batch_size:>5d} | {avg_out:>7.0f} | {out_tps:>10.1f}/s | {total_tps:>10.1f}/s | {elapsed:>5.1f}s")

    print(f"\n=== {mode} benchmark done ===")


if __name__ == "__main__":
    main()
