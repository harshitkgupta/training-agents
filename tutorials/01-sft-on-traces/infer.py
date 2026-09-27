# /// script
# dependencies = [
#   "torch>=2.4.0",
#   "transformers>=4.48.0",
#   "peft>=0.13.0",
#   "accelerate>=0.34.0",
# ]
# ///

"""Test generation with a trained LoRA adapter on Apple Silicon MPS or CUDA."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Safeguard Apple Silicon MPS allocators before importing torch
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with trained LoRA adapter.")
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        default="Qwen/Qwen3-0.6B",
        help="Base model ID.",
    )
    parser.add_argument(
        "--adapter-path",
        "--output-dir",
        dest="adapter_path",
        default="outputs/qwen3-0-6b-pi-mono-lora",
        help="Path to trained adapter directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Inspect the git repository status and list modified files.",
        help="User prompt to send to the agent.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Device to run on.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        device = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    else:
        device = args.device

    print(f"phase=load device={device} adapter={args.adapter_path}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.float16 if device == "mps" else torch.bfloat16 if device == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
    }
    if device in ("mps", "cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    if device == "mps":
        model.to("mps")
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful coding agent with access to shell commands and tool calling."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print(f"\n--- Prompt ---\n{args.prompt}\n", flush=True)
    print("--- Generating Response ---\n", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=False)
    print(response, flush=True)


if __name__ == "__main__":
    main()
