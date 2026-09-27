# /// script
# dependencies = [
#   "accelerate>=1.13.0",
#   "datasets>=4.4.0",
#   "huggingface-hub>=1.1.0",
#   "peft>=0.18.0",
#   "torch>=2.12.0",
#   "trackio>=0.3.0",
#   "transformers>=5.11.0",
#   "trl>=1.6.0",
#   "mlx>=0.22.0; platform_system == 'Darwin'",
#   "mlx-lm>=0.21.0; platform_system == 'Darwin'",
# ]
# ///

"""Unified runner and orchestrator for agent post-training experiments on Apple Silicon.

Supports:
  1. Parameterized training across PyTorch MPS and Apple MLX.
  2. Batch execution of curated experiment suites (framework comparison, model scaling, context scaling).
  3. Serving and execution benchmarks (MPS vs. MLX vs. GGUF) measuring TTFT and generation tokens/sec.
  4. Unified Trackio telemetry to training-agents-sft.db.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Default Trackio directory to current working directory so database is created in pwd
os.environ.setdefault("TRACKIO_DIR", os.path.abspath("."))
# Safeguard Apple Silicon MPS allocators
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

MODEL_ALIASES: dict[str, dict[str, str]] = {
    "0.5b": {
        "mps": "Qwen/Qwen2.5-0.5B-Instruct",
        "mlx_16bit": "Qwen/Qwen2.5-0.5B-Instruct",
        "mlx_4bit": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "gguf": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    },
    "1.5b": {
        "mps": "Qwen/Qwen2.5-1.5B-Instruct",
        "mlx_16bit": "Qwen/Qwen2.5-1.5B-Instruct",
        "mlx_4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "gguf": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    },
    "3b": {
        "mps": "Qwen/Qwen2.5-3B-Instruct",
        "mlx_16bit": "Qwen/Qwen2.5-3B-Instruct",
        "mlx_4bit": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "gguf": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    },
}

DEFAULT_PROJECT = "training-agents-sft"


def resolve_model_id(model_arg: str, backend: str, quant: str = "4bit") -> str:
    key = model_arg.lower()
    if key in MODEL_ALIASES:
        if backend == "mps":
            return MODEL_ALIASES[key]["mps"]
        if backend == "mlx":
            return MODEL_ALIASES[key]["mlx_4bit"] if quant == "4bit" else MODEL_ALIASES[key]["mlx_16bit"]
        if backend == "gguf":
            return MODEL_ALIASES[key]["gguf"]
    return model_arg


def run_command_stream(cmd: list[str]) -> int:
    print(f"\n[EXEC] {' '.join(cmd)}\n", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return proc.wait()


def check_experiment_status(backend: str, output_dir: Path, target_steps: int) -> tuple[str, str]:
    """Inspects output_dir to determine if task is 'completed', 'resume', or 'fresh'."""
    if not output_dir.exists():
        return "fresh", ""

    # Check for explicit completion marker with step verification
    completed_marker = output_dir / "COMPLETED"
    if completed_marker.exists():
        try:
            with open(completed_marker) as f:
                data = json.load(f)
                done_steps = data.get("global_step") or data.get("iters") or 0
                if done_steps >= target_steps:
                    return "completed", ""
        except Exception:
            return "completed", ""

    if backend == "mps":
        # Check trainer_state.json
        state_file = output_dir / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    st = json.load(f)
                    if st.get("global_step", 0) >= target_steps:
                        return "completed", ""
            except Exception:
                pass

        # Check for checkpoint directories
        ckpts = sorted(
            [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if ckpts:
            latest_ckpt = ckpts[-1]
            return "resume", str(latest_ckpt)

    elif backend == "mlx":
        final_adapter = output_dir / "adapters.safetensors"
        ckpts = sorted(
            [f for f in output_dir.glob("[0-9]*_adapters.safetensors") if f.is_file()],
            key=lambda p: int(p.name.split("_")[0]) if p.name.split("_")[0].isdigit() else 0,
        )
        if ckpts:
            latest_ckpt = ckpts[-1]
            step_num = int(latest_ckpt.name.split("_")[0])
            if step_num >= target_steps:
                return "completed", ""
            return "resume", str(latest_ckpt)
        elif final_adapter.exists():
            return "completed", ""

    return "fresh", ""


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SFT_SCRIPT = str(SCRIPT_DIR / "train_sft.py")
TRAIN_MLX_SCRIPT = str(SCRIPT_DIR / "train_mlx.py")
COMPARE_RUNS_SCRIPT = str(SCRIPT_DIR / "compare_runs.py")


# ---------------------------------------------------------------------------
# Training Dispatcher
# ---------------------------------------------------------------------------

def run_single_train(
    backend: str,
    model: str,
    max_length: int = 2048,
    steps: int = 509,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_layers: int = 24,
    trackio_project: str = DEFAULT_PROJECT,
    trackio_dir: str = ".",
    run_name: str = "",
    dry_run: bool = False,
    force_restart: bool = False,
) -> int:
    backend = backend.lower()
    clean_model_tag = model.split("/")[-1].lower().replace(".", "-")
    auto_run_name = run_name or f"{clean_model_tag}-{backend}-{max_length}ctx-{steps}steps"
    output_dir = Path(f"outputs/{auto_run_name}")

    status, resume_arg = "fresh", ""
    if not force_restart:
        status, resume_arg = check_experiment_status(backend, output_dir, steps)

    if status == "completed":
        print(f"\n[SKIP] Task '{auto_run_name}' already COMPLETED (marker found in {output_dir}). Skipping!", flush=True)
        return 0

    resolved_trackio_dir = str(Path(trackio_dir or os.environ.get("TRACKIO_DIR", ".")).resolve())

    if backend == "mps":
        model_id = resolve_model_id(model, "mps")
        cmd = [
            sys.executable,
            "-u",
            TRAIN_SFT_SCRIPT,
            "--model-id", model_id,
            "--device", "mps",
            "--completion-only-loss",
            "--max-length", str(max_length),
            "--max-steps", str(steps),
            "--per-device-train-batch-size", "1",
            "--gradient-accumulation-steps", str(max(1, batch_size)),
            "--learning-rate", str(learning_rate),
            "--lora-r", str(lora_r),
            "--lora-alpha", str(lora_alpha),
            "--logging-steps", "10",
            "--eval-steps", "50",
            "--eval-size", "32",
            "--save-steps", "100",
            "--save-total-limit", "2",
            "--output-dir", str(output_dir),
            "--trackio-project", trackio_project,
            "--run-name", auto_run_name,
        ]
        if resolved_trackio_dir:
            cmd.extend(["--trackio-dir", resolved_trackio_dir])
        if status == "resume":
            print(f"\n[RESUME] Task '{auto_run_name}' found breakpoint at {resume_arg}. Resuming from checkpoint!", flush=True)
            cmd.extend(["--resume-from-checkpoint", resume_arg])

    elif backend == "mlx":
        model_id = resolve_model_id(model, "mlx")
        micro_batch = 1 if max_length >= 2048 else min(2, batch_size)
        grad_accum = max(1, batch_size // micro_batch)
        cmd = [
            sys.executable,
            "-u",
            TRAIN_MLX_SCRIPT,
            "--model", model_id,
            "--iters", str(steps),
            "--batch-size", str(micro_batch),
            "--grad-accumulation-steps", str(grad_accum),
            "--learning-rate", str(learning_rate),
            "--lora-layers", str(lora_layers),
            "--max-seq-length", str(max_length),
            "--adapter-path", str(output_dir),
            "--trackio-project", trackio_project,
            "--trackio-run-name", auto_run_name,
        ]
        if resolved_trackio_dir:
            cmd.extend(["--trackio-dir", resolved_trackio_dir])
        for candidate_pattern in (f"workspaces/*{clean_model_tag}*-pi-mono-sft/mlx_data", "workspaces/*-pi-mono-sft/mlx_data"):
            matched_dirs = sorted(glob.glob(candidate_pattern))
            if matched_dirs:
                cmd.extend(["--data", matched_dirs[-1]])
                break
        if status == "resume":
            print(f"\n[RESUME] Task '{auto_run_name}' found breakpoint at {resume_arg}. Resuming from checkpoint!", flush=True)
            cmd.extend(["--resume-adapter-file", resume_arg])
    else:
        sys.exit(f"Unsupported backend: {backend}. Use 'mps' or 'mlx'.")

    if dry_run:
        print(f"[DRY-RUN] Status: {status.upper()} | Would run:\n  {' '.join(cmd)}")
        return 0

    return run_command_stream(cmd)


# ---------------------------------------------------------------------------
# Suite Runner
# ---------------------------------------------------------------------------

SUITES: dict[str, list[dict[str, Any]]] = {
    "framework_comparison": [
        {
            "name": "Exp 1: Qwen2.5-0.5B PyTorch MPS 2k",
            "backend": "mps",
            "model": "0.5b",
            "max_length": 2048,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-0.5b-mps-2k",
        },
        {
            "name": "Exp 2: Qwen2.5-0.5B Apple MLX 2k",
            "backend": "mlx",
            "model": "0.5b",
            "max_length": 2048,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-0.5b-mlx-2k",
        },
    ],
    "model_scaling": [
        {
            "name": "Exp 3: Qwen2.5-1.5B PyTorch MPS 2k",
            "backend": "mps",
            "model": "1.5b",
            "max_length": 2048,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-1.5b-mps-2k",
        },
        {
            "name": "Exp 4: Qwen2.5-1.5B Apple MLX 2k",
            "backend": "mlx",
            "model": "1.5b",
            "max_length": 2048,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-1.5b-mlx-2k",
        },
        {
            "name": "Exp 5: Qwen2.5-3B Apple MLX 2k",
            "backend": "mlx",
            "model": "3b",
            "max_length": 2048,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-3b-mlx-2k",
        },
    ],
    "context_scaling": [
        {
            "name": "Exp 6: Qwen2.5-1.5B Apple MLX 1k Context",
            "backend": "mlx",
            "model": "1.5b",
            "max_length": 1024,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-1.5b-mlx-1k",
        },
        {
            "name": "Exp 7: Qwen2.5-1.5B Apple MLX 4k Context",
            "backend": "mlx",
            "model": "1.5b",
            "max_length": 4096,
            "steps": 509,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "run_name": "qwen-1.5b-mlx-4k",
        },
        {
            "name": "Exp 8: Qwen2.5-1.5B PyTorch MPS 4k Context",
            "backend": "mps",
            "model": "1.5b",
            "max_length": 4096,
            "steps": 509,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "run_name": "qwen-1.5b-mps-4k",
        },
    ],
}


def run_suite(suite_name: str, dry_run: bool = False, force_restart: bool = False, trackio_dir: str = ".") -> None:
    if suite_name == "all":
        items = SUITES["framework_comparison"] + SUITES["model_scaling"] + SUITES["context_scaling"]
    elif suite_name in SUITES:
        items = SUITES[suite_name]
    else:
        sys.exit(f"Unknown suite '{suite_name}'. Choose from: {list(SUITES.keys()) + ['all']}")

    print(f"===========================================================")
    print(f"Executing Suite: {suite_name} ({len(items)} experiments)")
    print(f"===========================================================")

    for i, exp in enumerate(items, start=1):
        print(f"\n---> Starting [{i}/{len(items)}]: {exp['name']}")
        code = run_single_train(
            backend=exp["backend"],
            model=exp["model"],
            max_length=exp.get("max_length", 2048),
            steps=exp.get("steps", 509),
            batch_size=exp.get("batch_size", 4),
            learning_rate=exp.get("learning_rate", 1e-4),
            trackio_dir=trackio_dir,
            run_name=exp.get("run_name", ""),
            dry_run=dry_run,
            force_restart=force_restart,
        )
        if code != 0 and not dry_run:
            print(f"[ERROR] Experiment failed with code {code}. Aborting suite.")
            sys.exit(code)

    print("\n[SUCCESS] Suite completed! Running comparison table:")
    comp_cmd = [sys.executable, COMPARE_RUNS_SCRIPT]
    target_dir = trackio_dir or os.environ.get("TRACKIO_DIR", ".")
    resolved_trackio_dir = str(Path(target_dir).resolve()) if target_dir else ""
    if resolved_trackio_dir:
        comp_cmd.extend(["--trackio-dir", resolved_trackio_dir])
    subprocess.run(comp_cmd)


# ---------------------------------------------------------------------------
# Serving Benchmark (MPS vs MLX vs GGUF)
# ---------------------------------------------------------------------------

def benchmark_serving(
    model_alias: str = "0.5b",
    backends: list[str] | None = None,
    prompt_lengths: list[int] | None = None,
    gen_tokens: int = 128,
) -> None:
    if backends is None:
        backends = ["mps", "mlx"]
    if prompt_lengths is None:
        prompt_lengths = [256, 1024, 2048]

    print("=" * 88)
    print(f"{'SERVING & EXECUTION BENCHMARK':^88}")
    print(f"{'Model: ' + model_alias + ' | Output Tokens: ' + str(gen_tokens):^88}")
    print("=" * 88)
    print(f"{'Backend':<12} | {'Prompt Len':<12} | {'TTFT (ms)':<14} | {'Tokens/Sec':<14} | {'Total Time (s)':<14}")
    print("-" * 88)

    sample_text = "You are a coding assistant. Read the current directory and list files. " * 100

    for backend in backends:
        for plen in prompt_lengths:
            prompt = sample_text[: plen * 4]  # Approx tokens

            ttft_ms = 0.0
            tok_sec = 0.0
            total_s = 0.0

            if backend == "mlx":
                try:
                    import mlx.core as mx
                    from mlx_lm import load, generate

                    model_id = resolve_model_id(model_alias, "mlx")
                    model, tokenizer = load(model_id)

                    # Warmup
                    generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)

                    # Accurately measure TTFT (prefill + first token decode)
                    t0 = time.perf_counter()
                    _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
                    ttft_ms = (time.perf_counter() - t0) * 1000.0

                    start_t = time.perf_counter()
                    response = generate(model, tokenizer, prompt=prompt, max_tokens=gen_tokens, verbose=False)
                    total_s = time.perf_counter() - start_t
                    tok_sec = gen_tokens / max(total_s, 0.001)
                except Exception as exc:
                    print(f"{backend:<12} | {plen:<12} | ERROR: {exc}")
                    continue

            elif backend == "mps":
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    model_id = resolve_model_id(model_alias, "mps")
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        attn_implementation="sdpa",
                    ).to("mps")

                    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

                    # Warmup
                    _ = model.generate(**tokenizer("Hello", return_tensors="pt").to("mps"), max_new_tokens=5)

                    # Accurately measure TTFT (prefill + first token decode) with Metal synchronization
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = model.generate(**inputs, max_new_tokens=1)
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    ttft_ms = (time.perf_counter() - t0) * 1000.0

                    start_t = time.perf_counter()
                    with torch.no_grad():
                        _ = model.generate(**inputs, max_new_tokens=gen_tokens)
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    total_s = time.perf_counter() - start_t
                    tok_sec = gen_tokens / max(total_s, 0.001)
                except Exception as exc:
                    print(f"{backend:<12} | {plen:<12} | ERROR: {exc}")
                    continue

            elif backend == "gguf":
                print(f"{backend:<12} | {plen:<12} | [Requires llama.cpp / llama-cpp-python]")
                continue

            print(f"{backend:<12} | {plen:<12} | {ttft_ms:<14.1f} | {tok_sec:<14.2f} | {total_s:<14.3f}")

    print("=" * 88)


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Agent Post-Training & Benchmark Runner.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", aliases=["t"], help="Run a single parameterized training experiment.")
    train_p.add_argument("--backend", choices=["mps", "mlx"], default="mlx", help="Framework backend.")
    train_p.add_argument("--model", default="0.5b", help="Model alias (0.5b, 1.5b, 3b) or HuggingFace repo.")
    train_p.add_argument("--max-length", type=int, default=2048, help="Max sequence length.")
    train_p.add_argument("--steps", type=int, default=509, help="Training steps/iterations (default: 509 = 1 full epoch).")
    train_p.add_argument("--batch-size", type=int, default=4, help="Effective batch size.")
    train_p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    train_p.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    train_p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    train_p.add_argument("--lora-layers", type=int, default=24, help="LoRA layers for MLX.")
    train_p.add_argument("--trackio-project", default=DEFAULT_PROJECT, help="Trackio project name.")
    train_p.add_argument(
        "--trackio-dir",
        default=os.environ.get("TRACKIO_DIR", "."),
        help="Directory for Trackio SQLite databases (defaults to current directory or TRACKIO_DIR).",
    )
    train_p.add_argument("--run-name", default="", help="Custom run name.")
    train_p.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    train_p.add_argument(
        "--force",
        "--force-restart",
        dest="force_restart",
        action="store_true",
        help="Force restart experiment even if COMPLETED marker exists.",
    )

    # Suite subcommand (alias 's', defaults to running 'all' experiments)
    suite_p = subparsers.add_parser("suite", aliases=["s"], help="Run a curated suite of experiments (defaults to 'all').")
    suite_p.add_argument(
        "--name",
        choices=list(SUITES.keys()) + ["all"],
        default="all",
        help="Suite name to run (default: all).",
    )
    suite_p.add_argument(
        "--trackio-dir",
        default=os.environ.get("TRACKIO_DIR", "."),
        help="Directory for Trackio SQLite databases (defaults to current directory or TRACKIO_DIR).",
    )
    suite_p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    suite_p.add_argument(
        "--force",
        "--force-restart",
        dest="force_restart",
        action="store_true",
        help="Force restart experiments even if COMPLETED marker exists.",
    )

    # Serving benchmark subcommand
    bench_p = subparsers.add_parser("bench-serving", aliases=["b", "bench"], help="Benchmark serving latency and throughput.")
    bench_p.add_argument("--model", default="0.5b", help="Model alias (0.5b, 1.5b, 3b).")
    bench_p.add_argument("--backends", nargs="+", default=["mps", "mlx"], help="Backends to benchmark.")
    bench_p.add_argument("--prompt-lengths", nargs="+", type=int, default=[256, 1024, 2048])
    bench_p.add_argument("--gen-tokens", type=int, default=128, help="Output tokens.")

    # Compare subcommand
    compare_p = subparsers.add_parser("compare", aliases=["c"], help="Display comparison table of all Trackio runs.")
    compare_p.add_argument(
        "--trackio-dir",
        default=os.environ.get("TRACKIO_DIR", "."),
        help="Directory to search for Trackio SQLite databases (defaults to current directory or TRACKIO_DIR).",
    )

    args = parser.parse_args()

    if args.subcommand in ("train", "t"):
        code = run_single_train(
            backend=args.backend,
            model=args.model,
            max_length=args.max_length,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_layers=args.lora_layers,
            trackio_project=args.trackio_project,
            trackio_dir=args.trackio_dir,
            run_name=args.run_name,
            dry_run=args.dry_run,
            force_restart=args.force_restart,
        )
        sys.exit(code)

    elif args.subcommand in ("suite", "s"):
        run_suite(args.name, dry_run=args.dry_run, force_restart=args.force_restart, trackio_dir=args.trackio_dir)

    elif args.subcommand in ("bench-serving", "b", "bench"):
        benchmark_serving(
            model_alias=args.model,
            backends=args.backends,
            prompt_lengths=args.prompt_lengths,
            gen_tokens=args.gen_tokens,
        )

    elif args.subcommand in ("compare", "c"):
        comp_cmd = [sys.executable, COMPARE_RUNS_SCRIPT]
        if args.trackio_dir:
            comp_cmd.extend(["--trackio-dir", args.trackio_dir])
        subprocess.run(comp_cmd)


if __name__ == "__main__":
    main()
