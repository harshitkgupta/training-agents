# Pi-Mono SFT

Reusable SFT example for fine-tuning a command-selected causal LM on
`badlogicgames/pi-mono` coding-agent traces with TRL, PEFT LoRA, Hugging Face
Jobs, and Trackio. Pass `--model-id`; the script infers LoRA targets, chat
template kwargs, work/output dirs, and run name from the model family
(Gemma, Qwen, Llama, or generic).

The verified Jobs sweep used `google/gemma-4-E2B-it`. Local MacBook smokes
usually pass `Qwen/Qwen3-0.6B`.

This example is intentionally lightweight: it checks in the runnable script and
commands, not datasets, checkpoints, logs, or generated outputs.

## What It Does

- downloads raw `*.jsonl` traces from `badlogicgames/pi-mono`
- converts visible user, assistant, tool-call, and tool-result turns into
  prompt/completion rows
- excludes hidden reasoning by default
- omits images as `[image omitted]`
- trains with TRL `SFTTrainer` and completion-only loss
- applies LoRA to the decoder for the selected `--model-id` family
  (Gemma: `language_model.layers.*`; Qwen/Llama: `layers.*` attention/MLP)
- logs remote runs to Trackio Space `burtenshaw/pi-mono-sft-trackio`
- pushes adapters to private Hub model repos
- the Gemma 4 sweep mirrored the selected best adapter to
  `burtenshaw/gemma-4-E2B-it-pi-mono-lora`
- evaluates the final adapter with Inspect AI HumanEval and MBPP coding
  benchmarks

## Files

- `train_sft.py`: TRL SFTTrainer script for PyTorch MPS and CUDA, supporting completion-only loss, PEFT LoRA, and Trackio.
- `train_mlx.py`: Apple MLX 4-bit QLoRA fine-tuning script with real-time Trackio telemetry streaming.
- `run_experiments.py`: Orchestrator and suite runner across MPS, MLX, and serving benchmarks (TTFT & tokens/sec).
- `compare_runs.py`: Cross-framework Trackio run inspector and metric comparison table generator.
- `infer.py`: Interactive and programmatic test generation script for trained LoRA adapters.
- `blog-apple-silicon-sft.md`: Publication-ready Hugging Face Community Blog post.

## Apple Silicon SFT Experiments & Benchmark Suites

You can run individual training runs, curated experiment suites, serving benchmarks, and comparison reports using `run_experiments.py`.

### 1. Run Curated Experiment Suites

All experiments automatically handle checkpoint detection and auto-resumption:

```bash
# Run all 8 benchmark experiments across MPS and MLX (509 steps = 1 full epoch)
caffeinate -dimsu uv run tutorials/01-sft-on-traces/run_experiments.py suite --name all

# Or run targeted sub-suites:
# Framework Comparison (0.5B PyTorch MPS vs Apple MLX at 2k context)
caffeinate -dimsu uv run tutorials/01-sft-on-traces/run_experiments.py suite --name framework_comparison

# Model Scaling (1.5B MPS, 1.5B MLX, 3B MLX at 2k context)
caffeinate -dimsu uv run tutorials/01-sft-on-traces/run_experiments.py suite --name model_scaling

# Context Scaling (1.5B MLX at 1k & 4k context; 1.5B MPS at 4k context)
caffeinate -dimsu uv run tutorials/01-sft-on-traces/run_experiments.py suite --name context_scaling
```

### 2. Run Individual Parameterized Experiments

```bash
# Run PyTorch MPS 1.5B at 2k context (with active MPSTelemetryCallback VRAM tracking):
uv run tutorials/01-sft-on-traces/run_experiments.py train \
  --backend mps --model 1.5b --max-length 2048 --steps 509 --run-name qwen-1.5b-mps-2k

# Run Apple MLX 3B at 2k context (native 4-bit QLoRA, micro-batching, cache clearing):
uv run tutorials/01-sft-on-traces/run_experiments.py train \
  --backend mlx --model 3b --max-length 2048 --steps 509 --run-name qwen-3b-mlx-2k
```

### 3. Compare All Runs & Generate Metrics Table

Inspect local SQLite Trackio databases and print the consolidated comparison table:

```bash
# Inspects Trackio database (*.db) in current working directory by default:
uv run tutorials/01-sft-on-traces/compare_runs.py

# Or inspect an explicit directory or specific .db file:
uv run tutorials/01-sft-on-traces/compare_runs.py --trackio-dir /path/to/dir
```

#### Trackio Database Location (Current Folder by Default)

All training runs and suite experiments create and log metrics to `<project>.db` (e.g. `training-agents-sft.db`) directly in the **current working directory (`$PWD`)** by default.

- To override the database location across all scripts, pass `--trackio-dir <path>` or set `export TRACKIO_DIR="/custom/path"`.
- `compare_runs.py` strictly inspects the configured location (`--trackio-dir` or `TRACKIO_DIR`, defaulting to `$PWD`), with no hidden cache searching.


### 4. Benchmark Serving Latency & Throughput (MPS vs MLX)

Measures Time-To-First-Token (TTFT) and decode tokens/second across prompt lengths (256, 1024, 2048 tokens):

```bash
uv run tutorials/01-sft-on-traces/run_experiments.py bench-serving --model 0.5b
```

### 5. Test Inference with Trained Adapters

```bash
uv run tutorials/01-sft-on-traces/infer.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path outputs/qwen-0.5b-mps-2k \
  --prompt "Read file main.py and list the functions."
```

## Local Smoke

`--model-id` is required. On a MacBook, start with Qwen 0.6B:

```bash
uv run tutorials/01-sft-on-traces/train_sft.py \
  --model-id Qwen/Qwen3-0.6B \
  --dataset-id badlogicgames/pi-mono \
  --device mps \
  --max-examples 64 \
  --eval-size 8 \
  --max-length 1024 \
  --max-steps 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --eval-steps 1 \
  --save-steps 1 \
  --trackio-project training-agents-sft \
  --trackio-group pi-mono-sft-smoke

# View local metrics in the interactive dashboard
uv run trackio show --project training-agents-sft
```

### Native 4-bit QLoRA on Apple Silicon with MLX (`mlx-lm`)

For native 4-bit quantized fine-tuning (QLoRA) on Apple Silicon, use the companion script `train_mlx.py` powered by Apple's `mlx-lm`:

```bash
# 1. Prepare data splits (creates workspaces/qwen3-0-6b-pi-mono-sft/mlx_data/train.jsonl)
uv run tutorials/01-sft-on-traces/train_sft.py \
  --model-id Qwen/Qwen3-0.6B \
  --prepare-only

# 2. Fine-tune with MLX 4-bit QLoRA
uv run tutorials/01-sft-on-traces/train_mlx.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --iters 100 \
  --batch-size 1 \
  --adapter-path outputs/qwen-mlx-lora \
  --prompt "Read file main.py and list the functions."
```

Gemma 4 E2B on CUDA or HF Jobs:

```bash
uv run tutorials/01-sft-on-traces/train_sft.py \
  --model-id google/gemma-4-E2B-it \
  --dataset-id badlogicgames/pi-mono \
  --max-examples 512 \
  --eval-size 32 \
  --max-length 2048 \
  --max-steps 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --eval-steps 1 \
  --save-steps 1 \
  --trackio-project training-agents-sft \
  --trackio-space-id burtenshaw/pi-mono-sft-trackio \
  --trackio-group pi-mono-sft-smoke \
  --run-name gemma4-e2b-pi-mono-smoke-targets
```

## HF Jobs Sweep

Use `hf auth whoami` first, then pass the local token by name with
`--secrets HF_TOKEN`. Do not print token values.

Baseline command:

```bash
hf jobs uv run \
  --flavor l40sx1 \
  --timeout 3h \
  -d \
  --secrets HF_TOKEN \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --label project=pi-mono-sft \
  --label sweep=trackio-v1 \
  --label kind=sweep \
  --label variant=lr2e4-r16-len4k \
  tutorials/01-sft-on-traces/train_sft.py \
  --model-id google/gemma-4-E2B-it \
  --dataset-id badlogicgames/pi-mono \
  --output-dir outputs/gemma4-e2b-pi-mono-lr2e4-r16-len4k \
  --hub-model-id burtenshaw/gemma-4-E2B-it-pi-mono-lora-lr2e4-r16-len4k \
  --push-to-hub \
  --hub-private-repo \
  --max-length 4096 \
  --max-steps 200 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-accumulation-steps 16 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --eval-size 256 \
  --logging-steps 5 \
  --eval-steps 50 \
  --save-steps 100 \
  --trackio-project training-agents-sft \
  --trackio-space-id burtenshaw/pi-mono-sft-trackio \
  --trackio-group pi-mono-sft-sweep-20260612-trackio \
  --run-name gemma4-e2b-pi-mono-sft-lr2e4-r16-len4k
```

Sweep variants:

| Variant | Learning rate | LoRA r/alpha | Hub model repo |
| --- | ---: | ---: | --- |
| `lr2e4-r16-len4k` | `2e-4` | `16/32` | `burtenshaw/gemma-4-E2B-it-pi-mono-lora-lr2e4-r16-len4k` |
| `lr1e4-r16-len4k` | `1e-4` | `16/32` | `burtenshaw/gemma-4-E2B-it-pi-mono-lora-lr1e4-r16-len4k` |
| `lr2e4-r8-len4k` | `2e-4` | `8/16` | `burtenshaw/gemma-4-E2B-it-pi-mono-lora-lr2e4-r8-len4k` |

## Verified Run Record

Dashboard: https://huggingface.co/spaces/burtenshaw/pi-mono-sft-trackio

Final adapter repo:
https://huggingface.co/burtenshaw/gemma-4-E2B-it-pi-mono-lora

Selected variant: `lr2e4-r16-len4k`, chosen by lowest held-out eval loss.
Final adapter mirror commit: `9e7aa2bf9c0ee7d76e3a09cada4ca9f5e34f9efc`.
Model card eval-table commit: `de66c52f690f38d97d104bd8a19e03cfdeb6a467`.

Smoke Job:

- `6a2bec397c68f455eff13786`: completed 1 training step and verified 205
  Gemma 4 language decoder LoRA targets.

Sweep Jobs, verified completed on 2026-06-15:

| Job | Variant | Train loss | Final eval loss | Final eval token accuracy | Runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| `6a2becfe7c68f455eff13796` | `lr2e4-r16-len4k` | `0.6465` | `0.5506` | `0.8624` | `3897s` |
| `6a2becfe7c68f455eff13798` | `lr1e4-r16-len4k` | `0.6914` | `0.5805` | `0.8585` | `3919s` |
| `6a2becfe871c005b5352b1cd` | `lr2e4-r8-len4k` | `0.6731` | `0.5648` | `0.8594` | `3880s` |

Dataset conversion for the 4k sweep:

- raw files: 627
- message events: 33,879
- assistant examples before length filtering: 15,251
- kept after `max_length=4096`: 6,727
- train/eval split: 6,471 / 256 with seed 42
- reasoning parts ignored: 12,451
- image parts omitted: 48

## Inspect AI Coding Evals

Use HumanEval and MBPP as the default post-SFT coding benchmarks for this
example. They are small enough to run on Hugging Face Jobs with the Inspect
local sandbox and directly test code generation with unit tests.

Run the evals against the final LoRA adapter using Inspect's vLLM LoRA target
syntax:

```bash
hf jobs run \
  --flavor a10g-large \
  --timeout 3h \
  --detach \
  --secrets HF_TOKEN \
  --env HF_XET_HIGH_PERFORMANCE=1 \
  --label project=pi-mono-sft \
  --label kind=inspect-eval \
  --label benchmark=humaneval \
  vllm/vllm-openai:latest \
  bash -lc '
set -euo pipefail
ln -sf "$(command -v python3)" /usr/local/bin/python
python3 -m pip install -q --upgrade pip
python3 -m pip install -q inspect-evals huggingface_hub pytest
mkdir -p /tmp/inspect-logs
inspect eval inspect_evals/humaneval \
  --model vllm/google/gemma-4-E2B-it:burtenshaw/gemma-4-E2B-it-pi-mono-lora \
  --sandbox local \
  --max-samples 8 \
  --max-connections 8 \
  --max-tokens 768 \
  --temperature 0 \
  --seed 42 \
  --no-fail-on-error \
  --log-format json \
  --log-level warning \
  --log-dir /tmp/inspect-logs \
  -M max_model_len=4096 \
  -M gpu_memory_utilization=0.85 \
  -M trust_remote_code=true \
  -M dtype=float16
python3 - <<'"'"'PY'"'"'
import json, pathlib
rows = []
for path in sorted(pathlib.Path("/tmp/inspect-logs").glob("*.json")):
    data = json.loads(path.read_text())
    row = {
        "task": data["eval"]["task"],
        "dataset": data["eval"]["dataset"]["name"],
        "samples": data["results"]["total_samples"],
        "completed_samples": data["results"]["completed_samples"],
        "status": data["status"],
        "model": data["eval"]["model"],
        "metrics": {},
    }
    for score in data["results"].get("scores", []):
        for name, metric in score.get("metrics", {}).items():
            row["metrics"][name] = metric.get("value")
    rows.append(row)
print("INSPECT_SUMMARY_JSON=" + json.dumps(rows, sort_keys=True))
PY
'
```

For MBPP, use the same command shape with these changes:

- set `--timeout 4h`
- change `--label benchmark=mbpp`
- change the task to `inspect_evals/mbpp`
- remove `--temperature 0` so the Inspect MBPP default remains
  `temperature=0.5` with five epochs for pass@k

Fetch results with:

```bash
hf jobs logs <job-id> --tail 3000
```

Copy the final scores into the model repo README table. Verified Inspect
results for the selected adapter:

| Benchmark | Dataset / split | Setting | Score | Job |
| --- | --- | --- | ---: | --- |
| HumanEval | `openai/openai_humaneval` test, 164 samples | accuracy / pass@1, `temperature=0.0` | `0.744 +/- 0.034` | `6a313894fb114ff24a387ee4` |
| MBPP | `google-research-datasets/mbpp` sanitized test, 257 x 5 samples | mean accuracy, `temperature=0.5` | `0.651 +/- 0.030` | `6a31389efb114ff24a387ee6` |
| MBPP | `google-research-datasets/mbpp` sanitized test, 257 x 5 samples | pass@1, `temperature=0.5` | `0.651 +/- 0.030` | `6a31389efb114ff24a387ee6` |
| MBPP | `google-research-datasets/mbpp` sanitized test, 257 x 5 samples | pass@2, `temperature=0.5` | `0.653 +/- 0.030` | `6a31389efb114ff24a387ee6` |
| MBPP | `google-research-datasets/mbpp` sanitized test, 257 x 5 samples | pass@5, `temperature=0.5` | `0.654 +/- 0.030` | `6a31389efb114ff24a387ee6` |

BigCodeBench and SWE-bench are not part of this default example because their
Inspect tasks require Docker-backed execution. Standard Hugging Face Jobs in
this environment did not expose a Docker daemon.

## Known Limits

- Training metrics are held-out prompt/completion loss and token accuracy.
  Use the Inspect AI coding evals above for code-generation benchmark scores.
- Raw traces should still be audited for secrets, private code, and policy
  concerns before treating this as a production recipe.
- Overlength filtering drops many long-context examples at 4k.
- The adapters are private Hub repos by default.
