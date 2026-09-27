---
name: apple-silicon-ml
description: Set up and run local ML, fine-tuning, or inference work on Apple-Silicon Macs using PyTorch MPS or Apple MLX. Use for M1–M5 Mac device selection, memory sizing, CUDA-to-MPS adaptation, and MLX 4-bit QLoRA workflows.
---

# Apple Silicon ML

Use this skill when running local ML, fine-tuning, or inference on Apple Silicon (M1–M5) Macs.

## 1. Choose the Framework: MPS vs. MLX

Apple Silicon offers two distinct post-training execution paths with significant trade-offs:

| Criterion | PyTorch MPS (`torch.device("mps")`) | Apple MLX (`mlx-lm`) |
| :--- | :--- | :--- |
| **Quantization** | FP16/BF16 base (no bitsandbytes 4-bit support) | Native 4-bit QLoRA (`mlx-community` weights) |
| **Max Model Size** | 0.5B – 3B comfortably on 16GB–36GB Macs | 7B – 27B+ via 4-bit weight compression |
| **Throughput (Short Seq)** | **Ultra-fast** (~5.2 it/s on 0.5B, ~2.7 it/s on 1.5B) | Slower (~0.15–0.45 it/s due to quantized matmuls) |
| **Long Context (4k+)** | Memory cliff / abrupt Metal allocator OOM | Flat, linear memory scaling; no memory cliff |
| **Best For** | Fast iteration on <=3B models, <=2k tokens | 7B+ models, 4k+ long-context traces on constrained RAM |

## 2. PyTorch MPS Environment & Allocator Gotchas

### Memory Watermark Crash Rule
To disable PyTorch MPS allocation limits without crashing, **both** watermark ratios must be set to `0.0` in the environment **before** `import torch`:
```python
# CRITICAL: Setting only HIGH watermark to 0.0 raises "RuntimeError: invalid low watermark ratio"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
import torch
```

### Memory Duplication & DataLoader Rules
- **No pinned memory**: Set `pin_memory=False` in PyTorch DataLoaders. Pinned memory targets CUDA host-to-device transfers and wastes unified RAM on Apple Silicon.
- **Single worker**: Set `num_workers=0` (or `1`). macOS multiprocessing uses `spawn`, which duplicates process memory and inflates RAM usage.

### HF Trainer Callback Ordering for MPS Telemetry
When integrating custom MPS memory telemetry (e.g. `torch.mps.current_allocated_memory()`) with `TrackioCallback` or other loggers:
- Upstream `Trainer` runs built-in integration callbacks *before* user callbacks.
- User callbacks appending metrics during `on_step_end` miss the logging step.
- **Fix**: Prepend your memory callback to the beginning of the handler:
  ```python
  trainer.callback_handler.callbacks.insert(0, MPSMemoryCallback())
  ```

## 3. Execution & OS-Level Safeguards

### Prevent System Sleep and GPU Throttling
macOS aggressively throttles background processes and engages App Nap or sleep during unattended training runs, dropping GPU utilization:
```bash
# Wrap training runs with caffeinate to prevent display/system sleep and disk idle
caffeinate -dimsu uv run python train_sft.py ...
```

### Detect Unified Memory Swap Thrashing
Unified memory is shared dynamically between macOS, display buffers, and ML allocations.
- Monitor swap usage during runs:
  ```bash
  sysctl vm.swapusage
  ```
- If memory pressure causes macOS to page Metal GPU buffers to SSD, throughput collapses by 100x–1000x. If swap usage increases, immediately reduce `per_device_train_batch_size` or `max_seq_length`.

## 4. MLX Training & Artifact Handling

### Step-Based Checkpoint Sorting Bug
MLX checkpoint directories follow the pattern `<step>_adapters` (e.g., `90_adapters`, `500_adapters`). Lexicographical sorting incorrectly ranks `90_adapters` after `500_adapters`.
- Always parse the step numerically when discovering or loading checkpoints:
  ```python
  checkpoints = sorted(
      adapters_dir.glob("*_adapters"),
      key=lambda p: int(p.name.split("_")[0])
  )
  final_adapter = checkpoints[-1]
  ```

### Data Formatting
MLX `mlx-lm` requires `.jsonl` files formatted with a `"text"` key containing pre-templated chat strings (e.g., formatted via `tokenizer.apply_chat_template`), placed in train/valid/test splits.

## 5. Trackio Observability on Apple Silicon

### Pre-Import Configuration
`trackio` binds `TRACKIO_DIR` at module import time. Set the target directory before importing `trackio`:
```python
import os
os.environ.setdefault("TRACKIO_DIR", os.path.abspath("."))
import trackio
```

### Deterministic DB Discovery
Never rely on multi-directory heuristic searches. Configure the database directory explicitly via `--trackio-dir` or `TRACKIO_DIR`, and force SQLite WAL flushes with `PRAGMA wal_checkpoint(FULL)` if reading metrics during active runs.

## 6. References

- [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX Framework GitHub](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Apple Silicon Unified Memory Architecture](https://developer.apple.com/documentation/apple-silicon)
