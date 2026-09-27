# /// script
# dependencies = [
#   "huggingface-hub>=1.1.0",
# ]
# ///

"""Publish already-trained LoRA adapters under outputs/ to the Hugging Face Hub.

This is a batch, retroactive publisher for runs that have already completed --
it does not train anything and does not invoke train_sft.py or train_mlx.py.
It scans outputs/<run-name>/ directories, auto-detects whether each is an MLX
adapter (train_mlx.py) or a PEFT/MPS adapter (train_sft.py) by file signature,
pulls that run's logged config and final metrics from the local Trackio
database (reusing compare_runs.py's own DB-reading code, not a reimplementation
of it), writes a model card if one doesn't already exist, and pushes the final
adapter only -- intermediate/resume checkpoints are excluded per backend.

Usage:
  uv run tutorials/01-sft-on-traces/publish_adapters.py --dry-run
  uv run tutorials/01-sft-on-traces/publish_adapters.py
  uv run tutorials/01-sft-on-traces/publish_adapters.py --runs qwen-0.5b-mlx-2k qwen-1.5b-mps-2k
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import get_trackio_dbs, inspect_db  # noqa: E402
from train_mlx import MLX_LORA_DEFAULTS  # noqa: E402

BACKEND_EXCLUDES = {
    "mlx": ["[0-9]*_adapters.safetensors"],
    "mps": ["checkpoint-*"],
}


def detect_backend(run_dir: Path) -> str | None:
    if (run_dir / "adapters.safetensors").exists():
        return "mlx"
    if (run_dir / "adapter_config.json").exists():
        return "mps"
    return None


def is_completed(run_dir: Path) -> bool:
    return (run_dir / "COMPLETED").exists()


def load_trackio_runs(trackio_dir: str) -> dict[str, dict[str, Any]]:
    """run_name -> the richest inspect_db() row seen for that name, across all
    local .db files, matching compare_runs.py's own dedup rule (most logged
    steps wins) so this stays consistent with what `compare_runs.py` prints."""
    by_name: dict[str, dict[str, Any]] = {}
    for db_path in get_trackio_dbs(trackio_dir):
        for row in inspect_db(db_path):
            name = row.get("run_name")
            if not name:
                continue
            existing = by_name.get(name)
            if existing is None or row.get("total_logged_steps", 0) > existing.get("total_logged_steps", 0):
                by_name[name] = row
    return by_name


def load_raw_configs(trackio_dir: str, trackio_runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """run_name -> full logged config dict (not just the narrow fields
    inspect_db() extracts), so LoRA rank/alpha/scale can be read per run
    when a training script actually logged them, instead of assumed.

    Reads the config only from each run's winning db_path (the one
    load_trackio_runs() already picked via "most logged steps wins"), rather
    than the first db file this function happens to iterate to. Otherwise, if
    the same run_name exists in two local .db files (the exact scenario
    compare_runs.py --diff / --find-all exists to catch), the config and the
    metrics shown for a run could silently come from two different rows.
    """
    by_name: dict[str, dict[str, Any]] = {}
    configs_by_db: dict[Path, dict[str, str]] = {}
    for db_path in get_trackio_dbs(trackio_dir):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception:
            continue
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            if "configs" not in tables:
                continue
            configs_by_db[db_path] = {
                run_name: config_raw
                for run_name, config_raw in conn.execute("SELECT run_name, config FROM configs;").fetchall()
                if run_name
            }
        except Exception:
            continue
        finally:
            conn.close()

    for run_name, row in trackio_runs.items():
        winning_db = row.get("db_path")
        if not winning_db:
            continue
        raw = configs_by_db.get(Path(winning_db), {}).get(run_name)
        if raw is None:
            continue
        try:
            by_name[run_name] = json.loads(raw)
        except Exception:
            continue
    return by_name


def discover_runs(outputs_dir: Path, force: bool) -> dict[str, str]:
    runs: dict[str, str] = {}
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        backend = detect_backend(child)
        if backend is None:
            continue
        if not force and not is_completed(child):
            print(f"phase=skip_incomplete run={child.name}", flush=True)
            continue
        runs[child.name] = backend
    return runs


def build_model_card(
    run_name: str,
    hub_model_id: str,
    backend: str,
    run_dir: Path,
    trackio_row: dict[str, Any] | None,
    raw_config: dict[str, Any] | None,
    trackio_space_id: str,
) -> str:
    base_model = "unknown"
    max_steps = "unknown"
    learning_rate = "unknown"
    eval_loss = "unknown"

    if trackio_row:
        base_model = trackio_row.get("model") or base_model
        max_steps = trackio_row.get("max_steps") or max_steps
        learning_rate = trackio_row.get("learning_rate") or learning_rate
        if trackio_row.get("final_eval_loss") is not None:
            eval_loss = f"{trackio_row['final_eval_loss']:.4f}"

    raw_config = raw_config or {}

    if backend == "mlx":
        # Older runs never logged rank/scale/dropout at all (see
        # MLX_LORA_DEFAULTS above); newer ones do, via train_mlx.py's own
        # trackio config -- prefer the logged values when present.
        rank = raw_config.get("lora_rank", MLX_LORA_DEFAULTS["rank"])
        scale = raw_config.get("lora_scale", MLX_LORA_DEFAULTS["scale"])
        dropout = raw_config.get("lora_dropout", MLX_LORA_DEFAULTS["dropout"])
        source = "logged in this run's config" if "lora_rank" in raw_config else "mlx_lm.lora defaults; not logged by this run"
        lora_line = f"- LoRA: rank {rank}, scale {scale}, dropout {dropout} ({source})\n"
        # "mlx16" is this project's own run-naming convention for the
        # non-quantized comparison runs (see run_experiments.py's SUITES) --
        # more reliable here than pattern-matching the base model id, which
        # doesn't carry a "-4bit" suffix for the unquantized runs either way.
        precision = "16-bit (bf16), unquantized" if "mlx16" in run_name else "4-bit QLoRA"
    else:
        rank = raw_config.get("lora_r", 16)
        alpha = raw_config.get("lora_alpha", 32)
        source = "logged in this run's config" if "lora_r" in raw_config else "train_sft.py default, not found in this run's logged config"
        lora_line = f"- LoRA: rank {rank}, alpha {alpha} ({source})\n"
        precision = "FP16"

    dashboard_line = (
        f"Trackio dashboard: https://huggingface.co/spaces/{trackio_space_id}\n\n" if trackio_space_id else ""
    )

    return (
        "---\n"
        f"base_model: {base_model}\n"
        f"library_name: {'mlx' if backend == 'mlx' else 'peft'}\n"
        "pipeline_tag: text-generation\n"
        "tags:\n"
        "  - lora\n"
        "  - pi-mono\n"
        f"  - {backend}\n"
        "---\n\n"
        f"# {hub_model_id.split('/')[-1]}\n\n"
        f"LoRA adapter fine-tuned on `badlogicgames/pi-mono` coding-agent traces "
        f"with {'Apple MLX' if backend == 'mlx' else 'PyTorch MPS'} ({precision}).\n\n"
        f"- Base model: `{base_model}`\n"
        f"- Backend: {backend}\n"
        f"- Training steps: {max_steps}\n"
        f"- Learning rate: {learning_rate}\n"
        f"- Final eval loss: {eval_loss}\n"
        f"{lora_line}\n"
        f"{dashboard_line}"
        "Trained with [training-agents](https://github.com/harshitkgupta/training-agents) "
        f"`tutorials/01-sft-on-traces/{'train_mlx.py' if backend == 'mlx' else 'train_sft.py'}`, "
        f"run name `{run_name}`.\n"
    )


def list_uploadable_files(run_dir: Path, exclude: list[str]) -> list[tuple[str, int]]:
    """Relative paths (and sizes) that would actually be uploaded, using the
    same filtering huggingface_hub applies internally -- not a re-derived
    approximation of it."""
    from huggingface_hub.utils import filter_repo_objects

    all_files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    kept = list(filter_repo_objects(all_files, ignore_patterns=exclude))
    return [(rel, (run_dir / rel).stat().st_size) for rel in kept]


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def push_run(
    run_name: str,
    run_dir: Path,
    backend: str,
    hub_model_id: str,
    private: bool,
    trackio_row: dict[str, Any] | None,
    raw_config: dict[str, Any] | None,
    trackio_space_id: str,
    overwrite_cards: bool,
    dry_run: bool,
) -> bool:
    exclude = BACKEND_EXCLUDES[backend]
    readme_path = run_dir / "README.md"
    card_written = False
    card_exists = readme_path.exists()
    preview_card_text: str | None = None
    if overwrite_cards or not card_exists:
        card = build_model_card(run_name, hub_model_id, backend, run_dir, trackio_row, raw_config, trackio_space_id)
        action = "overwrite" if card_exists else "write"
        preview_card_text = card
        if dry_run:
            print(f"phase=would_{action}_card run={run_name} path={readme_path}", flush=True)
        else:
            readme_path.write_text(card, encoding="utf-8")
            card_written = True
    else:
        print(f"phase=keep_existing_card run={run_name} path={readme_path}", flush=True)
        if dry_run:
            preview_card_text = readme_path.read_text(encoding="utf-8")

    if dry_run:
        print(
            f"[DRY-RUN] would push {run_dir} -> {hub_model_id} "
            f"(backend={backend}, private={private}, exclude={exclude})",
            flush=True,
        )
        files = list_uploadable_files(run_dir, exclude)
        total = sum(size for _, size in files)
        print(f"  files ({len(files)}, {format_size(total)} total):", flush=True)
        for rel_path, size in files:
            print(f"    {rel_path}  ({format_size(size)})", flush=True)
        if preview_card_text is not None:
            label = "existing (kept)" if not (overwrite_cards or not card_exists) else ("new" if not card_exists else "overwritten")
            print(f"  README.md [{label}]:", flush=True)
            print("  " + "-" * 76, flush=True)
            for line in preview_card_text.splitlines():
                print(f"  | {line}", flush=True)
            print("  " + "-" * 76, flush=True)
        print(flush=True)
        return True

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hub_model_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=hub_model_id,
            repo_type="model",
            ignore_patterns=exclude,
            commit_message=f"Push {run_name} adapter",
        )
        print(f"phase=push_done run={run_name} url=https://huggingface.co/{hub_model_id}", flush=True)
        return True
    except Exception as exc:
        if card_written:
            print(f"phase=push_failed_card_kept run={run_name} readme={readme_path}", flush=True)
        print(f"phase=push_failed run={run_name} type={type(exc).__name__} message={exc}", flush=True)
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing run-name subdirectories.")
    parser.add_argument("--trackio-dir", default=".", help="Directory holding the Trackio .db file(s) (defaults to current directory).")
    parser.add_argument("--namespace", default="harshitkgupta", help="HF username/org to push under.")
    parser.add_argument("--suffix", default="-lora", help="Suffix appended to the run name for the Hub repo name.")
    parser.add_argument("--trackio-space-id", default="", help="Optional hosted Trackio Space id to link in generated model cards.")
    parser.add_argument("--runs", nargs="+", default=None, help="Specific run names to push (default: every completed run found under --outputs-dir).")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True, help="Create Hub repos as private (default: yes).")
    parser.add_argument(
        "--overwrite-cards",
        action="store_true",
        help="Regenerate README.md even if one already exists (e.g. PEFT's auto-generated one for train_sft.py runs), so every run ends up with the same, richer card format.",
    )
    parser.add_argument("--force", action="store_true", help="Push even if the COMPLETED marker is missing.")
    parser.add_argument(
        "--set-visibility",
        choices=["public", "private"],
        default=None,
        help="Skip the push entirely and just change visibility on the Hub repos for the selected runs (uses the same --runs/--namespace/--suffix resolution as a normal push).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be pushed without touching the Hub.")
    return parser.parse_args(argv)


def set_visibility(hub_model_id: str, private: bool, dry_run: bool) -> bool:
    if dry_run:
        print(f"[DRY-RUN] would set {hub_model_id} private={private}", flush=True)
        return True
    try:
        from huggingface_hub import HfApi

        HfApi().update_repo_settings(repo_id=hub_model_id, private=private, repo_type="model")
        print(f"phase=visibility_done repo={hub_model_id} private={private}", flush=True)
        return True
    except Exception as exc:
        print(f"phase=visibility_failed repo={hub_model_id} type={type(exc).__name__} message={exc}", flush=True)
        return False


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        sys.exit(f"outputs dir not found: {outputs_dir}")

    if args.runs:
        runs: dict[str, str] = {}
        for name in args.runs:
            run_dir = outputs_dir / name
            if not run_dir.exists():
                print(f"phase=skip_missing run={name}", flush=True)
                continue
            backend = detect_backend(run_dir)
            if backend is None:
                print(f"phase=skip_unrecognized run={name}", flush=True)
                continue
            if not args.force and not is_completed(run_dir):
                print(f"phase=skip_incomplete run={name}", flush=True)
                continue
            runs[name] = backend
    else:
        runs = discover_runs(outputs_dir, args.force)

    if not runs:
        sys.exit("No completed, recognizable runs found to push.")

    if args.set_visibility is not None:
        private = args.set_visibility == "private"
        print(f"phase=plan_visibility runs={len(runs)} set={args.set_visibility}", flush=True)
        results = {
            run_name: set_visibility(f"{args.namespace}/{run_name}{args.suffix}", private, args.dry_run)
            for run_name in runs
        }
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            print(f"phase=done_with_failures updated={len(results) - len(failed)} failed={failed}", flush=True)
            sys.exit(1)
        print(f"phase=done updated={len(results)}", flush=True)
        return

    trackio_runs = load_trackio_runs(args.trackio_dir)
    raw_configs = load_raw_configs(args.trackio_dir, trackio_runs)
    print(f"phase=plan runs={len(runs)} namespace={args.namespace} trackio_rows_found={len(trackio_runs)}", flush=True)

    results: dict[str, bool] = {}
    for run_name, backend in runs.items():
        hub_model_id = f"{args.namespace}/{run_name}{args.suffix}"
        results[run_name] = push_run(
            run_name=run_name,
            run_dir=outputs_dir / run_name,
            backend=backend,
            hub_model_id=hub_model_id,
            private=args.private,
            trackio_row=trackio_runs.get(run_name),
            raw_config=raw_configs.get(run_name),
            trackio_space_id=args.trackio_space_id,
            overwrite_cards=args.overwrite_cards,
            dry_run=args.dry_run,
        )

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"phase=done_with_failures pushed={len(results) - len(failed)} failed={failed}", flush=True)
        sys.exit(1)
    print(f"phase=done pushed={len(results)}", flush=True)


if __name__ == "__main__":
    main()
