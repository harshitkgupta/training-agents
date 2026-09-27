# /// script
# dependencies = []
# ///

"""Inspect and compare all Trackio training runs from local SQLite databases."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path


def get_trackio_dbs(target_path: Path | str | None = None) -> list[Path]:
    """Get Trackio SQLite database(s) strictly from the configured location.

    Only inspects the path provided via CLI argument or TRACKIO_DIR environment variable
    (defaulting to the current working directory). No guessing or searching in cache directories.
    """
    raw_path = target_path if target_path else os.environ.get("TRACKIO_DIR", ".")
    target = Path(raw_path).resolve()

    if not target.exists():
        return []

    if target.is_file():
        return [target] if target.suffix == ".db" else []

    # If directory, find .db files directly in that directory
    return sorted([
        p for p in target.glob("*.db")
        if not p.name.endswith(("-shm", "-wal"))
    ])


def inspect_db(db_path: Path) -> list[dict]:
    project = db_path.stem
    runs_data = []

    try:
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            except Exception:
                pass
        except Exception:
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            except Exception:
                return []

        with conn:
            cursor = conn.cursor()
        
        # Check tables
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        if "metrics" not in tables:
            return []

        # Find distinct runs
        cursor.execute("SELECT DISTINCT run_id, run_name FROM metrics;")
        runs = cursor.fetchall()
        if not runs and "configs" in tables:
            cursor.execute("SELECT DISTINCT run_id, run_name FROM configs;")
            runs = cursor.fetchall()

        for run_id, run_name in runs:
            run_info: dict = {
                "project": project,
                "db_path": str(db_path),
                "run_id": run_id,
                "run_name": run_name,
                "model": "unknown",
                "learning_rate": "unknown",
                "max_steps": 0,
                "total_logged_steps": 0,
                "initial_train_loss": None,
                "final_train_loss": None,
                "best_train_loss": None,
                "final_eval_loss": None,
                "best_eval_loss": None,
                "final_token_acc": None,
                "peak_token_acc": None,
                "peak_mps_mb": None,
                "avg_it_sec": None,
                "created_at": None,
            }

            # Get config if exists
            if "configs" in tables:
                cursor.execute("SELECT config, created_at FROM configs WHERE run_id = ? LIMIT 1;", (run_id,))
                cfg_row = cursor.fetchone()
                if cfg_row:
                    try:
                        cfg = json.loads(cfg_row[0])
                        run_info["model"] = cfg.get("model") or cfg.get("model_id") or "unknown"
                        run_info["learning_rate"] = cfg.get("learning_rate") or "unknown"
                        run_info["max_steps"] = cfg.get("max_steps") or cfg.get("iters") or 0
                    except Exception:
                        pass
                    run_info["created_at"] = cfg_row[1]

            # Query all metric steps ordered
            cursor.execute(
                "SELECT step, metrics, timestamp FROM metrics WHERE run_id = ? ORDER BY step ASC;",
                (run_id,)
            )
            rows = cursor.fetchall()
            run_info["total_logged_steps"] = len(rows)

            train_losses = []
            eval_losses = []
            token_accs = []
            mps_mems = []
            it_secs = []

            for step, metrics_raw, ts in rows:
                if not run_info["created_at"]:
                    run_info["created_at"] = ts
                if isinstance(metrics_raw, bytes):
                    try:
                        metrics_raw = metrics_raw.decode("utf-8")
                    except Exception:
                        pass
                try:
                    m = json.loads(metrics_raw)
                except Exception:
                    continue

                # Train loss
                tl = (
                    m.get("train/loss")
                    or m.get("train_loss")
                    or m.get("loss")
                    or m.get("training_loss")
                )
                if tl is not None:
                    try:
                        train_losses.append((step, float(tl)))
                    except (ValueError, TypeError):
                        pass

                # Eval loss
                el = (
                    m.get("eval/loss")
                    or m.get("eval_loss")
                    or m.get("val_loss")
                    or m.get("validation_loss")
                )
                if el is not None:
                    try:
                        eval_losses.append((step, float(el)))
                    except (ValueError, TypeError):
                        pass

                # Token accuracy
                ta = (
                    m.get("eval/mean_token_accuracy")
                    or m.get("train/mean_token_accuracy")
                    or m.get("eval_mean_token_accuracy")
                    or m.get("mean_token_accuracy")
                )
                if ta is not None:
                    try:
                        token_accs.append((step, float(ta)))
                    except (ValueError, TypeError):
                        pass

                # Memory (handle both MB-scale and GB-scale keys cleanly)
                mem_mb = None
                for k in ("mps_peak_mb", "train/mps_peak_mb", "peak_mps_allocated_mb", "mps_allocated_mb", "train/mps_allocated_mb"):
                    if m.get(k) is not None:
                        try:
                            mem_mb = float(m[k])
                            break
                        except (ValueError, TypeError):
                            pass
                if mem_mb is None and m.get("peak_mem_gb") is not None:
                    try:
                        mem_mb = float(m["peak_mem_gb"]) * 1024.0
                    except (ValueError, TypeError):
                        pass
                if mem_mb is not None:
                    mps_mems.append(mem_mb)

                # Throughput speed
                spd = (
                    m.get("it_sec")
                    or m.get("train/train_steps_per_second")
                    or m.get("train_steps_per_second")
                    or m.get("eval/steps_per_second")
                )
                if spd is not None:
                    try:
                        it_secs.append(float(spd))
                    except (ValueError, TypeError):
                        pass

            if not it_secs and len(rows) >= 2:
                try:
                    from datetime import datetime
                    t_start = datetime.fromisoformat(rows[0][2])
                    t_end = datetime.fromisoformat(rows[-1][2])
                    duration = (t_end - t_start).total_seconds()
                    step_delta = rows[-1][0] - rows[0][0]
                    if duration > 0 and step_delta > 0:
                        it_secs.append(step_delta / duration)
                except Exception:
                    pass

            if train_losses:
                run_info["initial_train_loss"] = round(train_losses[0][1], 4)
                run_info["final_train_loss"] = round(train_losses[-1][1], 4)
                run_info["best_train_loss"] = round(min(v for _, v in train_losses), 4)

            if eval_losses:
                run_info["final_eval_loss"] = round(eval_losses[-1][1], 4)
                run_info["best_eval_loss"] = round(min(v for _, v in eval_losses), 4)

            if token_accs:
                run_info["final_token_acc"] = round(token_accs[-1][1] * (100 if token_accs[-1][1] <= 1.0 else 1), 2)
                run_info["peak_token_acc"] = round(max(v for _, v in token_accs) * (100 if max(v for _, v in token_accs) <= 1.0 else 1), 2)

            if mps_mems:
                run_info["peak_mps_mb"] = round(max(mps_mems), 1)

            if it_secs:
                run_info["avg_it_sec"] = round(sum(it_secs) / len(it_secs), 2)

            runs_data.append(run_info)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return runs_data


def print_comparison_table(all_runs: list[dict]) -> None:
    if not all_runs:
        print("No Trackio runs with metrics found.")
        return

    # Deduplicate runs by (project, run_name), keeping the run with the most logged steps
    deduped: dict[tuple[str, str], dict] = {}
    for r in all_runs:
        key = (r["project"], r["run_name"])
        if key not in deduped or r["total_logged_steps"] > deduped[key]["total_logged_steps"]:
            deduped[key] = r

    runs_to_show = sorted(deduped.values(), key=lambda r: (r["project"], r["run_name"]))

    print("=" * 128)
    print(f"{'TRACKIO RUNS COMPARISON':^128}")
    print("=" * 128)
    header = f"{'Project':<20} | {'Run Name':<28} | {'Steps':<6} | {'Train Loss (Beg->End)':<22} | {'Eval Loss':<10} | {'Token Acc':<10} | {'Speed':<9} | {'Peak VRAM':<9}"
    print(header)
    print("-" * 128)

    for r in runs_to_show:
        proj = r["project"][:19]
        name = r["run_name"][:27]
        steps = f"{r['total_logged_steps']}"

        tl_beg = f"{r['initial_train_loss']:.3f}" if r['initial_train_loss'] is not None else "-"
        tl_end = f"{r['final_train_loss']:.3f}" if r['final_train_loss'] is not None else "-"
        tl_str = f"{tl_beg} -> {tl_end}" if tl_beg != "-" else "-"

        el_str = f"{r['final_eval_loss']:.4f}" if r['final_eval_loss'] is not None else "-"
        acc_str = f"{r['final_token_acc']:.1f}%" if r['final_token_acc'] is not None else "-"
        spd_str = f"{r['avg_it_sec']:.2f} it/s" if r.get('avg_it_sec') is not None else "-"
        vram_str = f"{r['peak_mps_mb']:.0f} MB" if r['peak_mps_mb'] is not None else "-"

        print(f"{proj:<20} | {name:<28} | {steps:<6} | {tl_str:<22} | {el_str:<10} | {acc_str:<10} | {spd_str:<9} | {vram_str:<9}")

    print("=" * 128)
    # Print distinct database sources so user knows exactly which DB files were used
    db_sources = sorted({r.get("db_path", "") for r in runs_to_show if r.get("db_path")})
    if db_sources:
        print("Database source(s):")
        for src in db_sources:
            print(f"  • {src}")
        print("=" * 128)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and compare Trackio training runs from local SQLite database.")
    parser.add_argument(
        "--trackio-dir",
        default=os.environ.get("TRACKIO_DIR", "."),
        help="Directory or .db file path to inspect (defaults to TRACKIO_DIR env var or current directory).",
    )
    args = parser.parse_args()

    target_path = Path(args.trackio_dir).resolve()
    db_files = get_trackio_dbs(target_path)
    if not db_files:
        print(f"No Trackio databases (*.db) found in configured location: {target_path}")
        print("Run training in this directory first, or specify --trackio-dir <path>.")
        return

    print(f"Found {len(db_files)} Trackio database(s) in {target_path}:")
    for f in db_files:
        print(f"  - {f}")
    print()

    all_runs = []
    for db in db_files:
        all_runs.extend(inspect_db(db))

    print_comparison_table(all_runs)


if __name__ == "__main__":
    main()
