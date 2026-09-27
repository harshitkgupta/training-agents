# /// script
# dependencies = [
#   "accelerate>=1.13.0",
#   "bitsandbytes>=0.48.0; platform_system == 'Linux'",
#   "datasets>=4.4.0",
#   "huggingface-hub>=1.1.0",
#   "jinja2>=3.1.0",
#   "peft>=0.18.0",
#   "torch>=2.12.0",
#   "trackio>=0.3.0",
#   "transformers>=5.11.0",
#   "trl>=1.6.0",
# ]
# ///

"""SFT a command-selected causal LM on badlogicgames/pi-mono coding-agent traces.

This script intentionally avoids datasets.load_dataset("badlogicgames/pi-mono"):
the Hub dataset is raw session JSONL and the dataset-server table generation can
fail on mixed session content. It downloads raw *.jsonl traces, converts visible
assistant/tool-call turns to prompt/completion examples, and uses TRL
completion-only loss so user prompts and tool outputs are not training targets.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress PyTorch MPS DataLoader pin_memory warning (host pinned memory is unsupported and unneeded on unified memory)
warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*",
    category=UserWarning,
)

# Ensure PyTorch MPS watermark ratios are safe before any torch module initialization
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

# Default Trackio directory to the current working directory so database is created in pwd
os.environ.setdefault("TRACKIO_DIR", os.path.abspath("."))


DEFAULT_DATASET_ID = "badlogicgames/pi-mono"
DEFAULT_PROJECT = "training-agents-sft"
DECODER_LORA_SUFFIX = (
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
LORA_TARGET_REGEX_BY_FAMILY = {
    "gemma": rf".*(?:language_model\.|model\.)?layers\.\d+\.{DECODER_LORA_SUFFIX}",
    "qwen": rf".*(?:model\.)?layers\.\d+\.{DECODER_LORA_SUFFIX}",
    "llama": rf".*(?:model\.)?layers\.\d+\.{DECODER_LORA_SUFFIX}",
    "generic": rf".*layers\.\d+\.{DECODER_LORA_SUFFIX}",
}
GENERIC_LORA_FALLBACK_REGEX = (
    r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


KNOWN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "cmd": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "number", "description": "Optional timeout in milliseconds."},
            },
            "required": [],
        },
    },
    "read": {
        "description": "Read a file or image from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read."},
                "file": {"type": "string", "description": "Path to read."},
                "offset": {"type": "number", "description": "Optional starting line."},
                "limit": {"type": "number", "description": "Optional line limit."},
                "start": {"type": "number", "description": "Optional starting line."},
                "end": {"type": "number", "description": "Optional ending line."},
            },
            "required": [],
        },
    },
    "edit": {
        "description": "Edit a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to edit."},
                "oldText": {"type": "string", "description": "Text to replace."},
                "newText": {"type": "string", "description": "Replacement text."},
                "edits": {"type": "array", "description": "Structured edits."},
                "patch": {"type": "string", "description": "Patch content."},
            },
            "required": [],
        },
    },
    "write": {
        "description": "Write content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": [],
        },
    },
    "grep": {
        "description": "Search text in files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
                "literal": {"type": "boolean", "description": "Treat pattern literally."},
                "context": {"type": "number", "description": "Context lines."},
            },
            "required": [],
        },
    },
    "find": {
        "description": "Find files or text in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to find."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "ls": {
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "todo": {
        "description": "Manage a lightweight task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Task-list action."},
                "text": {"type": "string", "description": "Task text."},
                "id": {"type": "string", "description": "Task identifier."},
            },
            "required": [],
        },
    },
}


@dataclass
class ConversionStats:
    files: int = 0
    events: int = 0
    message_events: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_messages: int = 0
    skipped_messages: int = 0
    json_errors: int = 0
    image_parts: int = 0
    thinking_parts: int = 0
    assistant_examples: int = 0
    skipped_render_errors: int = 0
    skipped_prefix_mismatch: int = 0
    skipped_empty_completion: int = 0


def infer_model_family(model_id: str) -> str:
    slug = model_id.lower()
    if "gemma" in slug:
        return "gemma"
    if "qwen" in slug:
        return "qwen"
    if "llama" in slug:
        return "llama"
    return "generic"


def model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")


def apply_model_runtime_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "model_family", ""):
        args.model_family = infer_model_family(args.model_id)
    slug = model_slug(args.model_id)
    if not args.work_dir:
        args.work_dir = f"workspaces/{slug}-pi-mono-sft"
    if not args.output_dir:
        args.output_dir = f"outputs/{slug}-pi-mono-lora"
    if not args.run_name:
        args.run_name = f"{slug}-pi-mono-lora"
    print(
        "phase=model_profile "
        f"family={args.model_family} model_id={args.model_id} "
        f"work_dir={args.work_dir} output_dir={args.output_dir} run_name={args.run_name}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        required=True,
        help="Hub model id, for example Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen3-0.6B, or google/gemma-4-E2B-it.",
    )
    parser.add_argument(
        "--model-family",
        default="",
        choices=("", "gemma", "qwen", "llama", "generic"),
        help="Optional model family override. Inferred from --model-id if empty.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hub-private-repo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-messages", type=int, default=18)
    parser.add_argument("--max-tool-result-chars", type=int, default=12000)
    parser.add_argument("--max-user-chars", type=int, default=12000)
    parser.add_argument("--max-assistant-chars", type=int, default=12000)
    parser.add_argument("--max-prompt-chars", type=int, default=64000)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--completion-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--filter-overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="",
        help="Comma-separated LoRA module names. Leave empty to infer from --model-id.",
    )
    parser.add_argument(
        "--target-modules-regex",
        default="",
        help="LoRA module regex. Leave empty to infer from --model-id family.",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Training device. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--trackio-project", default=DEFAULT_PROJECT)
    parser.add_argument("--trackio-space-id", default="")
    parser.add_argument("--trackio-group", default="pi-mono-sft-sweep")
    parser.add_argument("--trackio-private-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trackio-dir",
        default=os.environ.get("TRACKIO_DIR", "."),
        help="Directory to store Trackio SQLite databases and media (defaults to current directory or TRACKIO_DIR).",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Path to checkpoint directory, or 'auto' / 'true' to resume from latest checkpoint.",
    )
    return parser.parse_args(argv)


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n\n[... omitted {omitted} chars ...]\n\n{text[-tail:]}"


def stable_example_id(file_name: str, message_id: str, index: int) -> str:
    key = f"{file_name}:{message_id}:{index}".encode("utf-8", errors="replace")
    return hashlib.sha1(key).hexdigest()[:16]


def extract_text_parts(parts: Any, stats: ConversionStats, max_chars: int) -> str:
    if isinstance(parts, str):
        return clip_text(parts.strip(), max_chars)
    if not isinstance(parts, list):
        return ""

    out: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            value = str(part.get("text") or "").strip()
            if value:
                out.append(value)
        elif part_type == "image":
            stats.image_parts += 1
            out.append("[image omitted]")
        elif part_type == "thinking":
            stats.thinking_parts += 1
        elif part_type == "toolCall":
            continue
    return clip_text("\n".join(out).strip(), max_chars)


def convert_tool_call(part: dict[str, Any]) -> dict[str, Any] | None:
    name = part.get("name")
    if not name:
        return None
    arguments = part.get("arguments")
    if arguments is None:
        arguments = {}
    return {
        "id": str(part.get("id") or f"call_{hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]}"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": arguments,
        },
    }


def extract_assistant_message(raw_message: dict[str, Any], stats: ConversionStats, args: argparse.Namespace) -> dict[str, Any] | None:
    parts = raw_message.get("content") or []
    text = extract_text_parts(parts, stats, args.max_assistant_chars)
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[str] = []

    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "toolCall":
                call = convert_tool_call(part)
                if call is not None:
                    tool_calls.append(call)
            elif args.include_reasoning and part.get("type") == "thinking":
                thinking = str(part.get("thinking") or "").strip()
                if thinking:
                    reasoning.append(thinking)

    if not text and not tool_calls:
        return None

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    return message


def raw_event_to_chat_message(event: dict[str, Any], stats: ConversionStats, args: argparse.Namespace) -> dict[str, Any] | None:
    if event.get("type") != "message":
        return None
    stats.message_events += 1
    raw_message = event.get("message") or {}
    role = raw_message.get("role")

    if role == "user":
        content = extract_text_parts(raw_message.get("content") or [], stats, args.max_user_chars)
        if not content:
            stats.skipped_messages += 1
            return None
        stats.user_messages += 1
        return {"role": "user", "content": content}

    if role == "assistant":
        message = extract_assistant_message(raw_message, stats, args)
        if message is None:
            stats.skipped_messages += 1
            return None
        stats.assistant_messages += 1
        return message

    if role == "toolResult":
        content = extract_text_parts(raw_message.get("content") or [], stats, args.max_tool_result_chars)
        if not content:
            content = "[empty tool result]"
        stats.tool_messages += 1
        return {
            "role": "tool",
            "tool_call_id": str(raw_message.get("toolCallId") or ""),
            "name": str(raw_message.get("toolName") or "unknown"),
            "content": content,
        }

    stats.skipped_messages += 1
    return None


def generic_tool_schema(name: str) -> dict[str, Any]:
    return {
        "description": f"Pi coding-agent tool named {name}.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def make_tool_definitions(tool_names: set[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for name in sorted(n for n in tool_names if n):
        schema = KNOWN_TOOL_SCHEMAS.get(name, generic_tool_schema(name))
        tools.append({"type": "function", "function": {"name": name, **schema}})
    return tools


def trim_context(messages: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    original = [copy.deepcopy(message) for message in messages]
    context = list(original)
    if args.max_context_messages > 0 and len(context) > args.max_context_messages:
        tail = context[-args.max_context_messages :]
        if not any(message.get("role") == "user" for message in tail):
            last_user_index = max(
                (index for index, message in enumerate(context) if message.get("role") == "user"),
                default=-1,
            )
            if last_user_index >= 0:
                user_anchor = context[last_user_index]
                tail = [user_anchor] + context[-max(1, args.max_context_messages - 1) :]
        context = tail

    if not any(message.get("role") == "user" for message in context):
        context = [{"role": "user", "content": "Continue the coding-agent session from the preceding context."}] + context

    def compact_value(value: Any, max_chars: int) -> Any:
        if isinstance(value, str):
            return clip_text(value, max_chars)
        if isinstance(value, dict):
            return {key: compact_value(item, max_chars) for key, item in value.items()}
        if isinstance(value, list):
            return [compact_value(item, max_chars) for item in value[:32]]
        return value

    def compact_messages(rows: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
        compacted = [copy.deepcopy(row) for row in rows]
        for row in compacted:
            if isinstance(row.get("content"), str):
                row["content"] = clip_text(row["content"], max_chars)
            if row.get("tool_calls"):
                row["tool_calls"] = compact_value(row["tool_calls"], max_chars)
        return compacted

    for per_field_limit in (4000, 2000, 1000, 500):
        compacted = compact_messages(context, per_field_limit)
        if len(json.dumps(compacted, ensure_ascii=False, default=str)) <= args.max_prompt_chars:
            return compacted
        context = compacted

    while len(json.dumps(context, ensure_ascii=False, default=str)) > args.max_prompt_chars and len(context) > 2:
        del context[1]
    return context


def load_raw_events(path: Path, stats: ConversionStats) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats.events += 1
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                stats.json_errors += 1
    return events


def apply_chat_template(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return processor.apply_chat_template(messages, **kwargs)


def render_prompt_completion(
    processor: Any,
    context: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tools: list[dict[str, Any]],
    model_family: str,
) -> tuple[str, str]:
    kwargs: dict[str, Any] = {"tokenize": False}
    if tools:
        kwargs["tools"] = tools
    if model_family == "qwen":
        kwargs["enable_thinking"] = False
    prompt = apply_chat_template(processor, context, add_generation_prompt=True, **kwargs)
    full = apply_chat_template(
        processor, context + [assistant_message], add_generation_prompt=False, **kwargs
    )
    if not full.startswith(prompt):
        raise ValueError("rendered full conversation does not start with rendered prompt")
    return prompt, full[len(prompt) :]


def build_examples(raw_dir: Path, processor: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], ConversionStats]:
    stats = ConversionStats()
    examples: list[dict[str, Any]] = []
    files = sorted(raw_dir.glob("*.jsonl"))
    if args.max_files > 0:
        files = files[: args.max_files]

    for file_index, path in enumerate(files):
        stats.files += 1
        events = load_raw_events(path, stats)
        tool_names: set[str] = set()
        for event in events:
            if event.get("type") != "message":
                continue
            raw_message = event.get("message") or {}
            for part in raw_message.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "toolCall" and part.get("name"):
                    tool_names.add(str(part["name"]))
        tools = make_tool_definitions(tool_names)

        conversation: list[dict[str, Any]] = []
        for message_index, event in enumerate(events):
            message = raw_event_to_chat_message(event, stats, args)
            if message is None:
                continue

            if message["role"] == "assistant" and any(m.get("role") == "user" for m in conversation):
                context = trim_context(conversation, args)
                try:
                    prompt, completion = render_prompt_completion(
                        processor, context, message, tools, args.model_family
                    )
                except Exception:
                    stats.skipped_render_errors += 1
                else:
                    if not completion.strip():
                        stats.skipped_empty_completion += 1
                    elif not prompt:
                        stats.skipped_prefix_mismatch += 1
                    else:
                        examples.append(
                            {
                                "id": stable_example_id(path.name, str(event.get("id") or ""), message_index),
                                "source_file": path.name,
                                "prompt": prompt,
                                "completion": completion,
                            }
                        )
                        stats.assistant_examples += 1
                        if args.max_examples > 0 and len(examples) >= args.max_examples:
                            return examples, stats

            conversation.append(message)

        if (file_index + 1) % 100 == 0:
            print(f"phase=convert files={file_index + 1} examples={len(examples)}", flush=True)

    return examples, stats


def download_dataset(args: argparse.Namespace) -> Path:
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(raw_dir)
        return raw_dir

    from huggingface_hub import snapshot_download

    raw_dir = Path(args.work_dir) / "raw"
    snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        local_dir=str(raw_dir),
    )
    return raw_dir


def load_processor(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenization_check(processor: Any, examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    sample = examples[: min(32, len(examples))]
    lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in sample:
        encoded = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)
        comp_encoded = tokenizer(row["completion"], add_special_tokens=False)
        lengths.append(len(encoded["input_ids"]))
        completion_lengths.append(len(comp_encoded["input_ids"]))
    over = sum(length > args.max_length for length in lengths)
    return {
        "checked": len(sample),
        "max_length": args.max_length,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "over_max_length": over,
        "min_completion_tokens": min(completion_lengths) if completion_lengths else 0,
        "max_completion_tokens": max(completion_lengths) if completion_lengths else 0,
    }


def filter_examples_by_length(
    processor: Any, examples: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.filter_overlength:
        return examples, {"enabled": False, "kept": len(examples), "dropped": 0}

    tokenizer = getattr(processor, "tokenizer", processor)
    kept: list[dict[str, Any]] = []
    dropped = 0
    lengths: list[int] = []
    for index, row in enumerate(examples):
        encoded = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)
        length = len(encoded["input_ids"])
        lengths.append(length)
        if length <= args.max_length:
            kept.append(row)
        else:
            dropped += 1
        if (index + 1) % 2000 == 0:
            print(
                f"phase=filter_lengths checked={index + 1} kept={len(kept)} dropped={dropped}",
                flush=True,
            )

    return kept, {
        "enabled": True,
        "max_length": args.max_length,
        "checked": len(examples),
        "kept": len(kept),
        "dropped": dropped,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }


def make_dataset_splits(examples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Any, Any]:
    from datasets import Dataset

    rng = random.Random(args.seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    eval_size = min(args.eval_size, max(1, len(shuffled) // 20))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def match_lora_modules(model: Any, pattern: re.Pattern[str]) -> list[tuple[str, Any]]:
    return [(name, module) for name, module in model.named_modules() if pattern.fullmatch(name)]


def resolve_lora_target_modules(model: Any, args: argparse.Namespace) -> list[str]:
    modules = [part.strip() for part in args.target_modules.split(",") if part.strip()]
    if modules:
        print(
            "phase=lora_targets "
            f"mode=explicit count={len(modules)} family={args.model_family} "
            f"modules={json.dumps(modules, sort_keys=True)}",
            flush=True,
        )
        return modules

    regexes: list[str] = []
    if args.target_modules_regex:
        regexes.append(args.target_modules_regex)
    else:
        regexes.append(LORA_TARGET_REGEX_BY_FAMILY[args.model_family])
        if args.model_family != "generic":
            regexes.append(LORA_TARGET_REGEX_BY_FAMILY["generic"])
        regexes.append(GENERIC_LORA_FALLBACK_REGEX)

    seen: set[str] = set()
    unique_regexes = []
    for regex in regexes:
        if regex not in seen:
            seen.add(regex)
            unique_regexes.append(regex)

    for regex in unique_regexes:
        pattern = re.compile(regex)
        matched_modules = match_lora_modules(model, pattern)
        if not matched_modules:
            print(f"phase=lora_targets_miss family={args.model_family} regex={regex}", flush=True)
            continue
        matches = [name for name, _module in matched_modules]
        sample = [
            {"name": name, "type": type(module).__name__}
            for name, module in matched_modules[:8]
        ]
        print(
            "phase=lora_targets "
            f"mode=regex family={args.model_family} count={len(matches)} "
            f"regex={regex} sample={json.dumps(sample, sort_keys=True)}",
            flush=True,
        )
        return matches

    raise RuntimeError(
        f"no LoRA target modules matched for family={args.model_family} model_id={args.model_id}"
    )


def get_system_device_info(runtime_device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"runtime_device": runtime_device}
    if runtime_device == "mps":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["chip"] = brand
        except Exception:
            pass
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["unified_memory_gb"] = round(mem_bytes / (1024**3), 2)
        except Exception:
            pass
    return info


def make_mps_telemetry_callback() -> Any:
    import torch
    from transformers import TrainerCallback

    class MPSTelemetryCallback(TrainerCallback):
        def __init__(self) -> None:
            self.peak_allocated_mb: float = 0.0

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if torch.backends.mps.is_available():
                current_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                self.peak_allocated_mb = max(self.peak_allocated_mb, current_mb)
                if torch.mps.driver_allocated_memory() > 6 * 1024 * 1024 * 1024:
                    torch.mps.empty_cache()

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is not None and torch.backends.mps.is_available():
                current_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
                driver_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
                self.peak_allocated_mb = max(self.peak_allocated_mb, current_mb)
                logs["mps_allocated_mb"] = current_mb
                logs["mps_driver_mb"] = driver_mb
                logs["mps_peak_mb"] = round(self.peak_allocated_mb, 2)

    return MPSTelemetryCallback()


def train(examples: list[dict[str, Any]], processor: Any, stats: ConversionStats, args: argparse.Namespace) -> None:
    if args.trackio_dir:
        resolved_trackio_dir = str(Path(args.trackio_dir).resolve())
        os.environ["TRACKIO_DIR"] = resolved_trackio_dir
        Path(resolved_trackio_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_PROJECT"] = args.trackio_project
    if args.trackio_space_id:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space_id

    import trackio
    import torch
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if args.device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    else:
        runtime_device = args.device
    if runtime_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if runtime_device == "mps" and not mps_available:
        raise RuntimeError("--device mps was requested, but Apple MPS is not available")

    # bitsandbytes 4-bit loading and paged optimizers require CUDA. Apple MPS
    # uses native PyTorch FP16 instead, so make the portable default safe.
    runtime_load_in_4bit = args.load_in_4bit
    runtime_bf16 = args.bf16
    if runtime_device == "mps":
        if runtime_load_in_4bit:
            print("phase=device_adjustment device=mps load_in_4bit=false", flush=True)
            runtime_load_in_4bit = False
        if runtime_bf16:
            print("phase=device_adjustment device=mps bf16=false fp16=true", flush=True)
            runtime_bf16 = False
    if runtime_device == "cpu" and runtime_bf16:
        print("phase=device_adjustment device=cpu bf16=false", flush=True)
        runtime_bf16 = False
    print(f"phase=device device={runtime_device}", flush=True)
    if runtime_device == "mps":
        sys_info = get_system_device_info(runtime_device)
        print(
            "phase=device_info "
            f"device=mps chip={json.dumps(sys_info.get('chip', 'Apple Silicon'))} "
            f"unified_memory_gb={sys_info.get('unified_memory_gb', 'unknown')}",
            flush=True,
        )

    def finish_trackio_safely() -> None:
        try:
            trackio.finish()
        except RuntimeError as exc:
            if "Call trackio.init() before trackio.finish()" in str(exc):
                pass
            else:
                print(f"phase=trackio_finish_warning type={type(exc).__name__} message={exc}", flush=True)
        except Exception as exc:
            print(f"phase=trackio_finish_warning type={type(exc).__name__} message={exc}", flush=True)
        try:
            import glob, sqlite3
            db_dirs = [os.path.expanduser("~/.cache/huggingface/trackio")]
            if os.environ.get("TRACKIO_DIR"):
                db_dirs.insert(0, os.environ["TRACKIO_DIR"])
            for db_dir in db_dirs:
                for db_file in glob.glob(os.path.join(db_dir, "*.db")):
                    try:
                        c = sqlite3.connect(db_file)
                        c.execute("PRAGMA wal_checkpoint(FULL);")
                        c.close()
                    except Exception:
                        pass
        except Exception:
            pass

    trackio_config = {
        "model": args.model_id,
        "dataset": args.dataset_id,
        "method": "sft_lora",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "device": runtime_device,
        "load_in_4bit": runtime_load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "model_family": args.model_family,
        "target_modules_regex": args.target_modules_regex,
        "target_modules": args.target_modules,
        "hub_model_id": args.hub_model_id,
        "completion_only_loss": args.completion_only_loss,
    }
    print(
        "phase=trackio_init "
        f"project={args.trackio_project} run={args.run_name} "
        f"group={args.trackio_group} space_id={args.trackio_space_id or 'local'}",
        flush=True,
    )
    trackio.init(
        project=args.trackio_project,
        name=args.run_name,
        group=args.trackio_group,
        space_id=args.trackio_space_id or None,
        private=args.trackio_private_space if args.trackio_space_id else None,
        config=trackio_config,
    )
    atexit.register(finish_trackio_safely)

    if args.push_to_hub and args.hub_model_id:
        HfApi().create_repo(
            repo_id=args.hub_model_id,
            repo_type="model",
            private=args.hub_private_repo,
            exist_ok=True,
        )

    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if runtime_load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if runtime_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if runtime_bf16 else torch.float16 if runtime_device == "mps" else "auto"
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto" if runtime_device == "cuda" else None,
        "quantization_config": quantization_config,
    }
    if runtime_device in ("mps", "cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    print("phase=load_model", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except (ValueError, TypeError):
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if runtime_device == "mps":
        model.to("mps")
    model.config.use_cache = False

    if runtime_device == "mps" and torch.backends.mps.is_available():
        alloc_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        driver_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
        print(f"phase=mps_memory_post_load allocated_mb={alloc_mb} driver_allocated_mb={driver_mb}", flush=True)

    train_ds, eval_ds = make_dataset_splits(examples, args)
    lora_target_modules = resolve_lora_target_modules(model, args)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    optim = "paged_adamw_8bit" if runtime_load_in_4bit else "adamw_torch"
    training_args = SFTConfig(
        output_dir=args.output_dir,
        project=args.trackio_project,
        trackio_space_id=args.trackio_space_id or None,
        max_length=args.max_length,
        completion_only_loss=args.completion_only_loss,
        packing=False,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=0,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if len(eval_ds) else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=runtime_bf16,
        fp16=(not runtime_bf16) and runtime_device in ("mps", "cuda"),
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * args.max_steps)) if args.max_steps > 0 else 10,
        report_to="trackio",
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
        hub_private_repo=args.hub_private_repo,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=True,
        dataloader_pin_memory=False if runtime_device == "mps" else True,
    )

    write_json(
        Path(args.output_dir) / "conversion_stats.json",
        {
            **stats.__dict__,
            "model_id": args.model_id,
            "model_family": args.model_family,
            "dataset_id": args.dataset_id,
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds),
            "max_length": args.max_length,
            "completion_only_loss": args.completion_only_loss,
            "include_reasoning": args.include_reasoning,
        },
    )

    print(
        "phase=train_start "
        f"train_examples={len(train_ds)} eval_examples={len(eval_ds)} "
        f"model={args.model_id} family={args.model_family} hub_model_id={args.hub_model_id or 'none'}",
        flush=True,
    )
    callbacks = []
    mps_callback = None
    if runtime_device == "mps":
        mps_callback = make_mps_telemetry_callback()
        callbacks.append(mps_callback)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Ensure MPSTelemetryCallback executes before integration callbacks (e.g. TrackioCallback)
    if mps_callback is not None and hasattr(trainer, "callback_handler"):
        if mps_callback in trainer.callback_handler.callbacks:
            trainer.callback_handler.callbacks.remove(mps_callback)
        trainer.callback_handler.callbacks.insert(0, mps_callback)
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() in ("true", "auto", "1"):
            resume_checkpoint = True
        else:
            resume_checkpoint = args.resume_from_checkpoint
        print(f"phase=resume_from_checkpoint target={resume_checkpoint}", flush=True)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    metrics = dict(train_result.metrics)
    metrics["train_examples"] = len(train_ds)
    metrics["eval_examples"] = len(eval_ds)
    if runtime_device == "mps" and torch.backends.mps.is_available():
        final_alloc_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        final_driver_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
        peak_mb = round(getattr(mps_callback, "peak_allocated_mb", final_alloc_mb), 2)
        metrics["peak_mps_allocated_mb"] = peak_mb
        metrics["final_mps_driver_mb"] = final_driver_mb
        torch.mps.empty_cache()
    trainer.save_metrics("train", metrics)

    if len(eval_ds):
        final_eval_metrics = next(
            (
                row
                for row in reversed(trainer.state.log_history)
                if any(str(key).startswith("eval_") for key in row)
            ),
            None,
        )
        if final_eval_metrics is None:
            try:
                final_eval_metrics = trainer.evaluate()
            except Exception:
                pass
        if final_eval_metrics:
            trainer.save_metrics("eval", final_eval_metrics)

    if args.push_to_hub:
        print("phase=push_to_hub", flush=True)
        trainer.push_to_hub()

    try:
        write_json(
            Path(args.output_dir) / "COMPLETED",
            {
                "run_name": args.run_name,
                "global_step": int(trainer.state.global_step),
                "max_steps": args.max_steps,
                "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
                "completed": True,
            },
        )
    except Exception:
        pass

    finish_trackio_safely()
    atexit.unregister(finish_trackio_safely)
    print("phase=done", json.dumps(metrics, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    apply_model_runtime_defaults(args)
    random.seed(args.seed)
    raw_dir = download_dataset(args)
    processor = load_processor(args.model_id)
    examples, stats = build_examples(raw_dir, processor, args)
    examples, length_filter = filter_examples_by_length(processor, examples, args)
    if len(examples) < 2:
        raise RuntimeError(f"not enough examples after conversion: {len(examples)}")

    work_dir = Path(args.work_dir)
    write_jsonl(work_dir / "prepared_examples.sample.jsonl", examples[: min(100, len(examples))])
    shuffled_rows = list(examples)
    random.Random(args.seed).shuffle(shuffled_rows)
    eval_size = min(args.eval_size, max(1, len(shuffled_rows) // 20)) if args.eval_size > 0 else 0
    mlx_dir = work_dir / "mlx_data"
    mlx_rows = [
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
            "text": row["prompt"] + row["completion"],
        }
        for row in shuffled_rows
    ]
    write_jsonl(mlx_dir / "train.jsonl", mlx_rows[eval_size:])
    write_jsonl(mlx_dir / "valid.jsonl", mlx_rows[:eval_size])
    summary: dict[str, Any] = {
        **stats.__dict__,
        "model_id": args.model_id,
        "model_family": args.model_family,
        "dataset_id": args.dataset_id,
        "raw_dir": str(raw_dir),
        "examples": len(examples),
        "length_filter": length_filter,
        "include_reasoning": args.include_reasoning,
    }
    if not args.skip_tokenizer_check:
        summary["tokenization_check"] = tokenization_check(processor, examples, args)
    write_json(work_dir / "conversion_summary.json", summary)
    print("phase=conversion_summary", json.dumps(summary, sort_keys=True), flush=True)

    if args.prepare_only:
        return

    train(examples, processor, stats, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"phase=error type={type(exc).__name__} message={exc}", file=sys.stderr, flush=True)
        raise
