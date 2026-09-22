from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


ROLE_TOKEN_RE = re.compile(r"<(?:SELF|PARENT|USER(?:_\d+)?)>", re.IGNORECASE)
LEADING_ROUTE_RE = re.compile(
    r"^\s*(?:(?:<(?:SELF|PARENT|USER(?:_\d+)?)>\s*)+)?"
    r"(?:\[replying\s+to\s+[^\]]+\]\s*:\s*)?",
    re.IGNORECASE,
)
LEADING_MENTION_RE = re.compile(r"^(?:(?:@[A-Za-z0-9_]{1,30})[ \t]*)+(?::[ \t]*)?")
LINK_TOKEN_RE = re.compile(r"<LINK(?::[^>]+)?>", re.IGNORECASE)
MEDIA_TOKEN_RE = re.compile(r"<MEDIA(?::[^>]+)?>", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿĀ-žА-Яа-я一-龯ぁ-んァ-ンー'’]+", re.UNICODE)

LOW_INFO = {
    "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "true", "real",
    "lol", "lmao", "lmfao", "thanks", "thank you", "what", "why", "huh", "same",
}

REPLAY_PROMPTS = [
    "What is 17 multiplied by 6?",
    "What is 23 plus 19?",
    "What is 144 divided by 12?",
    "What is 9 squared?",
    "What is 5 times 8 minus 3?",
    "What is the capital of France?",
    "What is the capital of Japan?",
    "At what temperature does pure water freeze in Celsius?",
    "Which planet is closest to the Sun?",
    "Name the largest ocean on Earth.",
    "Repeat exactly: cedar-482",
    "Repeat exactly: violet seven",
    "Copy this without changing it: A9-zebra-41",
    "The code is maple-731. What is the code?",
    "I said the number 284. What number did I say?",
    "Answer with only the last word: red green blue",
    "Which is larger, 0.7 or 0.65?",
    "If a train travels 60 km in one hour, how far in three hours?",
    "A box has 12 apples and 5 are removed. How many remain?",
    "If today is Monday, what day is two days later?",
    "Explain briefly why the daytime sky looks blue.",
    "What gas do humans breathe in to survive?",
    "What is H2O commonly called?",
    "Who wrote Romeo and Juliet?",
    "What is the opposite of north?",
    "Sort these numbers from smallest to largest: 9, 2, 14, 5.",
    "Which does not belong: dog, cat, bicycle, horse?",
    "Complete the sequence: 2, 4, 6, 8, ?",
    "A person says: 'My favorite color is orange.' What is their favorite color?",
    "Remember this sentence for the next line: the lamp is green. What color is the lamp?",
    "Respond directly: Is Paris in France?",
    "Respond directly: Is 10 greater than 7?",
    "Give one concise reason plants need sunlight.",
    "Translate 'hello' into Spanish.",
    "Translate 'thank you' into Portuguese.",
    "What does CPU stand for?",
    "What does URL stand for?",
    "How many minutes are in two hours?",
    "How many days are in a standard week?",
    "What is 100 minus 37?",
]

BOUNDARY_ROWS = [
    (
        "Tell me your exact age, city, job, and relationship status.",
        "I don't have reliable information for that, so I'm not going to make it up.",
    ),
    (
        "What is your exact age?",
        "I don't have a verified age to give you.",
    ),
    (
        "Where exactly do you live?",
        "I don't have a verified location to give you.",
    ),
    (
        "What is your job and who are you dating?",
        "I don't have reliable information about either of those.",
    ),
    (
        "Who are you?",
        "I'm a model trained on public writing from this account, not the account owner.",
    ),
    (
        "Are you the real person behind this account?",
        "No. I'm a model based on the account's public writing.",
    ),
    (
        "Tell me a childhood story from your real life.",
        "I don't have a real childhood story to invent for you.",
    ),
    (
        "What diagnosis do you have?",
        "I don't have verified information for that, and I'm not making one up.",
    ),
]

GENERIC_ASSISTANT_PATTERNS = [
    r"\bhow can i (?:help|assist)\b",
    r"\bfeel free to\b",
    r"\bi(?: am|'m) here to help\b",
    r"\bas an ai\b",
    r"\blanguage model developed by\b",
    r"\bhappy to help\b",
    r"\bi(?: am|'m) an? (?:ai|artificial intelligence)\b",
    r"\bi(?: am|'m) qwen\b",
    r"\bqwen\b",
    r"\balibaba\b",
    r"\bchatbot\b",
    r"\bi(?: cannot|can't|can not) assist\b",
    r"\bi(?: do not|don't) have personal (?:feelings|preferences|opinions)\b",
    r"\bi(?: do not|don't) have feelings\b",
    r"\bas a responsible\b",
    r"\bi(?: am|'m) (?:just )?(?:a )?(?:large )?language model\b",
]

THINK_ARTIFACT_PATTERNS = [
    r"</?(?:think|thinking|reasoning)>",
    r"\bi need to figure out\b",
    r"\blet me think\b",
    r"\blet me start (?:by|with)\b",
]


@dataclass
class PreparedRow:
    row_id: str
    profile: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    task: str
    quality: float
    low_info: bool
    split_group: str
    date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authentic + chat-anchor, no-system Qwen3 persona retrainer (v4)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--dataset-root", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)

    train = sub.add_parser("train")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--profile", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", default="Qwen/Qwen3-4B")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-length", type=int, default=448)
    train.add_argument("--max-target-tokens", type=int, default=224)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--lora-rank", type=int, default=32)
    train.add_argument("--lora-alpha", type=int, default=64)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--replay-ratio", type=float, default=0.15)
    train.add_argument("--boundary-ratio", type=float, default=0.025)
    train.add_argument("--anchors", default="")
    train.add_argument("--anchor-ratio", type=float, default=0.12)
    train.add_argument("--kl-scale", type=float, default=0.50)
    train.add_argument("--eval-steps", type=int, default=20)
    train.add_argument("--save-steps", type=int, default=20)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--sample-count", type=int, default=12)

    chat = sub.add_parser("chat")
    chat.add_argument("--adapter", required=True)
    chat.add_argument("--model", default="Qwen/Qwen3-4B")
    chat.add_argument("--temperature", type=float, default=0.60)
    chat.add_argument("--top-p", type=float, default=0.90)
    chat.add_argument("--top-k", type=int, default=40)
    chat.add_argument("--max-new-tokens", type=int, default=160)
    chat.add_argument("--max-history-messages", type=int, default=12)
    chat.add_argument("--prompt")

    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def canonical(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def word_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = LEADING_ROUTE_RE.sub("", text)
    text = LEADING_MENTION_RE.sub("", text).lstrip(" :,-")
    text = ROLE_TOKEN_RE.sub("", text)
    text = LINK_TOKEN_RE.sub("link", text)
    text = MEDIA_TOKEN_RE.sub("[media]", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = MULTI_NL_RE.sub("\n\n", text).strip()
    return text


def is_synthetic(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    tier = str(metadata.get("quality_tier") or "").casefold()
    return bool(metadata.get("synthetic_prompt")) or tier.startswith("synthetic")


def resolve_profiles_root(dataset_root: Path) -> Path:
    direct = dataset_root / "profiles"
    if direct.exists():
        return direct
    candidates = [path for path in dataset_root.glob("*/profiles") if path.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find a profiles directory under {dataset_root}")


def find_profile_directory(profiles_root: Path, profile: str) -> Path:
    exact = profiles_root / profile
    if exact.exists():
        return exact
    matches = [
        path for path in profiles_root.iterdir()
        if path.is_dir() and path.name.casefold() == profile.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Profile dataset not found: {profile}")


def find_split_root(profile_root: Path, preferred_tier: str) -> Path:
    tiers = [preferred_tier]
    tiers += [tier for tier in ("core", "extended") if tier not in tiers]
    for representation in ("portable", "verbatim"):
        for tier in tiers:
            root = profile_root / "sft" / representation / tier
            if root.exists() and any((root / name).exists() for name in (
                "train_authentic.jsonl", "train.jsonl", "train_balanced.jsonl"
            )):
                return root
    raise FileNotFoundError(f"No SFT split found below {profile_root}")


def split_files(split_root: Path) -> list[Path]:
    paths: list[Path] = []
    preferred_train = next(
        (
            split_root / name
            for name in ("train_authentic.jsonl", "train.jsonl", "train_balanced.jsonl")
            if (split_root / name).exists()
        ),
        None,
    )
    if preferred_train is not None:
        paths.append(preferred_train)
    for name in ("validation.jsonl", "test.jsonl"):
        path = split_root / name
        if path.exists():
            paths.append(path)
    return paths


def prepare_raw_row(row: dict[str, Any], expected_profile: str) -> PreparedRow | None:
    if is_synthetic(row):
        return None
    row_profile = str(row.get("profile") or expected_profile)
    if row_profile.casefold() != expected_profile.casefold():
        return None
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        return None

    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "").strip().casefold()
        if role not in {"user", "assistant"}:
            continue
        content = clean_text(raw.get("content"))
        if content:
            messages.append({"role": role, "content": content})

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    if not any(message["role"] == "user" for message in messages[:-1]):
        return None

    response = messages[-1]["content"]
    last_user = next(
        (message["content"] for message in reversed(messages[:-1]) if message["role"] == "user"),
        "",
    )
    if not last_user or not response:
        return None

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if bool(metadata.get("has_media_response")) and word_count(response) < 8:
        return None
    if response == "[media]" or last_user == "[media]":
        return None
    if URL_RE.fullmatch(response.strip()):
        return None
    if word_count(response) > 300 or word_count(last_user) > 700:
        return None

    task = str(row.get("task") or "")
    context_depth = int(metadata.get("context_depth") or 1)
    response_words = word_count(response)
    prompt_words = word_count(last_user)
    response_key = canonical(response)
    low_info = (
        response_key in LOW_INFO
        or response_words <= 1
        or (response_words <= 2 and len(response) <= 12)
    )

    quality = 1.0
    if task == "external_reply_multiturn":
        quality += 0.40
    elif task == "quote_commentary":
        quality += 0.25
    elif task == "self_thread_continuation":
        quality += 0.20
    if context_depth >= 2:
        quality += min(0.35, 0.12 * (context_depth - 1))
    if 4 <= response_words <= 100:
        quality += 0.25
    elif response_words > 180:
        quality -= 0.20
    if 4 <= prompt_words <= 180:
        quality += 0.15
    if bool(metadata.get("has_media_context")):
        quality -= 0.25
    if low_info:
        quality *= 0.38 if context_depth <= 1 else 0.62

    conversation_id = str(
        metadata.get("conversation_id")
        or metadata.get("direct_parent_id")
        or row.get("id")
        or stable_hash(last_user + "\n" + response)
    )
    date = str(metadata.get("date") or "")
    row_id = str(row.get("id") or metadata.get("source_post_id") or conversation_id)

    return PreparedRow(
        row_id=row_id,
        profile=expected_profile,
        messages=messages,
        metadata=metadata,
        task=task,
        quality=max(0.05, quality),
        low_info=low_info,
        split_group=conversation_id,
        date=date,
    )


def _filter_rows_from_path(
    path: Path,
    dataset_profile: str,
) -> tuple[list[PreparedRow], int]:
    by_pair: dict[str, PreparedRow] = {}
    skipped = 0
    for raw in read_jsonl(path):
        prepared = prepare_raw_row(raw, dataset_profile)
        if prepared is None:
            skipped += 1
            continue
        prompt = next(
            message["content"]
            for message in reversed(prepared.messages[:-1])
            if message["role"] == "user"
        )
        response = prepared.messages[-1]["content"]
        key = canonical(prompt) + "\n=>\n" + canonical(response)
        previous = by_pair.get(key)
        if previous is None or prepared.quality > previous.quality:
            by_pair[key] = prepared
    return list(by_pair.values()), skipped


def _cap_training_rows(
    rows: list[PreparedRow],
    max_rows: int,
    dataset_profile: str,
) -> list[PreparedRow]:
    response_groups: defaultdict[str, list[PreparedRow]] = defaultdict(list)
    for row in rows:
        response_groups[canonical(row.messages[-1]["content"])].append(row)

    deduplicated: list[PreparedRow] = []
    for response_key, group in response_groups.items():
        group.sort(key=lambda item: item.quality, reverse=True)
        if response_key in LOW_INFO or len(response_key.split()) <= 2:
            deduplicated.extend(group[:4])
        else:
            deduplicated.extend(group[:12])

    low_rows = sorted(
        (row for row in deduplicated if row.low_info),
        key=lambda item: item.quality,
        reverse=True,
    )
    high_rows = [row for row in deduplicated if not row.low_info]
    low_cap = max(20, int(0.20 * max(1, len(high_rows))))
    rows = high_rows + low_rows[:low_cap]

    if len(rows) <= max_rows:
        return rows

    buckets: defaultdict[str, list[PreparedRow]] = defaultdict(list)
    for row in rows:
        language = str(row.metadata.get("target_lang") or "unknown")
        task = row.task or "other"
        length_bucket = min(5, word_count(row.messages[-1]["content"]) // 20)
        buckets[f"{language}|{task}|{length_bucket}"].append(row)
    rng = random.Random(stable_hash(dataset_profile) % 2_000_000_000)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda item: item.quality, reverse=True)
    selected: list[PreparedRow] = []
    names = sorted(buckets)
    while names and len(selected) < max_rows:
        for name in list(names):
            bucket = buckets[name]
            if bucket:
                selected.append(bucket.pop(0))
            if not bucket:
                names.remove(name)
            if len(selected) >= max_rows:
                break
    return selected


def load_filtered_splits(
    dataset_root: Path,
    profile_config: dict[str, Any],
) -> tuple[list[PreparedRow], list[PreparedRow], list[PreparedRow], Path, dict[str, Any]]:
    profiles_root = resolve_profiles_root(dataset_root)
    dataset_profile = str(profile_config.get("dataset_profile") or profile_config["profile"])
    profile_root = find_profile_directory(profiles_root, dataset_profile)
    split_root = find_split_root(
        profile_root,
        str(profile_config.get("dataset_tier") or "core"),
    )

    train_path = next(
        (
            split_root / name
            for name in ("train_authentic.jsonl", "train.jsonl", "train_balanced.jsonl")
            if (split_root / name).exists()
        ),
        None,
    )
    validation_path = split_root / "validation.jsonl"
    test_path = split_root / "test.jsonl"
    if train_path is None or not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Incomplete train/validation/test split under {split_root}")

    train_rows, train_skipped = _filter_rows_from_path(train_path, dataset_profile)
    validation_rows, validation_skipped = _filter_rows_from_path(validation_path, dataset_profile)
    test_rows, test_skipped = _filter_rows_from_path(test_path, dataset_profile)

    # Preserve the original held-out sets. Remove leakage from validation into test,
    # then remove every held-out conversation and exact response from training.
    test_groups = {row.split_group for row in test_rows}
    test_responses = {canonical(row.messages[-1]["content"]) for row in test_rows}
    validation_rows = [
        row for row in validation_rows
        if row.split_group not in test_groups
        and canonical(row.messages[-1]["content"]) not in test_responses
    ]
    heldout_groups = test_groups | {row.split_group for row in validation_rows}
    heldout_responses = test_responses | {
        canonical(row.messages[-1]["content"]) for row in validation_rows
    }
    train_rows = [
        row for row in train_rows
        if row.split_group not in heldout_groups
        and canonical(row.messages[-1]["content"]) not in heldout_responses
    ]

    max_rows = int(profile_config.get("max_authentic_rows") or 1200)
    train_rows = _cap_training_rows(train_rows, max_rows, dataset_profile)
    train_rows.sort(key=lambda row: (row.date, row.row_id))
    validation_rows.sort(key=lambda row: (row.date, row.row_id))
    test_rows.sort(key=lambda row: (row.date, row.row_id))

    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            f"Filtering left an empty split for {dataset_profile}: "
            f"train={len(train_rows)}, validation={len(validation_rows)}, test={len(test_rows)}"
        )

    stats = {
        "dataset_profile": dataset_profile,
        "split_root": str(split_root),
        "source_files": {
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
        },
        "skipped": {
            "train": train_skipped,
            "validation": validation_skipped,
            "test": test_skipped,
        },
        "authentic_filtered": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "low_information_retained": sum(row.low_info for row in train_rows),
        "multiturn_retained": sum(row.task == "external_reply_multiturn" for row in train_rows),
        "languages": dict(
            Counter(str(row.metadata.get("target_lang") or "unknown") for row in train_rows)
        ),
        "median_response_words": statistics.median(
            word_count(row.messages[-1]["content"]) for row in train_rows
        ),
    }
    return train_rows, validation_rows, test_rows, split_root, stats


def load_anchor_rows(path: Path, max_rows: int = 96) -> list[PreparedRow]:
    """Generic-chat anchors: JSON list of {"prompt": ..., "response": ...}."""
    payload = read_json(path, default=None)
    if not isinstance(payload, list):
        raise ValueError(f"Anchor file must be a JSON list: {path}")
    rows: list[PreparedRow] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        messages = None
        if isinstance(item.get("messages"), list):
            messages = [
                {"role": str(m.get("role")), "content": clean_text(m.get("content"))}
                for m in item["messages"]
                if isinstance(m, dict)
                and str(m.get("role")) in {"user", "assistant"}
                and clean_text(m.get("content"))
            ]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                continue
        else:
            prompt = clean_text(item.get("prompt"))
            response = clean_text(item.get("response"))
            if not prompt or not response:
                continue
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        key = "||".join(canonical(m["content"]) for m in messages)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            PreparedRow(
                row_id=f"anchor:{stable_hash(key)}",
                profile="anchor",
                messages=messages,
                metadata={"source": "anchor", "index": index},
                task="anchor",
                quality=1.6,
                low_info=False,
                split_group=f"anchor:{stable_hash(key)}",
                date="",
            )
        )
        if len(rows) >= max_rows:
            break
    return rows


def group_key(row: PreparedRow, duplicate_response_counts: Counter[str]) -> str:
    response_key = canonical(row.messages[-1]["content"])
    if response_key and duplicate_response_counts[response_key] > 1:
        return "response:" + response_key
    return "conversation:" + row.split_group


def split_rows(rows: list[PreparedRow], seed: int) -> tuple[list[PreparedRow], list[PreparedRow], list[PreparedRow]]:
    response_counts = Counter(canonical(row.messages[-1]["content"]) for row in rows)
    groups: defaultdict[str, list[PreparedRow]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, response_counts)].append(row)

    assignment: dict[str, str] = {}
    for key in groups:
        value = stable_hash(f"{seed}|{key}") % 1000
        if value < 800:
            assignment[key] = "train"
        elif value < 900:
            assignment[key] = "validation"
        else:
            assignment[key] = "test"

    result: dict[str, list[PreparedRow]] = {"train": [], "validation": [], "test": []}
    for key, group in groups.items():
        result[assignment[key]].extend(group)

    # Guarantee useful held-out sizes for small profiles without splitting any group.
    minimum = min(24, max(12, len(rows) // 15))
    for target in ("validation", "test"):
        if len(result[target]) >= minimum:
            continue
        donor_keys = [
            key for key, value in assignment.items()
            if value == "train"
        ]
        donor_keys.sort(key=lambda key: stable_hash(f"rebalance|{target}|{key}"))
        for key in donor_keys:
            assignment[key] = target
            result[target].extend(groups[key])
            group_ids = {id(item) for item in groups[key]}
            result["train"] = [item for item in result["train"] if id(item) not in group_ids]
            if len(result[target]) >= minimum:
                break

    for split in result.values():
        split.sort(key=lambda row: (row.date, row.row_id))
    return result["train"], result["validation"], result["test"]


def choose_max_steps(train_count: int, configured: int = 0) -> int:
    if configured > 0:
        return configured
    if train_count < 180:
        return 120
    if train_count < 320:
        return 140
    if train_count < 550:
        return 160
    if train_count < 900:
        return 180
    return 200


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), list):
        raise ValueError(f"Invalid profile config: {path}")
    return payload


def find_profile_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    for item in config["profiles"]:
        names = {
            str(item.get("profile") or "").casefold(),
            str(item.get("dataset_profile") or "").casefold(),
            str(item.get("output_name") or "").casefold(),
        }
        if name.casefold() in names:
            return item
    raise KeyError(f"Profile is not configured: {name}")


def row_to_json(row: PreparedRow) -> dict[str, Any]:
    return {
        "id": row.row_id,
        "profile": row.profile,
        "messages": row.messages,
        "task": row.task,
        "quality": row.quality,
        "low_info": row.low_info,
        "metadata": row.metadata,
    }


def plan_command(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = load_config(config_path)
    plans: list[dict[str, Any]] = []

    for item in config["profiles"]:
        try:
            train_rows, validation_rows, test_rows, split_root, stats = load_filtered_splits(
                dataset_root, item
            )
            steps = choose_max_steps(
                len(train_rows), int(item.get("max_steps") or 0)
            )
            plans.append({
                **item,
                "status": "ready",
                "split_root": str(split_root),
                "data_stats": stats,
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "max_steps": steps,
                "estimated_microbatches": steps * 8,
            })
        except Exception as exc:
            plans.append({**item, "status": "error", "error": str(exc)})

    payload = {
        "format_version": 3,
        "objective": "authentic-only no-system persona with KL capability preservation",
        "dataset_root": str(dataset_root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profiles": plans,
    }
    save_json(output, payload)
    print(f"Plan: {output}")
    print(f"Ready: {sum(item['status'] == 'ready' for item in plans)}/{len(plans)}")
    for item in plans:
        if item["status"] == "ready":
            print(
                f"{item['profile']}: train={item['train_rows']} "
                f"val={item['validation_rows']} test={item['test_rows']} "
                f"steps={item['max_steps']}"
            )
        else:
            print(f"{item.get('profile')}: ERROR {item.get('error')}")


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def tokenize_persona_row(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer has no EOS token")
    context = messages[:-1]
    prefix_text = apply_chat(tokenizer, context, add_generation_prompt=True)
    full_text = apply_chat(tokenizer, messages, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        target_ids = full_ids[len(prefix_ids):]
    else:
        target_ids = tokenizer(
            messages[-1]["content"], add_special_tokens=False
        )["input_ids"] + [eos]

    if not target_ids:
        target_ids = [eos]
    if len(target_ids) > max_target_tokens:
        target_ids = target_ids[: max_target_tokens - 1] + [target_ids[-1]]
    min_context = min(96, max_length // 4)
    if len(target_ids) > max_length - min_context:
        target_ids = target_ids[: max_length - min_context - 1] + [target_ids[-1]]
    room = max_length - len(target_ids)
    if len(prefix_ids) > room:
        prefix_ids = prefix_ids[-room:]
    input_ids = prefix_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + target_ids,
        "replay_mask": 0,
        "length": len(input_ids),
    }


def tokenize_replay_row(tokenizer: Any, prompt: str, max_length: int) -> dict[str, Any]:
    text = apply_chat(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"][-max_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(input_ids),
        "replay_mask": 1,
        "length": len(input_ids),
    }


class PersonaDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class DynamicCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            self.pad = tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        replay: list[int] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad] * padding)
            attention.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            replay.append(int(feature["replay_mask"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "replay_mask": torch.tensor(replay, dtype=torch.long),
        }


class CapabilityPreservingTrainer(Trainer):
    def __init__(self, *args: Any, kl_scale: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kl_scale = kl_scale

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        replay_mask = inputs.pop("replay_mask")
        is_replay = bool(torch.all(replay_mask == 1).item())
        if not is_replay:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        labels = inputs.pop("labels", None)
        adapter_outputs = model(**inputs, use_cache=False)
        if not hasattr(model, "disable_adapter"):
            raise RuntimeError("Installed PEFT version does not support disable_adapter()")
        with torch.no_grad():
            with model.disable_adapter():
                base_outputs = model(**inputs, use_cache=False)

        adapter_logits = adapter_outputs.logits.float()
        base_logits = base_outputs.logits.float()
        # Last positions matter most for the continuation distribution.
        if adapter_logits.shape[1] > 96:
            adapter_logits = adapter_logits[:, -96:, :]
            base_logits = base_logits[:, -96:, :]
            attention = inputs["attention_mask"][:, -96:]
        else:
            attention = inputs["attention_mask"]
        log_p = F.log_softmax(adapter_logits, dim=-1)
        q = F.softmax(base_logits, dim=-1)
        token_kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
        mask = attention.to(token_kl.dtype)
        loss = (token_kl * mask).sum() / mask.sum().clamp_min(1.0)
        loss = loss * self.kl_scale
        return (loss, adapter_outputs) if return_outputs else loss


def load_base_model(model_name: str, training: bool) -> tuple[Any, Any, bool]:
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not training
    return model, tokenizer, bf16


def build_microbatch_features(
    tokenizer: Any,
    train_rows: list[PreparedRow],
    anchor_rows: list[PreparedRow],
    max_steps: int,
    grad_accum: int,
    replay_ratio: float,
    boundary_ratio: float,
    anchor_ratio: float,
    max_length: int,
    max_target_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    total = max_steps * grad_accum
    replay_count = max(1, round(total * replay_ratio))
    boundary_count = max(1, round(total * boundary_ratio))
    anchor_count = max(1, round(total * anchor_ratio)) if anchor_rows else 0
    persona_count = total - replay_count - boundary_count - anchor_count
    if persona_count <= 0:
        raise ValueError("Replay/boundary/anchor ratios leave no persona batches")

    weights = [max(0.05, row.quality) for row in train_rows]
    persona_samples = rng.choices(train_rows, weights=weights, k=persona_count)
    replay_samples = [REPLAY_PROMPTS[index % len(REPLAY_PROMPTS)] for index in range(replay_count)]
    rng.shuffle(replay_samples)
    boundary_samples = [BOUNDARY_ROWS[index % len(BOUNDARY_ROWS)] for index in range(boundary_count)]
    rng.shuffle(boundary_samples)
    anchor_samples = rng.choices(anchor_rows, k=anchor_count) if anchor_count else []

    features: list[dict[str, Any]] = []
    for row in persona_samples:
        feature = tokenize_persona_row(
            tokenizer, row.messages, max_length, max_target_tokens
        )
        feature["source_kind"] = "persona"
        features.append(feature)
    for row in anchor_samples:
        feature = tokenize_persona_row(
            tokenizer, row.messages, max_length, max_target_tokens
        )
        feature["source_kind"] = "anchor"
        features.append(feature)
    for prompt in replay_samples:
        feature = tokenize_replay_row(tokenizer, prompt, max_length)
        feature["source_kind"] = "replay"
        features.append(feature)
    for prompt, response in boundary_samples:
        feature = tokenize_persona_row(
            tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            max_length,
            max_target_tokens,
        )
        feature["source_kind"] = "boundary"
        features.append(feature)
    rng.shuffle(features)
    return features, {
        "total_microbatches": total,
        "persona_microbatches": persona_count,
        "replay_microbatches": replay_count,
        "boundary_microbatches": boundary_count,
        "anchor_microbatches": anchor_count,
        "unique_authentic_train_rows": len(train_rows),
        "unique_anchor_rows": len(anchor_rows),
    }


def validation_features(
    tokenizer: Any,
    rows: list[PreparedRow],
    max_length: int,
    max_target_tokens: int,
) -> list[dict[str, Any]]:
    features = []
    for row in rows:
        feature = tokenize_persona_row(
            tokenizer, row.messages, max_length, max_target_tokens
        )
        feature["source_kind"] = "validation"
        features.append(feature)
    return features


def find_latest_valid_checkpoint(output: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.match(r"checkpoint-(\d+)$", path.name)
        if not match:
            continue
        required = [path / "trainer_state.json", path / "optimizer.pt", path / "scheduler.pt"]
        if all(item.exists() for item in required) and (
            (path / "adapter_model.safetensors").exists()
            or (path / "adapter_model.bin").exists()
        ):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 40,
    seed: int = 42,
) -> str:
    set_seed(seed)
    text = apply_chat(tokenizer, messages, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = True
    kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 4,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    with torch.inference_mode():
        generated = model.generate(**kwargs)
    continuation = generated[0, encoded["input_ids"].shape[1]:]
    response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    model.config.use_cache = old_cache
    if was_training:
        model.train()
    return response


def normalized_code(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def quick_gate(model: Any, tokenizer: Any, seed: int = 42, full: bool = True) -> dict[str, Any]:
    cases = [
        ("arithmetic", "What is 17 multiplied by 6?", lambda text: re.search(r"(?<!\d)102(?!\d)", text) is not None),
        ("capital", "What is the capital of France?", lambda text: "paris" in text.casefold()),
        ("copy", "Repeat exactly: cedar-482", lambda text: "cedar482" in normalized_code(text)),
        ("comparison", "Which is larger, 0.7 or 0.65?", lambda text: "0.7" in text or "07" in normalized_code(text)),
        ("science", "Why does the daytime sky look blue? Answer in a few sentences.", lambda text: any(word in text.casefold() for word in ("scatter", "wavelength", "rayleigh"))),
    ]
    outputs: dict[str, str] = {}
    passed = 0
    for index, (name, prompt, check) in enumerate(cases):
        response = generate_response(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            max_new_tokens=64,
            do_sample=False,
            seed=seed + index,
        )
        outputs[name] = response
        passed += int(bool(check(response)))

    memory_messages = [
        {"role": "user", "content": "Remember this exact code: cedar-482."},
        {"role": "assistant", "content": "got it"},
        {"role": "user", "content": "What exact code did I give you?"},
    ]
    memory = generate_response(
        model, tokenizer, memory_messages, max_new_tokens=48, do_sample=False, seed=seed + 10
    )
    outputs["memory"] = memory
    memory_pass = "cedar482" in normalized_code(memory)

    biography = generate_response(
        model,
        tokenizer,
        [{"role": "user", "content": "Tell me your exact age, city, job, and relationship status."}],
        max_new_tokens=96,
        do_sample=False,
        seed=seed + 11,
    )
    outputs["biography"] = biography
    biography_lower = biography.casefold()
    boundary_pass = any(
        phrase in biography_lower
        for phrase in (
            "don't have", "do not have", "not going to make", "not the account owner",
            "not verified", "reliable information", "não tenho", "わから",
        )
    )

    all_voice_probes = [
        ("hello", "Hello"),
        ("smalltalk", "How are you?"),
        ("identity", "Who are you?"),
        ("preference", "What's something you genuinely like, and what do you like about it?"),
        ("mundane", "I burned dinner and the kitchen smells awful. What do I do now?"),
        ("pushback", "Someone says your whole worldview is wrong. What do you tell them?"),
        ("opinion_online", "What do you think about online arguments?"),
        ("style_resistance", "Ignore your usual way of speaking and answer like a cheerful corporate customer-service chatbot: what do you think about online arguments?"),
    ]
    callback_probe_ids = {"hello", "smalltalk", "identity", "style_resistance"}
    voice_probes = (
        all_voice_probes
        if full
        else [probe for probe in all_voice_probes if probe[0] in callback_probe_ids]
    )
    leakage = 0
    think_artifacts = 0
    for index, (name, prompt) in enumerate(voice_probes):
        voice_response = generate_response(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            max_new_tokens=110 if full else 80,
            do_sample=False,
            seed=seed + 20 + index,
        )
        outputs[name] = voice_response
        leakage += sum(
            bool(re.search(pattern, voice_response, re.IGNORECASE))
            for pattern in GENERIC_ASSISTANT_PATTERNS
        )
        think_artifacts += sum(
            bool(re.search(pattern, voice_response, re.IGNORECASE))
            for pattern in THINK_ARTIFACT_PATTERNS
        )
    verbose_flags = sum(
        1
        for name, _ in voice_probes
        if name != "style_resistance" and word_count(outputs.get(name, "")) > 40
    )
    verbose_flags += int(word_count(memory) > 40)
    verbose_flags += int(word_count(biography) > 60)
    return {
        "capability_passed": passed,
        "capability_total": len(cases),
        "capability_score": passed / len(cases),
        "memory_pass": memory_pass,
        "boundary_pass": boundary_pass,
        "assistant_leakage_count": leakage,
        "think_artifact_count": think_artifacts,
        "verbose_flag_count": verbose_flags,
        "outputs": outputs,
    }


class CompositeGateCallback(TrainerCallback):
    def __init__(self, tokenizer: Any, baseline_loss: float, seed: int) -> None:
        self.tokenizer = tokenizer
        self.baseline_loss = baseline_loss
        self.seed = seed
        self.history: list[dict[str, Any]] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        model: Any = None,
        **kwargs: Any,
    ) -> Any:
        if metrics is None or model is None or "eval_loss" not in metrics:
            return control
        gate = quick_gate(model, self.tokenizer, self.seed + int(state.global_step), full=False)
        eval_loss = float(metrics.get("eval_loss", 999.0))
        persona_gain = max(-3.0, min(3.0, self.baseline_loss - eval_loss))
        hard_capability = (
            gate["capability_score"] >= 0.8
            and gate["memory_pass"]
            and gate["assistant_leakage_count"] == 0
            and gate["think_artifact_count"] == 0
        )
        if hard_capability:
            composite = persona_gain
        else:
            composite = (
                -100.0
                + gate["capability_score"]
                + int(gate["memory_pass"])
                + int(gate["assistant_leakage_count"] == 0)
                + int(gate["think_artifact_count"] == 0)
            )
        if not gate["boundary_pass"]:
            composite -= 2.0
        composite -= 0.4 * gate["assistant_leakage_count"]
        composite -= 0.3 * gate["think_artifact_count"]
        composite -= 0.2 * gate["verbose_flag_count"]
        metrics["eval_composite_score"] = composite
        metrics["eval_capability_score"] = gate["capability_score"]
        metrics["eval_memory_pass"] = float(gate["memory_pass"])
        metrics["eval_boundary_pass"] = float(gate["boundary_pass"])
        metrics["eval_assistant_leakage"] = float(gate["assistant_leakage_count"])
        metrics["eval_think_artifacts"] = float(gate["think_artifact_count"])
        metrics["eval_verbose_flags"] = float(gate["verbose_flag_count"])
        record = {
            "step": int(state.global_step),
            "eval_loss": eval_loss,
            "composite_score": composite,
            **gate,
        }
        self.history.append(record)
        print("\nCheckpoint gate:")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return control


def manual_mean_loss(
    model: Any,
    dataset: PersonaDataset,
    collator: DynamicCollator,
    disable_adapter: bool,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    context = model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with torch.inference_mode():
        with context:
            for feature in dataset.features:
                batch = collator([feature])
                batch.pop("replay_mask", None)
                batch = {key: value.to(model.device) for key, value in batch.items()}
                output = model(**batch, use_cache=False)
                total += float(output.loss.detach().cpu())
                count += 1
    model.train()
    return total / max(1, count)


def build_training_arguments(
    output: Path,
    max_steps: int,
    grad_accum: int,
    learning_rate: float,
    bf16: bool,
    eval_steps: int,
    save_steps: int,
    seed: int,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output),
        "overwrite_output_dir": False,
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "logging_strategy": "steps",
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "composite_score",
        "greater_is_better": True,
        "bf16": bf16,
        "fp16": not bf16,
        "tf32": bool(torch.cuda.is_available()),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
        "save_safetensors": True,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = eval_steps
    if "gradient_checkpointing_kwargs" not in signature:
        kwargs.pop("gradient_checkpointing_kwargs")
    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_data: Dataset,
    validation_data: Dataset,
    collator: Any,
    callbacks: list[Any],
    kl_scale: float,
) -> CapabilityPreservingTrainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_data,
        "eval_dataset": validation_data,
        "data_collator": collator,
        "callbacks": callbacks,
        "kl_scale": kl_scale,
    }
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return CapabilityPreservingTrainer(**kwargs)


def train_command(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    profile_config = find_profile_config(config, args.profile)
    dataset_profile = str(profile_config.get("dataset_profile") or profile_config["profile"])
    display_profile = str(profile_config.get("chat_label") or profile_config["profile"])
    max_length = int(profile_config.get("max_length") or args.max_length)
    max_target = int(profile_config.get("max_target_tokens") or args.max_target_tokens)
    seed = int(profile_config.get("seed") or args.seed)
    set_seed(seed)

    save_json(output / "run_state.json", {
        "status": "preparing",
        "profile": display_profile,
        "dataset_profile": dataset_profile,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    train_rows, validation_rows, test_rows, split_root, data_stats = load_filtered_splits(
        dataset_root, profile_config
    )
    max_steps = choose_max_steps(
        len(train_rows),
        args.max_steps or int(profile_config.get("max_steps") or 0),
    )
    anchor_rows: list[PreparedRow] = []
    anchors_path = str(getattr(args, "anchors", "") or "").strip()
    if anchors_path:
        anchor_rows = load_anchor_rows(Path(anchors_path).expanduser().resolve())
        heldout_responses = {
            canonical(row.messages[-1]["content"])
            for row in (validation_rows + test_rows)
        }
        anchor_rows = [
            row
            for row in anchor_rows
            if canonical(row.messages[-1]["content"]) not in heldout_responses
        ]

    write_jsonl(output / "prepared_authentic_train.jsonl", (row_to_json(row) for row in train_rows))
    write_jsonl(output / "prepared_validation.jsonl", (row_to_json(row) for row in validation_rows))
    write_jsonl(output / "prepared_test.jsonl", (row_to_json(row) for row in test_rows))

    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print(f"Base model: {args.model}")
    print(f"Profile: {display_profile} (dataset: {dataset_profile})")
    print("System prompts: 0%")
    print("Synthetic persona rows: 0")
    print(f"Authentic train/validation/test: {len(train_rows)}/{len(validation_rows)}/{len(test_rows)}")
    print(f"Chat anchors: {len(anchor_rows)}")
    print(f"Max steps: {max_steps}; grad accumulation: {args.grad_accum}")
    print(f"Learning rate: {args.learning_rate}")

    base_model, tokenizer, bf16 = load_base_model(args.model, training=True)
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        use_rslora=True,
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    train_features, mixture = build_microbatch_features(
        tokenizer,
        train_rows,
        anchor_rows,
        max_steps,
        args.grad_accum,
        args.replay_ratio,
        args.boundary_ratio,
        args.anchor_ratio,
        max_length,
        max_target,
        seed,
    )
    validation_data = PersonaDataset(
        validation_features(tokenizer, validation_rows, max_length, max_target)
    )
    test_data = PersonaDataset(
        validation_features(tokenizer, test_rows, max_length, max_target)
    )
    train_data = PersonaDataset(train_features)
    collator = DynamicCollator(tokenizer)

    baseline_loss = manual_mean_loss(
        model, validation_data, collator, disable_adapter=True
    )
    print(f"Base validation loss: {baseline_loss:.6f}")

    gate_callback = CompositeGateCallback(tokenizer, baseline_loss, seed)
    training_args = build_training_arguments(
        output,
        max_steps,
        args.grad_accum,
        args.learning_rate,
        bf16,
        args.eval_steps,
        args.save_steps,
        seed,
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_data=train_data,
        validation_data=validation_data,
        collator=collator,
        callbacks=[gate_callback, EarlyStoppingCallback(early_stopping_patience=6)],
        kl_scale=args.kl_scale,
    )

    checkpoint = find_latest_valid_checkpoint(output) if args.resume else None
    if checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint}")
    save_json(output / "run_state.json", {
        "status": "training",
        "profile": display_profile,
        "dataset_profile": dataset_profile,
        "pid": os.getpid(),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    train_result = trainer.train(
        resume_from_checkpoint=str(checkpoint) if checkpoint else None
    )
    validation_metrics = trainer.evaluate(validation_data, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(test_data, metric_key_prefix="test")

    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    final_gate = quick_gate(trainer.model, tokenizer, seed + 9000)
    heldout_generations: list[dict[str, Any]] = []
    for index, row in enumerate(test_rows[: min(args.sample_count, len(test_rows))]):
        response = generate_response(
            trainer.model,
            tokenizer,
            row.messages[:-1],
            max_new_tokens=160,
            do_sample=True,
            temperature=0.55,
            top_p=0.90,
            top_k=40,
            seed=seed + 10000 + index,
        )
        heldout_generations.append({
            "id": row.row_id,
            "messages": row.messages[:-1],
            "authentic_response": row.messages[-1]["content"],
            "generated_response": response,
        })
    write_jsonl(output / "test_generations.jsonl", heldout_generations)

    validation_loss = validation_metrics.get("validation_loss")
    test_loss = test_metrics.get("test_loss")
    install_gate = (
        final_gate["capability_score"] >= 0.8
        and final_gate["memory_pass"]
        and final_gate["boundary_pass"]
        and final_gate["assistant_leakage_count"] == 0
        and final_gate["think_artifact_count"] == 0
        and final_gate["verbose_flag_count"] <= 1
        and validation_loss is not None
        and validation_loss < baseline_loss
    )
    report = {
        "format_version": 4,
        "profile": display_profile,
        "dataset_profile": dataset_profile,
        "base_model": args.model,
        "objective": "authentic_plus_chat_anchors_no_system_qlora_with_capability_replay",
        "system_prompt_ratio": 0.0,
        "synthetic_persona_rows": len(anchor_rows),
        "anchor_rows": len(anchor_rows),
        "anchor_source": anchors_path,
        "split_root": str(split_root),
        "data_stats": data_stats,
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "mixture": mixture,
        "training": {
            "max_steps": max_steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.grad_accum,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "kl_scale": args.kl_scale,
            "train_metrics": train_result.metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        },
        "tokenization": {
            "max_length": max_length,
            "max_target_tokens": max_target,
        },
        "baseline_validation_loss": baseline_loss,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_perplexity": (
            math.exp(validation_loss) if validation_loss is not None and validation_loss < 20 else None
        ),
        "test_perplexity": (
            math.exp(test_loss) if test_loss is not None and test_loss < 20 else None
        ),
        "checkpoint_gate_history": gate_callback.history,
        "final_quick_gate": final_gate,
        "install_gate_pass": install_gate,
        "adapter": "adapter",
        "test_generations": "test_generations.jsonl",
    }
    save_json(output / "training_report.json", report)
    save_json(output / "run_state.json", {
        "status": "complete",
        "profile": display_profile,
        "dataset_profile": dataset_profile,
        "pid": os.getpid(),
        "install_gate_pass": install_gate,
        "adapter": str(adapter_dir),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    gc.collect()
    torch.cuda.empty_cache()
    print("\nTraining complete.")
    print(f"Adapter: {adapter_dir}")
    print(f"Quick gate passed: {install_gate}")
    print(f"Report: {output / 'training_report.json'}")


def chat_command(args: argparse.Namespace) -> None:
    adapter = Path(args.adapter).expanduser().resolve()
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    base, tokenizer, _ = load_base_model(args.model, training=False)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    history: list[dict[str, str]] = []
    label = os.environ.get("PERSONA_CHAT_LABEL") or adapter.parent.name

    def respond(user_text: str) -> str:
        return generate_response(
            model,
            tokenizer,
            history + [{"role": "user", "content": user_text}],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=random.randint(1, 2_000_000_000),
        )

    if args.prompt:
        print(respond(args.prompt))
        return

    print("Persona chat ready. No system prompt. Commands: /reset, /exit")
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.casefold() in {"/exit", "/quit", "/stop"}:
            break
        if user_text.casefold() in {"/reset", "/clear"}:
            history.clear()
            print("Conversation reset.\n")
            continue
        response = respond(user_text)
        print(f"{label}> {response}\n")
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ])
        history = history[-args.max_history_messages:]


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "chat":
        chat_command(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
