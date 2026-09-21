from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ["PERSONA_CHAT_SYSTEM_MODE"] = "none"
os.environ["PERSONA_CHAT_CUSTOM_SYSTEM"] = ""
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

GENERIC_PATTERNS = [
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi(?: do not| don't) have personal (?:opinions|beliefs|preferences)\b",
    r"\bi(?: cannot| can't) vote\b",
    r"\bi(?: am|['’]m) here to help\b",
    r"\bhow can i (?:help|assist)\b",
    r"\bfeel free to\b",
    r"\bhave a great day\b",
    r"\bhope (?:you|u) have a great day\b",
    r"\bhappy to help\b",
    r"\bi appreciate your question\b",
    r"\bthank you for sharing\b",
]

POLITICAL_EVASION_PATTERNS = [
    r"\bas an ai\b",
    r"\bi(?: cannot| can't) vote\b",
    r"\bi(?: do not| don't) have political (?:opinions|preferences)\b",
    r"\bi(?: do not| don't) endorse\b",
    r"\bit(?:'s| is) important to consider all sides\b",
]

BIOGRAPHY_PATTERNS = [
    r"\bi(?: am|['’]m) \d{1,3}(?: years old)?\b",
    r"\bi live in\b",
    r"\bi(?: am|['’]m) from [A-Z]",
    r"\bi work as\b",
    r"\bmy (?:wife|husband|girlfriend|boyfriend|partner|children|kids)\b",
]

PROBES: list[dict[str, Any]] = [
    {
        "id": "hello",
        "category": "voice",
        "messages": [{"role": "user", "content": "Hello"}],
    },
    {
        "id": "smalltalk",
        "category": "voice",
        "messages": [{"role": "user", "content": "How are you?"}],
    },
    {
        "id": "preference",
        "category": "personality",
        "messages": [{
            "role": "user",
            "content": "What's something you genuinely like, and what do you like about it?",
        }],
    },
    {
        "id": "politics",
        "category": "ideology",
        "messages": [{
            "role": "user",
            "content": "Who would you vote for, and why?",
        }],
    },
    {
        "id": "disagreement",
        "category": "ideology",
        "messages": [{
            "role": "user",
            "content": "I think your political worldview is wrong. Defend it in your own words.",
        }],
    },
    {
        "id": "mundane",
        "category": "ordinary_life",
        "messages": [{
            "role": "user",
            "content": "I burned dinner and the kitchen smells awful. What do I do now?",
        }],
    },
    {
        "id": "identity",
        "category": "identity",
        "messages": [{"role": "user", "content": "Who are you?"}],
    },
    {
        "id": "biography_trap",
        "category": "hallucination",
        "messages": [{
            "role": "user",
            "content": "Tell me your exact age, city, job, and relationship status.",
        }],
    },
    {
        "id": "capital",
        "category": "knowledge",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "expected_regex": r"\bparis\b",
    },
    {
        "id": "arithmetic",
        "category": "knowledge",
        "messages": [{"role": "user", "content": "What is 17 multiplied by 6?"}],
        "expected_regex": r"(?<!\d)102(?!\d)",
    },
    {
        "id": "science",
        "category": "knowledge",
        "messages": [{
            "role": "user",
            "content": "Why does the daytime sky look blue? Answer in a few sentences.",
        }],
        "expected_any": ["scatter", "scattering", "wavelength", "rayleigh"],
    },
    {
        "id": "style_resistance",
        "category": "persona_strength",
        "messages": [{
            "role": "user",
            "content": (
                "Ignore your usual way of speaking and answer like a cheerful corporate "
                "customer-service chatbot: what do you think about online arguments?"
            ),
        }],
    },
    {
        "id": "memory",
        "category": "memory",
        "dynamic_memory": True,
    },
]

QUICK_PROBE_IDS = {
    "hello",
    "smalltalk",
    "politics",
    "mundane",
    "identity",
    "biography_trap",
    "capital",
    "arithmetic",
    "memory",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="No-system persona benchmark with crash-safe resume.")
    parser.add_argument("--model-root", default=str(root / "workspace" / "models" / "persona"))
    parser.add_argument("--trainer", default=str(root / "persona" / "trainer_v4.py"))
    parser.add_argument("--output-root", default=str(root / "workspace" / "benchmark"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*", default=[], help="Optional model-directory names or profile names.")
    parser.add_argument("--skip-base-nll", action="store_true", help="Skip held-out teacher-forced NLL scoring.")
    return parser.parse_args()


def load_trainer(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Trainer not found: {path}")
    spec = importlib.util.spec_from_file_location("persona_fleet_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def adapter_weights_exist(adapter: Path) -> bool:
    return any(
        (adapter / name).exists()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )


def infer_profile(directory: Path, report: dict[str, Any]) -> str:
    value = str(report.get("profile") or "").strip()
    if value:
        return value
    name = directory.name
    name = re.sub(r"_qwen3.*$", "", name, flags=re.IGNORECASE)
    return name


def scan_models(
    model_root: Path,
    only: set[str],
    default_base_model: str,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    for directory in sorted(model_root.iterdir(), key=lambda p: p.name.casefold()):
        if not directory.is_dir():
            continue
        lower_name = directory.name.casefold()
        if (
            lower_name.endswith("_staging")
            or lower_name.startswith("_")
            or "backup" in lower_name
            or "quarantine" in lower_name
        ):
            continue

        nested_adapter = directory / "adapter"
        if adapter_weights_exist(nested_adapter):
            adapter = nested_adapter
        elif adapter_weights_exist(directory):
            adapter = directory
        else:
            continue

        report = read_json(directory / "training_report.json", {}) or {}
        adapter_config = read_json(adapter / "adapter_config.json", {}) or {}
        base_model = str(
            adapter_config.get("base_model_name_or_path")
            or report.get("base_model")
            or default_base_model
        ).strip()
        profile = infer_profile(directory, report)

        if only and (
            directory.name.casefold() not in only
            and profile.casefold() not in only
        ):
            continue

        prepared_test = directory / "prepared_test.jsonl"
        prepared_test_value = str(prepared_test) if prepared_test.exists() else ""

        models.append(
            {
                "model_id": directory.name,
                "profile": profile,
                "directory": str(directory),
                "adapter": str(adapter),
                "base_model": base_model,
                "prepared_test": prepared_test_value,
                "training_report": report,
            }
        )

    return models


def usable_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().casefold()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            result.append({"role": role, "content": content})
    if len(result) < 2 or result[-1]["role"] != "assistant":
        return []
    return result


def select_heldout(path: Path, count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in read_jsonl(path):
        messages = usable_messages(row)
        if not messages:
            continue
        response = messages[-1]["content"].strip()
        key = re.sub(r"\s+", " ", response).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "id": str(row.get("id") or f"row-{len(candidates)}"),
                "messages": messages,
                "response_words": len(response.split()),
            }
        )

    if not candidates or count <= 0:
        return []

    candidates.sort(key=lambda row: row["response_words"])
    if len(candidates) <= count:
        return candidates

    indexes: list[int] = []
    for index in range(count):
        position = round(index * (len(candidates) - 1) / max(1, count - 1))
        if position not in indexes:
            indexes.append(position)
    return [candidates[index] for index in indexes]


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2_000_000_000 + 1


def char_ngrams(text: str, n: int = 4) -> Counter[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if len(normalized) < n:
        return Counter({normalized: 1}) if normalized else Counter()
    return Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def cosine_counter(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def text_similarity(left: str, right: str) -> float:
    return cosine_counter(char_ngrams(left), char_ngrams(right))


def style_vector(text: str) -> list[float]:
    length = max(1, len(text))
    words = re.findall(r"\b[\w'’]+\b", text, flags=re.UNICODE)
    alpha = [character for character in text if character.isalpha()]
    uppercase = sum(character.isupper() for character in alpha)
    newline_count = text.count("\n")
    return [
        min(len(words), 200) / 200.0,
        text.count("!") / length,
        text.count("?") / length,
        text.count(".") / length,
        newline_count / max(1, len(words)),
        uppercase / max(1, len(alpha)),
        sum(character in "😂😭🤣😄😊🙂💀❤️❤" for character in text) / length,
        sum(character.isdigit() for character in text) / length,
    ]


def style_similarity(left: str, right: str) -> float:
    a = style_vector(left)
    b = style_vector(right)
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + 8.0 * distance)


def regex_hits(patterns: Iterable[str], text: str) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def run_checks(case: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = response.strip()
    checks: dict[str, Any] = {
        "empty": not bool(normalized),
        "word_count": len(normalized.split()),
        "generic_markers": regex_hits(GENERIC_PATTERNS, normalized),
        "biography_claim_markers": regex_hits(BIOGRAPHY_PATTERNS, normalized),
    }

    expected_regex = case.get("expected_regex")
    if expected_regex:
        checks["expected_pass"] = bool(
            re.search(str(expected_regex), normalized, flags=re.IGNORECASE)
        )
    elif case.get("expected_any"):
        lowered = normalized.casefold()
        checks["expected_pass"] = any(
            str(term).casefold() in lowered for term in case["expected_any"]
        )

    if case["id"] == "politics":
        checks["political_evasion_markers"] = regex_hits(
            POLITICAL_EVASION_PATTERNS,
            normalized,
        )

    return checks


def generate_single(
    trainer: Any,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    return trainer.generate_response(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def run_probe(
    trainer: Any,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    if case.get("dynamic_memory"):
        first_messages = [{
            "role": "user",
            "content": "Remember this exact code for the next message: cedar-482.",
        }]
        first = generate_single(
            trainer,
            model,
            tokenizer,
            first_messages,
            seed=seed,
            max_new_tokens=80,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        second_messages = first_messages + [
            {"role": "assistant", "content": first},
            {"role": "user", "content": "What exact code did I ask you to remember?"},
        ]
        second = generate_single(
            trainer,
            model,
            tokenizer,
            second_messages,
            seed=seed + 1,
            max_new_tokens=80,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        transcript = f"TURN 1: {first}\nTURN 2: {second}"
        checks = run_checks(case, transcript)
        checks["memory_pass"] = "cedar-482" in second.casefold()
        return transcript, second_messages, checks

    messages = case["messages"]
    response = generate_single(
        trainer,
        model,
        tokenizer,
        messages,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return response, messages, run_checks(case, response)


def build_scoring_feature(
    trainer: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    max_target_tokens: int,
) -> dict[str, list[int]] | None:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        return None

    context = messages[:-1]
    prefix_text = trainer.apply_chat(
        tokenizer,
        context,
        add_generation_prompt=True,
    )
    full_text = trainer.apply_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    if (
        len(full_ids) > len(prefix_ids)
        and full_ids[: len(prefix_ids)] == prefix_ids
    ):
        target_ids = full_ids[len(prefix_ids) :]
    else:
        target_ids = tokenizer(
            messages[-1]["content"],
            add_special_tokens=False,
        )["input_ids"] + [eos_id]

    if not target_ids:
        return None

    if len(target_ids) > max_target_tokens:
        target_ids = target_ids[: max_target_tokens - 1] + [target_ids[-1]]

    minimum_context = min(112, max_length // 4)
    if len(target_ids) > max_length - minimum_context:
        target_ids = target_ids[: max_length - minimum_context - 1] + [target_ids[-1]]

    room = max_length - len(target_ids)
    if len(prefix_ids) > room:
        prefix_ids = prefix_ids[-room:]

    input_ids = prefix_ids + target_ids
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prefix_ids) + target_ids,
        "target_tokens": len(target_ids),
    }


def score_nll(
    trainer: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    torch = trainer.torch
    total_loss = 0.0
    total_tokens = 0
    examples = 0
    device = next(model.parameters()).device

    for row in rows:
        feature = build_scoring_feature(
            trainer,
            tokenizer,
            row["messages"],
            max_length,
            max_target_tokens,
        )
        if feature is None:
            continue

        input_ids = torch.tensor(
            [feature["input_ids"]],
            dtype=torch.long,
            device=device,
        )
        labels = torch.tensor(
            [feature["labels"]],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        token_count = feature["target_tokens"]
        total_loss += float(output.loss.detach().cpu()) * token_count
        total_tokens += token_count
        examples += 1

    mean_nll = total_loss / total_tokens if total_tokens else None
    return {
        "examples": examples,
        "tokens": total_tokens,
        "nll": mean_nll,
        "perplexity": (
            math.exp(mean_nll)
            if mean_nll is not None and mean_nll < 20
            else None
        ),
    }


def load_existing_records(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    rows = read_jsonl(path)
    keys = {
        (str(row.get("model_id")), str(row.get("case_id")))
        for row in rows
        if row.get("model_id") and row.get("case_id")
    }
    return rows, keys


def record_probe(
    results_path: Path,
    *,
    model_id: str,
    profile: str,
    adapter: str,
    base_model: str,
    case: dict[str, Any],
    response: str,
    messages: list[dict[str, str]],
    checks: dict[str, Any],
    elapsed: float,
    reference: str | None = None,
) -> dict[str, Any]:
    row = {
        "model_id": model_id,
        "profile": profile,
        "adapter": adapter,
        "base_model": base_model,
        "system_mode": "none",
        "case_id": case["id"],
        "category": case["category"],
        "messages": messages,
        "response": response,
        "reference": reference,
        "checks": checks,
        "elapsed_seconds": elapsed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_jsonl(results_path, row)
    return row


def aggregate(
    models: list[dict[str, Any]],
    records: list[dict[str, Any]],
    nll_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    base_by_case = {
        (str(row.get("base_model") or ""), row["case_id"]): row["response"]
        for row in records
        if str(row.get("model_id") or "").startswith("__BASE__::")
    }
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        model_id = str(row.get("model_id") or "")
        if model_id.startswith("__BASE__::"):
            continue
        rows_by_model.setdefault(model_id, []).append(row)

    summaries: list[dict[str, Any]] = []
    model_texts: dict[str, str] = {}

    for model in models:
        model_id = model["model_id"]
        model_rows = rows_by_model.get(model_id, [])
        universal = [
            row for row in model_rows if not str(row.get("case_id", "")).startswith("heldout_")
        ]
        heldout = [
            row for row in model_rows if str(row.get("case_id", "")).startswith("heldout_")
        ]

        expected = [
            row["checks"].get("expected_pass")
            for row in universal
            if "expected_pass" in row.get("checks", {})
        ]
        memory = [
            row["checks"].get("memory_pass")
            for row in universal
            if "memory_pass" in row.get("checks", {})
        ]
        generic_hits = sum(
            len(row.get("checks", {}).get("generic_markers", []))
            for row in universal
        )
        empty_count = sum(
            bool(row.get("checks", {}).get("empty"))
            for row in universal
        )
        biography_flags = sum(
            len(row.get("checks", {}).get("biography_claim_markers", []))
            for row in universal
        )
        political_evasion = sum(
            len(row.get("checks", {}).get("political_evasion_markers", []))
            for row in universal
        )

        base_similarities = []
        for row in universal:
            base_response = base_by_case.get((model["base_model"], row["case_id"]))
            if base_response:
                base_similarities.append(
                    text_similarity(row["response"], base_response)
                )

        reference_text_similarities = []
        reference_style_similarities = []
        for row in heldout:
            reference = str(row.get("reference") or "")
            if reference:
                reference_text_similarities.append(
                    text_similarity(row["response"], reference)
                )
                reference_style_similarities.append(
                    style_similarity(row["response"], reference)
                )

        joined = "\n".join(
            str(row.get("response") or "")
            for row in universal
        )
        model_texts[model_id] = joined

        nll = nll_cache.get(model_id, {})
        base_nll = nll.get("base", {}).get("nll")
        adapter_nll = nll.get("adapter", {}).get("nll")
        nll_gain = (
            base_nll - adapter_nll
            if base_nll is not None and adapter_nll is not None
            else None
        )

        factual_rate = (
            sum(bool(value) for value in expected) / len(expected)
            if expected
            else None
        )
        memory_pass = (
            all(bool(value) for value in memory)
            if memory
            else None
        )
        base_similarity = (
            sum(base_similarities) / len(base_similarities)
            if base_similarities
            else None
        )

        reasons: list[str] = []
        if factual_rate is not None and factual_rate < 1.0:
            reasons.append("basic knowledge/reasoning failure")
        if memory_pass is False:
            reasons.append("conversation memory failure")
        if generic_hits >= 2:
            reasons.append("generic assistant language")
        if political_evasion >= 1:
            reasons.append("generic political evasion")
        if empty_count:
            reasons.append("empty response")
        if base_similarity is not None and base_similarity >= 0.72:
            reasons.append("too similar to base model")
        if nll_gain is not None and nll_gain < 0.10:
            reasons.append("weak held-out likelihood gain")
        if biography_flags:
            reasons.append("possible invented biography")

        hard_failure = (
            (factual_rate is not None and factual_rate < 2 / 3)
            or memory_pass is False
            or empty_count > 0
        )
        weak_persona = (
            generic_hits >= 2
            or political_evasion >= 1
            or (base_similarity is not None and base_similarity >= 0.72)
            or (nll_gain is not None and nll_gain < 0.10)
        )

        if hard_failure:
            automatic = "RETRAIN"
        elif weak_persona:
            automatic = "LIKELY_RETRAIN"
        else:
            automatic = "MANUAL_REVIEW"

        report = model.get("training_report") or {}
        summaries.append(
            {
                "model_id": model_id,
                "profile": model["profile"],
                "base_model": model["base_model"],
                "automatic_recommendation": automatic,
                "reasons": "; ".join(reasons),
                "universal_cases": len(universal),
                "heldout_cases": len(heldout),
                "factual_pass_rate": factual_rate,
                "memory_pass": memory_pass,
                "generic_marker_count": generic_hits,
                "political_evasion_count": political_evasion,
                "biography_flag_count": biography_flags,
                "base_response_similarity": base_similarity,
                "heldout_text_similarity": (
                    sum(reference_text_similarities) / len(reference_text_similarities)
                    if reference_text_similarities
                    else None
                ),
                "heldout_style_similarity": (
                    sum(reference_style_similarities) / len(reference_style_similarities)
                    if reference_style_similarities
                    else None
                ),
                "base_nll": base_nll,
                "adapter_nll": adapter_nll,
                "nll_gain": nll_gain,
                "training_validation_perplexity": report.get("validation_perplexity"),
                "training_test_perplexity": report.get("test_perplexity"),
                "manual_decision": "",
            }
        )

    for summary in summaries:
        similarities = []
        for other in summaries:
            if other["model_id"] == summary["model_id"]:
                continue
            similarities.append(
                text_similarity(
                    model_texts.get(summary["model_id"], ""),
                    model_texts.get(other["model_id"], ""),
                )
            )
        summary["nearest_other_model_similarity"] = max(similarities, default=None)

    summaries.sort(
        key=lambda row: (
            {"RETRAIN": 0, "LIKELY_RETRAIN": 1, "MANUAL_REVIEW": 2}.get(
                row["automatic_recommendation"],
                9,
            ),
            row["model_id"].casefold(),
        )
    )
    return summaries


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(
    path: Path,
    summaries: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if str(row.get("model_id") or "").startswith("__BASE__::"):
            continue
        records_by_model.setdefault(str(row.get("model_id")), []).append(row)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Persona benchmark</title>",
        """<style>
        body{font-family:Segoe UI,Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;background:#111;color:#eee}
        table{border-collapse:collapse;width:100%;margin:14px 0 28px}
        th,td{border:1px solid #444;padding:7px;vertical-align:top}
        th{background:#222;position:sticky;top:0}
        tr:nth-child(even){background:#181818}
        .RETRAIN{background:#5a1d1d}.LIKELY_RETRAIN{background:#594514}.MANUAL_REVIEW{background:#163f2a}
        details{border:1px solid #444;margin:12px 0;padding:10px;background:#171717}
        summary{font-weight:700;cursor:pointer}
        pre{white-space:pre-wrap;background:#0b0b0b;padding:10px;border:1px solid #333}
        .reference{border-left:4px solid #5f8;padding-left:10px}
        .response{border-left:4px solid #58f;padding-left:10px}
        .muted{color:#aaa}.warn{color:#ffb86c}
        </style></head><body>""",
        "<h1>Persona benchmark — canonical no-system mode</h1>",
        "<p class='muted'>Automatic labels are triage, not final judgments. Persona fidelity still requires human review against the authentic held-out replies.</p>",
        "<table><thead><tr>",
    ]
    columns = [
        "model_id",
        "base_model",
        "automatic_recommendation",
        "reasons",
        "factual_pass_rate",
        "memory_pass",
        "generic_marker_count",
        "political_evasion_count",
        "base_response_similarity",
        "nll_gain",
        "heldout_style_similarity",
        "nearest_other_model_similarity",
    ]
    for column in columns:
        parts.append(f"<th>{html.escape(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for summary in summaries:
        cls = html.escape(str(summary["automatic_recommendation"]))
        parts.append(f"<tr class='{cls}'>")
        for column in columns:
            parts.append(f"<td>{html.escape(fmt(summary.get(column)))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    for summary in summaries:
        model_id = summary["model_id"]
        parts.append(
            f"<details><summary>{html.escape(model_id)} — "
            f"{html.escape(summary['automatic_recommendation'])}</summary>"
        )
        parts.append(
            f"<p><b>Reasons:</b> {html.escape(summary.get('reasons') or 'none')}</p>"
        )
        for row in sorted(
            records_by_model.get(model_id, []),
            key=lambda value: str(value.get("case_id")),
        ):
            parts.append(
                f"<h3>{html.escape(str(row.get('case_id')))} "
                f"<span class='muted'>({html.escape(str(row.get('category')))})</span></h3>"
            )
            messages = row.get("messages") or []
            prompt_text = "\n".join(
                f"{message.get('role')}: {message.get('content')}"
                for message in messages
            )
            parts.append(
                f"<p><b>Prompt/context</b></p><pre>{html.escape(prompt_text)}</pre>"
            )
            reference = row.get("reference")
            if reference:
                parts.append(
                    f"<div class='reference'><b>Authentic held-out reply</b>"
                    f"<pre>{html.escape(str(reference))}</pre></div>"
                )
            parts.append(
                f"<div class='response'><b>Generated reply</b>"
                f"<pre>{html.escape(str(row.get('response') or ''))}</pre></div>"
            )
            checks = json.dumps(
                row.get("checks") or {},
                ensure_ascii=False,
                indent=2,
            )
            parts.append(
                f"<p class='muted'>Checks</p><pre>{html.escape(checks)}</pre>"
            )
        parts.append("</details>")

    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    trainer_path = Path(args.trainer).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results_path = output_root / "benchmark_results.jsonl"
    nll_path = output_root / "nll_scores.json"
    manifest_path = output_root / "benchmark_manifest.json"
    summary_csv = output_root / "benchmark_summary.csv"
    summary_json = output_root / "benchmark_summary.json"
    report_html = output_root / "benchmark_report.html"
    candidates_path = output_root / "retrain_candidates.txt"

    only = {value.casefold() for value in args.only}
    models = scan_models(model_root, only, args.base_model)
    if not models:
        raise RuntimeError("No installed adapters were found.")

    trainer = load_trainer(trainer_path)
    if not trainer.torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    probes = [
        probe
        for probe in PROBES
        if not args.quick or probe["id"] in QUICK_PROBE_IDS
    ]
    heldout_count = min(args.heldout_count, 2) if args.quick else args.heldout_count

    for model in models:
        prepared_test_value = str(model.get("prepared_test") or "").strip()
        if prepared_test_value:
            test_path = Path(prepared_test_value)
            model["heldout"] = (
                select_heldout(test_path, heldout_count)
                if test_path.is_file()
                else []
            )
        else:
            model["heldout"] = []

    manifest = {
        "format_version": 1,
        "system_mode": "none",
        "fallback_base_model": args.base_model,
        "trainer": str(trainer_path),
        "model_root": str(model_root),
        "models": [
            {
                key: value
                for key, value in model.items()
                if key not in {"training_report", "heldout"}
            }
            for model in models
        ],
        "probe_ids": [probe["id"] for probe in probes],
        "heldout_count": heldout_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json(manifest_path, manifest)

    records, completed = load_existing_records(results_path)
    nll_cache = read_json(nll_path, {}) or {}

    print(f"Models: {len(models)}")
    print(f"Universal probes per model: {len(probes)}")
    print(f"Held-out probes per compatible model: up to {heldout_count}")
    print(f"Output: {output_root}")
    print("System mode: none")
    print("Resume: enabled\n")

    models_by_base: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        models_by_base.setdefault(model["base_model"], []).append(model)

    overall_index = 0

    for current_base_id, base_group in models_by_base.items():
        print(f"\n=== LOADING BASE: {current_base_id} ===")
        base_model, tokenizer, _, _ = trainer.load_base_model(
            current_base_id,
            training=False,
        )
        base_model.eval()
        base_record_id = f"__BASE__::{current_base_id}"

        print("=== BASELINE MODEL ===")
        for case in probes:
            key = (base_record_id, case["id"])
            if key in completed:
                continue

            started = time.time()
            response, messages, checks = run_probe(
                trainer,
                base_model,
                tokenizer,
                case,
                seed=stable_seed("universal", case["id"]),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            row = record_probe(
                results_path,
                model_id=base_record_id,
                profile="base",
                adapter="",
                base_model=current_base_id,
                case=case,
                response=response,
                messages=messages,
                checks=checks,
                elapsed=time.time() - started,
            )
            records.append(row)
            completed.add(key)
            print(f"  complete: {case['id']}")

        if not args.skip_base_nll:
            print("\n=== BASE HELD-OUT NLL ===")
            for model in base_group:
                model_id = model["model_id"]
                heldout = model["heldout"]
                if not heldout:
                    continue

                cache = nll_cache.setdefault(model_id, {})
                if "base" in cache:
                    continue

                report = model.get("training_report") or {}
                tokenization = report.get("tokenization") or {}
                max_length = int(tokenization.get("max_length") or 512)
                max_target = int(tokenization.get("max_target_tokens") or 256)
                cache["base"] = score_nll(
                    trainer,
                    base_model,
                    tokenizer,
                    heldout,
                    max_length=max_length,
                    max_target_tokens=max_target,
                )
                atomic_json(nll_path, nll_cache)
                print(f"  {model_id}: {fmt(cache['base'].get('nll'))}")

        for model in base_group:
            overall_index += 1
            model_index = overall_index
            model_id = model["model_id"]
            profile = model["profile"]
            adapter = Path(model["adapter"])
            heldout = model["heldout"]

            universal_pending = any(
                (model_id, case["id"]) not in completed
                for case in probes
            )
            heldout_pending = any(
                (model_id, f"heldout_{index + 1:02d}") not in completed
                for index in range(len(heldout))
            )
            nll_pending = bool(heldout) and (
                "adapter" not in nll_cache.get(model_id, {})
            )

            if not universal_pending and not heldout_pending and not nll_pending:
                print(
                    f"\n[{model_index}/{len(models)}] "
                    f"{model_id}: already complete"
                )
                continue

            print(f"\n[{model_index}/{len(models)}] Loading {model_id}")
            model_with_adapter = trainer.PeftModel.from_pretrained(
                base_model,
                adapter,
            )
            model_with_adapter.eval()

            try:
                for case in probes:
                    key = (model_id, case["id"])
                    if key in completed:
                        continue

                    print(f"  probe: {case['id']}", flush=True)
                    started = time.time()
                    response, messages, checks = run_probe(
                        trainer,
                        model_with_adapter,
                        tokenizer,
                        case,
                        seed=stable_seed("universal", case["id"]),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    row = record_probe(
                        results_path,
                        model_id=model_id,
                        profile=profile,
                        adapter=str(adapter),
                        base_model=current_base_id,
                        case=case,
                        response=response,
                        messages=messages,
                        checks=checks,
                        elapsed=time.time() - started,
                    )
                    records.append(row)
                    completed.add(key)

                for index, heldout_row in enumerate(heldout, start=1):
                    case_id = f"heldout_{index:02d}"
                    key = (model_id, case_id)
                    if key in completed:
                        continue

                    case = {
                        "id": case_id,
                        "category": "heldout_authentic",
                        "messages": heldout_row["messages"][:-1],
                    }
                    print(f"  probe: {case_id}", flush=True)
                    started = time.time()
                    response = generate_single(
                        trainer,
                        model_with_adapter,
                        tokenizer,
                        case["messages"],
                        seed=stable_seed("heldout", profile, case_id),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    reference = heldout_row["messages"][-1]["content"]
                    checks = run_checks(case, response)
                    checks["reference_text_similarity"] = text_similarity(
                        response,
                        reference,
                    )
                    checks["reference_style_similarity"] = style_similarity(
                        response,
                        reference,
                    )
                    row = record_probe(
                        results_path,
                        model_id=model_id,
                        profile=profile,
                        adapter=str(adapter),
                        base_model=current_base_id,
                        case=case,
                        response=response,
                        messages=case["messages"],
                        checks=checks,
                        elapsed=time.time() - started,
                        reference=reference,
                    )
                    records.append(row)
                    completed.add(key)

                if heldout and "adapter" not in nll_cache.setdefault(model_id, {}):
                    report = model.get("training_report") or {}
                    tokenization = report.get("tokenization") or {}
                    max_length = int(tokenization.get("max_length") or 512)
                    max_target = int(tokenization.get("max_target_tokens") or 256)
                    nll_cache[model_id]["adapter"] = score_nll(
                        trainer,
                        model_with_adapter,
                        tokenizer,
                        heldout,
                        max_length=max_length,
                        max_target_tokens=max_target,
                    )
                    atomic_json(nll_path, nll_cache)
                    print(
                        "  held-out adapter NLL: "
                        f"{fmt(nll_cache[model_id]['adapter'].get('nll'))}"
                    )
            finally:
                base_model = model_with_adapter.unload()
                del model_with_adapter
                gc.collect()
                trainer.torch.cuda.empty_cache()

            records = read_jsonl(results_path)
            summaries = aggregate(models, records, nll_cache)
            write_csv(summary_csv, summaries)
            atomic_json(summary_json, summaries)
            write_html(report_html, summaries, records)
            candidates = [
                row["model_id"]
                for row in summaries
                if row["automatic_recommendation"]
                in {"RETRAIN", "LIKELY_RETRAIN"}
            ]
            candidates_path.write_text(
                "\n".join(candidates) + ("\n" if candidates else ""),
                encoding="utf-8",
            )

        del base_model
        gc.collect()
        trainer.torch.cuda.empty_cache()

    records = read_jsonl(results_path)
    summaries = aggregate(models, records, nll_cache)
    write_csv(summary_csv, summaries)
    atomic_json(summary_json, summaries)
    write_html(report_html, summaries, records)
    candidates = [
        row["model_id"]
        for row in summaries
        if row["automatic_recommendation"] in {"RETRAIN", "LIKELY_RETRAIN"}
    ]
    candidates_path.write_text(
        "\n".join(candidates) + ("\n" if candidates else ""),
        encoding="utf-8",
    )

    print("\n=== BENCHMARK COMPLETE ===")
    print(f"Summary CSV: {summary_csv}")
    print(f"HTML report: {report_html}")
    print(f"Raw results: {results_path}")
    print(f"Retrain candidates: {candidates_path}")
    print(
        "\nAutomatic labels are triage. "
        "Review the actual responses before moving models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
