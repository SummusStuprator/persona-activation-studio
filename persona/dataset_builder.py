#!/usr/bin/env python3
"""Build authentic, context-aware per-profile fine-tuning datasets from xscra exports.

The default SFT outputs contain only observed X/Twitter relationships:
- replies with a recovered direct parent (one-shot or reconstructed multi-turn)
- quote posts with a recovered quoted source

No synthetic user prompt is inserted into the canonical SFT files. Standalone posts,
replies with missing parents, unresolved quotes, self-thread continuations, and media-only
examples are preserved in separate corpora/queues rather than silently discarded.

Standard-library only. Supports either an xscra project directory or a share-safe ZIP.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urlparse

MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{1,50})", re.UNICODE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TCO_RE = re.compile(r"https?://t\.co/\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t]+")
LOOSE_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TOKEN_LINK_RE = re.compile(r"<LINK(?::[^>]+)?>")
VALID_RESPONSE_CLASSES = {"substantive", "emoji_or_symbol", "punctuation_reaction"}
VALID_PROMPT_CLASSES = {"substantive", "emoji_or_symbol"}


def jd(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=pretty,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []


def parse_date(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.min


def normalize_username(value: Any) -> str:
    return str(value or "").strip().lstrip("@").lower()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unknown"


def media_info(row: dict[str, Any]) -> dict[str, Any]:
    photos = parse_list(row.get("photos"))
    videos = parse_list(row.get("videos"))
    gifs = parse_list(row.get("animated_gifs"))
    links = parse_list(row.get("links"))
    return {
        "images": len(photos),
        "videos": len(videos),
        "gifs": len(gifs),
        "links": len(links),
        "has_media": bool(photos or videos or gifs),
        "has_links": bool(links),
        "expanded_links": [str(x) for x in links],
    }


def link_markers(row: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for value in parse_list(row.get("links")):
        url = str(value)
        try:
            host = (urlparse(url).hostname or "").lower()
        except Exception:
            host = ""
        host = host.removeprefix("www.")
        if host in {"youtube.com", "youtu.be", "m.youtube.com"}:
            marker = "<LINK:youtube>"
        elif host in {"open.spotify.com", "spotify.com"}:
            marker = "<LINK:spotify>"
        elif host in {"x.com", "twitter.com", "mobile.twitter.com"}:
            marker = "<LINK:x>"
        elif host:
            marker = f"<LINK:{host}>"
        else:
            marker = "<LINK>"
        if marker not in markers:
            markers.append(marker)
    return markers


def normalize_transport_text(text: Any, row: dict[str, Any]) -> str:
    """Remove encoding/transport artifacts without rewriting a person's wording."""
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFC", value)
    kept: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if char in "\n\t":
            kept.append(char)
        elif category == "Cc":
            continue
        elif category == "Cf" and char not in {"\u200d", "\ufe0f"}:
            continue
        else:
            kept.append(char)
    value = "".join(kept)

    # t.co links are transport pointers. Preserve the link *type* via expanded metadata,
    # but do not teach transient hashes. Attached media remains in metadata.
    value = TCO_RE.sub("", value)
    value = URL_RE.sub("<LINK>", value)
    for marker in link_markers(row):
        if marker not in value:
            value = f"{value} {marker}".strip()

    value = SPACE_RE.sub(" ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def mojibake_score(value: str) -> int:
    if "\ufffd" in value or any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        return 100
    markers = ["ÔÇ", "Ã", "Â", "â€™", "â€œ", "â€", "Ń", "ň", "╣", "▒", "▓"]
    return sum(value.count(marker) for marker in markers)


def is_probably_emoji_or_symbol(char: str) -> bool:
    code = ord(char)
    if 0x1F000 <= code <= 0x1FAFF:
        return True
    if 0x2600 <= code <= 0x27BF:
        return True
    return unicodedata.category(char) in {"So", "Sk"}


def classify_text(text: str, row: dict[str, Any]) -> str:
    if mojibake_score(text) >= 3:
        return "unintelligible"
    if not text:
        return "media_only" if media_info(row)["has_media"] else "empty"

    without_links = TOKEN_LINK_RE.sub(" ", text)
    without_mentions = MENTION_RE.sub(" ", without_links)
    if any(ch.isalnum() for ch in without_mentions):
        return "substantive"

    compact = "".join(ch for ch in without_mentions if not ch.isspace())
    if any(is_probably_emoji_or_symbol(ch) for ch in compact):
        return "emoji_or_symbol"

    no_mentions = MENTION_RE.sub("", TOKEN_LINK_RE.sub("", text)).strip()
    if not no_mentions and MENTION_RE.search(text):
        return "mention_only"
    if not TOKEN_LINK_RE.sub("", text).strip() and TOKEN_LINK_RE.search(text):
        return "link_only"
    if compact and len(compact) <= 24 and all(unicodedata.category(ch).startswith("P") for ch in compact):
        return "punctuation_reaction"
    if media_info(row)["has_media"]:
        return "media_or_low_information"
    return "other_low_information"


def canonical_exact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = SPACE_RE.sub(" ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def canonical_loose(value: str) -> str:
    value = canonical_exact(value)
    value = MENTION_RE.sub("@user", value)
    value = TOKEN_LINK_RE.sub("<link>", value)
    value = LOOSE_NUMBER_RE.sub("<n>", value)
    value = re.sub(r"[^\w\s@<>]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


@dataclass
class Post:
    id: str
    username: str
    display_name: str
    date: str
    kind: str
    text_raw: str
    text_clean: str
    text_class: str
    url: str
    lang: str
    conversation_id: str
    parent_id: str
    reply_to_username: str
    quote_id: str
    quote_url: str
    source: str
    row: dict[str, Any] = field(repr=False)
    media: dict[str, Any] = field(default_factory=dict)

    @property
    def username_key(self) -> str:
        return normalize_username(self.username)

    @property
    def dt(self) -> datetime:
        return parse_date(self.date)

    @property
    def group_id(self) -> str:
        return self.conversation_id or self.id


class InputStore:
    def __init__(self, source: Path, exports: Path | None = None):
        self.source = source
        self.exports = exports
        if exports is not None and source.is_file():
            raise ValueError("An external exports directory requires a directory input.")
        self.zf: zipfile.ZipFile | None = None
        if source.is_file() and source.suffix.lower() == ".zip":
            self.zf = zipfile.ZipFile(source, "r")

    def close(self) -> None:
        if self.zf:
            self.zf.close()

    def locate(self, suffixes: Sequence[str]) -> str:
        normalized = [s.replace("\\", "/").lstrip("./") for s in suffixes]
        if self.zf:
            names = self.zf.namelist()
            for suffix in normalized:
                matches = [
                    name
                    for name in names
                    if name.replace("\\", "/").lstrip("./").endswith(suffix)
                ]
                if matches:
                    return sorted(matches, key=len)[0]
        else:
            for suffix in normalized:
                if self.exports is not None and suffix.startswith('x_exports/'):
                    candidate = self.exports / suffix.removeprefix('x_exports/')
                    if candidate.is_file():
                        return str(candidate)
                    continue
                direct = self.source / suffix
                if direct.is_file():
                    return str(direct)
                matches = list(self.source.rglob(Path(suffix).name))
                matches = [
                    p
                    for p in matches
                    if p.as_posix().replace("\\", "/").endswith(suffix)
                ]
                if matches:
                    return str(sorted(matches, key=lambda p: len(str(p)))[0])
        raise FileNotFoundError(f"Could not find any of: {', '.join(suffixes)}")

    def open_text(self, member: str, encoding: str = "utf-8-sig"):
        if self.zf:
            return io.TextIOWrapper(self.zf.open(member, "r"), encoding=encoding, newline="")
        return open(member, "r", encoding=encoding, newline="")

    def read_csv(self, suffixes: Sequence[str]) -> list[dict[str, str]]:
        member = self.locate(suffixes)
        with self.open_text(member, "utf-8-sig") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    def read_jsonl(self, suffixes: Sequence[str]) -> list[dict[str, Any]]:
        member = self.locate(suffixes)
        rows: list[dict[str, Any]] = []
        with self.open_text(member, "utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {member}:{line_number}: {exc}") from exc
        return rows


def make_post(row: dict[str, Any], source: str) -> Post:
    clean = normalize_transport_text(row.get("text", ""), row)
    return Post(
        id=str(row.get("id", "")).strip(),
        username=str(row.get("username", "")).strip(),
        display_name=str(row.get("display_name", "")).strip(),
        date=str(row.get("date", "")).strip(),
        kind=str(row.get("kind", "tweet")).strip() or "tweet",
        text_raw=str(row.get("text", "")),
        text_clean=clean,
        text_class=classify_text(clean, row),
        url=str(row.get("url", "")).strip(),
        lang=str(row.get("lang", "")).strip(),
        conversation_id=str(row.get("conversation_id", "")).strip(),
        parent_id=str(row.get("in_reply_to_tweet_id", "")).strip(),
        reply_to_username=str(row.get("in_reply_to_username", "")).strip(),
        quote_id=str(row.get("quoted_tweet_id", "")).strip(),
        quote_url=str(row.get("quoted_tweet_url", "")).strip(),
        source=source,
        row=row,
        media=media_info(row),
    )


def merge_posts(preferred: Post, fallback: Post) -> Post:
    """Keep authored data but fill missing fields from cache data."""
    for attr in (
        "display_name",
        "date",
        "kind",
        "text_raw",
        "text_clean",
        "url",
        "lang",
        "conversation_id",
        "parent_id",
        "reply_to_username",
        "quote_id",
        "quote_url",
    ):
        if not getattr(preferred, attr) and getattr(fallback, attr):
            setattr(preferred, attr, getattr(fallback, attr))
    return preferred


def collect_chain(final: Post, nodes: dict[str, Post], max_posts: int) -> list[Post]:
    chain: list[Post] = [final]
    seen = {final.id}
    current = final
    while current.parent_id and len(chain) < max_posts:
        parent = nodes.get(current.parent_id)
        if not parent or parent.id in seen:
            break
        chain.append(parent)
        seen.add(parent.id)
        current = parent
    return list(reversed(chain))


def participant_mapping(posts: Sequence[Post], target: str, direct_parent: Post | None) -> dict[str, str]:
    mapping: dict[str, str] = {normalize_username(target): "<SELF>"}
    if direct_parent and direct_parent.username_key and direct_parent.username_key not in mapping:
        mapping[direct_parent.username_key] = "<PARENT>"
    index = 2
    for post in posts:
        key = post.username_key
        if key and key not in mapping:
            mapping[key] = f"<USER_{index}>"
            index += 1
        for match in MENTION_RE.finditer(post.text_clean):
            mention_key = normalize_username(match.group(1))
            if mention_key and mention_key not in mapping:
                mapping[mention_key] = f"<USER_{index}>"
                index += 1
    return mapping


def replace_mentions(value: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        username = normalize_username(match.group(1))
        return mapping.get(username, "<USER>")

    return MENTION_RE.sub(repl, value)


def username_label(username: str, mapping: dict[str, str], portable: bool) -> str:
    if portable:
        return mapping.get(normalize_username(username), "<USER>")
    return f"@{username}" if username else "@unknown"


def render_post(
    post: Post,
    nodes: dict[str, Post],
    mapping: dict[str, str],
    portable: bool,
) -> str:
    author = username_label(post.username, mapping, portable)
    routing = ""
    if post.reply_to_username:
        routed = username_label(post.reply_to_username, mapping, portable)
        routing = f" [replying to {routed}]"
    elif post.kind == "quote":
        routing = " [quote post]"
    text = replace_mentions(post.text_clean, mapping) if portable else post.text_clean
    parts = [f"{author}{routing}: {text}".rstrip()]

    if post.quote_id:
        quoted = nodes.get(post.quote_id)
        if quoted and quoted.text_class in VALID_PROMPT_CLASSES:
            q_author = username_label(quoted.username, mapping, portable)
            q_text = replace_mentions(quoted.text_clean, mapping) if portable else quoted.text_clean
            parts.append(f"↳ quoted {q_author}: {q_text}")
    return "\n".join(parts)


def merge_role_messages(items: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for role, content in items:
        if not content.strip():
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n---\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


def example_fingerprint(messages: Sequence[dict[str, str]], task: str, loose: bool = False) -> str:
    normalizer = canonical_loose if loose else canonical_exact
    payload = {
        "task": task,
        "messages": [
            {"role": m["role"], "content": normalizer(m["content"])} for m in messages
        ],
    }
    return sha256_text(jd(payload))


def render_example_variant(
    context: Sequence[Post],
    response: Post,
    nodes: dict[str, Post],
    task: str,
    portable: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    direct_parent = nodes.get(response.parent_id) if response.parent_id else nodes.get(response.quote_id)
    mapping = participant_mapping([*context, response], response.username, direct_parent)
    role_items: list[tuple[str, str]] = []
    target_key = response.username_key
    for post in context:
        role = "assistant" if post.username_key == target_key else "user"
        role_items.append((role, render_post(post, nodes, mapping, portable)))
    messages = merge_role_messages(role_items)

    # A canonical external-reply example must begin from an observed external turn and
    # finish immediately before a user turn. Trim unusable leading target-only history.
    while messages and messages[0]["role"] == "assistant":
        messages.pop(0)
    if task.startswith("external_reply"):
        while messages and messages[-1]["role"] == "assistant":
            messages.pop()

    response_text = replace_mentions(response.text_clean, mapping) if portable else response.text_clean
    messages.append({"role": "assistant", "content": response_text})
    return messages, {token: username for username, token in mapping.items()}


def build_reply_example(
    post: Post,
    nodes: dict[str, Post],
    max_context_posts: int,
) -> dict[str, Any] | None:
    parent = nodes.get(post.parent_id)
    if not parent:
        return None
    if parent.username_key == post.username_key:
        return None
    if post.text_class not in VALID_RESPONSE_CLASSES:
        return None
    if parent.text_class not in VALID_PROMPT_CLASSES:
        return None

    chain = collect_chain(post, nodes, max_context_posts)
    context = chain[:-1]
    # Keep textual ancestors; direct parent has already passed the prompt gate.
    filtered: list[Post] = []
    for ancestor in context:
        if ancestor.id == parent.id or ancestor.text_class in VALID_PROMPT_CLASSES | VALID_RESPONSE_CLASSES:
            filtered.append(ancestor)
    context = filtered[-(max_context_posts - 1) :]

    target_key = post.username_key
    prior_target_turn = any(p.username_key == target_key for p in context)
    task = "external_reply_multiturn" if len(context) >= 2 and prior_target_turn else "external_reply_oneshot"
    portable_messages, mention_map = render_example_variant(context, post, nodes, task, True)
    verbatim_messages, _ = render_example_variant(context, post, nodes, task, False)
    if len(portable_messages) < 2 or portable_messages[-2]["role"] != "user":
        return None

    used_posts = [p.id for p in context] + [post.id]
    used_conversations = sorted({p.group_id for p in [*context, post] if p.group_id})
    has_media_context = any(p.media["has_media"] for p in context)
    has_media_response = post.media["has_media"]
    quality_tier = "extended_text_with_media" if has_media_context or has_media_response else "core_text"

    return {
        "id": f"reply:{post.id}",
        "profile": post.username,
        "task": task,
        "date": post.date,
        "conversation_id": post.group_id,
        "source_post_id": post.id,
        "context_post_ids": [p.id for p in context],
        "all_post_ids": used_posts,
        "conversation_keys": used_conversations,
        "direct_parent_id": parent.id,
        "direct_parent_username": parent.username,
        "messages_portable": portable_messages,
        "messages_verbatim": verbatim_messages,
        "mention_map": mention_map,
        "quality_tier": quality_tier,
        "context_depth": len(context),
        "external_authors": sorted({p.username for p in context if p.username_key != target_key}),
        "has_media_context": has_media_context,
        "has_media_response": has_media_response,
        "target_lang": post.lang,
        "source_url": post.url,
    }


def build_quote_example(post: Post, nodes: dict[str, Post]) -> dict[str, Any] | None:
    quoted = nodes.get(post.quote_id)
    if not quoted:
        return None
    # Quoting one's own prior post is an authentic self-thread-like continuation, not
    # an external user prompt. Keep it out of canonical reply SFT.
    if quoted.username_key == post.username_key:
        return None
    if post.text_class not in VALID_RESPONSE_CLASSES or quoted.text_class not in VALID_PROMPT_CLASSES:
        return None
    task = "quote_commentary"
    portable_messages, mention_map = render_example_variant([quoted], post, nodes, task, True)
    verbatim_messages, _ = render_example_variant([quoted], post, nodes, task, False)
    has_media_context = quoted.media["has_media"]
    has_media_response = post.media["has_media"]
    return {
        "id": f"quote:{post.id}",
        "profile": post.username,
        "task": task,
        "date": post.date,
        "conversation_id": post.id,
        "source_post_id": post.id,
        "context_post_ids": [quoted.id],
        "all_post_ids": [quoted.id, post.id],
        "conversation_keys": sorted({quoted.group_id, post.group_id}),
        "direct_parent_id": quoted.id,
        "direct_parent_username": quoted.username,
        "messages_portable": portable_messages,
        "messages_verbatim": verbatim_messages,
        "mention_map": mention_map,
        "quality_tier": "extended_text_with_media" if has_media_context or has_media_response else "core_text",
        "context_depth": 1,
        "external_authors": [quoted.username],
        "has_media_context": has_media_context,
        "has_media_response": has_media_response,
        "target_lang": post.lang,
        "source_url": post.url,
    }


def public_example(example: dict[str, Any], portable: bool) -> dict[str, Any]:
    messages = example["messages_portable"] if portable else example["messages_verbatim"]
    return {
        "id": example["id"],
        "profile": example["profile"],
        "task": example["task"],
        "messages": messages,
        "metadata": {
            "date": example["date"],
            "conversation_id": example["conversation_id"],
            "source_post_id": example["source_post_id"],
            "context_post_ids": example["context_post_ids"],
            "direct_parent_id": example["direct_parent_id"],
            "direct_parent_username": example["direct_parent_username"],
            "context_depth": example["context_depth"],
            "quality_tier": example["quality_tier"],
            "has_media_context": example["has_media_context"],
            "has_media_response": example["has_media_response"],
            "target_lang": example["target_lang"],
            "source_url": example["source_url"],
            "mention_map": example["mention_map"] if portable else {},
            "duplicate_count": example.get("duplicate_count", 1),
            "duplicate_source_ids": example.get("duplicate_source_ids", []),
        },
    }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def deduplicate_examples(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        fp = example_fingerprint(ex["messages_portable"], ex["task"], loose=False)
        ex["exact_fingerprint"] = fp
        ex["loose_fingerprint"] = example_fingerprint(ex["messages_portable"], ex["task"], loose=True)
        groups[fp].append(ex)

    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for fp, group in groups.items():
        group.sort(
            key=lambda x: (
                x["quality_tier"] != "core_text",
                -int(x["context_depth"]),
                parse_date(x["date"]),
                x["id"],
            )
        )
        winner = group[0]
        winner["duplicate_count"] = len(group)
        winner["duplicate_source_ids"] = [x["source_post_id"] for x in group[1:]]
        kept.append(winner)
        if len(group) > 1:
            duplicates.append(
                {
                    "fingerprint": fp,
                    "kept_id": winner["id"],
                    "duplicate_ids": [x["id"] for x in group[1:]],
                    "source_post_ids": [x["source_post_id"] for x in group],
                    "count": len(group),
                }
            )
    kept.sort(key=lambda x: (parse_date(x["date"]), x["id"]))
    return kept, duplicates


def assign_splits(
    examples: list[dict[str, Any]], train_ratio: float, validation_ratio: float
) -> dict[str, str]:
    if not examples:
        return {}
    uf = UnionFind(len(examples))
    key_owner: dict[str, int] = {}
    for idx, ex in enumerate(examples):
        keys: list[str] = []
        keys.extend(f"post:{x}" for x in ex["all_post_ids"] if x)
        keys.extend(f"conv:{x}" for x in ex["conversation_keys"] if x)
        keys.append(f"loose:{ex['loose_fingerprint']}")
        for key in keys:
            if key in key_owner:
                uf.union(idx, key_owner[key])
            else:
                key_owner[key] = idx

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(examples)):
        components[uf.find(idx)].append(idx)

    ordered = sorted(
        components.values(),
        key=lambda indices: (
            max(parse_date(examples[i]["date"]) for i in indices),
            min(examples[i]["id"] for i in indices),
        ),
    )
    total = len(examples)
    train_target = int(round(total * train_ratio))
    val_target = int(round(total * validation_ratio))
    if total >= 10:
        train_target = min(max(train_target, 1), total - 2)
        val_target = min(max(val_target, 1), total - train_target - 1)

    result: dict[str, str] = {}
    assigned = 0
    for component in ordered:
        if assigned < train_target:
            split = "train"
        elif assigned < train_target + val_target:
            split = "validation"
        else:
            split = "test"
        for idx in component:
            result[examples[idx]["id"]] = split
        assigned += len(component)
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(jd(row) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def queue_record(post: Post, reason: str, nodes: dict[str, Post]) -> dict[str, Any]:
    parent = nodes.get(post.parent_id) if post.parent_id else None
    quoted = nodes.get(post.quote_id) if post.quote_id else None
    return {
        "id": f"queue:{post.id}",
        "profile": post.username,
        "source_post_id": post.id,
        "task_source": post.kind,
        "reason": reason,
        "observed_output": post.text_clean,
        "output_class": post.text_class,
        "date": post.date,
        "conversation_id": post.group_id,
        "in_reply_to_tweet_id": post.parent_id,
        "in_reply_to_username": post.reply_to_username,
        "quoted_tweet_id": post.quote_id,
        "known_parent_text": parent.text_clean if parent else "",
        "known_parent_username": parent.username if parent else "",
        "known_quote_text": quoted.text_clean if quoted else "",
        "known_quote_username": quoted.username if quoted else "",
        "media": post.media,
        "source_url": post.url,
        "synthetic_prompt": None,
        "review_status": "unreviewed",
    }


def self_thread_record(post: Post, nodes: dict[str, Post], max_context_posts: int) -> dict[str, Any]:
    chain = collect_chain(post, nodes, max_context_posts)
    context = chain[:-1]
    return {
        "id": f"self-thread:{post.id}",
        "profile": post.username,
        "task": "self_thread_continuation",
        "context": [
            {
                "post_id": p.id,
                "username": p.username,
                "text": p.text_clean,
                "date": p.date,
                "reply_to_username": p.reply_to_username,
                "media": p.media,
            }
            for p in context
        ],
        "continuation": post.text_clean,
        "metadata": {
            "date": post.date,
            "conversation_id": post.group_id,
            "source_post_id": post.id,
            "source_url": post.url,
            "response_class": post.text_class,
            "not_in_canonical_sft": True,
        },
    }


def authored_corpus_record(post: Post) -> dict[str, Any]:
    return {
        "id": f"corpus:{post.id}",
        "profile": post.username,
        "text": post.text_clean,
        "metadata": {
            "post_id": post.id,
            "date": post.date,
            "kind": post.kind,
            "lang": post.lang,
            "conversation_id": post.group_id,
            "text_class": post.text_class,
            "media": post.media,
            "url": post.url,
        },
    }


def validate_profile_outputs(profile_dir: Path) -> dict[str, Any]:
    seen_ids: set[str] = set()
    split_posts: dict[str, set[str]] = defaultdict(set)
    split_convs: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for variant in ("portable", "verbatim"):
        for tier in ("core", "extended"):
            for split in ("train", "validation", "test"):
                path = profile_dir / "sft" / variant / tier / f"{split}.jsonl"
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as fh:
                    for line_number, line in enumerate(fh, 1):
                        row = json.loads(line)
                        messages = row.get("messages", [])
                        if (
                            len(messages) < 2
                            or messages[0].get("role") != "user"
                            or messages[-1].get("role") != "assistant"
                        ):
                            raise ValueError(f"Invalid canonical role sequence in {path}:{line_number}")
                        if any(not str(m.get("content", "")).strip() for m in messages):
                            raise ValueError(f"Blank message in {path}:{line_number}")
                        if variant == "portable" and tier == "extended":
                            record_id = row["id"]
                            if record_id in seen_ids:
                                raise ValueError(f"Duplicate record id in extended SFT: {record_id}")
                            seen_ids.add(record_id)
                            meta = row["metadata"]
                            split_posts[split].update([meta["source_post_id"], *meta["context_post_ids"]])
                            split_convs[split].add(meta["conversation_id"])
                            counts[split] += 1
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_posts[a] & split_posts[b]
        if overlap:
            raise ValueError(f"Post-id leakage between {a} and {b}: {len(overlap)}")
        conv_overlap = split_convs[a] & split_convs[b]
        if conv_overlap:
            raise ValueError(f"Conversation leakage between {a} and {b}: {len(conv_overlap)}")
    return {"extended_split_counts": dict(counts), "validated_ids": len(seen_ids)}


def build(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        if output == source or source.is_relative_to(output):
            raise ValueError('The output directory contains the input source.')
        import uuid
        previous = output.with_name(output.name + '.backup-' + uuid.uuid4().hex[:8])
        output.rename(previous)
        print('Previous dataset:', previous)
    output.mkdir(parents=True)

    exports = getattr(args, 'exports', None)
    store = InputStore(source, Path(exports).expanduser().resolve() if exports else None)
    try:
        authored_rows = store.read_csv(
            ["x_exports/all_posts_with_context.csv", "x_exports/all_posts.csv"]
        )
        try:
            parent_rows = store.read_jsonl(["x_exports/reply_context/parents.jsonl"])
        except FileNotFoundError:
            parent_rows = store.read_csv(["x_exports/reply_context/parents.csv"])
        try:
            target_status_rows = store.read_csv(["x_exports/external_target_status.csv"])
        except FileNotFoundError:
            target_status_rows = []
    finally:
        store.close()

    authored = [make_post(row, "authored") for row in authored_rows if str(row.get("id", "")).strip()]
    parents = [make_post(row, "parent_cache") for row in parent_rows if str(row.get("id", "")).strip()]

    nodes: dict[str, Post] = {}
    for post in parents:
        if post.id in nodes:
            nodes[post.id] = merge_posts(nodes[post.id], post)
        else:
            nodes[post.id] = post
    for post in authored:
        if post.id in nodes:
            nodes[post.id] = merge_posts(post, nodes[post.id])
        else:
            nodes[post.id] = post

    represented_profiles = {p.username for p in authored if p.username}
    configured_profiles = {
        str(row.get("profile", "")).strip()
        for row in target_status_rows
        if str(row.get("profile", "")).strip()
    }
    profiles = sorted(represented_profiles | configured_profiles, key=str.casefold)
    authored_by_profile: dict[str, list[Post]] = defaultdict(list)
    for post in authored:
        authored_by_profile[post.username].append(post)

    summary_rows: list[dict[str, Any]] = []
    all_duplicate_rows: list[dict[str, Any]] = []
    global_counts = Counter()
    validation_results: dict[str, Any] = {}

    for profile in profiles:
        posts = sorted(authored_by_profile[profile], key=lambda p: (p.dt, p.id))
        profile_key = normalize_username(profile)
        examples_raw: list[dict[str, Any]] = []
        self_threads: list[dict[str, Any]] = []
        prompt_queue: list[dict[str, Any]] = []
        multimodal_queue: list[dict[str, Any]] = []
        low_information_queue: list[dict[str, Any]] = []
        corrupt_queue: list[dict[str, Any]] = []
        corpus: list[dict[str, Any]] = []
        stats = Counter()

        for post in posts:
            stats["authored_posts"] += 1
            stats[f"authored_kind_{post.kind}"] += 1
            stats[f"authored_class_{post.text_class}"] += 1
            if post.text_class in VALID_RESPONSE_CLASSES:
                corpus.append(authored_corpus_record(post))

            if post.text_class == "unintelligible":
                corrupt_queue.append(queue_record(post, "unintelligible_text", nodes))
                continue

            if post.kind == "reply" or post.parent_id:
                stats["replies_total"] += 1
                parent = nodes.get(post.parent_id)
                if not parent:
                    prompt_queue.append(queue_record(post, "missing_direct_parent", nodes))
                    stats["replies_missing_parent"] += 1
                    continue
                stats["replies_parent_found"] += 1
                if parent.username_key == profile_key:
                    self_threads.append(self_thread_record(post, nodes, args.max_context_posts))
                    stats["self_thread_replies"] += 1
                    continue
                if post.text_class not in VALID_RESPONSE_CLASSES:
                    target = multimodal_queue if post.media["has_media"] else low_information_queue
                    target.append(queue_record(post, f"unusable_response_{post.text_class}", nodes))
                    stats["replies_unusable_response"] += 1
                    continue
                if parent.text_class not in VALID_PROMPT_CLASSES:
                    target = multimodal_queue if parent.media["has_media"] else low_information_queue
                    target.append(queue_record(post, f"unusable_parent_{parent.text_class}", nodes))
                    stats["replies_unusable_parent"] += 1
                    continue
                example = build_reply_example(post, nodes, args.max_context_posts)
                if example:
                    examples_raw.append(example)
                    stats[f"sft_{example['task']}"] += 1
                else:
                    low_information_queue.append(queue_record(post, "reply_chain_not_representable", nodes))
                    stats["reply_chain_not_representable"] += 1
                continue

            if post.kind == "quote" or post.quote_id:
                stats["quotes_total"] += 1
                quoted = nodes.get(post.quote_id)
                if not quoted:
                    prompt_queue.append(queue_record(post, "missing_quoted_source", nodes))
                    stats["quotes_missing_source"] += 1
                    continue
                if post.text_class not in VALID_RESPONSE_CLASSES:
                    target = multimodal_queue if post.media["has_media"] else low_information_queue
                    target.append(queue_record(post, f"unusable_quote_commentary_{post.text_class}", nodes))
                    continue
                if quoted.text_class not in VALID_PROMPT_CLASSES:
                    target = multimodal_queue if quoted.media["has_media"] else low_information_queue
                    target.append(queue_record(post, f"unusable_quoted_source_{quoted.text_class}", nodes))
                    continue
                if quoted.username_key == profile_key:
                    self_threads.append(
                        {
                            "id": f"self-quote:{post.id}",
                            "profile": post.username,
                            "task": "self_quote_commentary",
                            "context": [
                                {
                                    "post_id": quoted.id,
                                    "username": quoted.username,
                                    "text": quoted.text_clean,
                                    "date": quoted.date,
                                    "media": quoted.media,
                                }
                            ],
                            "continuation": post.text_clean,
                            "metadata": {
                                "date": post.date,
                                "source_post_id": post.id,
                                "quoted_post_id": quoted.id,
                                "source_url": post.url,
                                "not_in_canonical_sft": True,
                            },
                        }
                    )
                    stats["self_quote_commentary"] += 1
                    continue
                example = build_quote_example(post, nodes)
                if example:
                    examples_raw.append(example)
                    stats["sft_quote_commentary"] += 1
                continue

            # Standalone posts have no observed prompt. Preserve them for optional DAPT and
            # for a local-model prompt-generation queue, but never put them in canonical SFT.
            if post.text_class in VALID_RESPONSE_CLASSES:
                prompt_queue.append(queue_record(post, "standalone_no_observed_prompt", nodes))
                stats["standalone_prompt_queue"] += 1
            elif post.media["has_media"]:
                multimodal_queue.append(queue_record(post, "standalone_media_only_or_low_text", nodes))
            else:
                low_information_queue.append(queue_record(post, f"standalone_{post.text_class}", nodes))

        examples, duplicate_rows = deduplicate_examples(examples_raw)
        for row in duplicate_rows:
            row["profile"] = profile
        all_duplicate_rows.extend(duplicate_rows)
        stats["sft_raw"] = len(examples_raw)
        stats["sft_deduplicated"] = len(examples)
        stats["sft_exact_duplicates_removed"] = len(examples_raw) - len(examples)
        stats["sft_core"] = sum(x["quality_tier"] == "core_text" for x in examples)
        stats["sft_extended"] = len(examples)
        stats["unique_external_authors"] = len(
            {author.casefold() for ex in examples for author in ex["external_authors"] if author}
        )

        split_map = assign_splits(examples, args.train_ratio, args.validation_ratio)
        profile_dir = output / "profiles" / safe_name(profile)

        for variant, portable in (("portable", True), ("verbatim", False)):
            for tier in ("core", "extended"):
                tier_examples = [
                    ex for ex in examples if tier == "extended" or ex["quality_tier"] == "core_text"
                ]
                for split in ("train", "validation", "test"):
                    rows = [
                        public_example(ex, portable)
                        for ex in tier_examples
                        if split_map.get(ex["id"]) == split
                    ]
                    write_jsonl(profile_dir / "sft" / variant / tier / f"{split}.jsonl", rows)

        write_jsonl(profile_dir / "auxiliary" / "self_thread_continuations.jsonl", self_threads)
        write_jsonl(profile_dir / "auxiliary" / "authored_style_corpus.jsonl", corpus)
        write_jsonl(profile_dir / "queues" / "prompt_generation.jsonl", prompt_queue)
        write_jsonl(profile_dir / "queues" / "multimodal_or_media_dependent.jsonl", multimodal_queue)
        write_jsonl(profile_dir / "queues" / "low_information.jsonl", low_information_queue)
        write_jsonl(profile_dir / "queues" / "unintelligible_or_corrupt.jsonl", corrupt_queue)

        validation_results[profile] = validate_profile_outputs(profile_dir)
        split_counts = Counter(split_map.values())
        summary = {
            "profile": profile,
            "dataset_status": "ready" if posts else "no_authored_posts",
            **{key: stats[key] for key in sorted(stats)},
            "self_thread_records": len(self_threads),
            "authored_style_corpus_records": len(corpus),
            "prompt_generation_queue": len(prompt_queue),
            "multimodal_queue": len(multimodal_queue),
            "low_information_queue": len(low_information_queue),
            "corrupt_queue": len(corrupt_queue),
            "train_examples": split_counts["train"],
            "validation_examples": split_counts["validation"],
            "test_examples": split_counts["test"],
        }
        summary_rows.append(summary)
        global_counts.update(stats)

    write_csv(output / "summary_by_profile.csv", summary_rows)
    write_csv(output / "duplicate_groups.csv", all_duplicate_rows)

    manifest = {
        "builder": "build_x_persona_datasets.py",
        "input": str(source),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "principles": {
            "canonical_sft_is_authentic_only": True,
            "synthetic_prompts_in_canonical_sft": False,
            "separate_model_per_profile": True,
            "content_category_filtering": False,
            "transport_artifacts_normalized": True,
            "media_only_examples_quarantined": True,
            "conversation_grouped_chronological_splits": True,
            "exact_pair_deduplication": True,
            "loose_duplicate_split_grouping": True,
        },
        "counts": {
            "authored_posts": len(authored),
            "cached_parent_posts": len(parents),
            "canonical_nodes": len(nodes),
            "profiles": len(profiles),
            "sft_examples": sum(int(row.get("sft_deduplicated", 0)) for row in summary_rows),
            "core_sft_examples": sum(int(row.get("sft_core", 0)) for row in summary_rows),
            "exact_duplicate_examples_removed": sum(
                int(row.get("sft_exact_duplicates_removed", 0)) for row in summary_rows
            ),
        },
        "parameters": {
            "max_context_posts": args.max_context_posts,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
        },
        "validation": validation_results,
    }
    (output / "manifest.json").write_text(jd(manifest, pretty=True), encoding="utf-8")

    readme = f"""# Authentic per-profile X/Twitter fine-tuning datasets

Built from `{source.name}` with **no synthetic prompts in canonical SFT**.

## Recommended files

For each profile:

- `profiles/<name>/sft/portable/core/train.jsonl`
- `profiles/<name>/sft/portable/core/validation.jsonl`
- `profiles/<name>/sft/portable/core/test.jsonl`

`portable` preserves conversational routing while replacing real handles with role tokens
such as `<SELF>`, `<PARENT>`, and `<USER_2>`. `verbatim` keeps observed handles.

`core` excludes examples with attached media in the prompt or response. `extended` adds
text-bearing posts that also carried media. Media-only inputs and outputs are never
pretended to be text conversations.

Each SFT row uses generic chat messages and ends in the observed profile response:

```json
{{"id":"reply:...","profile":"...","task":"external_reply_oneshot","messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}],"metadata":{{...}}}}
```

## Authentic data kept outside canonical SFT

- `auxiliary/self_thread_continuations.jsonl`: observed self-reply chains; kept separate
  because mapping a person's own previous post to a `user` turn changes the task.
- `auxiliary/authored_style_corpus.jsonl`: observed authored text for optional continued
  pretraining/style adaptation. It has no invented user prompt.
- `queues/prompt_generation.jsonl`: standalone posts, unresolved replies, and unresolved
  quotes that may later receive locally generated candidate prompts.
- `queues/multimodal_or_media_dependent.jsonl`: examples requiring image/video context.
- `queues/low_information.jsonl`: mention-only, link-only, or otherwise non-defensible text.
- `queues/unintelligible_or_corrupt.jsonl`: obvious encoding corruption only.

Generated prompt candidates should remain supplemental and should never be merged into
canonical SFT without review. Studio does not include a prompt-generation command for
these queues. Portable handles do not anonymize post text or source metadata.

## Deduplication and leakage controls

- one canonical node per post ID
- exact normalized context-response duplicates collapsed, with source IDs retained
- same-conversation and loose-near-duplicate examples forced into one split
- chronological 80/10/10 component-grouped splits
- post IDs and conversation IDs validated not to cross train/validation/test

See `summary_by_profile.csv`, `duplicate_groups.csv`, and `manifest.json`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    # Final parse audit over every JSONL file.
    jsonl_files = list(output.rglob("*.jsonl"))
    total_lines = 0
    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                json.loads(line)
                total_lines += 1
    print(f"Built {len(profiles)} profile datasets in {output}")
    print(f"Authentic deduplicated SFT examples: {manifest['counts']['sft_examples']}")
    print(f"Core text-only SFT examples: {manifest['counts']['core_sft_examples']}")
    print(f"Validated JSONL records: {total_lines}")
    print(f"Summary: {output / 'summary_by_profile.csv'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="xscra project directory or share-safe ZIP")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument("--exports", help="External X exports directory")
    parser.add_argument("--max-context-posts", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_context_posts < 2:
        parser.error("--max-context-posts must be at least 2")
    if not (0 < args.train_ratio < 1):
        parser.error("--train-ratio must be between 0 and 1")
    if not (0 <= args.validation_ratio < 1):
        parser.error("--validation-ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        parser.error("train + validation ratios must be less than 1")
    return args


def main() -> None:
    try:
        build(parse_args())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
