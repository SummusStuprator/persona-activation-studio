#!/usr/bin/env python3
"""
Export selected X accounts, split their post types, and cache reply parents.

This uses the unofficial `twscrape` client with an X session that you own.
It only requests content visible to that logged-in account. It does not bypass
protected-account access, blocks, suspensions, or other access controls.

Python: 3.11+
"""

from __future__ import annotations

import os

# Use twscrape's more browser-like HTTP backend and disable optional telemetry.
os.environ.setdefault("TWS_HTTP_BACKEND", "curl")
os.environ.setdefault("TWS_TELEMETRY", "0")
os.environ.setdefault("TWS_RAISE_WHEN_NO_ACCOUNT", "0")

import argparse
import asyncio
import csv
import getpass
import json
import random
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from twscrape import API
from twscrape.logger import set_log_level


DEFAULT_HANDLES: list[str] = []

TWITTER_EARLIEST_DATE = date(2006, 3, 21)
CSV_COLUMNS = [
    "id",
    "username",
    "display_name",
    "date",
    "kind",
    "text",
    "url",
    "lang",
    "conversation_id",
    "in_reply_to_tweet_id",
    "in_reply_to_username",
    "quoted_tweet_id",
    "quoted_tweet_url",
    "reply_count",
    "repost_count",
    "like_count",
    "quote_count",
    "bookmark_count",
    "view_count",
    "possibly_sensitive",
    "hashtags",
    "cashtags",
    "mentioned_users",
    "links",
    "photos",
    "videos",
    "animated_gifs",
]

PARENT_COLUMNS = [f"parent_{column}" for column in CSV_COLUMNS]
REPLY_CONTEXT_COLUMNS = CSV_COLUMNS + ["parent_status"] + PARENT_COLUMNS
UNAVAILABLE_COLUMNS = [
    "parent_tweet_id",
    "status",
    "error",
    "last_attempt",
]


def read_windows_clipboard() -> str:
    """Read text the user explicitly copied to the Windows clipboard."""
    commands = [
        ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
        ["pwsh.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
    ]
    errors: list[str] = []

    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            value = result.stdout.strip()
            if value:
                return value
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            errors.append(f"{command[0]}: {exc}")

    detail = "; ".join(errors) if errors else "clipboard was empty"
    raise RuntimeError(f"Could not read the Windows clipboard: {detail}")


def extract_x_cookie_pair(cookie_text: str) -> tuple[str, str]:
    """
    Parse auth_token and ct0 from a Cookie request header or cookie string.

    The user must explicitly copy the header from Chrome DevTools. This
    function does not read or decrypt Chrome's cookie database.
    """
    values: dict[str, str] = {}
    for name in ("auth_token", "ct0"):
        match = re.search(
            rf"(?i)(?:^|[;\s]){re.escape(name)}=([^;\r\n]+)",
            cookie_text,
        )
        if match:
            values[name] = match.group(1).strip()

    missing = [name for name in ("auth_token", "ct0") if not values.get(name)]
    if missing:
        raise ValueError(
            "Clipboard does not contain both auth_token and ct0. "
            "In Chrome DevTools, open Network, select an x.com request, "
            "copy the complete Cookie request header, then retry."
        )

    return values["auth_token"], values["ct0"]


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalized_handle(handle: str) -> str:
    handle = handle.strip()
    return handle[1:] if handle.startswith("@") else handle


def window_key(handle: str, start: date, end: date) -> str:
    return f"{handle.casefold()}:{start.isoformat()}:{end.isoformat()}"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return default


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def append_error(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().isoformat()
    with (output_dir / "errors.log").open("a", encoding="utf-8") as file:
        file.write(f"[{stamp}] {message}\n")


def jsonl_path(output_dir: Path, handle: str) -> Path:
    return output_dir / "jsonl" / f"{handle}.jsonl"


def csv_path(output_dir: Path, handle: str) -> Path:
    return output_dir / "csv" / f"{handle}.csv"


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                item = json.loads(line)
                tweet_id = str(item.get("id", ""))
                if tweet_id:
                    ids.add(tweet_id)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed JSONL line "
                    f"{path}:{line_number}",
                    file=sys.stderr,
                )
    return ids


def append_records(
    output_dir: Path,
    handle: str,
    records: Iterable[dict[str, Any]],
    known_ids: set[str],
) -> int:
    path = jsonl_path(output_dir, handle)
    path.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            tweet_id = str(record["id"])
            if tweet_id in known_ids:
                continue
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            known_ids.add(tweet_id)
            added += 1
    return added


def best_video_url(video: Any) -> str | None:
    variants = list(getattr(video, "variants", []) or [])
    if not variants:
        return None
    variants.sort(key=lambda item: int(getattr(item, "bitrate", 0) or 0))
    return getattr(variants[-1], "url", None)


def tweet_to_any_record(tweet: Any) -> dict[str, Any] | None:
    """Convert any parsed twscrape Tweet into the export schema."""
    author = getattr(tweet, "user", None)
    username = getattr(author, "username", "")
    if not username:
        return None

    created = getattr(tweet, "date", None)
    if created is None:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created = created.astimezone(timezone.utc)

    reply_to_id = getattr(tweet, "inReplyToTweetIdStr", None)
    if reply_to_id is None:
        raw_reply_id = getattr(tweet, "inReplyToTweetId", None)
        reply_to_id = str(raw_reply_id) if raw_reply_id is not None else None

    quoted = getattr(tweet, "quotedTweet", None)
    retweeted = getattr(tweet, "retweetedTweet", None)
    if retweeted is not None:
        kind = "repost"
    elif reply_to_id:
        kind = "reply"
    elif quoted is not None:
        kind = "quote"
    else:
        kind = "tweet"

    media = getattr(tweet, "media", None)
    photos = [
        getattr(item, "url", "")
        for item in (getattr(media, "photos", []) or [])
        if getattr(item, "url", "")
    ]
    videos = [
        url
        for item in (getattr(media, "videos", []) or [])
        if (url := best_video_url(item))
    ]
    animated_gifs = [
        getattr(item, "videoUrl", "")
        for item in (getattr(media, "animated", []) or [])
        if getattr(item, "videoUrl", "")
    ]

    reply_user = getattr(tweet, "inReplyToUser", None)
    mentioned = [
        getattr(item, "username", "")
        for item in (getattr(tweet, "mentionedUsers", []) or [])
        if getattr(item, "username", "")
    ]
    links = [
        getattr(item, "url", "")
        for item in (getattr(tweet, "links", []) or [])
        if getattr(item, "url", "")
    ]

    return {
        "id": str(getattr(tweet, "id_str", getattr(tweet, "id", ""))),
        "username": username,
        "display_name": getattr(author, "displayname", ""),
        "date": created.isoformat(),
        "kind": kind,
        "text": getattr(tweet, "rawContent", ""),
        "url": getattr(tweet, "url", ""),
        "lang": getattr(tweet, "lang", ""),
        "conversation_id": str(
            getattr(
                tweet,
                "conversationIdStr",
                getattr(tweet, "conversationId", ""),
            )
        ),
        "in_reply_to_tweet_id": reply_to_id,
        "in_reply_to_username": (
            getattr(reply_user, "username", None)
            or getattr(tweet, "inReplyToScreenName", None)
        ),
        "quoted_tweet_id": (
            str(getattr(quoted, "id_str", getattr(quoted, "id", "")))
            if quoted is not None
            else None
        ),
        "quoted_tweet_url": (
            getattr(quoted, "url", None) if quoted is not None else None
        ),
        "reply_count": getattr(tweet, "replyCount", 0),
        "repost_count": getattr(tweet, "retweetCount", 0),
        "like_count": getattr(tweet, "likeCount", 0),
        "quote_count": getattr(tweet, "quoteCount", 0),
        "bookmark_count": getattr(tweet, "bookmarkedCount", 0),
        "view_count": getattr(tweet, "viewCount", None),
        "possibly_sensitive": getattr(tweet, "possibly_sensitive", None),
        "hashtags": list(getattr(tweet, "hashtags", []) or []),
        "cashtags": list(getattr(tweet, "cashtags", []) or []),
        "mentioned_users": mentioned,
        "links": links,
        "photos": photos,
        "videos": videos,
        "animated_gifs": animated_gifs,
    }


def tweet_to_record(tweet: Any, requested_handle: str) -> dict[str, Any] | None:
    """Convert a target account's authored post, excluding native reposts."""
    record = tweet_to_any_record(tweet)
    if record is None:
        return None
    if record["username"].casefold() != requested_handle.casefold():
        return None
    if record["kind"] == "repost":
        return None
    return record

def record_in_window(record: dict[str, Any], start: date, end: date) -> bool:
    try:
        created = datetime.fromisoformat(record["date"]).date()
    except (KeyError, TypeError, ValueError):
        return False
    return start <= created < end


async def collect_generator(
    generator: Any,
    handle: str,
    timeout_seconds: int,
    start: date | None = None,
    end: date | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    async def consume() -> None:
        async for tweet in generator:
            record = tweet_to_record(tweet, handle)
            if record is None:
                continue
            if start is not None and end is not None:
                if not record_in_window(record, start, end):
                    continue
            records.append(record)

    async with asyncio.timeout(timeout_seconds):
        await consume()

    # Search can occasionally repeat pages. Deduplicate locally before writing.
    return list({record["id"]: record for record in records}.values())


async def scrape_recent_timeline(
    api: API,
    handle: str,
    user_id: int,
    output_dir: Path,
    known_ids: set[str],
    limit: int,
    timeout_seconds: int,
) -> tuple[int, bool]:
    print(f"  recent timeline: @{handle}")
    try:
        generator = api.user_tweets_and_replies(user_id, limit=limit)
        records = await collect_generator(
            generator=generator,
            handle=handle,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        append_error(output_dir, f"@{handle}: recent timeline timed out")
        print("    timed out; rerun later to retry")
        return 0, False
    except Exception as exc:
        append_error(
            output_dir,
            f"@{handle}: recent timeline failed: {type(exc).__name__}: {exc}",
        )
        print(f"    failed: {type(exc).__name__}: {exc}")
        return 0, False

    added = append_records(output_dir, handle, records, known_ids)
    print(f"    received {len(records)}, added {added}")
    return added, True


async def scrape_search_window(
    api: API,
    handle: str,
    start: date,
    end: date,
    output_dir: Path,
    known_ids: set[str],
    state: dict[str, Any],
    state_path: Path,
    search_limit: int,
    split_threshold: int,
    timeout_seconds: int,
    min_delay: float,
    max_delay: float,
) -> int:
    key = window_key(handle, start, end)
    completed = set(state.setdefault("completed_windows", []))
    if key in completed:
        return 0

    query = f"from:{handle} since:{start.isoformat()} until:{end.isoformat()}"
    print(f"  history: {start} → {end}  @{handle}")

    try:
        generator = api.search(query, limit=search_limit)
        records = await collect_generator(
            generator=generator,
            handle=handle,
            timeout_seconds=timeout_seconds,
            start=start,
            end=end,
        )
    except TimeoutError:
        append_error(
            output_dir,
            f"@{handle}: search timed out for {start} to {end}",
        )
        print("    timed out; window left incomplete")
        return 0
    except Exception as exc:
        append_error(
            output_dir,
            f"@{handle}: search failed for {start} to {end}: "
            f"{type(exc).__name__}: {exc}",
        )
        print(f"    failed: {type(exc).__name__}: {exc}")
        return 0

    added = append_records(output_dir, handle, records, known_ids)
    print(f"    received {len(records)}, added {added}")

    span_days = (end - start).days
    total_added = added

    # A nearly-full search window may have been capped. Split it until the
    # windows are one day wide, preserving completed state for resume support.
    if len(records) >= split_threshold and span_days > 1:
        midpoint = start + timedelta(days=max(1, span_days // 2))
        print(
            f"    dense window ({len(records)} results); "
            f"splitting at {midpoint}"
        )
        total_added += await scrape_search_window(
            api,
            handle,
            start,
            midpoint,
            output_dir,
            known_ids,
            state,
            state_path,
            search_limit,
            split_threshold,
            timeout_seconds,
            min_delay,
            max_delay,
        )
        total_added += await scrape_search_window(
            api,
            handle,
            midpoint,
            end,
            output_dir,
            known_ids,
            state,
            state_path,
            search_limit,
            split_threshold,
            timeout_seconds,
            min_delay,
            max_delay,
        )
    elif len(records) >= split_threshold and span_days <= 1:
        append_error(
            output_dir,
            f"@{handle}: one-day window {start} may be capped "
            f"({len(records)} results)",
        )
        print("    warning: one-day window may still be capped")

    completed = set(state.setdefault("completed_windows", []))
    completed.add(key)
    state["completed_windows"] = sorted(completed)
    save_json_atomic(state_path, state)

    if max_delay > 0:
        await asyncio.sleep(random.uniform(min_delay, max_delay))

    return total_added


def iter_windows(start: date, end: date, days: int) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=days), end)
        yield cursor, next_cursor
        cursor = next_cursor


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv_with_columns(
    path: Path,
    records: list[dict[str, Any]],
    columns: list[str],
    sort_columns: tuple[str, ...] = ("date", "id"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records.sort(
        key=lambda item: tuple(str(item.get(column, "")) for column in sort_columns)
    )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {column: csv_value(record.get(column)) for column in columns}
            )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    write_csv_with_columns(path, records, CSV_COLUMNS)


def selected_handles(raw_handles: list[str] | None) -> list[str]:
    handles = [
        normalized_handle(handle)
        for handle in (raw_handles or DEFAULT_HANDLES)
        if normalized_handle(handle)
    ]
    handles=list(dict.fromkeys(handles))
    if not handles:
        raise ValueError("No X handles configured. Pass --handle @name (repeatable) or use the studio CLI.")
    return handles


def load_all_target_records(
    output_dir: Path,
    handles: list[str],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for handle in handles:
        for record in load_records(jsonl_path(output_dir, handle)):
            tweet_id = str(record.get("id", ""))
            if tweet_id:
                records[tweet_id] = record
    return list(records.values())


def rebuild_csv_exports(output_dir: Path, handles: list[str]) -> int:
    all_records: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[dict[str, Any]]] = {
        "tweet": [],
        "reply": [],
        "quote": [],
    }

    for handle in handles:
        records = load_records(jsonl_path(output_dir, handle))
        write_csv(csv_path(output_dir, handle), records)

        account_dir = output_dir / "by_account" / handle
        kind_filenames = {
            "tweet": "tweets.csv",
            "reply": "replies.csv",
            "quote": "quotes.csv",
        }
        for kind, filename in kind_filenames.items():
            kind_records = [
                record for record in records if record.get("kind") == kind
            ]
            write_csv(account_dir / filename, kind_records)
            by_kind[kind].extend(kind_records)

        for record in records:
            all_records[str(record["id"])] = record

    combined = list(all_records.values())
    write_csv(output_dir / "all_posts.csv", combined)

    for kind, records in by_kind.items():
        write_csv(output_dir / "by_kind" / f"{kind}s.csv", records)

    return len(combined)


def reply_context_dir(output_dir: Path) -> Path:
    return output_dir / "reply_context"


def reply_parent_jsonl_path(output_dir: Path) -> Path:
    return reply_context_dir(output_dir) / "parents.jsonl"


def reply_context_state_path(output_dir: Path) -> Path:
    return reply_context_dir(output_dir) / "state.json"


def append_jsonl_record(
    path: Path,
    record: dict[str, Any],
    known_ids: set[str],
) -> bool:
    tweet_id = str(record.get("id", ""))
    if not tweet_id or tweet_id in known_ids:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    known_ids.add(tweet_id)
    return True


def enrich_with_parent_context(
    record: dict[str, Any],
    parent_map: dict[str, dict[str, Any]],
    unavailable: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = {column: record.get(column) for column in CSV_COLUMNS}

    if record.get("kind") != "reply" or not record.get("in_reply_to_tweet_id"):
        row["parent_status"] = "not_applicable"
        return row

    parent_id = str(record.get("in_reply_to_tweet_id", ""))
    parent = parent_map.get(parent_id)

    if parent is not None:
        row["parent_status"] = "found"
        for column in CSV_COLUMNS:
            row[f"parent_{column}"] = parent.get(column)
    elif parent_id in unavailable:
        row["parent_status"] = unavailable[parent_id].get(
            "status",
            "unavailable",
        )
    else:
        row["parent_status"] = "not_fetched"

    return row


def calculate_reply_context_status(
    output_dir: Path,
    handles: list[str],
) -> dict[str, Any]:
    target_records = load_all_target_records(output_dir, handles)
    target_ids = {
        str(record["id"])
        for record in target_records
        if record.get("id")
    }
    cached_records = load_records(reply_parent_jsonl_path(output_dir))
    cached_ids = {
        str(record["id"])
        for record in cached_records
        if record.get("id")
    }
    state = load_json(
        reply_context_state_path(output_dir),
        {"completed": [], "unavailable": {}, "version": 1},
    )
    unavailable: dict[str, dict[str, Any]] = state.get("unavailable", {})

    replies = [
        record
        for record in target_records
        if record.get("kind") == "reply"
        and record.get("in_reply_to_tweet_id")
    ]
    parent_ids = {
        str(record["in_reply_to_tweet_id"])
        for record in replies
    }

    available_ids = target_ids | cached_ids
    found_parent_ids = parent_ids & available_ids
    unavailable_ids = parent_ids & set(unavailable)
    pending_ids = parent_ids - found_parent_ids - unavailable_ids

    replies_with_found_parent = sum(
        str(record.get("in_reply_to_tweet_id", "")) in available_ids
        for record in replies
    )
    replies_unavailable = sum(
        str(record.get("in_reply_to_tweet_id", "")) in unavailable
        for record in replies
    )

    return {
        "target_records": len(target_records),
        "replies": len(replies),
        "unique_parent_ids": len(parent_ids),
        "parents_found": len(found_parent_ids),
        "parents_inside_target_dataset": len(parent_ids & target_ids),
        "parents_cached_separately": len(parent_ids & cached_ids),
        "parents_unavailable": len(unavailable_ids),
        "parents_pending": len(pending_ids),
        "replies_with_parent": replies_with_found_parent,
        "replies_unavailable": replies_unavailable,
        "replies_pending": len(replies) - replies_with_found_parent - replies_unavailable,
        "coverage_percent": (
            round((replies_with_found_parent / len(replies)) * 100, 2)
            if replies
            else 100.0
        ),
    }


def build_reply_context_exports(
    output_dir: Path,
    handles: list[str],
) -> dict[str, int | float]:
    target_records = load_all_target_records(output_dir, handles)
    fetched_parents = load_records(reply_parent_jsonl_path(output_dir))
    state = load_json(
        reply_context_state_path(output_dir),
        {"completed": [], "unavailable": {}, "version": 1},
    )
    unavailable: dict[str, dict[str, Any]] = state.get("unavailable", {})

    # A reply can point to another post already present in the target dataset.
    parent_map: dict[str, dict[str, Any]] = {
        str(record["id"]): record
        for record in target_records
        if record.get("id")
    }
    for record in fetched_parents:
        if record.get("id"):
            parent_map[str(record["id"])] = record

    enriched_all: list[dict[str, Any]] = []
    joined_replies: list[dict[str, Any]] = []
    per_handle_all: dict[str, list[dict[str, Any]]] = {
        handle.casefold(): [] for handle in handles
    }
    per_handle_replies: dict[str, list[dict[str, Any]]] = {
        handle.casefold(): [] for handle in handles
    }

    for record in target_records:
        row = enrich_with_parent_context(record, parent_map, unavailable)
        enriched_all.append(row)

        username = str(record.get("username", "")).casefold()
        per_handle_all.setdefault(username, []).append(row)

        if record.get("kind") == "reply":
            joined_replies.append(row)
            per_handle_replies.setdefault(username, []).append(row)

    context_dir = reply_context_dir(output_dir)
    write_csv(context_dir / "parents.csv", fetched_parents)

    write_csv_with_columns(
        output_dir / "all_posts_with_context.csv",
        enriched_all,
        REPLY_CONTEXT_COLUMNS,
    )
    write_csv_with_columns(
        output_dir / "replies_with_context.csv",
        joined_replies,
        REPLY_CONTEXT_COLUMNS,
    )

    for handle in handles:
        account_dir = output_dir / "by_account" / handle
        write_csv_with_columns(
            account_dir / "all_posts_with_context.csv",
            per_handle_all.get(handle.casefold(), []),
            REPLY_CONTEXT_COLUMNS,
        )
        write_csv_with_columns(
            account_dir / "replies_with_context.csv",
            per_handle_replies.get(handle.casefold(), []),
            REPLY_CONTEXT_COLUMNS,
        )

    unavailable_rows = [
        {
            "parent_tweet_id": parent_id,
            "status": details.get("status", "unavailable"),
            "error": details.get("error", ""),
            "last_attempt": details.get("last_attempt", ""),
        }
        for parent_id, details in unavailable.items()
    ]
    write_csv_with_columns(
        context_dir / "unavailable.csv",
        unavailable_rows,
        UNAVAILABLE_COLUMNS,
        sort_columns=("parent_tweet_id",),
    )

    status = calculate_reply_context_status(output_dir, handles)
    return {
        "replies": int(status["replies"]),
        "found": int(status["replies_with_parent"]),
        "unavailable": int(status["replies_unavailable"]),
        "not_fetched": int(status["replies_pending"]),
        "cached_parents": len(fetched_parents),
        "coverage_percent": float(status["coverage_percent"]),
    }


async def fetch_reply_context(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    handles = selected_handles(args.handle)
    target_records = load_all_target_records(output_dir, handles)

    if not target_records:
        print(
            "No scraped target records found. Run the scrape command first.",
            file=sys.stderr,
        )
        return 2

    api = API(str(Path(args.db).expanduser()), raise_when_no_account=False)
    accounts = await api.pool.get_all()
    if not any(account.active for account in accounts):
        print(
            "No active X session found. Run setup first.",
            file=sys.stderr,
        )
        return 2

    context_path = reply_parent_jsonl_path(output_dir)
    state_path = reply_context_state_path(output_dir)
    state = load_json(
        state_path,
        {"completed": [], "unavailable": {}, "version": 1},
    )
    completed = set(str(value) for value in state.setdefault("completed", []))
    unavailable: dict[str, dict[str, Any]] = state.setdefault("unavailable", {})

    target_map = {
        str(record["id"]): record
        for record in target_records
        if record.get("id")
    }
    cached_records = load_records(context_path)
    cached_ids = {
        str(record["id"])
        for record in cached_records
        if record.get("id")
    }

    parent_ids = {
        str(record["in_reply_to_tweet_id"])
        for record in target_records
        if record.get("kind") == "reply"
        and record.get("in_reply_to_tweet_id")
    }

    candidates = sorted(
        parent_id
        for parent_id in parent_ids
        if parent_id not in target_map
        and parent_id not in cached_ids
        and (
            args.retry_unavailable
            or parent_id not in unavailable
        )
    )

    print(f"Replies reference {len(parent_ids)} unique direct parent posts.")
    print(f"Already available inside target dataset: {len(parent_ids & target_map.keys())}")
    print(f"Already cached separately: {len(parent_ids & cached_ids)}")
    print(f"Parents queued for lookup: {len(candidates)}")

    if args.export_only:
        stats = build_reply_context_exports(output_dir, handles)
        print(
            "Rebuilt reply-context exports: "
            f"{stats['found']}/{stats['replies']} replies have parent content."
        )
        return 0

    deadline = (
        time.monotonic() + args.max_runtime
        if args.max_runtime > 0
        else None
    )
    added = 0
    attempted = 0

    for index, parent_id in enumerate(candidates, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            print("Context runtime limit reached; progress is resumable.")
            break

        attempted += 1
        print(f"[{index}/{len(candidates)}] parent {parent_id}")

        try:
            parent_tweet = await asyncio.wait_for(
                api.tweet_details(int(parent_id)),
                timeout=args.timeout,
            )
        except TimeoutError:
            append_error(
                output_dir,
                f"parent {parent_id}: tweet_details timed out",
            )
            print("  timed out; left pending")
            continue
        except Exception as exc:
            append_error(
                output_dir,
                f"parent {parent_id}: {type(exc).__name__}: {exc}",
            )
            print(f"  failed: {type(exc).__name__}: {exc}")
            continue

        now = utc_now().isoformat()
        if parent_tweet is None:
            unavailable[parent_id] = {
                "status": "deleted_or_not_visible",
                "error": "tweet_details returned no visible post",
                "last_attempt": now,
            }
            completed.add(parent_id)
            print("  unavailable, deleted, or not visible")
        else:
            record = tweet_to_any_record(parent_tweet)
            if record is None:
                unavailable[parent_id] = {
                    "status": "parse_failed",
                    "error": "tweet_details could not be converted",
                    "last_attempt": now,
                }
                completed.add(parent_id)
                print("  response could not be parsed")
            else:
                if append_jsonl_record(context_path, record, cached_ids):
                    added += 1
                    print(f"  cached @{record.get('username')}: {record.get('text', '')[:80]}")
                else:
                    print("  already cached")
                completed.add(parent_id)
                unavailable.pop(parent_id, None)

        state["completed"] = sorted(completed)
        state["unavailable"] = unavailable
        save_json_atomic(state_path, state)

        if args.max_delay > 0:
            await asyncio.sleep(
                random.uniform(args.min_delay, args.max_delay)
            )

    stats = build_reply_context_exports(output_dir, handles)
    print("")
    print(f"Context pass attempted {attempted}; cached {added} new parents.")
    print(
        f"Joined export: {stats['found']}/{stats['replies']} replies "
        "currently have parent content."
    )
    print(f"  {output_dir / 'replies_with_context.csv'}")
    print(f"  {output_dir / 'all_posts_with_context.csv'}")
    print("Rerun the context command to resume pending parent lookups.")
    return 0


async def setup_account(args: argparse.Namespace) -> int:
    db_path = str(Path(args.db).expanduser())
    api = API(db_path, raise_when_no_account=False)

    if args.replace:
        await api.pool.delete_accounts(args.alias)

    if args.from_clipboard:
        try:
            auth_token, ct0 = extract_x_cookie_pair(read_windows_clipboard())
        except (RuntimeError, ValueError) as exc:
            print(f"Clipboard setup failed: {exc}", file=sys.stderr)
            return 2
        print("Found auth_token and ct0 in the copied Cookie header.")
    else:
        auth_token = getpass.getpass("X auth_token (hidden): ").strip()
        ct0 = getpass.getpass("X ct0 (hidden): ").strip()

    if not auth_token or not ct0:
        print("Both auth_token and ct0 are required.", file=sys.stderr)
        return 2

    cookies = f"auth_token={auth_token}; ct0={ct0}"
    await api.pool.add_account_cookies(args.alias, cookies)
    account = await api.pool.get_account(args.alias)

    if account is None or not account.active:
        print("Account was not activated. Check the cookie values.", file=sys.stderr)
        return 2

    print(f"Saved an active local session as {args.alias!r} in {db_path}.")
    print("Keep that database private; it contains login cookies.")
    return 0


async def show_accounts(args: argparse.Namespace) -> int:
    api = API(str(Path(args.db).expanduser()), raise_when_no_account=False)
    items = await api.pool.accounts_info()
    if not items:
        print("No accounts configured.")
        return 0

    for item in items:
        print(
            f"{item['username']}: active={item['active']} "
            f"requests={item['total_req']} error={item['error_msg']}"
        )
    return 0


def local_scrape_status(
    output_dir: Path,
    handles: list[str],
) -> dict[str, Any]:
    records = load_all_target_records(output_dir, handles)
    scrape_state = load_json(
        output_dir / "state.json",
        {"completed_windows": [], "completed_recent": [], "version": 1},
    )

    totals_by_kind: dict[str, int] = {"tweet": 0, "reply": 0, "quote": 0}
    by_account: dict[str, dict[str, Any]] = {}

    for handle in handles:
        account_records = load_records(jsonl_path(output_dir, handle))
        kinds = {"tweet": 0, "reply": 0, "quote": 0}
        dates: list[str] = []

        for record in account_records:
            kind = str(record.get("kind", ""))
            if kind in kinds:
                kinds[kind] += 1
                totals_by_kind[kind] += 1
            if record.get("date"):
                dates.append(str(record["date"]))

        by_account[handle] = {
            "total": len(account_records),
            **kinds,
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None,
            "recent_timeline_completed": (
                handle.casefold()
                in {
                    str(value).casefold()
                    for value in scrape_state.get("completed_recent", [])
                }
            ),
        }

    context = calculate_reply_context_status(output_dir, handles)

    return {
        "output_dir": str(output_dir),
        "total_unique_posts": len(
            {
                str(record.get("id"))
                for record in records
                if record.get("id")
            }
        ),
        "by_kind": totals_by_kind,
        "completed_recent_timelines": len(
            scrape_state.get("completed_recent", [])
        ),
        "target_accounts": len(handles),
        "completed_history_windows": len(
            scrape_state.get("completed_windows", [])
        ),
        "by_account": by_account,
        "reply_context": context,
    }


async def show_status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    handles = selected_handles(args.handle)
    status = local_scrape_status(output_dir, handles)

    account_items: list[dict[str, Any]] = []
    db_path = Path(args.db).expanduser()
    if db_path.exists():
        try:
            api = API(str(db_path), raise_when_no_account=False)
            account_items = await api.pool.accounts_info()
        except Exception as exc:
            status["account_database_error"] = f"{type(exc).__name__}: {exc}"

    status["x_sessions"] = account_items

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        return 0

    print("=== Local scrape status ===")
    print(f"Unique target posts: {status['total_unique_posts']}")
    print(
        "Kinds: "
        f"tweets={status['by_kind']['tweet']}, "
        f"replies={status['by_kind']['reply']}, "
        f"quotes={status['by_kind']['quote']}"
    )
    print(
        f"Recent timelines completed: "
        f"{status['completed_recent_timelines']}/{status['target_accounts']}"
    )
    print(
        f"Historical windows completed: "
        f"{status['completed_history_windows']}"
    )

    print("")
    print("=== Reply-parent backfill ===")
    context = status["reply_context"]
    print(f"Replies with a direct parent ID: {context['replies']}")
    print(f"Unique direct parents referenced: {context['unique_parent_ids']}")
    print(
        f"Parents available: {context['parents_found']} "
        f"({context['parents_inside_target_dataset']} already among targets, "
        f"{context['parents_cached_separately']} separately cached)"
    )
    print(f"Parents unavailable/deleted: {context['parents_unavailable']}")
    print(f"Parents still pending: {context['parents_pending']}")
    print(
        f"Reply rows enriched: {context['replies_with_parent']}/"
        f"{context['replies']} ({context['coverage_percent']}%)"
    )

    print("")
    print("=== Per account ===")
    for handle in handles:
        item = status["by_account"][handle]
        recent_mark = "yes" if item["recent_timeline_completed"] else "no"
        print(
            f"@{handle}: total={item['total']} "
            f"tweets={item['tweet']} replies={item['reply']} "
            f"quotes={item['quote']} recent_done={recent_mark}"
        )

    if account_items:
        print("")
        print("=== X session ===")
        for item in account_items:
            print(
                f"{item['username']}: active={item['active']} "
                f"requests={item['total_req']} error={item['error_msg']}"
            )
    elif "account_database_error" in status:
        print("")
        print(f"Account DB error: {status['account_database_error']}")

    return 0


async def scrape(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = load_json(
        state_path,
        {"completed_windows": [], "completed_recent": [], "version": 1},
    )

    handles = selected_handles(args.handle)

    if args.min_delay < 0 or args.max_delay < args.min_delay:
        print("Delay values are invalid.", file=sys.stderr)
        return 2
    if args.split_threshold > args.search_limit:
        print(
            "--split-threshold must not exceed --search-limit.",
            file=sys.stderr,
        )
        return 2

    api = API(str(Path(args.db).expanduser()), raise_when_no_account=False)
    accounts = await api.pool.get_all()
    if not any(account.active for account in accounts):
        print(
            "No active X session found. Run the setup command first.",
            file=sys.stderr,
        )
        return 2

    known_ids = {
        handle: load_existing_ids(jsonl_path(output_dir, handle))
        for handle in handles
    }
    total_added = 0

    for index, handle in enumerate(handles, start=1):
        print(f"\n[{index}/{len(handles)}] @{handle}")

        try:
            user = await asyncio.wait_for(
                api.user_by_login(handle),
                timeout=args.timeout,
            )
        except Exception as exc:
            append_error(
                output_dir,
                f"@{handle}: profile lookup failed: {type(exc).__name__}: {exc}",
            )
            print(f"  profile lookup failed: {type(exc).__name__}: {exc}")
            continue

        if user is None:
            append_error(output_dir, f"@{handle}: profile not found or not visible")
            print("  profile not found or not visible to this account")
            continue

        recent_key = handle.casefold()
        recent_done = set(state.setdefault("completed_recent", []))
        if recent_key not in recent_done or args.refresh_recent:
            added, success = await scrape_recent_timeline(
                api=api,
                handle=handle,
                user_id=user.id,
                output_dir=output_dir,
                known_ids=known_ids[handle],
                limit=args.timeline_limit,
                timeout_seconds=args.timeout,
            )
            total_added += added
            if success:
                recent_done.add(recent_key)
                state["completed_recent"] = sorted(recent_done)
                save_json_atomic(state_path, state)
        else:
            print("  recent timeline already completed; use --refresh-recent to rerun")

        if args.recent_only:
            continue

        account_created = getattr(user, "created", None)
        if isinstance(account_created, datetime):
            default_start = max(account_created.date(), TWITTER_EARLIEST_DATE)
        else:
            default_start = TWITTER_EARLIEST_DATE

        start = args.start or default_start
        end = args.end or (utc_now().date() + timedelta(days=1))
        if start >= end:
            print(f"  skipped: start {start} is not before end {end}")
            continue

        for window_start, window_end in iter_windows(
            start,
            end,
            args.window_days,
        ):
            total_added += await scrape_search_window(
                api=api,
                handle=handle,
                start=window_start,
                end=window_end,
                output_dir=output_dir,
                known_ids=known_ids[handle],
                state=state,
                state_path=state_path,
                search_limit=args.search_limit,
                split_threshold=args.split_threshold,
                timeout_seconds=args.timeout,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
            )

    count = rebuild_csv_exports(output_dir, handles)
    print(f"\nDone. Added {total_added} new records.")
    print(f"Combined CSV contains {count} unique records:")
    print(f"  {output_dir / 'all_posts.csv'}")
    print("Rerun the same command to resume incomplete windows.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected X accounts, backfill direct reply parents, and "
            "build separated/context-enriched CSV files."
        )
    )
    parser.add_argument(
        "--db",
        default="x_accounts.db",
        help="Local twscrape account database (default: x_accounts.db)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable twscrape debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Store your X session cookies locally",
    )
    setup_parser.add_argument(
        "--alias",
        default="my_x_account",
        help="Local name for the session",
    )
    setup_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing session with the same alias",
    )
    setup_parser.add_argument(
        "--from-clipboard",
        action="store_true",
        help=(
            "Parse auth_token and ct0 from a Cookie request header that you "
            "explicitly copied to the Windows clipboard"
        ),
    )

    subparsers.add_parser("accounts", help="Show locally configured sessions")

    status_parser = subparsers.add_parser(
        "status",
        help="Show scrape progress and reply-parent backfill coverage",
    )
    status_parser.add_argument(
        "--handle",
        action="append",
        help="Handle to inspect; repeat for multiple. Requires --handle unless supplied by the studio CLI.",
    )
    status_parser.add_argument(
        "--output-dir",
        default="x_exports",
        help="Output directory (default: x_exports)",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    scrape_parser = subparsers.add_parser("scrape", help="Run or resume export")
    scrape_parser.add_argument(
        "--handle",
        action="append",
        help="Handle to export; repeat for multiple. Requires --handle unless supplied by the studio CLI.",
    )
    scrape_parser.add_argument(
        "--output-dir",
        default="x_exports",
        help="Output directory (default: x_exports)",
    )
    scrape_parser.add_argument(
        "--recent-only",
        action="store_true",
        help="Only use the recent profile timeline (roughly the newest 3,200)",
    )
    scrape_parser.add_argument(
        "--refresh-recent",
        action="store_true",
        help="Rerun recent timelines even if state marks them complete",
    )
    scrape_parser.add_argument(
        "--start",
        type=parse_iso_date,
        help="History start date, inclusive (YYYY-MM-DD)",
    )
    scrape_parser.add_argument(
        "--end",
        type=parse_iso_date,
        help="History end date, exclusive (YYYY-MM-DD; default: tomorrow)",
    )
    scrape_parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Initial historical search window size (default: 90)",
    )
    scrape_parser.add_argument(
        "--search-limit",
        type=int,
        default=1000,
        help="Maximum search results requested per window (default: 1000)",
    )
    scrape_parser.add_argument(
        "--split-threshold",
        type=int,
        default=850,
        help="Split windows returning at least this many records (default: 850)",
    )
    scrape_parser.add_argument(
        "--timeline-limit",
        type=int,
        default=5000,
        help="Requested recent-timeline limit; X usually caps near 3200",
    )
    scrape_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per profile/window in seconds (default: 300)",
    )
    scrape_parser.add_argument(
        "--min-delay",
        type=float,
        default=2.0,
        help="Minimum pause between successful search windows",
    )
    scrape_parser.add_argument(
        "--max-delay",
        type=float,
        default=5.0,
        help="Maximum pause between successful search windows",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Rebuild CSV files from saved JSONL without contacting X",
    )
    export_parser.add_argument(
        "--handle",
        action="append",
        help="Handle to export; repeat for multiple. Requires --handle unless supplied by the studio CLI.",
    )
    export_parser.add_argument(
        "--output-dir",
        default="x_exports",
        help="Output directory (default: x_exports)",
    )

    context_parser = subparsers.add_parser(
        "context",
        help="Fetch direct parent posts for saved replies and build joined CSVs",
    )
    context_parser.add_argument(
        "--handle",
        action="append",
        help="Handle to process; repeat for multiple. Requires --handle unless supplied by the studio CLI.",
    )
    context_parser.add_argument(
        "--output-dir",
        default="x_exports",
        help="Output directory containing the saved JSONL files",
    )
    context_parser.add_argument(
        "--max-runtime",
        type=int,
        default=21600,
        help="Maximum run time in seconds; default is six hours",
    )
    context_parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout for one parent lookup, including rate-limit waiting",
    )
    context_parser.add_argument(
        "--min-delay",
        type=float,
        default=1.5,
        help="Minimum pause between successful parent lookups",
    )
    context_parser.add_argument(
        "--max-delay",
        type=float,
        default=3.0,
        help="Maximum pause between successful parent lookups",
    )
    context_parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="Retry parents previously marked deleted, unavailable, or unparseable",
    )
    context_parser.add_argument(
        "--export-only",
        action="store_true",
        help="Rebuild joined CSVs from the current cache without contacting X",
    )

    return parser


def export_existing(args: argparse.Namespace) -> int:
    """Rebuild CSV files from already-saved JSONL without contacting X."""
    output_dir = Path(args.output_dir).expanduser()
    handles = selected_handles(args.handle)
    count = rebuild_csv_exports(output_dir, handles)
    context_stats = build_reply_context_exports(output_dir, handles)
    print(f"Rebuilt CSV exports with {count} unique records:")
    print(f"  {output_dir / 'all_posts.csv'}")
    print(
        f"Reply context: {context_stats['found']}/"
        f"{context_stats['replies']} replies have parent content."
    )
    return 0


async def async_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    set_log_level("DEBUG" if args.verbose else "INFO")

    if args.command == "setup":
        return await setup_account(args)
    if args.command == "accounts":
        return await show_accounts(args)
    if args.command == "status":
        return await show_status(args)
    if args.command == "export":
        return export_existing(args)
    if args.command == "context":
        if args.max_runtime < 0:
            parser.error("--max-runtime cannot be negative")
        if args.timeout < 1:
            parser.error("--timeout must be at least 1")
        if args.min_delay < 0 or args.max_delay < args.min_delay:
            parser.error("Context delay values are invalid")
        return await fetch_reply_context(args)
    if args.command == "scrape":
        if args.window_days < 1:
            parser.error("--window-days must be at least 1")
        if args.search_limit < 1:
            parser.error("--search-limit must be at least 1")
        if args.split_threshold < 1:
            parser.error("--split-threshold must be at least 1")
        if args.timeout < 1:
            parser.error("--timeout must be at least 1")
        return await scrape(args)

    parser.error("Unknown command")
    return 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        print("\nStopped. Progress already written can be resumed.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
