"""Public dataset integration — download & parse LongMemEval, LoCoMo, etc.

Supports:
  - LongMemEval (arXiv:2410.10813) — HuggingFace: THU/LongMemEval
  - LoCoMo (arXiv:2402.17753) — HuggingFace: THU/LoCoMo
  - Auto-download + cache to ~/.qmemory-bench/datasets/
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DATASET_SOURCES = {
    "longmemeval": {
        "url": "https://huggingface.co/datasets/THU-KEG/LongMemEval/resolve/main/data",
        "files": ["all_queries.json", "events.json"],
        "description": "LongMemEval — 6-dimensional long-term memory evaluation (500 queries)",
        "paper": "arXiv:2410.10813",
    },
    "locomo": {
        "url": "https://huggingface.co/datasets/THU-KEG/LoCoMo/resolve/main",
        "files": ["test.json"],
        "description": "LoCoMo — long conversation memory benchmark",
        "paper": "arXiv:2402.17753",
    },
}


def _cache_dir() -> Path:
    d = Path.home() / ".qmemory-bench" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def download_public_dataset(name: str, *, force: bool = False) -> Path:
    """Download a public dataset to local cache.

    Returns the cache directory containing the downloaded files.
    """
    import httpx

    if name not in DATASET_SOURCES:
        raise ValueError(f"Unknown public dataset: {name}. Available: {list(DATASET_SOURCES.keys())}")

    source = DATASET_SOURCES[name]
    cache = _cache_dir() / name
    cache.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for filename in source["files"]:
            local_path = cache / filename
            if local_path.exists() and not force:
                logger.info(f"  Cached: {local_path}")
                continue

            url = f"{source['url']}/{filename}"
            logger.info(f"  Downloading: {url}")
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                logger.info(f"  Saved: {local_path} ({len(resp.content)} bytes)")
            except Exception as e:
                logger.warning(f"  Failed to download {url}: {e}")

    return cache


def parse_longmemeval(cache_dir: Path, *, scale: str = "quick") -> dict:
    """Parse downloaded LongMemEval data into our dataset format.

    Converts the original format into sessions + questions.
    Scale: quick=20q, standard=100q, full=all.
    """
    queries_path = cache_dir / "all_queries.json"
    events_path = cache_dir / "events.json"

    if not queries_path.exists():
        raise FileNotFoundError(f"LongMemEval queries not found at {queries_path}")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []

    # Category mapping from LongMemEval dimensions
    CATEGORY_MAP = {
        "single-session-user": "single-session-user",
        "single-session-assistant": "single-session-assistant",
        "single-session-preference": "single-session-preference",
        "knowledge-update": "knowledge-update",
        "temporal-reasoning": "temporal-reasoning",
        "multi-session": "multi-session",
    }

    # Convert events to sessions
    sessions = []
    for i, evt in enumerate(events[:50]):  # Cap at 50 sessions
        messages = []
        for turn in evt.get("turns", evt.get("dialogue", [])):
            role = turn.get("role", "user")
            content = turn.get("content", turn.get("text", ""))
            if content:
                messages.append({"role": role, "content": content})
        if messages:
            sessions.append({
                "id": evt.get("id", f"lme_sess_{i}"),
                "messages": messages,
                "metadata": {"source": "LongMemEval", "timestamp": evt.get("timestamp", "")},
            })

    # Convert queries to questions
    limits = {"quick": 20, "standard": 100, "full": 999999}
    limit = limits.get(scale, 20)

    questions = []
    for i, q in enumerate(queries[:limit]):
        cat = q.get("category", q.get("type", "general"))
        mapped_cat = CATEGORY_MAP.get(cat, cat)
        questions.append({
            "id": q.get("id", f"lme_q_{i}"),
            "query": q.get("query", q.get("question", "")),
            "expected": q.get("answer", q.get("expected", "")),
            "category": mapped_cat,
            "difficulty": q.get("difficulty", "standard"),
        })

    return {
        "name": "longmemeval",
        "description": f"LongMemEval public dataset ({len(questions)} questions, {len(sessions)} sessions)",
        "version": "public-1.0",
        "sessions": sessions,
        "questions": questions,
    }


def parse_locomo(cache_dir: Path, *, scale: str = "quick") -> dict:
    """Parse downloaded LoCoMo data into our dataset format."""
    test_path = cache_dir / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"LoCoMo test data not found at {test_path}")

    raw = json.loads(test_path.read_text(encoding="utf-8"))
    limits = {"quick": 15, "standard": 50, "full": 999999}
    limit = limits.get(scale, 15)

    sessions = []
    questions = []

    for i, item in enumerate(raw[:limit]):
        # Each item typically has dialogue + questions
        dialogue = item.get("dialogue", item.get("conversation", []))
        messages = []
        for turn in dialogue:
            role = turn.get("role", "user")
            content = turn.get("content", turn.get("text", ""))
            if content:
                messages.append({"role": role, "content": content})

        if messages:
            sessions.append({
                "id": f"locomo_sess_{i}",
                "messages": messages,
            })

        for j, q in enumerate(item.get("questions", [])):
            questions.append({
                "id": f"locomo_q_{i}_{j}",
                "query": q.get("question", q.get("query", "")),
                "expected": q.get("answer", q.get("expected", "")),
                "category": q.get("type", "recall-accuracy"),
                "difficulty": "standard",
            })

    return {
        "name": "locomo",
        "description": f"LoCoMo public dataset ({len(questions)} questions)",
        "version": "public-1.0",
        "sessions": sessions,
        "questions": questions[:limit],
    }


def list_cached_public() -> list[dict[str, str]]:
    """List locally cached public datasets."""
    result = []
    cache = _cache_dir()
    for name, info in DATASET_SOURCES.items():
        ds_dir = cache / name
        if ds_dir.exists():
            files = list(ds_dir.glob("*.json"))
            result.append({
                "name": name,
                "description": info["description"],
                "cached": True,
                "file_count": len(files),
                "path": str(ds_dir),
            })
        else:
            result.append({
                "name": name,
                "description": info["description"],
                "cached": False,
            })
    return result
