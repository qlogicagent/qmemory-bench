#!/usr/bin/env python3
"""extract_eval.py — Offline extraction quality evaluator.

Evaluates whether QMemory's chunker + extractor captures the key facts
needed to answer benchmark questions, WITHOUT starting a server or running recall.

Pipeline:
  1. Load dataset sessions + questions
  2. Chunk each session's messages
  3. Run LLM extraction on each chunk
  4. For each question, check if extracted memories cover the expected answer
  5. Report coverage metrics

Usage:
  python scripts/extract_eval.py --dataset locomo --scale micro
  python scripts/extract_eval.py --preset micro --scale micro
  python scripts/extract_eval.py --dataset qmemory-chinese --scale quick -o report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure qmemory and qmemory_bench are importable
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "qmemory" / "src"))
sys.path.insert(0, str(_root / "qmemory-bench" / "src"))

from qmemory.core.chunker import chunk_messages, Chunk
from qmemory.core.extractor import extract_from_text, ExtractionResult, ExtractedMemory
from qmemory.llm.openai_compat import OpenAICompatProvider
from qmemory_bench.dataset import (
    Dataset,
    Question,
    load_dataset,
    resolve_dataset_selection,
    DATASET_PRESETS,
)

logger = logging.getLogger(__name__)

# ── Key-term extraction (for coverage matching) ─────────────────

_NUM_PATTERN = re.compile(r'\d[\d,.]*')
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff]+')


def extract_key_terms(text: str) -> set[str]:
    """Extract key terms from expected answer for coverage checking.

    Extracts: numbers, CJK word segments (2+ chars), English words (3+ chars).
    """
    terms: set[str] = set()
    # Numbers (prices, dates, counts)
    for m in _NUM_PATTERN.finditer(text):
        terms.add(m.group())
    # CJK segments (naive: consecutive CJK chars as one term)
    for m in _CJK_RANGE.finditer(text):
        seg = m.group()
        if len(seg) >= 2:
            terms.add(seg)
    # English words (3+ chars, lowered)
    for word in re.findall(r'[a-zA-Z]{3,}', text):
        terms.add(word.lower())
    return terms


def extract_numbers(text: str) -> set[str]:
    """Extract all numeric values from text."""
    return {m.group() for m in _NUM_PATTERN.finditer(text)}


# ── Coverage computation ────────────────────────────────────────

@dataclass
class QuestionCoverage:
    """Coverage metrics for a single question."""
    question_id: str
    query: str
    expected: str
    category: str
    dataset: str
    key_terms: set[str]
    matched_terms: set[str]
    numbers: set[str]
    matched_numbers: set[str]
    term_coverage: float  # 0-1
    number_coverage: float  # 0-1 (1.0 if no numbers expected)
    best_matching_memories: list[str]  # top memories that matched


@dataclass
class DatasetExtractionReport:
    """Extraction report for a single dataset."""
    name: str
    total_sessions: int
    total_chunks: int
    total_memories: int
    memories_per_chunk: float
    extraction_time: float  # seconds
    question_coverages: list[QuestionCoverage]
    avg_term_coverage: float
    avg_number_coverage: float


@dataclass
class ExtractionEvalReport:
    """Full extraction evaluation report."""
    datasets: dict[str, DatasetExtractionReport]
    overall_term_coverage: float
    overall_number_coverage: float
    total_memories: int
    total_chunks: int
    total_time: float


def compute_coverage(
    question: Question,
    all_memory_texts: list[str],
) -> QuestionCoverage:
    """Check how well extracted memories cover a question's expected answer."""
    key_terms = extract_key_terms(question.expected)
    numbers = extract_numbers(question.expected)

    # Build a combined text of all memories for term matching
    combined = "\n".join(all_memory_texts).lower()

    matched_terms: set[str] = set()
    for term in key_terms:
        if term.lower() in combined:
            matched_terms.add(term)

    matched_numbers: set[str] = set()
    for num in numbers:
        if num in combined:
            matched_numbers.add(num)

    term_coverage = len(matched_terms) / len(key_terms) if key_terms else 1.0
    number_coverage = len(matched_numbers) / len(numbers) if numbers else 1.0

    # Find top matching memories (by term overlap)
    best: list[tuple[int, str]] = []
    for mem_text in all_memory_texts:
        mem_lower = mem_text.lower()
        hits = sum(1 for t in key_terms if t.lower() in mem_lower)
        if hits > 0:
            best.append((hits, mem_text))
    best.sort(key=lambda x: x[0], reverse=True)

    return QuestionCoverage(
        question_id=question.id,
        query=question.query,
        expected=question.expected,
        category=question.category,
        dataset=question.dataset,
        key_terms=key_terms,
        matched_terms=matched_terms,
        numbers=numbers,
        matched_numbers=matched_numbers,
        term_coverage=term_coverage,
        number_coverage=number_coverage,
        best_matching_memories=[t for _, t in best[:3]],
    )


# ── Main extraction pipeline ───────────────────────────────────

async def extract_dataset(
    dataset: Dataset,
    llm_complete,
    *,
    concurrency: int = 5,
) -> DatasetExtractionReport:
    """Run extraction on all sessions in a dataset and compute coverage."""
    t0 = time.time()

    # Step 1: Chunk all sessions
    all_chunks: list[Chunk] = []
    for session in dataset.sessions:
        if session.messages:
            chunks = chunk_messages(session.messages)
            all_chunks.extend(chunks)

    logger.info(
        "Dataset %s: %d sessions → %d chunks",
        dataset.name, len(dataset.sessions), len(all_chunks),
    )

    # Step 2: Extract memories from chunks (with concurrency limit)
    semaphore = asyncio.Semaphore(concurrency)
    all_memories: list[ExtractedMemory] = []
    extracted_count = 0

    async def extract_one(chunk: Chunk) -> list[ExtractedMemory]:
        async with semaphore:
            result = await extract_from_text(
                chunk.text,
                llm_complete=llm_complete,
            )
            return result.memories

    tasks = [extract_one(c) for c in all_chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("Extraction failed for chunk %d: %s", i, r)
            continue
        all_memories.extend(r)
        extracted_count += 1

    extraction_time = time.time() - t0
    logger.info(
        "Dataset %s: extracted %d memories from %d/%d chunks in %.1fs",
        dataset.name, len(all_memories), extracted_count, len(all_chunks), extraction_time,
    )

    # Step 3: Compute coverage for each question
    all_memory_texts = [m.text for m in all_memories]
    coverages = [
        compute_coverage(q, all_memory_texts)
        for q in dataset.questions
    ]

    avg_term = (
        sum(c.term_coverage for c in coverages) / len(coverages)
        if coverages else 0.0
    )
    avg_num = (
        sum(c.number_coverage for c in coverages) / len(coverages)
        if coverages else 0.0
    )

    return DatasetExtractionReport(
        name=dataset.name,
        total_sessions=len(dataset.sessions),
        total_chunks=len(all_chunks),
        total_memories=len(all_memories),
        memories_per_chunk=len(all_memories) / len(all_chunks) if all_chunks else 0,
        extraction_time=extraction_time,
        question_coverages=coverages,
        avg_term_coverage=avg_term,
        avg_number_coverage=avg_num,
    )


# ── Report formatting ──────────────────────────────────────────

def print_report(report: ExtractionEvalReport) -> None:
    """Print human-readable extraction eval report."""
    print("\n" + "=" * 70)
    print("EXTRACTION QUALITY REPORT")
    print("=" * 70)
    print(f"Total: {report.total_chunks} chunks → {report.total_memories} memories")
    print(f"Overall term coverage:   {report.overall_term_coverage:.1%}")
    print(f"Overall number coverage: {report.overall_number_coverage:.1%}")
    print(f"Total time: {report.total_time:.1f}s")

    for ds_name, ds_report in report.datasets.items():
        print(f"\n{'─' * 60}")
        print(f"📦 {ds_name}")
        print(f"   Sessions: {ds_report.total_sessions}  Chunks: {ds_report.total_chunks}  "
              f"Memories: {ds_report.total_memories}  "
              f"Density: {ds_report.memories_per_chunk:.1f}/chunk")
        print(f"   Term coverage: {ds_report.avg_term_coverage:.1%}  "
              f"Number coverage: {ds_report.avg_number_coverage:.1%}  "
              f"Time: {ds_report.extraction_time:.1f}s")

        # Per-question details
        for cov in ds_report.question_coverages:
            status = "✓" if cov.term_coverage >= 0.6 else "✗"
            print(f"   {status} [{cov.category}] {cov.question_id}: "
                  f"term={cov.term_coverage:.0%} num={cov.number_coverage:.0%}")
            if cov.term_coverage < 0.6:
                missed = cov.key_terms - cov.matched_terms
                if missed:
                    display = list(missed)[:5]
                    print(f"     MISSED terms: {display}")
                if cov.best_matching_memories:
                    print(f"     Best match: {cov.best_matching_memories[0][:80]}")

    print("\n" + "=" * 70)


def report_to_json(report: ExtractionEvalReport) -> dict:
    """Convert report to JSON-serializable dict."""
    datasets = {}
    for ds_name, ds_report in report.datasets.items():
        questions = []
        for cov in ds_report.question_coverages:
            questions.append({
                "id": cov.question_id,
                "query": cov.query,
                "expected": cov.expected,
                "category": cov.category,
                "term_coverage": round(cov.term_coverage, 3),
                "number_coverage": round(cov.number_coverage, 3),
                "key_terms_count": len(cov.key_terms),
                "matched_terms_count": len(cov.matched_terms),
                "missed_terms": sorted(cov.key_terms - cov.matched_terms),
                "best_memories": cov.best_matching_memories[:3],
            })
        datasets[ds_name] = {
            "sessions": ds_report.total_sessions,
            "chunks": ds_report.total_chunks,
            "memories": ds_report.total_memories,
            "memories_per_chunk": round(ds_report.memories_per_chunk, 2),
            "extraction_time_s": round(ds_report.extraction_time, 1),
            "avg_term_coverage": round(ds_report.avg_term_coverage, 3),
            "avg_number_coverage": round(ds_report.avg_number_coverage, 3),
            "questions": questions,
        }
    return {
        "overall_term_coverage": round(report.overall_term_coverage, 3),
        "overall_number_coverage": round(report.overall_number_coverage, 3),
        "total_memories": report.total_memories,
        "total_chunks": report.total_chunks,
        "total_time_s": round(report.total_time, 1),
        "datasets": datasets,
    }


# ── CLI ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline extraction quality evaluator",
    )
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help="Single dataset name (e.g. locomo, qmemory-chinese)")
    p.add_argument("--preset", "-p", type=str, default=None,
                   help="Dataset preset (e.g. micro, release-full)")
    p.add_argument("--scale", "-s", type=str, default="micro",
                   choices=["micro", "quick", "standard", "full"],
                   help="Dataset scale (default: micro)")
    p.add_argument("--api-key", type=str, default=None,
                   help="LLM API key (default: $DEEPSEEK_API_KEY or $QMEMORY_LLM_API_KEY)")
    p.add_argument("--provider", type=str, default="deepseek",
                   help="LLM provider (default: deepseek)")
    p.add_argument("--model", type=str, default=None,
                   help="Override model name")
    p.add_argument("--concurrency", "-c", type=int, default=5,
                   help="Max concurrent extraction calls (default: 5)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Save JSON report to file")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _resolve_api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("QMEMORY_LLM_API_KEY", "")
    if not key:
        print("ERROR: No API key. Use --api-key or set DEEPSEEK_API_KEY env var.")
        sys.exit(1)
    return key


def _resolve_provider(args: argparse.Namespace) -> OpenAICompatProvider:
    """Create LLM provider from CLI args."""
    from qmemory.llm.registry import resolve_provider
    api_key = _resolve_api_key(args)
    base_url, model, _ = resolve_provider(args.provider)
    if args.model:
        model = args.model
    return OpenAICompatProvider(api_key=api_key, base_url=base_url, model=model)


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve datasets
    if args.dataset:
        dataset_names = [args.dataset]
    elif args.preset:
        dataset_names = DATASET_PRESETS.get(args.preset, [])
        if not dataset_names:
            print(f"ERROR: Unknown preset '{args.preset}'. Available: {list(DATASET_PRESETS.keys())}")
            sys.exit(1)
    else:
        print("ERROR: Specify --dataset or --preset")
        sys.exit(1)

    # Load datasets
    datasets: list[Dataset] = []
    for name in dataset_names:
        try:
            ds = load_dataset(name, args.scale)
            datasets.append(ds)
            logger.info("Loaded %s: %d sessions, %d questions", name, len(ds.sessions), len(ds.questions))
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", name, e)

    if not datasets:
        print("ERROR: No datasets loaded.")
        sys.exit(1)

    # Create LLM provider
    provider = _resolve_provider(args)
    logger.info("Using %s/%s", args.provider, provider.model)

    # Run extraction on each dataset
    t0 = time.time()
    ds_reports: dict[str, DatasetExtractionReport] = {}
    for ds in datasets:
        ds_report = await extract_dataset(ds, provider.complete, concurrency=args.concurrency)
        ds_reports[ds.name] = ds_report

    total_time = time.time() - t0

    # Aggregate
    all_coverages = [c for dr in ds_reports.values() for c in dr.question_coverages]
    overall_term = sum(c.term_coverage for c in all_coverages) / len(all_coverages) if all_coverages else 0
    overall_num = sum(c.number_coverage for c in all_coverages) / len(all_coverages) if all_coverages else 0

    report = ExtractionEvalReport(
        datasets=ds_reports,
        overall_term_coverage=overall_term,
        overall_number_coverage=overall_num,
        total_memories=sum(dr.total_memories for dr in ds_reports.values()),
        total_chunks=sum(dr.total_chunks for dr in ds_reports.values()),
        total_time=total_time,
    )

    # Output
    print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report_to_json(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
