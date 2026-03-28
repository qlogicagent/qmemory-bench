"""Benchmark Runner — orchestrates the full evaluation pipeline.

Pipeline:
  1. Connect to QMemory server
  2. Create isolated eval user
  3. For each dataset:
     a. Inject sessions (q.add)
     b. Ask questions (q.search)
     c. Judge answers (LLM)
  4. Clean up data
  5. Generate report

Supports progress callbacks for UI integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from qmemory_bench.dataset import (
    Dataset,
    Question,
    Session,
    load_dataset,
    resolve_dataset_selection,
)
from qmemory_bench.judge import JudgeResult, aggregate_scores, judge_single
from qmemory_bench.providers import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    target_url: str = "http://localhost:18800"
    provider: str = "deepseek"
    api_key: str = ""
    model: str = ""
    scale: str = "quick"      # quick | standard | full
    dataset_names: list[str] = field(default_factory=list)
    dataset_preset: str = "public-main"
    output_path: str | None = None


@dataclass
class ConcurrencyMetrics:
    """Latency / throughput under concurrent load."""
    concurrency: int = 0          # N parallel requests
    total_requests: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    max_ms: float = 0.0
    errors: int = 0               # failed requests
    accuracy_drop: float = 0.0    # score drop vs sequential (pp)


@dataclass
class DatasetReport:
    """Report for a single dataset."""
    name: str
    overall: float            # 0-100%
    overall_precision: float  # 0-1.0 precision
    categories: dict[str, Any]
    total_questions: int
    inject_time: float        # seconds
    eval_time: float          # seconds
    results: list[JudgeResult]
    concurrency: ConcurrencyMetrics | None = None


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    overall: float            # 0-100%
    datasets: dict[str, DatasetReport]
    timestamp: str
    qmemory_version: str
    llm_provider: str
    llm_model: str
    scale: str
    dataset_preset: str
    dataset_names: list[str]
    duration: float           # total seconds
    target_url: str


# ── Targets from PLAN Appendix A ────────────────────────────────

TARGETS = {
    # LongMemEval-S 6 dimensions
    "single-session-user": 95.0,
    "single-session-assistant": 93.0,
    "single-session-preference": 72.0,
    "knowledge-update": 88.0,
    "temporal-reasoning": 80.0,
    "multi-session": 73.0,
    # QMemory-Chinese dimensions
    "temporal-zh": 78.0,
    "idiom-zh": 75.0,
    "name-disambig": 70.0,
    "profile-zh": 80.0,
    # Multimodal v2 — 6 dimensions (strict)
    "mm-basic-recall": 85.0,
    "mm-precision-retrieve": 70.0,
    "mm-noise-resist": 65.0,
    "mm-abstract-query": 60.0,
    "mm-temporal-disambig": 65.0,
    "mm-cross-ref": 65.0,
    # LoCoMo — dialogue retrieval
    "recall-accuracy": 90.0,
    "multi-turn": 75.0,
    "logical-reasoning": 70.0,
    # Conflict-resolution — version chain
    "contradiction-detect": 80.0,
    "supersede-correctness": 75.0,
    "expired-info": 78.0,
    # Profile-accuracy — user profile extraction
    "preference-extract": 85.0,
    "habit-extract": 80.0,
    "fact-extract": 90.0,
    "timeliness": 70.0,
    # Stress-scale — large-scale retrieval
    "10k-latency": 85.0,
    "50k-latency": 80.0,
    "100k-accuracy": 75.0,
    # v2.0 新增: 系统性盲区覆盖
    # Preference-drift
    "preference-drift": 65.0,
    "preference-conflict": 65.0,
    # Implicit-memory
    "implicit-family": 70.0,
    "implicit-career": 65.0,
    "implicit-emotion": 60.0,
    "implicit-health": 65.0,
    "implicit-finance": 60.0,
    "implicit-location": 55.0,
    # Stress-latency
    "stress-recall": 85.0,
    "stress-precision": 75.0,
    # Shared
    "adversarial": 70.0,
    "noise-resist": 70.0,
    "temporal": 70.0,
    "multi-hop": 70.0,
    "recall-accuracy": 90.0,
    # Real LoCoMo (HuggingFace)
    "single-fact": 85.0,
    "open-ended": 60.0,
}


# ── Progress helpers ────────────────────────────────────────────

def _update_progress(progress: dict | None, **kw) -> None:
    """Thread-safe progress update (dict is atomic in CPython)."""
    if progress is not None:
        progress.update(kw)


# ── Runner ──────────────────────────────────────────────────────

async def run_benchmark(
    config: BenchmarkConfig,
    progress: dict | None = None,
) -> BenchmarkReport:
    """Execute the full benchmark pipeline.

    Args:
        config: Benchmark configuration.
        progress: Optional dict updated with progress info for UI.
            Keys: stage, pct, detail, dataset, question_i, question_n,
                  session_i, session_n.
    """
    start_time = time.time()
    _update_progress(progress, stage="connecting", pct=0.02, detail="连接 QMemory 服务...")

    # 1. Setup
    client = httpx.AsyncClient(
        base_url=config.target_url,
        timeout=120.0,
        proxy=None,
        trust_env=False,
    )
    llm = LLMJudge(
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )
    eval_user = f"eval_{uuid4().hex[:8]}"

    # Get QMemory version
    try:
        health_resp = await client.get("/v1/health/")
        qm_version = health_resp.json().get("version", "?")
        _update_progress(progress, stage="connected", pct=0.05,
                         detail=f"已连接 QMemory {qm_version}")
    except Exception as e:
        _update_progress(progress, stage="error", pct=0, detail=f"连接失败: {e}")
        raise

    # Build list of valid datasets
    dataset_preset, selected_datasets = resolve_dataset_selection(
        config.dataset_names,
        config.dataset_preset,
    )

    valid_datasets: list[tuple[str, Dataset]] = []
    for ds_name in selected_datasets:
        try:
            ds = load_dataset(ds_name, config.scale)
            valid_datasets.append((ds_name, ds))
        except FileNotFoundError:
            logger.warning(f"Dataset '{ds_name}' not found, skipping")

    if not valid_datasets:
        _update_progress(progress, stage="error", pct=0, detail="没有可用的数据集")
        raise FileNotFoundError("No valid datasets found")

    total_datasets = len(valid_datasets)
    dataset_reports: dict[str, DatasetReport] = {}

    # Pre-cleanup: ensure a clean slate for the eval user
    _update_progress(progress, stage="cleanup", pct=0.04, detail="清空评测用户数据...")
    try:
        await client.request(
            "DELETE", "/v1/memories/",
            params={"user_id": eval_user, "confirm": "true"},
        )
    except Exception:
        pass  # User didn't exist yet, that's fine

    # Build passthrough LLM config once
    llm_passthrough: dict | None = None
    if config.api_key:
        llm_passthrough = {
            "provider": config.provider,
            "api_key": config.api_key,
            "model": config.model or None,
        }

    for ds_idx, (ds_name, ds) in enumerate(valid_datasets):
        base_pct = 0.05 + (ds_idx / total_datasets) * 0.90
        _update_progress(progress, stage="running", pct=base_pct,
                         detail=f"数据集 {ds_name} ({ds_idx+1}/{total_datasets})",
                         dataset=ds_name)

        # Clean memories before each dataset to avoid cross-contamination
        if ds_idx > 0:
            _update_progress(progress, stage="cleanup", pct=base_pct,
                             detail=f"清理上一数据集记忆...")
            try:
                await client.request(
                    "DELETE", "/v1/memories/",
                    params={"user_id": eval_user, "confirm": "true"},
                )
            except Exception:
                pass

        report = await _run_single_dataset(
            ds, eval_user, client, llm,
            llm_config=llm_passthrough,
            progress=progress,
            base_pct=base_pct,
            pct_range=0.90 / total_datasets,
        )
        dataset_reports[ds_name] = report

    # Final cleanup
    _update_progress(progress, stage="cleanup", pct=0.97, detail="清理评测数据...")
    try:
        await client.request(
            "DELETE", "/v1/memories/",
            params={"user_id": eval_user, "confirm": "true"},
        )
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

    await llm.close()
    await client.aclose()

    # Compute weighted overall
    all_scores = [dr.overall for dr in dataset_reports.values()]
    weighted_overall = sum(all_scores) / len(all_scores) if all_scores else 0
    total_time = time.time() - start_time

    report = BenchmarkReport(
        overall=round(weighted_overall, 1),
        datasets=dataset_reports,
        timestamp=datetime.now().isoformat(),
        qmemory_version=qm_version,
        llm_provider=config.provider,
        llm_model=llm.model,
        scale=config.scale,
        dataset_preset=dataset_preset,
        dataset_names=[name for name, _ in valid_datasets],
        duration=round(total_time, 1),
        target_url=config.target_url,
    )

    _update_progress(progress, stage="done", pct=1.0,
                     detail=f"评测完成！耗时 {report.duration}s")

    # Save report if output path given
    if config.output_path:
        _save_report(report, Path(config.output_path))

    return report


async def _run_single_dataset(
    ds: Dataset,
    eval_user: str,
    client: httpx.AsyncClient,
    llm: LLMJudge,
    *,
    llm_config: dict | None = None,
    progress: dict | None = None,
    base_pct: float = 0.0,
    pct_range: float = 1.0,
) -> DatasetReport:
    """Run evaluation for a single dataset."""
    logger.info(f"Evaluating dataset: {ds.name} ({len(ds.questions)} questions)")
    total_sessions = len(ds.sessions)
    total_questions = len(ds.questions)

    # Phase 1: Inject sessions — parallel with semaphore
    inject_start = time.time()
    inject_sem = asyncio.Semaphore(3)  # limit concurrent injections
    inject_done = 0

    async def _inject_one(i: int, session: Session) -> None:
        nonlocal inject_done
        async with inject_sem:
            _update_progress(
                progress, stage="injecting",
                pct=base_pct + (inject_done / max(total_sessions, 1)) * pct_range * 0.35,
                detail=f"注入会话 {inject_done+1}/{total_sessions} ({ds.name})",
                session_i=inject_done + 1, session_n=total_sessions,
            )
            try:
                body: dict = {
                    "messages": session.messages,
                    "user_id": eval_user,
                    "session_id": session.id,
                }
                if llm_config:
                    body["llm_config"] = llm_config
                ts = session.metadata.get("timestamp")
                if ts:
                    body["created_at"] = ts
                await client.post("/v1/memories/", json=body, timeout=300.0)
            except Exception as e:
                logger.warning(f"Failed to inject session {session.id}: {type(e).__name__}: {e}")
            finally:
                inject_done += 1

    await asyncio.gather(*[_inject_one(i, s) for i, s in enumerate(ds.sessions)])
    inject_time = time.time() - inject_start

    # Phase 1.5: Verify injection — check how many memories were created
    mem_count = 0
    try:
        resp = await client.get("/v1/memories/",
                                params={"user_id": eval_user, "page_size": 1})
        mem_count = resp.json().get("total", 0)
    except Exception:
        pass
    _update_progress(
        progress, stage="injected",
        pct=base_pct + pct_range * 0.38,
        detail=f"注入完成: {total_sessions} 会话 → {mem_count} 条记忆 ({ds.name})",
    )
    logger.info(f"Injection done: {total_sessions} sessions → {mem_count} memories")

    # Phase 2: Evaluate questions — parallel search + judge
    eval_start = time.time()
    eval_sem = asyncio.Semaphore(5)  # limit concurrent search + judge calls
    eval_done = 0

    async def _eval_one(q: Question) -> tuple[JudgeResult, dict]:
        nonlocal eval_done
        async with eval_sem:
            _update_progress(
                progress, stage="evaluating",
                pct=base_pct + pct_range * 0.4 + (eval_done / max(total_questions, 1)) * pct_range * 0.6,
                detail=f"评测 {eval_done+1}/{total_questions} ({ds.name})",
                question_i=eval_done + 1, question_n=total_questions,
            )
            try:
                resp = await client.get("/v1/memories/search/", params={
                    "q": q.query,
                    "user_id": eval_user,
                    "limit": 10,
                    "hierarchy": "true",
                })
                recall = resp.json()
            except Exception as e:
                logger.warning(f"Search failed for {q.id}: {type(e).__name__}: {e}")
                recall = {"memories": [], "context": ""}

            result = await judge_single(
                question_id=q.id,
                query=q.query,
                expected=q.expected,
                category=q.category,
                recall_result=recall,
                llm=llm,
            )
            logger.info(f"  {q.id}: {result.score}/10 P={result.precision:.2f} ({q.category})")
            eval_done += 1
            return result, recall

    eval_outputs = await asyncio.gather(*[_eval_one(q) for q in ds.questions])
    results: list[JudgeResult] = [r for r, _ in eval_outputs]

    # Push Q&A entries to progress (maintain question order)
    if progress is not None:
        for (result, recall), q in zip(eval_outputs, ds.questions):
            mem_texts = [
                f"[{m.get('category', '?')}] {m.get('text', '')[:100]}"
                for m in recall.get("memories", [])[:5]
            ]
            progress.setdefault("qa_log", []).append({
                "id": q.id,
                "query": q.query,
                "expected": q.expected,
                "recall": result.context_preview,
                "memory_count": len(recall.get("memories", [])),
                "memory_texts": mem_texts,
                "score": result.score,
                "precision": result.precision,
                "reason": result.reason,
                "category": q.category,
                "dataset": ds.name,
                "is_adversarial": result.is_adversarial,
            })

    eval_time = time.time() - eval_start

    # Aggregate
    agg = aggregate_scores(results, ds.categories)

    # Phase 3: Concurrent stress test (if dataset has stress-related categories)
    conc_metrics = None
    stress_cats = {"stress-recall", "stress-precision", "adversarial", "noise-resist"}
    has_stress = bool(stress_cats & set(ds.categories))
    if has_stress and len(ds.questions) >= 3:
        _update_progress(
            progress, stage="stress-testing",
            pct=base_pct + pct_range * 0.95,
            detail=f"并发压力测试 ({ds.name})",
        )
        conc_metrics = await _concurrent_stress_test(
            client=client,
            eval_user=eval_user,
            questions=ds.questions,
            sequential_results=results,
        )

    return DatasetReport(
        name=ds.name,
        overall=agg["overall"],
        overall_precision=agg.get("overall_precision", 0.0),
        categories=agg["categories"],
        total_questions=agg["total_questions"],
        inject_time=round(inject_time, 1),
        eval_time=round(eval_time, 1),
        results=results,
        concurrency=conc_metrics,
    )


# ── Concurrent stress test ──────────────────────────────────────

async def _concurrent_stress_test(
    *,
    client: httpx.AsyncClient,
    eval_user: str,
    questions: list[Question],
    sequential_results: list[JudgeResult],
    concurrency: int = 8,
) -> ConcurrencyMetrics:
    """Fire N parallel search requests to measure latency degradation.

    Uses the same questions already evaluated sequentially, so we can
    compare concurrent accuracy against baseline.
    """
    import statistics

    queries = [q.query for q in questions]
    expected_map = {q.query: r.score for q, r in zip(questions, sequential_results)}

    latencies: list[float] = []
    errors = 0
    concurrent_scores: list[int] = []

    async def _fire_one(query: str) -> tuple[float, int | None]:
        """Send one search request, return (latency_ms, keyword_score|None)."""
        t0 = time.monotonic()
        try:
            resp = await client.get("/v1/memories/search/", params={
                "q": query,
                "user_id": eval_user,
                "limit": 10,
                "hierarchy": "true",
            })
            lat = (time.monotonic() - t0) * 1000
            data = resp.json()
            # Quick keyword-based consistency check
            context = data.get("context", "")
            return lat, 1 if context else 0
        except Exception:
            lat = (time.monotonic() - t0) * 1000
            return lat, None

    # Run all queries in batches of `concurrency`
    for batch_start in range(0, len(queries), concurrency):
        batch = queries[batch_start:batch_start + concurrency]
        results = await asyncio.gather(*[_fire_one(q) for q in batch])
        for lat, score in results:
            latencies.append(lat)
            if score is None:
                errors += 1
            else:
                concurrent_scores.append(score)

    if not latencies:
        return ConcurrencyMetrics()

    latencies.sort()
    n = len(latencies)

    # Compare concurrent hit-rate vs sequential average
    seq_avg = sum(r.score for r in sequential_results) / len(sequential_results) * 10 if sequential_results else 0
    conc_hit_rate = (sum(concurrent_scores) / len(concurrent_scores) * 100) if concurrent_scores else 0
    accuracy_drop = max(0, seq_avg - conc_hit_rate)

    metrics = ConcurrencyMetrics(
        concurrency=concurrency,
        total_requests=n,
        p50_ms=round(latencies[n // 2], 1),
        p95_ms=round(latencies[int(n * 0.95)], 1) if n >= 2 else round(latencies[-1], 1),
        p99_ms=round(latencies[int(n * 0.99)], 1) if n >= 2 else round(latencies[-1], 1),
        avg_ms=round(statistics.mean(latencies), 1),
        max_ms=round(latencies[-1], 1),
        errors=errors,
        accuracy_drop=round(accuracy_drop, 1),
    )
    logger.info(
        f"Concurrent stress: N={concurrency}, p50={metrics.p50_ms}ms, "
        f"p95={metrics.p95_ms}ms, errors={errors}, "
        f"accuracy_drop={accuracy_drop:.1f}pp"
    )
    return metrics


# ── Comparison runner ───────────────────────────────────────────

@dataclass
class ComparisonConfig:
    """Config for running comparison benchmarks across providers."""
    target_url: str = "http://localhost:18800"
    providers: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"provider": "deepseek", "api_key": "sk-...", "model": ""}
    scale: str = "quick"
    dataset_names: list[str] = field(default_factory=list)
    dataset_preset: str = "public-main"


@dataclass
class ComparisonReport:
    """Side-by-side comparison of multiple providers."""
    reports: dict[str, BenchmarkReport]   # provider_label → report
    timestamp: str
    scale: str
    dataset_preset: str
    dataset_names: list[str]
    target_url: str


async def run_comparison(
    config: ComparisonConfig,
    progress: dict | None = None,
) -> ComparisonReport:
    """Run the same benchmark with multiple providers and compare."""
    total = len(config.providers)
    reports: dict[str, BenchmarkReport] = {}

    for idx, prov in enumerate(config.providers):
        provider_key = prov["provider"]
        label = f"{provider_key}"
        if sum(1 for p in config.providers if p["provider"] == provider_key) > 1:
            label = f"{provider_key}#{idx+1}"

        _update_progress(progress, stage="compare", pct=idx / total,
                         detail=f"对比评测 {idx+1}/{total}: {provider_key}")

        bench_cfg = BenchmarkConfig(
            target_url=config.target_url,
            provider=prov["provider"],
            api_key=prov["api_key"],
            model=prov.get("model", ""),
            scale=config.scale,
            dataset_names=config.dataset_names,
            dataset_preset=config.dataset_preset,
        )

        sub_progress: dict[str, Any] = {}
        try:
            report = await run_benchmark(bench_cfg, sub_progress)
            reports[label] = report
        except Exception as e:
            logger.error(f"Comparison run failed for {provider_key}: {e}")
            _update_progress(progress, detail=f"{provider_key} 评测失败: {e}")

    _update_progress(progress, stage="done", pct=1.0,
                     detail=f"对比评测完成 ({len(reports)}/{total})")

    return ComparisonReport(
        reports=reports,
        timestamp=datetime.now().isoformat(),
        scale=config.scale,
        dataset_preset=config.dataset_preset,
        dataset_names=config.dataset_names,
        target_url=config.target_url,
    )


# ── Report Output ───────────────────────────────────────────────

def print_report(report: BenchmarkReport) -> None:
    """Print a formatted benchmark report to console."""
    try:
        from rich.console import Console
        from rich.table import Table
        _print_rich(report)
    except ImportError:
        _print_plain(report)


def _print_rich(report: BenchmarkReport) -> None:
    """Rich-formatted output."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()
    console.print("[bold]QMemory Benchmark Report[/bold]")
    console.print(f"  QMemory: {report.qmemory_version} @ {report.target_url}")
    console.print(f"  Judge: {report.llm_provider} / {report.llm_model}")
    console.print(f"  Preset: {report.dataset_preset} | Scale: {report.scale} | Duration: {report.duration}s")
    console.print()

    target_met = report.overall >= 85.0
    status = "[green]PASS[/green]" if target_met else "[red]FAIL[/red]"
    console.print(f"  Overall: [bold]{report.overall:.1f}%[/bold] (target ≥85%) {status}")
    console.print()

    for ds_name, ds_report in report.datasets.items():
        table = Table(title=f"Dataset: {ds_name}")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Count", justify="right")

        for cat, info in ds_report.categories.items():
            score = info["score"]
            target = TARGETS.get(cat, 80.0)
            met = score >= target
            status_str = "[green]✓[/green]" if met else "[red]✗[/red]"
            table.add_row(cat, f"{score:.1f}%", f"{target:.0f}%",
                          status_str, str(info["count"]))

        console.print(table)
        console.print()


def _print_plain(report: BenchmarkReport) -> None:
    """Plain text fallback."""
    print(f"\nQMemory Benchmark Report")
    print(f"  QMemory: {report.qmemory_version} @ {report.target_url}")
    print(f"  Judge: {report.llm_provider} / {report.llm_model}")
    print(f"  Preset: {report.dataset_preset}")
    print(f"  Overall: {report.overall:.1f}% (target >=85%)")
    print(f"  Duration: {report.duration}s")
    for ds_name, ds_report in report.datasets.items():
        print(f"\n  Dataset: {ds_name}")
        for cat, info in ds_report.categories.items():
            target = TARGETS.get(cat, 80.0)
            status = "PASS" if info["score"] >= target else "FAIL"
            print(f"    {cat:30s} {info['score']:6.1f}%  target={target:.0f}%  {status}")


def _save_report(report: BenchmarkReport, path: Path) -> None:
    """Save report to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report_to_dict(report)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Report saved to {path}")


def report_to_dict(obj: Any) -> Any:
    """Recursively serialize a report/dataclass to a plain dict."""
    if hasattr(obj, "__dict__"):
        return {k: report_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: report_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [report_to_dict(v) for v in obj]
    return obj
