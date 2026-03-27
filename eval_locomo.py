#!/usr/bin/env python3
"""
LoCoMo 评测脚本 — QMemory Phase 4 验证 (4.8 + 4.9)

对比 flat (L1-only) vs hierarchy (L2+L3) 两种检索模式:
  - Flat:      hierarchy=false, 仅向量+FTS检索
  - Hierarchy:  hierarchy=true,  注入后运行 consolidation (L1→L2→L3), 检索带层级

50 题 × 5 维度: recall-accuracy(10) + multi-hop(15) + temporal(10)
                 + logical-reasoning(10) + noise-resist(5)

用 DeepSeek 做 LLM judge, 0-10 评分.

Usage:
    python eval_locomo.py [--base-url http://127.0.0.1:18800] [--judge-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── Config ──────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "data" / "locomo_standard.json"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "http://127.0.0.1:18800"

USER_FLAT = "eval_locomo_flat"
USER_HIER = "eval_locomo_hier"


@dataclass
class QuestionResult:
    qid: str
    category: str
    difficulty: str
    query: str
    expected: str
    context_flat: str = ""
    context_hier: str = ""
    score_flat: int = 0
    score_hier: int = 0
    reason_flat: str = ""
    reason_hier: str = ""


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)
    flat_overall: float = 0.0
    hier_overall: float = 0.0
    flat_by_cat: dict[str, float] = field(default_factory=dict)
    hier_by_cat: dict[str, float] = field(default_factory=dict)
    inject_time_flat: float = 0.0
    inject_time_hier: float = 0.0
    consol_time: float = 0.0
    judge_time: float = 0.0
    flat_memory_count: int = 0
    hier_memory_count: int = 0
    hier_episode_count: int = 0
    hier_schema_count: int = 0


# ── Helpers ─────────────────────────────────────────────────────
def load_dataset() -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def require_deepseek_key() -> str:
    if DEEPSEEK_KEY:
        return DEEPSEEK_KEY
    raise RuntimeError("DEEPSEEK_API_KEY is required for LoCoMo evaluation.")


def llm_config_body() -> dict:
    """LLM config to pass through to QMemory for extraction/consolidation."""
    return {
        "provider": "openai_compat",
        "api_key": require_deepseek_key(),
        "base_url": DEEPSEEK_BASE,
        "model": DEEPSEEK_MODEL,
    }


def deepseek_judge(query: str, expected: str, context: str) -> tuple[int, str]:
    """Call DeepSeek to judge a recall result. Returns (score 0-10, reason)."""
    system = """你是记忆系统评测打分员。根据问题、期望答案和系统实际召回内容打分。
评分标准 (0-10):
- 10: 召回内容完整覆盖期望答案所有要点
- 8-9: 覆盖大部分要点，遗漏少量细节
- 5-7: 覆盖部分要点，有明显遗漏
- 3-4: 仅覆盖少量要点
- 1-2: 几乎没有相关信息但有轻微关联
- 0: 完全无关或错误

注意:
1. 语义等价即可，不要求完全字面匹配
2. 如果召回了期望答案的核心信息但细节略有不同，仍应给高分
3. 如果混入了不相关信息但核心答案正确，不扣太多分
4. 如果张冠李戴（如把A的信息说成B的），应严格扣分

输出严格JSON格式: {"score": <0-10整数>, "reason": "<简要理由>"}"""

    user_msg = f"""## 问题
{query}

## 期望答案
{expected}

## 系统召回内容
{context}

请打分:"""

    try:
        resp = httpx.post(
            f"{DEEPSEEK_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {require_deepseek_key()}"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Parse JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        obj = json.loads(text)
        return int(obj["score"]), obj.get("reason", "")
    except Exception as e:
        print(f"    [JUDGE ERROR] {e}")
        # Fallback: keyword matching
        return _keyword_fallback(expected, context), f"fallback: {e}"


def _keyword_fallback(expected: str, context: str) -> int:
    """Simple keyword matching fallback."""
    if not context:
        return 0
    keywords = [w for w in expected.replace("，", " ").replace("。", " ").split() if len(w) >= 2]
    if not keywords:
        return 0
    hits = sum(1 for k in keywords if k in context)
    ratio = hits / len(keywords)
    if ratio >= 0.8:
        return 8
    elif ratio >= 0.5:
        return 5
    elif ratio >= 0.2:
        return 3
    return 0


# ── Core eval functions ─────────────────────────────────────────
def cleanup_user(base_url: str, user_id: str):
    """Delete all memories for a user."""
    try:
        resp = httpx.delete(
            f"{base_url}/v1/memories/",
            params={"user_id": user_id, "confirm": "true"},
            timeout=30,
            trust_env=False,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Cleaned {user_id}: {data.get('memories_deleted', 0)} memories deleted")
        else:
            print(f"  Cleanup {user_id}: status {resp.status_code}")
    except Exception as e:
        print(f"  Cleanup {user_id} error: {e}")


def inject_sessions(base_url: str, user_id: str, sessions: list[dict]) -> float:
    """Inject all sessions into QMemory. Returns elapsed time."""
    t0 = time.time()
    total = len(sessions)
    for i, sess in enumerate(sessions, 1):
        sid = sess["id"]
        messages = sess["messages"]
        body = {
            "messages": messages,
            "user_id": user_id,
            "session_id": sid,
            "llm_config": llm_config_body(),
            "created_at": sess.get("metadata", {}).get("timestamp"),
        }
        try:
            resp = httpx.post(
                f"{base_url}/v1/memories/",
                json=body,
                timeout=120,
                trust_env=False,
            )
            if resp.status_code == 200:
                data = resp.json()
                added = data.get("results", [{}])[0].get("memories_added", "?") if data.get("results") else data.get("memories_added", "?")
                print(f"  [{i}/{total}] {sid}: +{added} memories")
            else:
                print(f"  [{i}/{total}] {sid}: HTTP {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  [{i}/{total}] {sid}: ERROR {e}")
    elapsed = time.time() - t0
    return elapsed


def run_consolidation(base_url: str, user_id: str) -> float:
    """Run L1→L2→L3 consolidation. Returns elapsed time."""
    t0 = time.time()
    body = {
        "user_id": user_id,
        "min_memories": 3,
        "min_episodes": 2,
    }
    try:
        resp = httpx.post(
            f"{base_url}/v1/admin/consolidate",
            json=body,
            timeout=300,
            trust_env=False,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Consolidation: episodes_created={data.get('episodes_created', 0)}, "
                  f"schemas_created={data.get('schemas_created', 0)}, "
                  f"memories_consolidated={data.get('memories_consolidated', 0)}")
        else:
            print(f"  Consolidation failed: HTTP {resp.status_code} - {resp.text[:300]}")
    except Exception as e:
        print(f"  Consolidation error: {e}")
    return time.time() - t0


def get_stats(base_url: str, user_id: str) -> dict:
    """Get memory/episode/schema counts for a user."""
    stats = {"memories": 0, "episodes": 0, "schemas": 0}
    try:
        resp = httpx.get(
            f"{base_url}/v1/memories/",
            params={"user_id": user_id, "page_size": 1},
            timeout=15,
            trust_env=False,
        )
        if resp.status_code == 200:
            stats["memories"] = resp.json().get("total", 0)
    except:
        pass
    # Try admin overview for episodes/schemas
    try:
        resp = httpx.get(
            f"{base_url}/v1/admin/overview",
            params={"user_id": user_id},
            timeout=15,
            trust_env=False,
        )
        if resp.status_code == 200:
            data = resp.json()
            stats["episodes"] = data.get("episodes", 0)
            stats["schemas"] = data.get("schemas", 0)
    except:
        pass
    return stats


def search_questions(
    base_url: str, user_id: str, questions: list[dict], hierarchy: bool
) -> dict[str, str]:
    """Search each question. Returns {qid: context_string}."""
    results = {}
    mode_label = "hierarchy" if hierarchy else "flat"
    total = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["query"]
        try:
            resp = httpx.get(
                f"{base_url}/v1/memories/search/",
                params={
                    "q": query,
                    "user_id": user_id,
                    "limit": 10,
                    "hierarchy": str(hierarchy).lower(),
                },
                timeout=120,
                trust_env=False,
            )
            if resp.status_code == 200:
                data = resp.json()
                context = data.get("context", "")
                mem_count = len(data.get("memories", []))
                results[qid] = context
                if i % 10 == 0 or i == total:
                    print(f"  [{mode_label}] Searched {i}/{total}")
            else:
                results[qid] = ""
                print(f"  [{mode_label}] {qid}: HTTP {resp.status_code}")
        except Exception as e:
            results[qid] = ""
            print(f"  [{mode_label}] {qid}: ERROR {e}")
    return results


def judge_all(questions: list[dict], flat_ctx: dict, hier_ctx: dict) -> list[QuestionResult]:
    """Judge all questions for both modes."""
    results = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        qr = QuestionResult(
            qid=qid,
            category=q["category"],
            difficulty=q.get("difficulty", "standard"),
            query=q["query"],
            expected=q["expected"],
            context_flat=flat_ctx.get(qid, ""),
            context_hier=hier_ctx.get(qid, ""),
        )

        # Judge flat
        score_f, reason_f = deepseek_judge(q["query"], q["expected"], qr.context_flat)
        qr.score_flat = score_f
        qr.reason_flat = reason_f

        # Judge hierarchy
        score_h, reason_h = deepseek_judge(q["query"], q["expected"], qr.context_hier)
        qr.score_hier = score_h
        qr.reason_hier = reason_h

        marker = "+" if score_h > score_f else ("=" if score_h == score_f else "-")
        print(f"  [{i}/{total}] {qid} ({q['category']}): flat={score_f} hier={score_h} [{marker}]")
        results.append(qr)
    return results


def compute_report(results: list[QuestionResult]) -> EvalReport:
    """Compute aggregate scores."""
    report = EvalReport(results=results)

    # Overall
    if results:
        report.flat_overall = sum(r.score_flat for r in results) / len(results) * 10
        report.hier_overall = sum(r.score_hier for r in results) / len(results) * 10

    # By category
    cats: dict[str, list[QuestionResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    for cat, rs in cats.items():
        report.flat_by_cat[cat] = sum(r.score_flat for r in rs) / len(rs) * 10
        report.hier_by_cat[cat] = sum(r.score_hier for r in rs) / len(rs) * 10

    return report


def print_report(report: EvalReport):
    """Pretty-print the eval report."""
    print("\n" + "=" * 72)
    print("  QMemory Phase 4 评测报告 — LoCoMo v2.0")
    print("=" * 72)

    print(f"\n{'指标':<25} {'Flat (L1)':<12} {'Hierarchy':<12} {'Delta':<10}")
    print("-" * 60)
    print(f"{'Overall Score':<25} {report.flat_overall:>8.1f}%    {report.hier_overall:>8.1f}%    {report.hier_overall - report.flat_overall:>+6.1f}%")

    cats_order = ["recall-accuracy", "multi-hop", "temporal", "logical-reasoning", "noise-resist"]
    for cat in cats_order:
        f = report.flat_by_cat.get(cat, 0)
        h = report.hier_by_cat.get(cat, 0)
        print(f"  {cat:<23} {f:>8.1f}%    {h:>8.1f}%    {h - f:>+6.1f}%")

    print(f"\n{'Memories':<25} {report.flat_memory_count:<12} {report.hier_memory_count:<12}")
    print(f"{'Episodes':<25} {'—':<12} {report.hier_episode_count:<12}")
    print(f"{'Schemas':<25} {'—':<12} {report.hier_schema_count:<12}")
    print(f"{'Inject Time':<25} {report.inject_time_flat:>8.1f}s    {report.inject_time_hier:>8.1f}s")
    print(f"{'Consolidation Time':<25} {'—':<12} {report.consol_time:>8.1f}s")
    print(f"{'Judge Time':<25} {report.judge_time:>8.1f}s")

    # Per-question detail
    print(f"\n{'ID':<12} {'Cat':<20} {'Diff':<8} {'Flat':>5} {'Hier':>5} {'Delta':>6}")
    print("-" * 60)
    for r in report.results:
        d = r.score_hier - r.score_flat
        marker = f"+{d}" if d > 0 else str(d)
        print(f"{r.qid:<12} {r.category:<20} {r.difficulty:<8} {r.score_flat:>5} {r.score_hier:>5} {marker:>6}")

    print("=" * 72)


def save_report(report: EvalReport, output_path: Path):
    """Save detailed report as JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "flat_overall": report.flat_overall,
        "hier_overall": report.hier_overall,
        "delta": report.hier_overall - report.flat_overall,
        "flat_by_category": report.flat_by_cat,
        "hier_by_category": report.hier_by_cat,
        "flat_memory_count": report.flat_memory_count,
        "hier_memory_count": report.hier_memory_count,
        "hier_episode_count": report.hier_episode_count,
        "hier_schema_count": report.hier_schema_count,
        "inject_time_flat_s": report.inject_time_flat,
        "inject_time_hier_s": report.inject_time_hier,
        "consolidation_time_s": report.consol_time,
        "judge_time_s": report.judge_time,
        "questions": [
            {
                "id": r.qid,
                "category": r.category,
                "difficulty": r.difficulty,
                "query": r.query,
                "expected": r.expected,
                "score_flat": r.score_flat,
                "score_hier": r.score_hier,
                "reason_flat": r.reason_flat,
                "reason_hier": r.reason_hier,
                "context_flat": r.context_flat[:500],
                "context_hier": r.context_hier[:500],
            }
            for r in report.results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {output_path}")


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QMemory LoCoMo Eval")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-only", action="store_true",
                        help="Skip injection, only run judge on existing data")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--categories", default="",
                        help="Comma-separated categories to evaluate, e.g. temporal,multi-hop")
    parser.add_argument("--question-limit", type=int, default=0,
                        help="Cap number of evaluated questions after filtering")
    parser.add_argument("--user-suffix", default="",
                        help="Append a suffix to eval user ids to avoid clobbering existing runs")
    args = parser.parse_args()

    base_url = args.base_url
    output = Path(args.output) if args.output else Path(__file__).parent / "results" / f"locomo_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    user_flat = USER_FLAT + (f"_{args.user_suffix}" if args.user_suffix else "")
    user_hier = USER_HIER + (f"_{args.user_suffix}" if args.user_suffix else "")

    print(f"QMemory LoCoMo Eval")
    print(f"  Server: {base_url}")
    print(f"  Judge:  DeepSeek ({DEEPSEEK_MODEL})")
    print(f"  Output: {output}")
    print(f"  Users:  flat={user_flat} hier={user_hier}")

    # Health check
    try:
        resp = httpx.get(f"{base_url}/v1/health/", timeout=10, trust_env=False)
        health = resp.json()
        print(f"  Server: {health.get('status')} (v{health.get('version')}, {health.get('memory_count')} memories)")
    except Exception as e:
        print(f"  ERROR: Cannot connect to server: {e}")
        sys.exit(1)

    # Load dataset
    ds = load_dataset()
    sessions = ds["sessions"]
    questions = ds["questions"]
    selected_categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    if selected_categories:
        wanted = set(selected_categories)
        questions = [q for q in questions if q.get("category") in wanted]
    if args.question_limit > 0:
        questions = questions[:args.question_limit]
    print(f"  Dataset: {ds['name']} v{ds['version']} — {len(sessions)} sessions, {len(questions)} questions")

    report = EvalReport()

    if not args.judge_only:
        # ── Phase 1: Flat mode ──────────────────────────────────
        print(f"\n{'='*50}")
        print("PHASE 1: FLAT MODE (hierarchy=false)")
        print(f"{'='*50}")

        print("\n[1.1] Cleaning up...")
        cleanup_user(base_url, user_flat)

        print("\n[1.2] Injecting sessions...")
        report.inject_time_flat = inject_sessions(base_url, user_flat, sessions)
        print(f"  Done in {report.inject_time_flat:.1f}s")

        stats_flat = get_stats(base_url, user_flat)
        report.flat_memory_count = stats_flat["memories"]
        print(f"  Flat stats: {stats_flat}")

        # ── Phase 2: Hierarchy mode ─────────────────────────────
        print(f"\n{'='*50}")
        print("PHASE 2: HIERARCHY MODE (hierarchy=true + consolidation)")
        print(f"{'='*50}")

        print("\n[2.1] Cleaning up...")
        cleanup_user(base_url, user_hier)

        print("\n[2.2] Injecting sessions...")
        report.inject_time_hier = inject_sessions(base_url, user_hier, sessions)
        print(f"  Done in {report.inject_time_hier:.1f}s")

        print("\n[2.3] Running consolidation (L1→L2→L3)...")
        report.consol_time = run_consolidation(base_url, user_hier)
        print(f"  Done in {report.consol_time:.1f}s")

        stats_hier = get_stats(base_url, user_hier)
        report.hier_memory_count = stats_hier["memories"]
        report.hier_episode_count = stats_hier["episodes"]
        report.hier_schema_count = stats_hier["schemas"]
        print(f"  Hierarchy stats: {stats_hier}")

    # ── Phase 3: Search + Judge ─────────────────────────────────
    print(f"\n{'='*50}")
    print("PHASE 3: SEARCH + JUDGE")
    print(f"{'='*50}")

    print("\n[3.1] Searching (flat)...")
    flat_ctx = search_questions(base_url, user_flat, questions, hierarchy=False)

    print("\n[3.2] Searching (hierarchy)...")
    hier_ctx = search_questions(base_url, user_hier, questions, hierarchy=True)

    print("\n[3.3] Judging (50 questions × 2 modes = 100 judge calls)...")
    t0 = time.time()
    report.results = judge_all(questions, flat_ctx, hier_ctx)
    report.judge_time = time.time() - t0

    # Compute aggregates (preserve timing data)
    agg = compute_report(report.results)
    report.flat_overall = agg.flat_overall
    report.hier_overall = agg.hier_overall
    report.flat_by_cat = agg.flat_by_cat
    report.hier_by_cat = agg.hier_by_cat

    print_report(report)
    save_report(report, output)

    print(f"\nDone! Total time: injection + consolidation + judge")


if __name__ == "__main__":
    main()
