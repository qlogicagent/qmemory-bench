#!/usr/bin/env python3
"""
QMemory 综合评测引擎 v2.0 — 6大改进方向集成

改进方向:
  1. 真实 LoCoMo 公开数据集 (locomo10.json from HuggingFace)
  2. 偏好漂移测试集 (preference_drift.json — 50题)
  3. 隐式记忆测试集 (implicit_memory.json — 30题)
  4. Precision 指标 (每题评估精确率, 不只看召回)
  5. 人工校准接口 (输出可供人工比对的JSON)
  6. 综合跑分 (多数据集聚合, 可与公开 benchmark 对比)

Usage:
    python eval_comprehensive.py                           # 全部数据集标准模式
    python eval_comprehensive.py --datasets locomo,pref    # 指定数据集
    python eval_comprehensive.py --quick                   # 快速模式 (每集≤15题)
    python eval_comprehensive.py --judge-only              # 仅评分
    python eval_comprehensive.py --human-calibration       # 输出人工校准文件
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ── Config ──────────────────────────────────────────────────────
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "http://127.0.0.1:18800"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

DATASET_REGISTRY = {
    "locomo": {
        "file": "locomo_standard.json",
        "name": "LoCoMo (自建中文)",
        "type": "standard",
    },
    "pref": {
        "file": "preference_drift.json",
        "name": "偏好漂移",
        "type": "standard",
    },
    "implicit": {
        "file": "implicit_memory.json",
        "name": "隐式记忆",
        "type": "standard",
    },
    "real_locomo": {
        "file": None,  # loaded from HuggingFace
        "name": "真实 LoCoMo (英文)",
        "type": "real_locomo",
    },
}

CATEGORY_META = {
    # recall dimensions
    "recall-accuracy":     {"weight": 1.0, "dim": "recall"},
    "single-fact":         {"weight": 1.0, "dim": "recall"},
    "preference-extract":  {"weight": 1.0, "dim": "recall"},
    "implicit-family":     {"weight": 1.0, "dim": "implicit"},
    "implicit-career":     {"weight": 1.0, "dim": "implicit"},
    "implicit-emotion":    {"weight": 1.0, "dim": "implicit"},
    "implicit-health":     {"weight": 1.0, "dim": "implicit"},
    "implicit-finance":    {"weight": 1.0, "dim": "implicit"},
    "implicit-location":   {"weight": 1.0, "dim": "implicit"},
    # reasoning
    "multi-hop":           {"weight": 1.2, "dim": "reasoning"},
    "open-ended":          {"weight": 1.2, "dim": "reasoning"},
    "logical-reasoning":   {"weight": 1.2, "dim": "reasoning"},
    # temporal
    "temporal":            {"weight": 1.0, "dim": "temporal"},
    # drift & conflict
    "preference-drift":    {"weight": 1.3, "dim": "drift"},
    "preference-conflict": {"weight": 1.3, "dim": "drift"},
    # robustness
    "noise-resist":        {"weight": 1.0, "dim": "robustness"},
    "adversarial":         {"weight": 1.2, "dim": "robustness"},
}


def require_deepseek_key() -> str:
    if DEEPSEEK_KEY:
        return DEEPSEEK_KEY
    raise RuntimeError("DEEPSEEK_API_KEY is required for comprehensive evaluation.")


# ── Data classes ────────────────────────────────────────────────
@dataclass
class QuestionResult:
    qid: str
    dataset: str
    category: str
    question: str
    expected: str
    score_flat: int = 0
    score_hier: int = 0
    reason_flat: str = ""
    reason_hier: str = ""
    precision_flat: float = 1.0
    precision_hier: float = 1.0
    context_flat: str = ""
    context_hier: str = ""
    is_adversarial: bool = False


@dataclass
class DatasetResult:
    name: str
    dataset_id: str
    flat_recall: float = 0.0
    hier_recall: float = 0.0
    flat_precision: float = 0.0
    hier_precision: float = 0.0
    question_count: int = 0
    by_category: dict[str, dict] = field(default_factory=dict)
    inject_time: float = 0.0
    consol_time: float = 0.0


@dataclass
class ComprehensiveReport:
    timestamp: str = ""
    datasets: list[DatasetResult] = field(default_factory=list)
    all_results: list[QuestionResult] = field(default_factory=list)
    # aggregated
    overall_flat_recall: float = 0.0
    overall_hier_recall: float = 0.0
    overall_flat_precision: float = 0.0
    overall_hier_precision: float = 0.0
    # by dimension
    by_dim: dict[str, dict] = field(default_factory=dict)
    total_time: float = 0.0


# ── LLM Judge ──────────────────────────────────────────────────
def judge_question(question: str, expected: str, context: str,
                   is_adversarial: bool = False) -> tuple[int, str, float]:
    """
    LLM judge: Returns (score 0-10, reason, precision 0.0-1.0).

    方向 4: Precision 指标 — 除了评分还评估召回内容中无关信息的比例.
    """
    if is_adversarial:
        system = """你是记忆系统评测员。此题为对抗性问题(adversarial)。
正确行为是系统不应该被误导。

评分标准 (0-10):
- 10: 正确识别陷阱或表示无相关信息
- 7-9: 未被误导, 给出正确信息
- 4-6: 部分被误导
- 0-3: 完全被误导

精确率 (precision 0.0-1.0):
- 1.0: 召回内容全部与问题相关
- 0.0: 全部无关

输出JSON: {"score": <0-10>, "reason": "<理由>", "precision": <0.0-1.0>}"""
    else:
        system = """你是记忆系统评测员。根据问题、期望答案、召回内容打分。

召回得分 (score 0-10):
- 10: 完整覆盖期望答案
- 8-9: 覆盖大部分
- 5-7: 覆盖部分
- 3-4: 仅少量
- 1-2: 微弱关联
- 0: 无关或错误

精确率 (precision 0.0-1.0):
- 1.0: 召回全部相关
- 0.7: 大部分相关
- 0.3: 大部分无关
- 0.0: 全部无关

规则:
1. 语义等价即可
2. 张冠李戴(A的事说成B的)严格扣分
3. 精确率独立于召回分: 即使score高, 夹杂大量无关内容则precision低
4. context为空时score=0, precision=1.0(没错误但也没信息)

输出JSON: {"score": <0-10>, "reason": "<理由>", "precision": <0.0-1.0>}"""

    user_msg = f"""## 问题
{question}

## 期望答案
{expected}

## 系统召回内容
{context if context else "(空 — 无召回)"}

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
                "max_tokens": 300,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        obj = json.loads(text)
        return (
            int(obj["score"]),
            obj.get("reason", ""),
            float(obj.get("precision", 0.7)),
        )
    except Exception as e:
        return 0, f"judge error: {e}", 0.5


# ── QMemory API ─────────────────────────────────────────────────
def llm_config() -> dict:
    return {
        "provider": "openai_compat",
        "api_key": require_deepseek_key(),
        "base_url": DEEPSEEK_BASE,
        "model": DEEPSEEK_MODEL,
    }


def api_cleanup(base_url: str, user_id: str):
    try:
        r = httpx.delete(f"{base_url}/v1/memories/",
                         params={"user_id": user_id, "confirm": "true"}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            print(f"    Cleaned {user_id}: {d.get('memories_deleted', 0)} memories")
    except Exception as e:
        print(f"    Cleanup error: {e}")


def api_inject(base_url: str, user_id: str, sessions: list[dict]) -> float:
    t0 = time.time()
    for i, s in enumerate(sessions, 1):
        try:
            r = httpx.post(f"{base_url}/v1/memories/", json={
                "messages": s["messages"],
                "user_id": user_id,
                "session_id": s["id"],
                "llm_config": llm_config(),
            }, timeout=180)
            if r.status_code == 200:
                data = r.json()
                added = "?"
                if data.get("results"):
                    added = data["results"][0].get("memories_added", "?")
                elif "memories_added" in data:
                    added = data["memories_added"]
                if i % 5 == 0 or i == len(sessions):
                    print(f"    Injected {i}/{len(sessions)} sessions")
            else:
                print(f"    Session {s['id']}: HTTP {r.status_code}")
        except Exception as e:
            print(f"    Session {s['id']}: ERROR {e}")
    return time.time() - t0


def api_consolidate(base_url: str, user_id: str) -> float:
    t0 = time.time()
    try:
        r = httpx.post(f"{base_url}/v1/admin/consolidate", json={
            "user_id": user_id, "min_memories": 3, "min_episodes": 2,
        }, timeout=600)
        if r.status_code == 200:
            d = r.json()
            print(f"    Consolidation: ep={d.get('episodes_created', 0)} "
                  f"sch={d.get('schemas_created', 0)}")
    except Exception as e:
        print(f"    Consolidation error: {e}")
    return time.time() - t0


def api_search(base_url: str, user_id: str, query: str, hierarchy: bool) -> str:
    try:
        r = httpx.get(f"{base_url}/v1/memories/search/", params={
            "q": query, "user_id": user_id, "limit": 10,
            "hierarchy": str(hierarchy).lower(),
        }, timeout=120)
        if r.status_code == 200:
            return r.json().get("context", "")
    except:
        pass
    return ""


def api_stats(base_url: str, user_id: str) -> dict:
    stats = {"memories": 0, "episodes": 0, "schemas": 0}
    try:
        r = httpx.get(f"{base_url}/v1/memories/",
                      params={"user_id": user_id, "page_size": 1}, timeout=15)
        if r.status_code == 200:
            stats["memories"] = r.json().get("total", 0)
    except:
        pass
    try:
        r = httpx.get(f"{base_url}/v1/admin/overview",
                      params={"user_id": user_id}, timeout=15)
        if r.status_code == 200:
            d = r.json()
            stats["episodes"] = d.get("episodes", 0)
            stats["schemas"] = d.get("schemas", 0)
    except:
        pass
    return stats


# ── Dataset loaders ─────────────────────────────────────────────
def load_standard_dataset(ds_id: str) -> tuple[list[dict], list[dict]]:
    """Load a standard JSON dataset. Returns (sessions, questions)."""
    info = DATASET_REGISTRY[ds_id]
    path = DATA_DIR / info["file"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = data["sessions"]
    questions = data["questions"]

    # Normalize question format
    normalized = []
    for q in questions:
        normalized.append({
            "id": q.get("id", q.get("qid", f"{ds_id}-{len(normalized)}")),
            "category": q.get("category", "general"),
            "question": q.get("query", q.get("question", "")),
            "expected": q.get("expected", q.get("answer", "")),
            "is_adversarial": q.get("is_adversarial", q.get("category") == "adversarial"),
        })

    return sessions, normalized


def load_real_locomo(sample_idx: int = 0, max_q: int = 50) -> tuple[list[dict], list[dict]]:
    """Load real LoCoMo from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("KimmoZZZ/locomo", "locomo10.json", repo_type="dataset")
    except ImportError:
        local = DATA_DIR / "locomo10.json"
        if not local.exists():
            raise FileNotFoundError("huggingface_hub not installed and locomo10.json not found")
        path = str(local)

    with open(path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    sample = all_data[sample_idx]
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]

    # Convert sessions
    sess_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda k: int(k.split("_")[1])
    )
    sessions = []
    for sk in sess_keys:
        messages = []
        for turn in conv[sk]:
            role = "user" if turn["speaker"] == speaker_a else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        sessions.append({
            "id": sk, "date": conv.get(sk + "_date_time", ""),
            "messages": messages,
        })

    # Convert questions
    cat_map = {1: "single-fact", 2: "temporal", 3: "open-ended", 4: "multi-hop", 5: "adversarial"}
    questions = []
    for i, qa in enumerate(sample["qa"]):
        cat_id = qa["category"]
        expected = str(qa.get("answer", qa.get("adversarial_answer", "")))
        questions.append({
            "id": f"rlocomo-{sample['sample_id']}-{i:03d}",
            "category": cat_map.get(cat_id, f"unknown-{cat_id}"),
            "question": qa["question"],
            "expected": expected,
            "is_adversarial": cat_id == 5,
        })

    if max_q and len(questions) > max_q:
        random.seed(42)
        by_cat: dict[str, list] = {}
        for q in questions:
            by_cat.setdefault(q["category"], []).append(q)
        sampled = []
        total = len(questions)
        for cat, qs in by_cat.items():
            n = max(1, int(len(qs) / total * max_q))
            sampled.extend(random.sample(qs, min(n, len(qs))))
        remaining = [q for q in questions if q not in sampled]
        random.shuffle(remaining)
        while len(sampled) < max_q and remaining:
            sampled.append(remaining.pop())
        questions = sampled[:max_q]

    return sessions, questions


# ── Eval pipeline ───────────────────────────────────────────────
def eval_dataset(
    base_url: str,
    ds_id: str,
    sessions: list[dict],
    questions: list[dict],
    skip_injection: bool = False,
    quick_mode: bool = False,
) -> tuple[DatasetResult, list[QuestionResult]]:
    """Evaluate one dataset. Returns (DatasetResult, list of QuestionResult)."""
    ds_name = DATASET_REGISTRY.get(ds_id, {}).get("name", ds_id)
    user_flat = f"compeval_{ds_id}_flat"
    user_hier = f"compeval_{ds_id}_hier"

    if quick_mode and len(questions) > 15:
        random.seed(42)
        questions = random.sample(questions, 15)

    ds_result = DatasetResult(name=ds_name, dataset_id=ds_id, question_count=len(questions))
    q_results: list[QuestionResult] = []

    print(f"\n{'─'*60}")
    print(f"  Dataset: {ds_name} ({ds_id})")
    print(f"  Sessions: {len(sessions)}, Questions: {len(questions)}")
    print(f"{'─'*60}")

    if not skip_injection:
        # Flat
        print(f"\n  [Flat] Cleanup + Inject")
        api_cleanup(base_url, user_flat)
        ds_result.inject_time = api_inject(base_url, user_flat, sessions)

        # Hierarchy
        print(f"  [Hier] Cleanup + Inject + Consolidate")
        api_cleanup(base_url, user_hier)
        api_inject(base_url, user_hier, sessions)
        ds_result.consol_time = api_consolidate(base_url, user_hier)

        stats_f = api_stats(base_url, user_flat)
        stats_h = api_stats(base_url, user_hier)
        print(f"  Stats: flat={stats_f}, hier={stats_h}")

    # Search + Judge
    print(f"\n  [Judge] {len(questions)} questions × 2 modes")
    for i, q in enumerate(questions, 1):
        ctx_flat = api_search(base_url, user_flat, q["question"], hierarchy=False)
        ctx_hier = api_search(base_url, user_hier, q["question"], hierarchy=True)

        sf, rf, pf = judge_question(q["question"], q["expected"], ctx_flat, q["is_adversarial"])
        sh, rh, ph = judge_question(q["question"], q["expected"], ctx_hier, q["is_adversarial"])

        qr = QuestionResult(
            qid=q["id"], dataset=ds_id, category=q["category"],
            question=q["question"], expected=q["expected"],
            score_flat=sf, score_hier=sh,
            reason_flat=rf, reason_hier=rh,
            precision_flat=pf, precision_hier=ph,
            context_flat=ctx_flat[:500], context_hier=ctx_hier[:500],
            is_adversarial=q["is_adversarial"],
        )
        q_results.append(qr)

        marker = "+" if sh > sf else ("=" if sh == sf else "-")
        if i % 5 == 0 or i == len(questions):
            print(f"    [{i}/{len(questions)}] last: {q['category']:<18} f={sf} h={sh} [{marker}]")

    # Compute aggregates
    if q_results:
        ds_result.flat_recall = sum(r.score_flat for r in q_results) / len(q_results) * 10
        ds_result.hier_recall = sum(r.score_hier for r in q_results) / len(q_results) * 10
        ds_result.flat_precision = sum(r.precision_flat for r in q_results) / len(q_results)
        ds_result.hier_precision = sum(r.precision_hier for r in q_results) / len(q_results)

        cats: dict[str, list[QuestionResult]] = {}
        for r in q_results:
            cats.setdefault(r.category, []).append(r)
        for cat, rs in cats.items():
            ds_result.by_category[cat] = {
                "count": len(rs),
                "flat_recall": sum(r.score_flat for r in rs) / len(rs) * 10,
                "hier_recall": sum(r.score_hier for r in rs) / len(rs) * 10,
                "flat_precision": sum(r.precision_flat for r in rs) / len(rs),
                "hier_precision": sum(r.precision_hier for r in rs) / len(rs),
            }

    return ds_result, q_results


# ── Report generation ───────────────────────────────────────────
def compute_comprehensive(
    ds_results: list[DatasetResult],
    all_q: list[QuestionResult],
) -> ComprehensiveReport:
    report = ComprehensiveReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        datasets=ds_results,
        all_results=all_q,
    )

    if all_q:
        report.overall_flat_recall = sum(r.score_flat for r in all_q) / len(all_q) * 10
        report.overall_hier_recall = sum(r.score_hier for r in all_q) / len(all_q) * 10
        report.overall_flat_precision = sum(r.precision_flat for r in all_q) / len(all_q)
        report.overall_hier_precision = sum(r.precision_hier for r in all_q) / len(all_q)

    # By dimension (using CATEGORY_META)
    dims: dict[str, list[QuestionResult]] = {}
    for r in all_q:
        meta = CATEGORY_META.get(r.category, {"dim": "other"})
        dim = meta["dim"]
        dims.setdefault(dim, []).append(r)

    for dim, rs in dims.items():
        weights = [CATEGORY_META.get(r.category, {}).get("weight", 1.0) for r in rs]
        total_w = sum(weights)
        report.by_dim[dim] = {
            "count": len(rs),
            "flat_recall": sum(r.score_flat * w for r, w in zip(rs, weights)) / total_w * 10 if total_w else 0,
            "hier_recall": sum(r.score_hier * w for r, w in zip(rs, weights)) / total_w * 10 if total_w else 0,
            "flat_precision": sum(r.precision_flat * w for r, w in zip(rs, weights)) / total_w if total_w else 0,
            "hier_precision": sum(r.precision_hier * w for r, w in zip(rs, weights)) / total_w if total_w else 0,
        }

    report.total_time = sum(d.inject_time + d.consol_time for d in ds_results)
    return report


def print_comprehensive(report: ComprehensiveReport):
    print(f"\n{'='*72}")
    print(f"  QMemory 综合评测报告 v2.0")
    print(f"  时间: {report.timestamp}")
    print(f"  总题数: {len(report.all_results)}")
    print(f"{'='*72}")

    # Overall
    print(f"\n  {'指标':<22} {'Flat':<14} {'Hierarchy':<14} {'Δ':<10}")
    print(f"  {'-'*60}")
    print(f"  {'Recall Score':<22} {report.overall_flat_recall:>8.1f}%     "
          f"{report.overall_hier_recall:>8.1f}%     "
          f"{report.overall_hier_recall - report.overall_flat_recall:>+6.1f}%")
    print(f"  {'Precision':<22} {report.overall_flat_precision:>8.1%}     "
          f"{report.overall_hier_precision:>8.1%}     "
          f"{report.overall_hier_precision - report.overall_flat_precision:>+6.1%}")

    # By dataset
    print(f"\n  ── 分数据集 ──")
    for ds in report.datasets:
        delta_r = ds.hier_recall - ds.flat_recall
        print(f"  {ds.name:<22} R: {ds.flat_recall:.1f}→{ds.hier_recall:.1f}% (Δ{delta_r:+.1f}) "
              f" P: {ds.flat_precision:.0%}→{ds.hier_precision:.0%}")

    # By dimension
    print(f"\n  ── 分维度 ──")
    for dim, stats in sorted(report.by_dim.items()):
        fr = stats["flat_recall"]
        hr = stats["hier_recall"]
        print(f"  {dim:<22} R: {fr:.1f}→{hr:.1f}% (Δ{hr-fr:+.1f}) "
              f" P: {stats['flat_precision']:.0%}→{stats['hier_precision']:.0%} "
              f" [{stats['count']}题]")

    print(f"\n  Total eval time: {report.total_time:.0f}s")
    print(f"{'='*72}")


def save_comprehensive(report: ComprehensiveReport, output: Path):
    """Save full report as JSON."""
    data: dict[str, Any] = {
        "type": "comprehensive_eval_v2",
        "timestamp": report.timestamp,
        "overall": {
            "flat_recall": report.overall_flat_recall,
            "hier_recall": report.overall_hier_recall,
            "delta_recall": report.overall_hier_recall - report.overall_flat_recall,
            "flat_precision": report.overall_flat_precision,
            "hier_precision": report.overall_hier_precision,
            "total_questions": len(report.all_results),
        },
        "by_dimension": report.by_dim,
        "datasets": [
            {
                "id": ds.dataset_id,
                "name": ds.name,
                "flat_recall": ds.flat_recall,
                "hier_recall": ds.hier_recall,
                "flat_precision": ds.flat_precision,
                "hier_precision": ds.hier_precision,
                "question_count": ds.question_count,
                "by_category": ds.by_category,
                "inject_time_s": ds.inject_time,
                "consol_time_s": ds.consol_time,
            }
            for ds in report.datasets
        ],
        "questions": [
            {
                "id": r.qid,
                "dataset": r.dataset,
                "category": r.category,
                "question": r.question,
                "expected": r.expected,
                "score_flat": r.score_flat,
                "score_hier": r.score_hier,
                "reason_flat": r.reason_flat,
                "reason_hier": r.reason_hier,
                "precision_flat": r.precision_flat,
                "precision_hier": r.precision_hier,
                "context_flat": r.context_flat,
                "context_hier": r.context_hier,
                "is_adversarial": r.is_adversarial,
            }
            for r in report.all_results
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {output}")


def export_human_calibration(report: ComprehensiveReport, output: Path):
    """
    方向 5: 人工校准接口 — 导出可供人工逐题打分的 JSON.

    人工打分后, 可对比 LLM judge 与人工评分的相关性.
    """
    random.seed(42)
    all_q = report.all_results

    # Stratified sample: 50 questions (or all if < 50)
    sample_size = min(50, len(all_q))
    by_cat: dict[str, list] = {}
    for r in all_q:
        by_cat.setdefault(r.category, []).append(r)

    sampled = []
    for cat, rs in by_cat.items():
        n = max(1, int(len(rs) / len(all_q) * sample_size))
        sampled.extend(random.sample(rs, min(n, len(rs))))
    remaining = [r for r in all_q if r not in sampled]
    random.shuffle(remaining)
    while len(sampled) < sample_size and remaining:
        sampled.append(remaining.pop())

    calibration_items = []
    for r in sampled[:sample_size]:
        calibration_items.append({
            "id": r.qid,
            "category": r.category,
            "question": r.question,
            "expected_answer": r.expected,
            "system_context_hier": r.context_hier,
            "llm_judge_score": r.score_hier,
            "llm_judge_reason": r.reason_hier,
            "llm_judge_precision": r.precision_hier,
            # Fields for human to fill:
            "human_score": None,          # 人工评分 0-10
            "human_precision": None,      # 人工精确率 0.0-1.0
            "human_notes": "",            # 人工备注
        })

    data = {
        "type": "human_calibration",
        "timestamp": report.timestamp,
        "instructions": (
            "请对每题打分:\n"
            "1. human_score: 0-10, 根据系统召回内容是否覆盖了期望答案\n"
            "2. human_precision: 0.0-1.0, 召回内容中相关信息占比\n"
            "3. human_notes: 可选备注\n"
            "\n打分后运行 python eval_comprehensive.py --calc-calibration <file> 计算校准结果"
        ),
        "sample_count": len(calibration_items),
        "items": calibration_items,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nHuman calibration file: {output}")
    print(f"  {len(calibration_items)} questions sampled for human review")


def calc_calibration(cal_file: Path):
    """Compare LLM judge scores with human scores in a calibration file."""
    with open(cal_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    pairs_score = []
    pairs_prec = []
    for item in items:
        if item["human_score"] is not None:
            pairs_score.append((item["llm_judge_score"], item["human_score"]))
        if item["human_precision"] is not None:
            pairs_prec.append((item["llm_judge_precision"], item["human_precision"]))

    if not pairs_score:
        print("No human scores found. Please fill in human_score fields first.")
        return

    # Pearson correlation
    def pearson(pairs):
        n = len(pairs)
        if n < 3:
            return 0.0
        sx = sum(p[0] for p in pairs)
        sy = sum(p[1] for p in pairs)
        sxx = sum(p[0]**2 for p in pairs)
        syy = sum(p[1]**2 for p in pairs)
        sxy = sum(p[0]*p[1] for p in pairs)
        denom = ((n*sxx - sx**2) * (n*syy - sy**2)) ** 0.5
        if denom == 0:
            return 0.0
        return (n*sxy - sx*sy) / denom

    # MAE
    def mae(pairs):
        return sum(abs(p[0] - p[1]) for p in pairs) / len(pairs)

    r_score = pearson(pairs_score)
    mae_score = mae(pairs_score)
    avg_llm = sum(p[0] for p in pairs_score) / len(pairs_score)
    avg_human = sum(p[1] for p in pairs_score) / len(pairs_score)

    print(f"\n{'='*50}")
    print(f"  人工校准结果")
    print(f"{'='*50}")
    print(f"  样本数:           {len(pairs_score)}")
    print(f"  LLM平均分:        {avg_llm:.2f}")
    print(f"  人工平均分:       {avg_human:.2f}")
    print(f"  Pearson相关系数:  {r_score:.3f}")
    print(f"  MAE (平均绝对误差): {mae_score:.2f}")
    print(f"  偏差方向:         {'LLM偏高' if avg_llm > avg_human else 'LLM偏低'}")

    if pairs_prec:
        r_prec = pearson(pairs_prec)
        print(f"\n  Precision 校准:")
        print(f"    Pearson: {r_prec:.3f}")
        print(f"    MAE: {mae(pairs_prec):.3f}")

    # Suggest calibration factor
    if avg_human > 0:
        cal_factor = avg_human / avg_llm if avg_llm > 0 else 1.0
        print(f"\n  建议校准系数: {cal_factor:.3f}")
        print(f"  (将 LLM 分数 × {cal_factor:.3f} ≈ 人工分数)")
    print(f"{'='*50}")


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QMemory Comprehensive Eval v2.0")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--datasets", default="locomo,pref,implicit",
                        help="Comma-separated: locomo,pref,implicit,real_locomo")
    parser.add_argument("--quick", action="store_true", help="Quick mode (≤15 per dataset)")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--human-calibration", action="store_true",
                        help="Export human calibration file after eval")
    parser.add_argument("--calc-calibration", default=None,
                        help="Calculate calibration from human-scored file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--real-locomo-sample", type=int, default=0)
    parser.add_argument("--real-locomo-max-q", type=int, default=50)
    args = parser.parse_args()

    # Handle calibration calculation
    if args.calc_calibration:
        calc_calibration(Path(args.calc_calibration))
        return

    base_url = args.base_url

    # Health check
    try:
        r = httpx.get(f"{base_url}/v1/health/", timeout=10)
        health = r.json()
        print(f"Server: {health.get('status')} v{health.get('version')} "
              f"({health.get('memory_count')} memories)")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        sys.exit(1)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_ds_results: list[DatasetResult] = []
    all_q_results: list[QuestionResult] = []

    t_start = time.time()

    for ds_id in datasets:
        if ds_id not in DATASET_REGISTRY:
            print(f"  WARN: Unknown dataset '{ds_id}', skipping")
            continue

        if ds_id == "real_locomo":
            sessions, questions = load_real_locomo(
                args.real_locomo_sample, args.real_locomo_max_q)
        else:
            sessions, questions = load_standard_dataset(ds_id)

        ds_result, q_results = eval_dataset(
            base_url, ds_id, sessions, questions,
            skip_injection=args.judge_only,
            quick_mode=args.quick,
        )

        all_ds_results.append(ds_result)
        all_q_results.extend(q_results)

    # ── Comprehensive report ──
    report = compute_comprehensive(all_ds_results, all_q_results)
    report.total_time = time.time() - t_start

    print_comprehensive(report)

    output = Path(args.output) if args.output else (
        RESULTS_DIR / f"comprehensive_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_comprehensive(report, output)

    # ── Human calibration ──
    if args.human_calibration:
        cal_output = output.with_name(output.stem + "_human_calibration.json")
        export_human_calibration(report, cal_output)

    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
