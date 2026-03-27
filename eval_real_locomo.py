#!/usr/bin/env python3
"""
真实 LoCoMo 数据集评测脚本 — 方向 1: 引入公开数据集

从 HuggingFace 加载 KimmoZZZ/locomo (locomo10.json), 自动转换为
QMemory bench 格式, 注入 → consolidation → search → judge.

数据集分类映射 (LoCoMo原始):
  1: single-fact    (单事实检索)
  2: temporal        (时间推理)
  3: open-ended      (推理/反事实)
  4: multi-hop       (多跳推理)
  5: adversarial     (对抗题, 有 adversarial_answer, 无 answer)

Usage:
    python eval_real_locomo.py [--sample 0] [--max-questions 50]
    python eval_real_locomo.py --all-samples --max-questions 30
    python eval_real_locomo.py --judge-only --result-file results/xxx.json
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
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "http://127.0.0.1:18800"

CATEGORY_MAP = {
    1: "single-fact",
    2: "temporal",
    3: "open-ended",
    4: "multi-hop",
    5: "adversarial",
}


def require_deepseek_key() -> str:
    if DEEPSEEK_KEY:
        return DEEPSEEK_KEY
    raise RuntimeError("DEEPSEEK_API_KEY is required for real LoCoMo evaluation.")


# ── Data classes ────────────────────────────────────────────────
@dataclass
class QResult:
    qid: str
    category: str
    question: str
    expected: str
    evidence: list[str]
    is_adversarial: bool = False
    context_flat: str = ""
    context_hier: str = ""
    score_flat: int = 0
    score_hier: int = 0
    reason_flat: str = ""
    reason_hier: str = ""
    # precision extras
    precision_flat: float = 1.0  # 1.0 = no false positives
    precision_hier: float = 1.0


@dataclass
class RealLocomoReport:
    sample_id: str = ""
    speakers: tuple[str, str] = ("", "")
    session_count: int = 0
    total_qa: int = 0
    results: list[QResult] = field(default_factory=list)
    flat_overall: float = 0.0
    hier_overall: float = 0.0
    flat_by_cat: dict[str, float] = field(default_factory=dict)
    hier_by_cat: dict[str, float] = field(default_factory=dict)
    flat_precision: float = 0.0
    hier_precision: float = 0.0
    inject_time: float = 0.0
    consol_time: float = 0.0
    judge_time: float = 0.0
    flat_memory_count: int = 0
    hier_memory_count: int = 0
    hier_episode_count: int = 0
    hier_schema_count: int = 0


# ── Load real LoCoMo ────────────────────────────────────────────
def load_locomo(sample_idx: int | None = None) -> list[dict]:
    """Load LoCoMo from HuggingFace cache or download."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("KimmoZZZ/locomo", "locomo10.json", repo_type="dataset")
    except ImportError:
        # Fallback: look in local data dir
        local = Path(__file__).parent / "data" / "locomo10.json"
        if local.exists():
            path = str(local)
        else:
            print("ERROR: huggingface_hub not installed and locomo10.json not found locally")
            sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if sample_idx is not None:
        return [data[sample_idx]]
    return data


def convert_sessions(conversation: dict) -> list[dict]:
    """Convert LoCoMo conversation format → QMemory session list."""
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    # Find all session keys
    sess_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda k: int(k.split("_")[1])
    )

    sessions = []
    for sk in sess_keys:
        turns = conversation[sk]
        date_key = sk + "_date_time"
        date_str = conversation.get(date_key, "")

        messages = []
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            # Map speaker to role: speaker_a = user, speaker_b = assistant
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({"role": role, "content": text})

        sessions.append({
            "id": sk,
            "date": date_str,
            "messages": messages,
        })

    return sessions


def convert_questions(qa_list: list[dict], max_questions: int = 0) -> list[dict]:
    """Convert LoCoMo QA → bench question format."""
    questions = []
    for i, qa in enumerate(qa_list):
        cat_id = qa["category"]
        cat_name = CATEGORY_MAP.get(cat_id, f"unknown-{cat_id}")
        is_adv = cat_id == 5

        # Adversarial questions have adversarial_answer instead of answer
        expected = qa.get("answer", qa.get("adversarial_answer", ""))
        expected = str(expected)

        questions.append({
            "id": f"locomo-{i:03d}",
            "category": cat_name,
            "question": qa["question"],
            "expected": expected,
            "evidence": qa.get("evidence", []),
            "is_adversarial": is_adv,
        })

    if max_questions and len(questions) > max_questions:
        # Stratified sample: proportional per category
        import random
        random.seed(42)
        by_cat: dict[str, list] = {}
        for q in questions:
            by_cat.setdefault(q["category"], []).append(q)

        sampled = []
        total = len(questions)
        for cat, qs in by_cat.items():
            n = max(1, int(len(qs) / total * max_questions))
            sampled.extend(random.sample(qs, min(n, len(qs))))

        # Fill remainder
        remaining = [q for q in questions if q not in sampled]
        random.shuffle(remaining)
        while len(sampled) < max_questions and remaining:
            sampled.append(remaining.pop())

        questions = sampled[:max_questions]

    return questions


# ── API calls ───────────────────────────────────────────────────
def llm_config_body() -> dict:
    return {
        "provider": "openai_compat",
        "api_key": require_deepseek_key(),
        "base_url": DEEPSEEK_BASE,
        "model": DEEPSEEK_MODEL,
    }


def cleanup_user(base_url: str, user_id: str):
    try:
        resp = httpx.delete(
            f"{base_url}/v1/memories/",
            params={"user_id": user_id, "confirm": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            d = resp.json()
            print(f"  Cleaned {user_id}: {d.get('memories_deleted', 0)} memories")
        else:
            print(f"  Cleanup {user_id}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Cleanup {user_id}: {e}")


def inject_sessions(base_url: str, user_id: str, sessions: list[dict]) -> float:
    t0 = time.time()
    total = len(sessions)
    for i, sess in enumerate(sessions, 1):
        body = {
            "messages": sess["messages"],
            "user_id": user_id,
            "session_id": sess["id"],
            "llm_config": llm_config_body(),
        }
        try:
            resp = httpx.post(f"{base_url}/v1/memories/", json=body, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                added = "?"
                if data.get("results"):
                    added = data["results"][0].get("memories_added", "?")
                elif "memories_added" in data:
                    added = data["memories_added"]
                print(f"  [{i}/{total}] {sess['id']} ({sess.get('date','')}): +{added} memories")
            else:
                print(f"  [{i}/{total}] {sess['id']}: HTTP {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  [{i}/{total}] {sess['id']}: ERROR {e}")
    return time.time() - t0


def run_consolidation(base_url: str, user_id: str) -> float:
    t0 = time.time()
    try:
        resp = httpx.post(
            f"{base_url}/v1/admin/consolidate",
            json={"user_id": user_id, "min_memories": 3, "min_episodes": 2},
            timeout=600,
        )
        if resp.status_code == 200:
            d = resp.json()
            print(f"  Consolidation: episodes={d.get('episodes_created', 0)}, "
                  f"schemas={d.get('schemas_created', 0)}")
        else:
            print(f"  Consolidation: HTTP {resp.status_code} - {resp.text[:300]}")
    except Exception as e:
        print(f"  Consolidation error: {e}")
    return time.time() - t0


def get_stats(base_url: str, user_id: str) -> dict:
    stats = {"memories": 0, "episodes": 0, "schemas": 0}
    try:
        resp = httpx.get(f"{base_url}/v1/memories/", params={"user_id": user_id, "page_size": 1}, timeout=15)
        if resp.status_code == 200:
            stats["memories"] = resp.json().get("total", 0)
    except:
        pass
    try:
        resp = httpx.get(f"{base_url}/v1/admin/overview", params={"user_id": user_id}, timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            stats["episodes"] = d.get("episodes", 0)
            stats["schemas"] = d.get("schemas", 0)
    except:
        pass
    return stats


def search_question(base_url: str, user_id: str, query: str, hierarchy: bool) -> str:
    try:
        resp = httpx.get(
            f"{base_url}/v1/memories/search/",
            params={"q": query, "user_id": user_id, "limit": 10, "hierarchy": str(hierarchy).lower()},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("context", "")
    except Exception as e:
        print(f"    Search error: {e}")
    return ""


# ── Judge (with precision) ──────────────────────────────────────
def deepseek_judge(question: str, expected: str, context: str,
                   is_adversarial: bool = False) -> tuple[int, str, float]:
    """Judge recall quality. Returns (score 0-10, reason, precision 0-1).

    For adversarial questions, check if system correctly resists false info.
    Precision: penalize if context contains clearly irrelevant/wrong info.
    """
    if is_adversarial:
        system = """你是记忆系统评测打分员。这是一个对抗性问题(adversarial), 正确行为是
系统不应该据此给出错误信息。

评分标准 (0-10):
- 10: 系统正确识别了问题中的陷阱/错误前提, 或表示无相关记忆
- 7-9: 系统给出了正确信息, 未被误导
- 4-6: 系统部分被误导但仍有正确信息
- 0-3: 系统完全被误导, 给出了错误信息

同时评估精确率 (precision 0.0-1.0):
- 1.0: 所有召回内容都与问题相关
- 0.5: 一半相关, 一半无关
- 0.0: 全部无关

输出严格JSON: {"score": <0-10>, "reason": "<理由>", "precision": <0.0-1.0>}"""
    else:
        system = """你是记忆系统评测打分员。根据问题、期望答案和系统召回内容打分。

评分标准 (0-10):
- 10: 完整覆盖期望答案所有要点
- 8-9: 覆盖大部分, 遗漏少量细节
- 5-7: 覆盖部分要点
- 3-4: 仅覆盖少量
- 1-2: 几乎无关但有轻微关联
- 0: 完全无关或错误

同时评估精确率 (precision 0.0-1.0):
- 1.0: 所有召回内容都与问题直接相关
- 0.7: 大部分相关, 少量噪声
- 0.5: 一半相关一半无关
- 0.3: 大部分无关
- 0.0: 全部无关

注意:
1. 语义等价即可
2. 张冠李戴严格扣分
3. 精确率独立于召回分数 — 即使分数高, 如果混入大量无关内容, precision 也应低

输出严格JSON: {"score": <0-10>, "reason": "<理由>", "precision": <0.0-1.0>}"""

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


# ── Main pipeline ───────────────────────────────────────────────
def run_eval(
    base_url: str,
    sample: dict,
    max_questions: int = 50,
    judge_only: bool = False,
    prev_result: dict | None = None,
) -> RealLocomoReport:
    """Run full eval on one LoCoMo sample."""
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]

    sessions = convert_sessions(conv)
    questions = convert_questions(sample["qa"], max_questions)

    report = RealLocomoReport(
        sample_id=sample_id,
        speakers=(speaker_a, speaker_b),
        session_count=len(sessions),
        total_qa=len(questions),
    )

    user_flat = f"locomo_real_{sample_id}_flat"
    user_hier = f"locomo_real_{sample_id}_hier"

    print(f"\n{'='*60}")
    print(f"  LoCoMo Sample: {sample_id}")
    print(f"  Speakers: {speaker_a} & {speaker_b}")
    print(f"  Sessions: {len(sessions)}, Questions: {len(questions)}")
    print(f"{'='*60}")

    if not judge_only:
        # ── Inject flat ──
        print("\n[1] FLAT: Cleanup + Inject")
        cleanup_user(base_url, user_flat)
        report.inject_time = inject_sessions(base_url, user_flat, sessions)
        print(f"  Flat inject done in {report.inject_time:.1f}s")

        # ── Inject hierarchy ──
        print("\n[2] HIERARCHY: Cleanup + Inject + Consolidate")
        cleanup_user(base_url, user_hier)
        inject_sessions(base_url, user_hier, sessions)

        print("  Running consolidation...")
        report.consol_time = run_consolidation(base_url, user_hier)
        print(f"  Consolidation done in {report.consol_time:.1f}s")

        stats_f = get_stats(base_url, user_flat)
        stats_h = get_stats(base_url, user_hier)
        report.flat_memory_count = stats_f["memories"]
        report.hier_memory_count = stats_h["memories"]
        report.hier_episode_count = stats_h["episodes"]
        report.hier_schema_count = stats_h["schemas"]
        print(f"  Flat: {stats_f}, Hier: {stats_h}")

    # ── Search + Judge ──
    print(f"\n[3] SEARCH + JUDGE ({len(questions)} questions × 2 modes)")
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        qr = QResult(
            qid=q["id"],
            category=q["category"],
            question=q["question"],
            expected=q["expected"],
            evidence=q["evidence"],
            is_adversarial=q["is_adversarial"],
        )

        # Search
        qr.context_flat = search_question(base_url, user_flat, q["question"], hierarchy=False)
        qr.context_hier = search_question(base_url, user_hier, q["question"], hierarchy=True)

        # Judge
        sf, rf, pf = deepseek_judge(q["question"], q["expected"], qr.context_flat, q["is_adversarial"])
        sh, rh, ph = deepseek_judge(q["question"], q["expected"], qr.context_hier, q["is_adversarial"])

        qr.score_flat = sf
        qr.score_hier = sh
        qr.reason_flat = rf
        qr.reason_hier = rh
        qr.precision_flat = pf
        qr.precision_hier = ph

        marker = "+" if sh > sf else ("=" if sh == sf else "-")
        print(f"  [{i}/{len(questions)}] {q['category']:<13} flat={sf} hier={sh} [{marker}] "
              f"prec={pf:.1f}/{ph:.1f}")

        report.results.append(qr)

    report.judge_time = time.time() - t0

    # ── Aggregate ──
    if report.results:
        report.flat_overall = sum(r.score_flat for r in report.results) / len(report.results) * 10
        report.hier_overall = sum(r.score_hier for r in report.results) / len(report.results) * 10
        report.flat_precision = sum(r.precision_flat for r in report.results) / len(report.results)
        report.hier_precision = sum(r.precision_hier for r in report.results) / len(report.results)

        cats: dict[str, list[QResult]] = {}
        for r in report.results:
            cats.setdefault(r.category, []).append(r)
        for cat, rs in cats.items():
            report.flat_by_cat[cat] = sum(r.score_flat for r in rs) / len(rs) * 10
            report.hier_by_cat[cat] = sum(r.score_hier for r in rs) / len(rs) * 10

    return report


def print_report(report: RealLocomoReport):
    print(f"\n{'='*72}")
    print(f"  真实 LoCoMo 评测报告 — Sample {report.sample_id}")
    print(f"  Speakers: {report.speakers[0]} & {report.speakers[1]}")
    print(f"{'='*72}")

    print(f"\n{'指标':<20} {'Flat':<12} {'Hierarchy':<12} {'Delta':<10}")
    print("-" * 55)
    print(f"{'Recall Score':<20} {report.flat_overall:>8.1f}%    {report.hier_overall:>8.1f}%    "
          f"{report.hier_overall - report.flat_overall:>+6.1f}%")
    print(f"{'Precision':<20} {report.flat_precision:>8.1%}    {report.hier_precision:>8.1%}    "
          f"{report.hier_precision - report.flat_precision:>+6.1%}")

    for cat in ["single-fact", "temporal", "open-ended", "multi-hop", "adversarial"]:
        f = report.flat_by_cat.get(cat, 0)
        h = report.hier_by_cat.get(cat, 0)
        if f or h:
            print(f"  {cat:<18} {f:>8.1f}%    {h:>8.1f}%    {h - f:>+6.1f}%")

    print(f"\nMemories: flat={report.flat_memory_count} hier={report.hier_memory_count}")
    print(f"Episodes: {report.hier_episode_count}, Schemas: {report.hier_schema_count}")
    print(f"Time: inject={report.inject_time:.0f}s consol={report.consol_time:.0f}s judge={report.judge_time:.0f}s")
    print("=" * 72)


def save_report(report: RealLocomoReport, output: Path):
    data = {
        "type": "real_locomo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_id": report.sample_id,
        "speakers": list(report.speakers),
        "session_count": report.session_count,
        "total_qa": report.total_qa,
        "flat_overall": report.flat_overall,
        "hier_overall": report.hier_overall,
        "delta": report.hier_overall - report.flat_overall,
        "flat_precision": report.flat_precision,
        "hier_precision": report.hier_precision,
        "flat_by_category": report.flat_by_cat,
        "hier_by_category": report.hier_by_cat,
        "flat_memory_count": report.flat_memory_count,
        "hier_memory_count": report.hier_memory_count,
        "hier_episode_count": report.hier_episode_count,
        "hier_schema_count": report.hier_schema_count,
        "inject_time_s": report.inject_time,
        "consolidation_time_s": report.consol_time,
        "judge_time_s": report.judge_time,
        "questions": [
            {
                "id": r.qid,
                "category": r.category,
                "question": r.question,
                "expected": r.expected,
                "evidence": r.evidence,
                "is_adversarial": r.is_adversarial,
                "score_flat": r.score_flat,
                "score_hier": r.score_hier,
                "reason_flat": r.reason_flat,
                "reason_hier": r.reason_hier,
                "precision_flat": r.precision_flat,
                "precision_hier": r.precision_hier,
                "context_flat": r.context_flat[:500],
                "context_hier": r.context_hier[:500],
            }
            for r in report.results
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Real LoCoMo Evaluation")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sample", type=int, default=0, help="Sample index (0-9)")
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--max-questions", type=int, default=50, help="Max questions per sample (0=all)")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_url = args.base_url

    # Health check
    try:
        resp = httpx.get(f"{base_url}/v1/health/", timeout=10)
        health = resp.json()
        print(f"Server: {health.get('status')} (v{health.get('version')}, {health.get('memory_count')} memories)")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        sys.exit(1)

    if args.all_samples:
        samples = load_locomo()
    else:
        samples = load_locomo(args.sample)

    all_reports = []
    for sample in samples:
        report = run_eval(base_url, sample, args.max_questions, args.judge_only)
        print_report(report)
        all_reports.append(report)

        out = Path(args.output) if args.output else (
            Path(__file__).parent / "results" /
            f"real_locomo_{sample['sample_id']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_report(report, out)

    # Summary if multiple
    if len(all_reports) > 1:
        print(f"\n{'='*72}")
        print("  AGGREGATE SUMMARY (all samples)")
        print(f"{'='*72}")
        all_f = sum(r.flat_overall for r in all_reports) / len(all_reports)
        all_h = sum(r.hier_overall for r in all_reports) / len(all_reports)
        all_pf = sum(r.flat_precision for r in all_reports) / len(all_reports)
        all_ph = sum(r.hier_precision for r in all_reports) / len(all_reports)
        print(f"  Recall:    Flat {all_f:.1f}% → Hier {all_h:.1f}% (Δ{all_h-all_f:+.1f}%)")
        print(f"  Precision: Flat {all_pf:.1%} → Hier {all_ph:.1%}")
        for cat in ["single-fact", "temporal", "open-ended", "multi-hop", "adversarial"]:
            fs = [r.flat_by_cat.get(cat, 0) for r in all_reports if cat in r.flat_by_cat]
            hs = [r.hier_by_cat.get(cat, 0) for r in all_reports if cat in r.hier_by_cat]
            if fs and hs:
                af = sum(fs) / len(fs)
                ah = sum(hs) / len(hs)
                print(f"    {cat:<15}: {af:.1f}% → {ah:.1f}% (Δ{ah-af:+.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
