"""LLM Judge — scores QMemory recall results against expected answers.

Uses an LLM to evaluate whether the recalled memories + context
contain the expected information, on a 0-10 scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from qmemory_bench.providers import LLMJudge

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are a strict evaluation judge for a memory retrieval system.
Your task: determine whether the RECALLED CONTEXT correctly answers the QUESTION,
compared to the EXPECTED ANSWER.

Important rules for Chinese content evaluation:
- Treat semantically equivalent Chinese expressions as correct (e.g. 字节跳动=ByteDance, 北京=Beijing)
- Chinese temporal expressions like 去年/上个月/三天前 should be evaluated by meaning, not literal match
- Chinese idioms, abbreviated forms, and colloquial expressions count as correct if meaning matches
- Names in Chinese (人名/地名) should match semantically, pinyin differences are acceptable
- 张冠李戴 (attributing A's info to B) should be strictly penalized

Score from 0 to 10:
- 10: Perfect match — all key information in expected answer is present
- 7-9: Good — most key information present, minor omissions
- 4-6: Partial — some key information present but incomplete
- 1-3: Poor — very little relevant information
- 0: No relevant information found

Also evaluate PRECISION — how much of the recalled content is actually relevant:
- 1.0: All recalled content is relevant to the question
- 0.7: Mostly relevant with some noise
- 0.5: Half relevant, half irrelevant
- 0.3: Mostly irrelevant
- 0.0: All irrelevant
- If context is empty, precision = 1.0 (no false positives, just no recall)

Output ONLY a JSON object: {"score": <int 0-10>, "reason": "<brief explanation>", "precision": <float 0.0-1.0>}
"""

JUDGE_SYSTEM_ADVERSARIAL = """\
You are evaluating a memory system on an ADVERSARIAL question — a question
containing a false premise, incorrect assumption, or deliberate trap.

The correct behavior is for the system to NOT be misled. Evaluate:
- Did the system resist the false premise?
- Did it provide correct information instead?
- Or was it tricked into confirming incorrect information?

Score from 0 to 10:
- 10: Correctly identified the trap or stated no matching info
- 7-9: Not misled, gave correct information
- 4-6: Partially misled but has some correct info
- 0-3: Completely misled, confirmed false premise

Also evaluate precision (same scale as above).

Output ONLY a JSON: {"score": <int 0-10>, "reason": "<brief>", "precision": <float 0.0-1.0>}
"""

JUDGE_PROMPT = """\
QUESTION: {query}

EXPECTED ANSWER: {expected}

RECALLED CONTEXT:
{context}

RECALLED MEMORIES:
{memories}

Score this recall result (0-10):"""


@dataclass
class JudgeResult:
    """Result of a single judge evaluation."""
    question_id: str
    query: str
    expected: str
    category: str
    score: int             # 0-10
    reason: str
    context_preview: str   # First 200 chars of recalled context
    raw_recall: dict       # Full recall response from QMemory
    precision: float = 1.0  # 0.0-1.0, ratio of relevant info in recall
    is_adversarial: bool = False  # adversarial question flag


async def judge_single(
    question_id: str,
    query: str,
    expected: str,
    category: str,
    recall_result: dict,
    llm: LLMJudge,
) -> JudgeResult:
    """Judge a single recall result against expected answer."""
    memories = recall_result.get("memories", [])
    context = recall_result.get("context", "")

    mem_text = "\n".join(
        f"- [{m.get('category', '?')}] {m.get('text', '')}" for m in memories
    ) or "(no memories found)"

    prompt = JUDGE_PROMPT.format(
        query=query,
        expected=expected,
        context=context[:2000] if context else "(no context)",
        memories=mem_text[:2000],
    )

    # Select judge system prompt based on category
    is_adv = category in ("adversarial",) or "adversarial" in category.lower()
    system_prompt = JUDGE_SYSTEM_ADVERSARIAL if is_adv else JUDGE_SYSTEM

    try:
        raw = await llm.complete(prompt, system=system_prompt, json_mode=True)
        parsed = _parse_judge_response(raw)
        score = parsed.get("score", 0)
        reason = parsed.get("reason", "")
        precision = float(parsed.get("precision", 0.7))
    except Exception as e:
        logger.warning(f"Judge failed for {question_id}: {e}")
        # Fallback: simple keyword matching
        score = _keyword_fallback_score(expected, memories, context)
        reason = f"LLM judge failed ({e}), used keyword fallback"
        precision = 0.5

    return JudgeResult(
        question_id=question_id,
        query=query,
        expected=expected,
        category=category,
        score=min(max(score, 0), 10),
        reason=reason,
        context_preview=context[:500] if context else "",
        raw_recall=recall_result,
        precision=min(max(precision, 0.0), 1.0),
        is_adversarial=is_adv,
    )


def _parse_judge_response(raw: str) -> dict:
    """Parse LLM judge JSON response."""
    text = raw.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _keyword_fallback_score(
    expected: str,
    memories: list[dict],
    context: str,
) -> int:
    """Fallback scoring via keyword presence when LLM judge fails."""
    # Extract key terms from expected
    import re
    terms = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", expected)
    if not terms:
        return 5  # Can't evaluate

    # Combine all text
    all_text = context + " ".join(m.get("text", "") for m in memories)

    found = sum(1 for t in terms if t in all_text)
    ratio = found / len(terms) if terms else 0

    if ratio >= 0.8:
        return 8
    elif ratio >= 0.5:
        return 5
    elif ratio >= 0.2:
        return 3
    else:
        return 0


def aggregate_scores(results: list[JudgeResult], categories: list[str]) -> dict[str, Any]:
    """Aggregate judge results into category scores with precision."""
    all_scores = [r.score for r in results]
    overall = sum(all_scores) / len(all_scores) * 10 if all_scores else 0
    overall_precision = sum(r.precision for r in results) / len(results) if results else 0

    by_category: dict[str, list[JudgeResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    cat_scores = {}
    for cat, rs in by_category.items():
        scores = [r.score for r in rs]
        avg = sum(scores) / len(scores) * 10 if scores else 0
        avg_prec = sum(r.precision for r in rs) / len(rs) if rs else 0
        cat_scores[cat] = {
            "score": round(avg, 1),
            "precision": round(avg_prec, 3),
            "count": len(scores),
            "scores": scores,
        }

    return {
        "overall": round(overall, 1),
        "overall_precision": round(overall_precision, 3),
        "categories": cat_scores,
        "total_questions": len(results),
    }
