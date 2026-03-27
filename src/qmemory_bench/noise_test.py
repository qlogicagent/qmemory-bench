"""Noise injection + cross-time-period stress testing.

Simulates real-world long-term usage:
  - Inject N sessions of random daily chatter over M months/years
  - Test recall accuracy after noise injection
  - Validate temporal reasoning with time gaps
  - Measure signal-to-noise ratio degradation
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


# ── Noise Templates ─────────────────────────────────────────────

NOISE_TOPICS = [
    # (topic, user_messages, assistant_responses)
    ("weather", [
        "今天天气{adj}，{temp}度",
        "看天气预报说{day}要{weather_event}",
        "这个{season}真是{adj2}",
    ], [
        "注意{advice}。",
        "是的，{season}就是这样。",
        "希望天气好转。",
    ]),
    ("food", [
        "中午吃了{food}，{taste}",
        "最近迷上了{cuisine}，特别是{dish}",
        "今晚想做{dish2}，你有什么建议吗",
    ], [
        "听起来不错，注意营养均衡。",
        "{cuisine}确实好吃。",
        "可以试试加点{ingredient}。",
    ]),
    ("entertainment", [
        "昨晚看了{media}，{评价}",
        "最近在追{show}，更新太慢了",
        "周末想去{place}逛逛",
    ], [
        "评价还不错，值得看吗？",
        "追剧就是这样，等更新很煎熬。",
        "周末出去走走挺好的。",
    ]),
    ("tech", [
        "看到{company}发布了{product}，觉得{opinion}",
        "我的{device}最近有点{problem}",
        "想升级一下{component}，预算{budget}左右",
    ], [
        "科技产品更新很快。",
        "可以试试{solution}。",
        "这个预算应该够了。",
    ]),
    ("daily", [
        "今天上班{event}了",
        "周末{weekend_plan}",
        "刚{action}回来，{feeling}",
    ], [
        "辛苦了，注意休息。",
        "放松一下挺好的。",
        "保持好心情。",
    ]),
]

# Fill-in values
FILL_VALUES = {
    "adj": ["不错", "糟糕", "挺好的", "一般"],
    "adj2": ["难熬", "舒服", "还行"],
    "temp": ["15", "28", "35", "-5", "8", "22"],
    "day": ["明天", "后天", "周末"],
    "weather_event": ["下雨", "下雪", "降温", "回暖"],
    "season": ["春天", "夏天", "秋天", "冬天"],
    "advice": ["带伞", "多穿点", "防晒", "注意保暖"],
    "food": ["黄焖鸡", "兰州拉面", "麦当劳", "食堂", "螺蛳粉", "沙县小吃"],
    "taste": ["一般般", "还不错", "有点咸", "挺好吃", "不太新鲜"],
    "cuisine": ["湖南菜", "粤菜", "东北菜", "火锅", "日料", "烤肉"],
    "dish": ["酸菜白肉", "麻辣香锅", "红烧排骨", "烤鱼", "宫保鸡丁"],
    "dish2": ["番茄炒蛋", "可乐鸡翅", "糖醋里脊", "清蒸鲈鱼"],
    "ingredient": ["蒜末", "生抽", "料酒", "花椒"],
    "media": ["一部电影", "一个纪录片", "一集综艺"],
    "评价": ["还行吧", "特别好看", "有点无聊", "推荐"],
    "show": ["一部韩剧", "一个国产剧", "一个美剧"],
    "place": ["商场", "公园", "书店", "咖啡厅", "市集"],
    "company": ["苹果", "华为", "小米", "三星", "谷歌"],
    "product": ["新手机", "新芯片", "新系统", "新耳机"],
    "opinion": ["挺厉害的", "没啥创新", "性价比不错", "太贵了"],
    "device": ["手机", "电脑", "耳机", "平板"],
    "problem": ["卡顿", "电池不行了", "发热严重", "内存不够"],
    "component": ["内存", "显卡", "硬盘", "显示器"],
    "budget": ["500", "1000", "2000", "3000"],
    "solution": ["重启试试", "清理缓存", "恢复出厂设置"],
    "event": ["迟到", "被表扬", "开了好多会", "写了一天代码"],
    "weekend_plan": ["躺平", "去健身", "约朋友吃饭", "宅家看剧"],
    "action": ["跑步", "散步", "买菜", "逛超市"],
    "feeling": ["挺累的", "心情不错", "好舒服", "累但值得"],
}


def _fill(template: str) -> str:
    """Fill a template string with random values."""
    import re
    def replacer(m):
        key = m.group(1)
        values = FILL_VALUES.get(key, [key])
        return random.choice(values)
    return re.sub(r'\{(\w+)\}', replacer, template)


def generate_noise_sessions(
    count: int = 50,
    span_days: int = 365,
    start_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Generate N noise sessions spread over a time period.

    Args:
        count: Number of noise sessions to generate.
        span_days: Time span in days (365 = 1 year).
        start_date: Start date (defaults to 1 year ago).

    Returns:
        List of session dicts ready for injection.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=span_days)

    sessions = []
    for i in range(count):
        # Random timestamp within the span
        offset = random.randint(0, span_days * 86400)
        ts = start_date + timedelta(seconds=offset)

        topic_idx = random.randint(0, len(NOISE_TOPICS) - 1)
        topic_name, user_tmpls, asst_tmpls = NOISE_TOPICS[topic_idx]

        # Generate 2-4 turns
        turn_count = random.randint(2, min(4, len(user_tmpls)))
        messages = []
        for j in range(turn_count):
            u_tmpl = random.choice(user_tmpls)
            a_tmpl = random.choice(asst_tmpls)
            messages.append({"role": "user", "content": _fill(u_tmpl)})
            messages.append({"role": "assistant", "content": _fill(a_tmpl)})

        sessions.append({
            "id": f"noise_{i:04d}_{topic_name}",
            "messages": messages,
            "metadata": {
                "type": "noise",
                "topic": topic_name,
                "timestamp": ts.isoformat(),
                "generated": True,
            },
        })

    # Sort by timestamp
    sessions.sort(key=lambda s: s["metadata"]["timestamp"])
    return sessions


# ── Noise-Augmented Benchmark ───────────────────────────────────

@dataclass
class NoiseTestConfig:
    """Configuration for noise-augmented testing."""
    target_url: str = "http://localhost:18800"
    noise_count: int = 100         # Number of noise sessions
    noise_span_days: int = 365     # Spread over N days
    inject_batch_size: int = 10    # Parallel injection batch
    provider: str = "deepseek"
    api_key: str = ""
    model: str = ""


@dataclass
class NoiseTestReport:
    """Report comparing accuracy before vs after noise injection."""
    baseline_scores: dict[str, float]   # Scores without noise
    noisy_scores: dict[str, float]      # Scores with noise
    degradation: dict[str, float]       # Score drop per category
    noise_sessions: int
    total_memories_before: int
    total_memories_after: int
    signal_to_noise_ratio: float        # core_memories / total_memories


async def run_noise_test(
    config: NoiseTestConfig,
    dataset_path: str | None = None,
) -> NoiseTestReport:
    """Run noise injection test: inject core → test → inject noise → retest.

    Measures how recall accuracy degrades with increasing noise.
    """
    from qmemory_bench.dataset import load_dataset
    from qmemory_bench.judge import aggregate_scores, judge_single
    from qmemory_bench.providers import LLMJudge

    client = httpx.AsyncClient(base_url=config.target_url, timeout=30.0)
    llm = LLMJudge(provider=config.provider, api_key=config.api_key, model=config.model)
    eval_user = f"noise_eval_{uuid4().hex[:8]}"

    ds = load_dataset("longmemeval-s", "quick")

    # Phase 1: Inject core sessions only
    logger.info(f"Phase 1: Injecting {len(ds.sessions)} core sessions...")
    for session in ds.sessions:
        try:
            await client.post("/v1/memories/", json={
                "messages": session.messages,
                "user_id": eval_user,
                "session_id": session.id,
            })
        except Exception as e:
            logger.warning(f"Inject failed: {e}")

    # Count baseline memories
    try:
        resp = await client.get("/v1/memories/", params={"user_id": eval_user, "page_size": 1})
        baseline_count = resp.json().get("total", 0)
    except Exception:
        baseline_count = 0

    # Phase 2: Baseline scoring
    logger.info("Phase 2: Baseline scoring (no noise)...")
    baseline_results = await _score_questions(ds, eval_user, client, llm)
    baseline_agg = aggregate_scores(baseline_results, ds.categories)

    # Phase 3: Inject noise sessions
    logger.info(f"Phase 3: Injecting {config.noise_count} noise sessions...")
    noise_sessions = generate_noise_sessions(
        count=config.noise_count,
        span_days=config.noise_span_days,
    )
    for session in noise_sessions:
        try:
            await client.post("/v1/memories/", json={
                "messages": session["messages"],
                "user_id": eval_user,
                "session_id": session["id"],
            })
        except Exception as e:
            logger.debug(f"Noise inject failed: {e}")

    # Count total memories after noise
    try:
        resp = await client.get("/v1/memories/", params={"user_id": eval_user, "page_size": 1})
        total_count = resp.json().get("total", 0)
    except Exception:
        total_count = 0

    # Phase 4: Re-score with noise present
    logger.info("Phase 4: Scoring with noise...")
    noisy_results = await _score_questions(ds, eval_user, client, llm)
    noisy_agg = aggregate_scores(noisy_results, ds.categories)

    # Phase 5: Cleanup
    try:
        await client.request("DELETE", "/v1/memories/",
                             params={"user_id": eval_user, "confirm": "true"})
    except Exception:
        pass

    await llm.close()
    await client.aclose()

    # Compute degradation
    baseline_cats = {cat: info["score"] for cat, info in baseline_agg["categories"].items()}
    noisy_cats = {cat: info["score"] for cat, info in noisy_agg["categories"].items()}
    degradation = {}
    for cat in baseline_cats:
        b = baseline_cats.get(cat, 0)
        n = noisy_cats.get(cat, 0)
        degradation[cat] = round(b - n, 1)

    snr = baseline_count / total_count if total_count > 0 else 1.0

    return NoiseTestReport(
        baseline_scores={"overall": baseline_agg["overall"], **baseline_cats},
        noisy_scores={"overall": noisy_agg["overall"], **noisy_cats},
        degradation=degradation,
        noise_sessions=config.noise_count,
        total_memories_before=baseline_count,
        total_memories_after=total_count,
        signal_to_noise_ratio=round(snr, 3),
    )


async def _score_questions(ds, eval_user, client, llm):
    """Score all questions in a dataset."""
    from qmemory_bench.judge import judge_single
    results = []
    for q in ds.questions:
        try:
            resp = await client.get("/v1/memories/search/", params={
                "q": q.query, "user_id": eval_user, "limit": 10,
            })
            recall = resp.json()
        except Exception:
            recall = {"memories": [], "context": ""}

        result = await judge_single(
            question_id=q.id, query=q.query, expected=q.expected,
            category=q.category, recall_result=recall, llm=llm,
        )
        results.append(result)
    return results
