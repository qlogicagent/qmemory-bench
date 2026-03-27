"""Dataset loader — manages evaluation datasets (local + remote update).

Datasets:
  - longmemeval-s: LongMemEval-S 6-dimensional long-term memory (core)
  - locomo: LoCoMo dialogue history retrieval
  - qmemory-chinese: Chinese-specific temporal / profile evaluation
  - conflict-resolution: Contradiction detection and version chain
  - profile-accuracy: User profile extraction accuracy
  - stress-scale: Large-scale stress test (10K/50K/100K)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A single evaluation question."""
    id: str
    query: str
    expected: str           # Expected answer (for judge matching)
    category: str           # Sub-dimension, e.g. "single-session-user"
    dataset: str            # Parent dataset name
    difficulty: str = "standard"  # "easy" | "standard" | "hard"
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """A conversation session to inject before evaluation."""
    id: str
    messages: list[dict[str, str]]
    metadata: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """A complete evaluation dataset."""
    name: str
    description: str
    sessions: list[Session]
    questions: list[Question]
    categories: list[str]
    version: str = "1.0"


# ── Available datasets registry ─────────────────────────────────

AVAILABLE_DATASETS: dict[str, dict[str, Any]] = {
    "longmemeval-s": {
        "description": "LongMemEval-S v2.0 — 六维度长期记忆评测（quick 30题/standard 150题）",
        "question_count": 150,
        "tier": "supporting",
        "authority": "paper-aligned",
        "categories": [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "knowledge-update",
            "temporal-reasoning",
            "multi-session",
        ],
        "source": "synthetic",
        "source_note": "v2.0 含针对性混淆噪声（相似职位/薪资/项目名等），维度对齐 arXiv:2410.10813",
        "citation": "Di Wu et al. 'LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.' arXiv:2410.10813, 2024.",
    },
    "locomo": {
        "description": "LoCoMo — 对话历史精准召回/多跳推理/时序推理/逻辑一致性/抗噪声（quick 30题/standard 50题）",
        "question_count": 50,
        "tier": "main",
        "authority": "public",
        "categories": ["recall-accuracy", "multi-hop", "temporal", "logical-reasoning", "noise-resist"],
        "source": "academic",
        "source_note": "基于 LoCoMo 公开论文设计 v2.0, 5维度扩展版, 含 multi-hop/temporal 层级检索验证子集",
        "citation": "Maharana et al. 'Evaluating Very Long-Term Conversational Memory of LLM Agents.' ACL 2024. arXiv:2402.17753",
    },
    "qmemory-chinese": {
        "description": "QMemory-Chinese v2.0 — 中文特有挑战：时间词/成语/同名/画像（quick 24题/standard 120题）",
        "question_count": 120,
        "tier": "supporting",
        "authority": "custom",
        "categories": ["temporal-zh", "idiom-zh", "name-disambig", "profile-zh"],
        "source": "custom",
        "source_note": "QMemory 团队自建 v2，含针对性噪声会话（近义时间词、同名人物、相似成语语境等中文混淆场景）",
    },
    "multimodal": {
        "description": "多模态精准检索 — 大量相似数据+长期噪声下精准定位（quick 30题/standard 150题）",
        "question_count": 150,
        "tier": "supporting",
        "authority": "custom",
        "categories": [
            "mm-basic-recall",
            "mm-precision-retrieve",
            "mm-noise-resist",
            "mm-abstract-query",
            "mm-temporal-disambig",
            "mm-cross-ref",
        ],
        "source": "custom",
        "source_note": "QMemory 团队自建 v2，以动画设计师 2 年工作流为场景，20 会话（12 核心 + 8 高混淆噪声），6 维度严格评测",
    },
    "conflict-resolution": {
        "description": "冲突检测 — 矛盾事实检测/版本链正确性/过期信息识别（quick 30题）",
        "question_count": 30,
        "tier": "regression",
        "authority": "custom",
        "categories": ["contradiction-detect", "supersede-correctness", "expired-info"],
        "source": "custom",
        "source_note": "QMemory 团队自建，评测记忆更新与冲突处理",
    },
    "profile-accuracy": {
        "description": "用户画像 — 偏好/习惯/事实提取/时效性（quick 30题）",
        "question_count": 30,
        "tier": "regression",
        "authority": "custom",
        "categories": ["preference-extract", "habit-extract", "fact-extract", "timeliness"],
        "source": "custom",
        "source_note": "QMemory 团队自建，评测 profile 自动提取质量",
    },
    "stress-scale": {
        "description": "压力测试 — 大规模记忆检索延迟与准确率（quick 30题）",
        "question_count": 30,
        "tier": "regression",
        "authority": "custom",
        "categories": ["10k-latency", "50k-latency", "100k-accuracy"],
        "source": "custom",
        "source_note": "QMemory 团队自建，评测大规模场景下性能",
    },
    # ── v2.0 新增: 系统性盲区覆盖 ──
    "preference-drift": {
        "description": "偏好漂移检测 — 偏好提取/更新追踪/冲突消解/对抗/时序（quick 20题）",
        "question_count": 50,
        "tier": "regression",
        "authority": "custom",
        "categories": [
            "preference-extract", "preference-drift", "preference-conflict",
            "temporal", "adversarial", "noise-resist",
        ],
        "source": "custom",
        "source_note": "盲区修复: 偏好系统严重欠测 → 6维度50题全覆盖; 含7步饮品变化链/6次运动更换/4城迁移等真实偏好漂移场景",
    },
    "implicit-memory": {
        "description": "隐式记忆 — 从暗示/间接表达/潜台词推断未直接说明的信息（quick 20题）",
        "question_count": 30,
        "tier": "regression",
        "authority": "custom",
        "categories": [
            "implicit-family", "implicit-career", "implicit-emotion",
            "implicit-health", "implicit-finance", "implicit-location",
        ],
        "source": "custom",
        "source_note": "盲区修复: 合成数据偏差(显式事实) → 隐式推理; 含'老妈走了'婉辞推断/职业间接推断/年龄推算/经济暗示等",
    },
    "stress-latency": {
        "description": "并发压力与延迟 — 高密度记忆下精准检索/对抗题/噪声抗干扰（quick 20题）",
        "question_count": 20,
        "tier": "regression",
        "authority": "custom",
        "categories": [
            "stress-recall", "stress-precision", "adversarial", "noise-resist",
        ],
        "source": "custom",
        "source_note": "盲区修复: 无并发/延迟测试 → 高密度记忆库(23会话/15核心+8噪声/多维度精准检索)+对抗题",
    },
    "locomo-real": {
        "description": "真实LoCoMo数据集 (HuggingFace KimmoZZZ/locomo) — 自动下载转换",
        "question_count": 30,
        "tier": "main",
        "authority": "public",
        "categories": ["single-fact", "temporal", "open-ended", "multi-hop", "adversarial"],
        "source": "KimmoZZZ/locomo @ HuggingFace",
        "source_note": "盲区修复: 不是真LoCoMo → 从HuggingFace拉取真实locomo10.json, 自动转为bench格式, 第一次加载会下载约5MB",
    },
}

DATASET_GROUP_LABELS: dict[str, str] = {
    "main": "主评测集",
    "supporting": "辅助评测集",
    "regression": "专项回归集",
}

DATASET_AUTHORITY_LABELS: dict[str, str] = {
    "public": "公开/权威",
    "paper-aligned": "论文对齐",
    "custom": "自建",
}

DATASET_PRESET_LABELS: dict[str, str] = {
    "public-main": "对外主评测",
    "release-full": "对外完整评测",
    "supporting": "辅助评测",
    "regression": "专项回归",
    "all": "全量评测",
}

DATASET_PRESETS: dict[str, list[str]] = {
    "public-main": ["locomo-real", "locomo"],
    "supporting": ["longmemeval-s", "qmemory-chinese", "multimodal"],
    "release-full": [
        "locomo-real",
        "locomo",
        "longmemeval-s",
        "qmemory-chinese",
        "multimodal",
    ],
    "regression": [
        "conflict-resolution",
        "profile-accuracy",
        "stress-scale",
        "preference-drift",
        "implicit-memory",
        "stress-latency",
    ],
}
DATASET_PRESETS["all"] = list(AVAILABLE_DATASETS.keys())


# ── Data directory ──────────────────────────────────────────────

def _data_dir() -> Path:
    """Built-in data directory (shipped with package)."""
    return Path(__file__).parent.parent.parent / "data"


def _cache_dir() -> Path:
    """User-level cache directory."""
    d = Path.home() / ".qmemory-bench" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_dataset_preset(name: str) -> list[str]:
    """Return dataset names for a named preset."""
    if name not in DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {name}")
    return list(DATASET_PRESETS[name])


def infer_dataset_preset(dataset_names: list[str]) -> str:
    """Return the matching preset name, or 'custom' if none matches exactly."""
    normalized = list(dict.fromkeys(dataset_names))
    for preset_name, preset_datasets in DATASET_PRESETS.items():
        if normalized == preset_datasets:
            return preset_name
    return "custom"


def resolve_dataset_selection(
    dataset_names: list[str] | None = None,
    dataset_preset: str | None = None,
) -> tuple[str, list[str]]:
    """Resolve preset or explicit dataset selection into a de-duplicated ordered list."""
    if dataset_names:
        resolved = [name for name in dataset_names if name in AVAILABLE_DATASETS]
        return infer_dataset_preset(resolved), list(dict.fromkeys(resolved))

    preset_name = dataset_preset or "public-main"
    if preset_name not in DATASET_PRESETS:
        preset_name = "public-main"
    return preset_name, get_dataset_preset(preset_name)


def get_grouped_datasets() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Return datasets grouped as main/supporting/regression."""
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "main": [],
        "supporting": [],
        "regression": [],
    }
    for name, info in AVAILABLE_DATASETS.items():
        grouped.setdefault(info.get("tier", "supporting"), []).append((name, info))
    return grouped


# ── Loader ──────────────────────────────────────────────────────

def load_dataset(name: str, scale: str = "quick") -> Dataset:
    """Load a dataset by name and scale.

    Scale controls how many questions are selected:
    - quick: ~12 questions (2 min)
    - standard: ~50 questions (10 min)
    - full: all questions (30 min)
    """
    # Special: locomo-real requires HuggingFace download & conversion
    if name == "locomo-real":
        return _load_locomo_real(scale)

    # Try built-in data first, with graceful degradation:
    # full → standard → quick (never silently skip to smallest)
    fallback_order = [scale]
    if scale == "full":
        fallback_order = ["full", "standard", "quick"]
    elif scale == "standard":
        fallback_order = ["standard", "quick"]

    data_path: Path | None = None
    for s in fallback_order:
        candidate = _data_dir() / f"{name}_{s}.json"
        if candidate.exists():
            if s != scale:
                logger.warning(
                    "Dataset %s has no '%s' scale, falling back to '%s'",
                    name, scale, s,
                )
            data_path = candidate
            break

    # Try cache if built-in not found
    if data_path is None:
        for s in fallback_order:
            candidate = _cache_dir() / name / f"{s}.json"
            if candidate.exists():
                if s != scale:
                    logger.warning(
                        "Dataset %s (cache) falling back from '%s' to '%s'",
                        name, scale, s,
                    )
                data_path = candidate
                break
    if data_path is None:
        raise FileNotFoundError(
            f"Dataset '{name}' not found for any scale in {fallback_order}. "
            f"Checked: {_data_dir()}, {_cache_dir() / name}"
        )

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return _parse_dataset(raw, name, scale)


def load_builtin_quick(name: str) -> Dataset:
    """Load the built-in quick dataset (always available)."""
    return load_dataset(name, "quick")


def _parse_dataset(raw: dict, name: str, scale: str) -> Dataset:
    """Parse raw JSON into a Dataset object."""
    sessions = [
        Session(
            id=s.get("id", f"sess_{i}"),
            messages=s.get("messages", []),
            metadata=s.get("metadata", {}),
        )
        for i, s in enumerate(raw.get("sessions", []))
    ]

    questions = [
        Question(
            id=q.get("id", f"q_{i}"),
            query=q["query"],
            expected=q["expected"],
            category=q.get("category", "general"),
            dataset=name,
            difficulty=q.get("difficulty", "standard"),
            metadata=q.get("metadata", {}),
        )
        for i, q in enumerate(raw.get("questions", []))
    ]

    categories = list({q.category for q in questions})

    return Dataset(
        name=name,
        description=raw.get("description", ""),
        sessions=sessions,
        questions=questions,
        categories=categories,
        version=raw.get("version", "1.0"),
    )


def list_local_datasets() -> list[str]:
    """List datasets available locally."""
    found = set()
    for d in (_data_dir(), _cache_dir()):
        if d.exists():
            for f in d.glob("*_quick.json"):
                found.add(f.stem.rsplit("_", 1)[0])
            for sub in d.iterdir():
                if sub.is_dir() and (sub / "quick.json").exists():
                    found.add(sub.name)
    return sorted(found)


# ── Real LoCoMo loader (HuggingFace) ───────────────────────────

_LOCOMO_CATEGORY_MAP = {
    1: "single-fact",
    2: "temporal",
    3: "open-ended",
    4: "multi-hop",
    5: "adversarial",
}


def _load_locomo_real(scale: str) -> Dataset:
    """Download real LoCoMo from HuggingFace, convert, cache, and return.

    Uses sample_0 for 'quick', sample_0-2 for 'standard', all 10 for 'full'.
    """
    cache_path = _cache_dir() / "locomo-real" / f"{scale}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return _parse_dataset(raw, "locomo-real", scale)

    # Download locomo10.json from HuggingFace
    raw_data = _download_locomo()

    # Determine how many samples to use
    if scale == "quick":
        samples = raw_data[:1]
        max_q = 30
    elif scale == "standard":
        samples = raw_data[:3]
        max_q = 100
    else:
        samples = raw_data
        max_q = 0  # all

    sessions: list[dict] = []
    questions: list[dict] = []
    q_idx = 0

    for s_idx, sample in enumerate(samples):
        conversation = sample.get("conversation", sample)

        # Extract speaker names
        speaker_a = conversation.get("speaker_a", "PersonA")
        speaker_b = conversation.get("speaker_b", "PersonB")

        # Convert sessions
        sess_keys = sorted(
            [k for k in conversation if k.startswith("session_")
             and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in sess_keys:
            turns = conversation[sk]
            date_key = sk + "_date_time"
            date_str = conversation.get(date_key, "")

            messages = []
            for turn in turns:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                role = "user" if speaker == speaker_a else "assistant"
                messages.append({"role": role, "content": text})

            sessions.append({
                "id": f"lc{s_idx}-{sk}",
                "messages": messages,
                "metadata": {"timestamp": date_str, "type": "core"},
            })

        # Convert questions
        qa_list = sample.get("qa", [])
        for qa in qa_list:
            cat_id = qa.get("category", 1)
            cat_name = _LOCOMO_CATEGORY_MAP.get(cat_id, f"unknown-{cat_id}")
            is_adv = cat_id == 5
            expected = str(qa.get("answer", qa.get("adversarial_answer", "")))

            questions.append({
                "id": f"lc-real-{q_idx:03d}",
                "query": qa["question"],
                "expected": expected,
                "category": cat_name,
                "difficulty": "standard",
            })
            q_idx += 1

    # Stratified sampling if needed
    if max_q and len(questions) > max_q:
        import random
        random.seed(42)
        by_cat: dict[str, list] = {}
        for q in questions:
            by_cat.setdefault(q["category"], []).append(q)
        sampled: list[dict] = []
        total = len(questions)
        for cat, qs in by_cat.items():
            n = max(1, int(len(qs) / total * max_q))
            sampled.extend(random.sample(qs, min(n, len(qs))))
        remaining = [q for q in questions if q not in sampled]
        random.shuffle(remaining)
        while len(sampled) < max_q and remaining:
            sampled.append(remaining.pop())
        questions = sampled[:max_q]

    # Build and cache the converted dataset
    converted = {
        "name": "locomo-real",
        "description": f"Real LoCoMo dataset (HuggingFace, {len(samples)} samples, {len(questions)} questions)",
        "version": "1.0",
        "source": "KimmoZZZ/locomo",
        "sessions": sessions,
        "questions": questions,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return _parse_dataset(converted, "locomo-real", scale)


def _download_locomo() -> list[dict]:
    """Download locomo10.json from HuggingFace or use local fallback."""
    # Try HuggingFace hub
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "KimmoZZZ/locomo", "locomo10.json", repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except ImportError:
        pass

    # Fallback: local data dir
    local = _data_dir() / "locomo10.json"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(
        "Cannot load real LoCoMo: huggingface_hub not installed and "
        "data/locomo10.json not found. Install with: pip install huggingface_hub"
    )
