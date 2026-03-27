"""Tests for qmemory-bench — providers, dataset, judge, runner, i18n."""

from __future__ import annotations

import json
import pytest

from qmemory_bench.providers import (
    PROVIDERS,
    LLMJudge,
    ProviderInfo,
    get_provider_info,
    list_providers,
)
from qmemory_bench.dataset import (
    AVAILABLE_DATASETS,
    Dataset,
    Question,
    Session,
    _parse_dataset,
    load_dataset,
    list_local_datasets,
)
from qmemory_bench.judge import (
    JudgeResult,
    _keyword_fallback_score,
    _parse_judge_response,
    aggregate_scores,
)
from qmemory_bench.runner import (
    BenchmarkConfig,
    BenchmarkReport,
    ComparisonConfig,
    ComparisonReport,
    TARGETS,
    report_to_dict,
)


# ====================================================================
# Provider tests
# ====================================================================

class TestProviders:
    def test_all_seven_providers(self):
        assert len(PROVIDERS) == 7
        for key in ["deepseek", "minimax", "zhipu", "kimi", "qwen", "doubao", "openai"]:
            assert key in PROVIDERS

    def test_provider_info_fields(self):
        for key, info in PROVIDERS.items():
            assert isinstance(info, ProviderInfo)
            assert info.name
            assert info.base_url.startswith("https://")
            assert info.default_model
            assert info.description

    def test_get_provider_info_existing(self):
        info = get_provider_info("deepseek")
        assert info.name == "DeepSeek V3.2"
        assert "deepseek" in info.base_url

    def test_get_provider_info_fallback(self):
        info = get_provider_info("nonexistent")
        assert info.name == "DeepSeek V3.2"

    def test_list_providers(self):
        items = list_providers()
        assert len(items) == 7
        keys = {item["key"] for item in items}
        assert "deepseek" in keys
        assert "openai" in keys
        for item in items:
            assert "key" in item
            assert "name" in item
            assert "default_model" in item
            assert "description" in item

    def test_llm_judge_init_default(self):
        judge = LLMJudge(provider="deepseek", api_key="test-key")
        assert "deepseek" in judge.base_url
        assert judge.model == "deepseek-chat"
        assert judge.api_key == "test-key"

    def test_llm_judge_init_custom(self):
        judge = LLMJudge(
            provider="zhipu",
            api_key="test",
            model="custom-model",
            base_url="https://custom.api.com/v1",
        )
        assert judge.base_url == "https://custom.api.com/v1"
        assert judge.model == "custom-model"

    def test_llm_judge_repr(self):
        judge = LLMJudge(provider="kimi", api_key="x")
        r = repr(judge)
        assert "Kimi" in r
        assert "moonshot" in r


# ====================================================================
# Dataset tests
# ====================================================================

class TestDatasets:
    def test_available_datasets_registry(self):
        assert len(AVAILABLE_DATASETS) == 7  # +1 multimodal
        for name in ["longmemeval-s", "locomo", "qmemory-chinese",
                      "multimodal", "conflict-resolution",
                      "profile-accuracy", "stress-scale"]:
            assert name in AVAILABLE_DATASETS
            info = AVAILABLE_DATASETS[name]
            assert "description" in info
            assert "question_count" in info
            assert "categories" in info
            # Description should be in Chinese
            assert any("\u4e00" <= ch <= "\u9fff" for ch in info["description"])

    def test_datasets_have_source_metadata(self):
        """All datasets must have source attribution."""
        for name, info in AVAILABLE_DATASETS.items():
            assert "source" in info, f"{name} missing 'source'"
            assert "source_note" in info, f"{name} missing 'source_note'"

    def test_longmemeval_has_citation(self):
        info = AVAILABLE_DATASETS["longmemeval-s"]
        assert "citation" in info
        assert "arXiv" in info["citation"]

    def test_multimodal_categories(self):
        cats = AVAILABLE_DATASETS["multimodal"]["categories"]
        assert "mm-basic-recall" in cats
        assert "mm-precision-retrieve" in cats
        assert "mm-noise-resist" in cats
        assert "mm-abstract-query" in cats
        assert "mm-temporal-disambig" in cats
        assert "mm-cross-ref" in cats

    def test_longmemeval_categories(self):
        cats = AVAILABLE_DATASETS["longmemeval-s"]["categories"]
        assert len(cats) == 6
        assert "single-session-user" in cats
        assert "temporal-reasoning" in cats

    def test_parse_dataset_basic(self):
        raw = {
            "description": "Test dataset",
            "version": "1.0",
            "sessions": [
                {"id": "s1", "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]}
            ],
            "questions": [
                {"id": "q1", "query": "What?", "expected": "Something",
                 "category": "test-cat"},
            ],
        }
        ds = _parse_dataset(raw, "test-ds", "quick")
        assert isinstance(ds, Dataset)
        assert ds.name == "test-ds"
        assert len(ds.sessions) == 1
        assert len(ds.questions) == 1
        assert ds.questions[0].dataset == "test-ds"
        assert "test-cat" in ds.categories

    def test_load_quick_dataset_exists(self):
        """LongMemEval-S v2: 25 sessions (10 core + 15 noise), 30 questions (5 per dim)."""
        ds = load_dataset("longmemeval-s", "quick")
        assert isinstance(ds, Dataset)
        assert ds.name == "longmemeval-s"
        assert len(ds.sessions) >= 25
        assert len(ds.questions) >= 30
        # Verify noise sessions exist (targeted confusion)
        noise = [s for s in ds.sessions if s.metadata.get("type") == "noise"]
        assert len(noise) >= 15, f"Expected >=15 noise sessions, got {len(noise)}"
        # Verify all 6 categories balanced at 5 each
        from collections import Counter
        cats = Counter(q.category for q in ds.questions)
        assert all(v >= 5 for v in cats.values()), f"Unbalanced categories: {dict(cats)}"

    def test_load_chinese_quick_dataset(self):
        """QMemory-Chinese v2: 12 sessions (6 core + 6 noise), 24 questions (6 per dim)."""
        ds = load_dataset("qmemory-chinese", "quick")
        assert isinstance(ds, Dataset)
        assert ds.name == "qmemory-chinese"
        assert len(ds.sessions) >= 12
        assert len(ds.questions) >= 24
        cats = {q.category for q in ds.questions}
        assert "temporal-zh" in cats
        assert "idiom-zh" in cats
        assert "name-disambig" in cats
        assert "profile-zh" in cats
        # Verify noise sessions exist (no more noise-free!)
        noise = [s for s in ds.sessions if s.metadata.get("type") == "noise"]
        assert len(noise) >= 6, f"Expected >=6 noise sessions, got {len(noise)}"
        # Verify all categories balanced at 6 each
        from collections import Counter
        cats_count = Counter(q.category for q in ds.questions)
        assert all(v >= 6 for v in cats_count.values()), f"Unbalanced: {dict(cats_count)}"

    def test_load_multimodal_quick_dataset(self):
        """The multimodal_quick.json should be loadable (v2: 20 sessions, 30 questions, 6 dims)."""
        ds = load_dataset("multimodal", "quick")
        assert isinstance(ds, Dataset)
        assert ds.name == "multimodal"
        assert len(ds.sessions) >= 20
        assert len(ds.questions) >= 30
        cats = {q.category for q in ds.questions}
        assert "mm-basic-recall" in cats
        assert "mm-precision-retrieve" in cats
        assert "mm-noise-resist" in cats
        assert "mm-abstract-query" in cats
        assert "mm-temporal-disambig" in cats
        assert "mm-cross-ref" in cats

    def test_multimodal_noise_sessions(self):
        """Multimodal v2 should have at least 8 noise sessions."""
        ds = load_dataset("multimodal", "quick")
        noise = [s for s in ds.sessions if s.metadata.get("type") == "noise"]
        core = [s for s in ds.sessions if s.metadata.get("type") == "core"]
        assert len(noise) >= 8, f"Expected >=8 noise sessions, got {len(noise)}"
        assert len(core) >= 12, f"Expected >=12 core sessions, got {len(core)}"

    def test_multimodal_hard_difficulty_ratio(self):
        """Most multimodal v2 questions should be hard difficulty."""
        ds = load_dataset("multimodal", "quick")
        hard = [q for q in ds.questions if q.difficulty == "hard"]
        assert len(hard) >= 20, f"Expected >=20 hard questions, got {len(hard)}"

    def test_load_dataset_not_found(self):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_dataset("nonexistent", "quick")

    def test_list_local_datasets(self):
        found = list_local_datasets()
        assert "longmemeval-s" in found
        # All 7 datasets should now have data files
        for ds_name in ["longmemeval-s", "locomo", "qmemory-chinese",
                        "multimodal", "conflict-resolution",
                        "profile-accuracy", "stress-scale"]:
            assert ds_name in found, f"Missing dataset file: {ds_name}"

    # ── New datasets: LoCoMo ──

    def test_load_locomo_quick_dataset(self):
        """LoCoMo: 25 sessions (15 core + 10 noise), 30 questions, 3 categories."""
        ds = load_dataset("locomo", "quick")
        assert ds.name == "locomo"
        assert len(ds.sessions) >= 25
        assert len(ds.questions) >= 30
        cats = {q.category for q in ds.questions}
        assert "recall-accuracy" in cats
        assert "multi-turn" in cats
        assert "logical-reasoning" in cats

    # ── New datasets: Conflict-Resolution ──

    def test_load_conflict_resolution_quick(self):
        """Conflict-resolution: 20 sessions, 30 questions, 3 categories."""
        ds = load_dataset("conflict-resolution", "quick")
        assert ds.name == "conflict-resolution"
        assert len(ds.sessions) >= 20
        assert len(ds.questions) >= 30
        cats = {q.category for q in ds.questions}
        assert "contradiction-detect" in cats
        assert "supersede-correctness" in cats
        assert "expired-info" in cats

    # ── New datasets: Profile-Accuracy ──

    def test_load_profile_accuracy_quick(self):
        """Profile-accuracy: 18 sessions, 30 questions, 4 categories."""
        ds = load_dataset("profile-accuracy", "quick")
        assert ds.name == "profile-accuracy"
        assert len(ds.sessions) >= 18
        assert len(ds.questions) >= 30
        cats = {q.category for q in ds.questions}
        assert "preference-extract" in cats
        assert "habit-extract" in cats
        assert "fact-extract" in cats
        assert "timeliness" in cats

    # ── New datasets: Stress-Scale ──

    def test_load_stress_scale_quick(self):
        """Stress-scale: 30 sessions, 30 questions, 3 categories."""
        ds = load_dataset("stress-scale", "quick")
        assert ds.name == "stress-scale"
        assert len(ds.sessions) >= 30
        assert len(ds.questions) >= 30
        cats = {q.category for q in ds.questions}
        assert "10k-latency" in cats
        assert "50k-latency" in cats
        assert "100k-accuracy" in cats

    # ── Standard-scale datasets ──

    def test_load_longmemeval_standard(self):
        """Standard: 125 sessions, 150 questions (5x quick)."""
        ds = load_dataset("longmemeval-s", "standard")
        assert len(ds.sessions) >= 125
        assert len(ds.questions) >= 150
        noise = [s for s in ds.sessions if s.metadata.get("type") == "noise"]
        assert len(noise) >= 100

    def test_load_chinese_standard(self):
        """Standard: 57 sessions, 120 questions (5x quick)."""
        ds = load_dataset("qmemory-chinese", "standard")
        assert len(ds.sessions) >= 50
        assert len(ds.questions) >= 120

    def test_load_multimodal_standard(self):
        """Standard: 100 sessions, 150 questions (5x quick)."""
        ds = load_dataset("multimodal", "standard")
        assert len(ds.sessions) >= 100
        assert len(ds.questions) >= 150

    def test_standard_is_superset_of_quick(self):
        """Standard datasets should contain all quick questions as a subset."""
        for ds_name in ["longmemeval-s", "qmemory-chinese", "multimodal"]:
            quick = load_dataset(ds_name, "quick")
            standard = load_dataset(ds_name, "standard")
            quick_ids = {q.id for q in quick.questions}
            standard_ids = {q.id for q in standard.questions}
            assert quick_ids.issubset(standard_ids), (
                f"{ds_name}: quick questions not subset of standard"
            )

    def test_question_dataclass(self):
        q = Question(
            id="q1", query="test", expected="ans",
            category="cat", dataset="ds",
        )
        assert q.difficulty == "standard"
        assert q.metadata == {}

    def test_session_dataclass(self):
        s = Session(id="s1", messages=[])
        assert s.metadata == {}


# ====================================================================
# Judge tests
# ====================================================================

class TestJudge:
    def test_parse_judge_response_clean_json(self):
        raw = '{"score": 8, "reason": "Good match"}'
        result = _parse_judge_response(raw)
        assert result["score"] == 8
        assert result["reason"] == "Good match"

    def test_parse_judge_response_markdown(self):
        raw = '```json\n{"score": 7, "reason": "ok"}\n```'
        result = _parse_judge_response(raw)
        assert result["score"] == 7

    def test_parse_judge_response_with_prefix(self):
        raw = 'Here is my evaluation: {"score": 5, "reason": "partial"}'
        result = _parse_judge_response(raw)
        assert result["score"] == 5

    def test_keyword_fallback_high_match(self):
        expected = "字节跳动 后端 工程师"
        memories = [{"text": "在字节跳动担任后端工程师"}]
        score = _keyword_fallback_score(expected, memories, "")
        assert score >= 5

    def test_keyword_fallback_no_match(self):
        expected = "用户喜欢爬山和游泳"
        memories = [{"text": "今天天气很好"}]
        score = _keyword_fallback_score(expected, memories, "")
        assert score <= 3

    def test_keyword_fallback_from_context(self):
        expected = "北京海淀"
        memories = []
        score = _keyword_fallback_score(expected, memories, "用户住在北京海淀区")
        assert score >= 5

    def test_aggregate_scores(self):
        results = [
            JudgeResult(
                question_id="q1", query="a", expected="b",
                category="cat-a", score=8, reason="",
                context_preview="", raw_recall={},
            ),
            JudgeResult(
                question_id="q2", query="a", expected="b",
                category="cat-a", score=6, reason="",
                context_preview="", raw_recall={},
            ),
            JudgeResult(
                question_id="q3", query="a", expected="b",
                category="cat-b", score=10, reason="",
                context_preview="", raw_recall={},
            ),
        ]
        agg = aggregate_scores(results, ["cat-a", "cat-b"])
        assert agg["total_questions"] == 3
        assert agg["overall"] == pytest.approx(80.0)
        assert agg["categories"]["cat-a"]["score"] == pytest.approx(70.0)
        assert agg["categories"]["cat-b"]["score"] == pytest.approx(100.0)

    def test_aggregate_empty(self):
        agg = aggregate_scores([], [])
        assert agg["overall"] == 0
        assert agg["total_questions"] == 0


# ====================================================================
# Runner tests
# ====================================================================

class TestRunner:
    def test_benchmark_config_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.target_url == "http://localhost:18800"
        assert cfg.provider == "deepseek"
        assert cfg.scale == "quick"
        assert "longmemeval-s" in cfg.dataset_names

    def test_targets_coverage(self):
        """TARGETS should cover all 7 dataset categories."""
        expected_keys = [
            # LongMemEval-S
            "single-session-user", "single-session-assistant",
            "single-session-preference", "knowledge-update",
            "temporal-reasoning", "multi-session",
            # Chinese
            "temporal-zh", "idiom-zh", "name-disambig", "profile-zh",
            # Multimodal v2
            "mm-basic-recall", "mm-precision-retrieve", "mm-noise-resist",
            "mm-abstract-query", "mm-temporal-disambig", "mm-cross-ref",
            # LoCoMo
            "recall-accuracy", "multi-turn", "logical-reasoning",
            # Conflict-resolution
            "contradiction-detect", "supersede-correctness", "expired-info",
            # Profile-accuracy
            "preference-extract", "habit-extract", "fact-extract", "timeliness",
            # Stress-scale
            "10k-latency", "50k-latency", "100k-accuracy",
        ]
        for key in expected_keys:
            assert key in TARGETS, f"Missing target: {key}"
            assert 0 < TARGETS[key] <= 100

    def test_targets_values(self):
        assert TARGETS["single-session-user"] >= 95
        assert TARGETS["single-session-assistant"] >= 93
        assert TARGETS["temporal-reasoning"] >= 80
        assert TARGETS["multi-session"] >= 73

    def test_benchmark_report_dataclass(self):
        report = BenchmarkReport(
            overall=85.0,
            datasets={},
            timestamp="2024-01-01T00:00:00",
            qmemory_version="0.1.0",
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            scale="quick",
            duration=10.0,
            target_url="http://localhost:18800",
        )
        assert report.overall == 85.0

    def test_comparison_config_dataclass(self):
        cfg = ComparisonConfig(
            providers=[
                {"provider": "deepseek", "api_key": "k1"},
                {"provider": "zhipu", "api_key": "k2"},
            ],
        )
        assert len(cfg.providers) == 2

    def test_comparison_report_dataclass(self):
        report = ComparisonReport(
            reports={},
            timestamp="2024-01-01T00:00:00",
            scale="quick",
            target_url="http://localhost:18800",
        )
        assert report.scale == "quick"

    def test_report_to_dict(self):
        report = BenchmarkReport(
            overall=85.0,
            datasets={},
            timestamp="2024-01-01T00:00:00",
            qmemory_version="0.1.0",
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            scale="quick",
            duration=10.0,
            target_url="http://localhost:18800",
        )
        d = report_to_dict(report)
        assert isinstance(d, dict)
        assert d["overall"] == 85.0
        assert d["llm_provider"] == "deepseek"


# ====================================================================
# i18n tests
# ====================================================================

class TestI18n:
    def test_default_language_is_chinese(self):
        from qmemory_bench.ui.i18n import get_lang, set_lang, t
        set_lang("zh")  # reset to default
        assert get_lang() == "zh"

    def test_chinese_translation(self):
        from qmemory_bench.ui.i18n import set_lang, t
        set_lang("zh")
        assert t("app_title") == "QMemory 评测工具"
        assert t("connected") == "已连接"

    def test_english_translation(self):
        from qmemory_bench.ui.i18n import set_lang, t
        set_lang("en")
        assert t("app_title") == "QMemory Benchmark"
        assert t("connected") == "Connected"
        set_lang("zh")  # restore

    def test_missing_key_returns_key(self):
        from qmemory_bench.ui.i18n import t
        assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"

    def test_set_invalid_lang_defaults_zh(self):
        from qmemory_bench.ui.i18n import set_lang, get_lang
        set_lang("fr")
        assert get_lang() == "zh"  # falls back to zh

    def test_all_keys_have_both_languages(self):
        from qmemory_bench.ui.i18n import TEXTS
        for key, langs in TEXTS.items():
            assert "zh" in langs, f"Key '{key}' missing 'zh'"
            assert "en" in langs, f"Key '{key}' missing 'en'"


# ====================================================================
# Tokenizer tests (qmemory side)
# ====================================================================

class TestTokenizerIntegration:
    def test_import_tokenizer(self):
        from qmemory.search.tokenizer import tokenize_query, tokenize_for_fts
        assert callable(tokenize_query)
        assert callable(tokenize_for_fts)

    def test_tokenize_chinese_query(self):
        from qmemory.search.tokenizer import tokenize_query
        result = tokenize_query("用户在哪里工作")
        assert result
        assert "OR" in result or '"' in result

    def test_tokenize_english_query(self):
        from qmemory.search.tokenizer import tokenize_query
        result = tokenize_query("where does the user work")
        assert result
        assert '"' in result

    def test_tokenize_for_fts_chinese(self):
        from qmemory.search.tokenizer import tokenize_for_fts
        result = tokenize_for_fts("我在字节跳动担任后端工程师")
        assert result
        assert " " in result

    def test_tokenize_empty(self):
        from qmemory.search.tokenizer import tokenize_query
        result = tokenize_query("")
        assert result == ""
