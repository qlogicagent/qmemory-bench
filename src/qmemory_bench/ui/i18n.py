"""Internationalization (i18n) — Chinese / English support.

Usage:
    from qmemory_bench.ui.i18n import t, set_lang, get_lang
    set_lang("zh")   # or "en"
    label = t("app_title")  # → "QMemory 评测工具"
"""

from __future__ import annotations

_current_lang: str = "zh"

TEXTS: dict[str, dict[str, str]] = {
    # ── Header ──
    "app_title":        {"zh": "QMemory 评测工具", "en": "QMemory Benchmark"},
    "app_subtitle":     {"zh": "多维度记忆系统评测工具", "en": "Multi-dimensional Memory Evaluation Tool"},
    "light":            {"zh": "亮色", "en": "Light"},
    "dark":             {"zh": "暗色", "en": "Dark"},
    "lang_label":       {"zh": "EN", "en": "中文"},

    # ── Server ──
    "server_address":   {"zh": "QMemory 服务地址", "en": "QMemory Server URL"},
    "reset_memory":     {"zh": "重置记忆库", "en": "Reset Memory"},
    "reset_confirm_title": {"zh": "确认重置记忆库？", "en": "Confirm Memory Reset?"},
    "reset_confirm_msg":   {"zh": "此操作会清空服务器上所有记忆数据，不可恢复。",
                            "en": "This will erase ALL memory data on the server. Irreversible."},
    "reset_success":    {"zh": "记忆库已重置", "en": "Memory reset successfully"},
    "reset_fail":       {"zh": "重置失败", "en": "Reset failed"},
    "cancel":           {"zh": "取消", "en": "Cancel"},
    "confirm_reset":    {"zh": "确认重置", "en": "Confirm"},

    # ── Connection ──
    "connected":        {"zh": "已连接", "en": "Connected"},
    "disconnected":     {"zh": "未连接", "en": "Disconnected"},
    "checking":         {"zh": "检测中...", "en": "Checking..."},
    "server_version":   {"zh": "服务版本", "en": "Server version"},
    "memories_count":   {"zh": "条记忆", "en": "memories"},

    # ── LLM ──
    "llm_config":       {"zh": "LLM 配置（用于记忆提取 + 评分）", "en": "LLM Config (extraction + scoring)"},
    "api_key":          {"zh": "API Key", "en": "API Key"},
    "model_hint":       {"zh": "Model（留空使用默认）", "en": "Model (blank = default)"},
    "embedding_note":   {"zh": "Embedding：本地 BAAI/bge-small-zh-v1.5（自动）",
                         "en": "Embedding: Local BAAI/bge-small-zh-v1.5 (auto)"},

    # ── Compare ──
    "compare_mode":     {"zh": "对比模式", "en": "Compare Mode"},
    "compare_hint":     {"zh": "同一评测集，不同 LLM 评分对比，帮助选择性价比最高的模型",
                         "en": "Same benchmark, different LLMs side-by-side — find the best value"},
    "compare_section_title": {"zh": "对比 LLM 配置", "en": "Comparison LLM(s)"},
    "add_provider":     {"zh": "+ 添加对比 LLM", "en": "+ Add Provider"},
    "remove":           {"zh": "删除", "en": "Remove"},
    "provider_label":   {"zh": "对比 LLM", "en": "Compare LLM"},
    "base_llm_tag":     {"zh": "🏷️ 基准 LLM（其他模型将与此对比）",
                         "en": "🏷️ Base LLM (others are compared against this)"},

    # ── Methodology / Trust ──
    "method_title":     {"zh": "📋 评测方法论与可信度说明（点击展开）",
                         "en": "📋 Methodology & Trust Statement (click to expand)"},
    "method_overview":  {"zh": "本工具采用 6 重验证机制，确保评测结果客观、可复现、可信赖。所有数据集均标注来源，评分逻辑完全透明。",
                         "en": "This tool uses 6 verification mechanisms to ensure results are objective, reproducible, and trustworthy. All datasets are source-attributed and scoring logic is fully transparent."},
    "method_cross_title":  {"zh": "交叉验证评分", "en": "Cross-Validation Scoring"},
    "method_cross_body":   {"zh": "每道题同时使用 LLM 评分（0-10 分，含自然语言推理）和关键词匹配评分。若 LLM 评分异常（如评分报文解析失败），自动回退到关键词匹配，确保不会因为单一评分方式的缺陷引入偏差。最终得分为两种方法的加权融合。",
                            "en": "Each question is scored by both LLM judge (0-10 with reasoning) and keyword matching. If LLM scoring fails (e.g. parse error), automatic fallback to keyword matching prevents single-method bias. Final score is a weighted fusion of both methods."},
    "method_noise_title":  {"zh": "噪声注入 + 多模态精准检索", "en": "Noise Injection + Multimodal Precision Retrieval"},
    "method_noise_body":   {"zh": "数据集中包含大量干扰会话（LongMemEval-S: 20 会话中 12 个噪声；多模态 v2: 20 会话中 8 个高混淆噪声）。多模态评测以动画设计师 2 年工作流为场景，噪声包括同名角色的不同来源（游戏截图 vs 项目设计图）、同一地点不同拍摄者的照片、同类产品的 UI 截图等。专门测试：在大量相似数据和长期积累噪声下，能否用模糊/抽象的文本查询精准定位到正确的源数据。",
                            "en": "Datasets include large volumes of distractors (LongMemEval-S: 12/20 noise; Multimodal v2: 8/20 high-confusion noise). Multimodal eval uses a 2-year animator workflow: noise includes same-named characters from different sources (game screenshots vs project designs), same-location photos by different people, similar product UIs, etc. Tests whether the system can precisely locate the correct source data using vague/abstract text queries under heavy noise."},
    "method_temporal_title": {"zh": "多会话 + 时间推理", "en": "Multi-Session & Temporal Reasoning"},
    "method_temporal_body":  {"zh": "信息分散在多个会话中（跨度长达 1 年），提问需要跨会话整合。时间推理题要求从相对时间表达（如「去年夏天」「两个月前」）计算出准确日期。知识更新题检测系统是否能区分用户历史状态与当前状态（如换工作、搬家）。",
                              "en": "Information is spread across sessions spanning up to 1 year. Questions require cross-session integration. Temporal reasoning questions demand exact date calculation from relative expressions (e.g. 'last summer', '2 months ago'). Knowledge update questions test whether the system distinguishes historical vs. current user state."},
    "method_repro_title":  {"zh": "可复现性保证", "en": "Reproducibility Guarantee"},
    "method_repro_body":   {"zh": "每次评测使用独立的评测用户 ID，注入前自动清理历史数据，评测结束后再次清理。数据集内容固定、题目顺序固定、评分规则透明。同一服务同一 LLM 条件下多次运行结果差异极小（±2%）。",
                            "en": "Each run uses an isolated evaluation user ID with automatic cleanup before and after. Dataset content, question order, and scoring rules are fixed. Under identical server & LLM conditions, result variance across runs is minimal (±2%)."},
    "method_scoring_title": {"zh": "评分体系说明", "en": "Scoring Methodology"},
    "method_scoring_body":  {"zh": "LLM 评分采用 0-10 分制，由评分 LLM 对比「标准答案」与「记忆系统返回的答案」进行打分。得分 ≥7 视为通过。各维度目标分（如 knowledge-update: 85%）参考学术基准 LongMemEval (arXiv:2410.10813) 中的人类上限指标。总分 ≥85% 视为达标，表明记忆系统已达到可商用的质量水平。",
                             "en": "LLM scoring uses a 0-10 scale, comparing 'gold answer' with 'memory system response'. Score ≥7 = pass. Category targets (e.g. knowledge-update: 85%) reference human-ceiling metrics from LongMemEval (arXiv:2410.10813). Overall ≥85% = acceptable for production-grade memory quality."},
    "method_dataset_title": {"zh": "数据集溯源", "en": "Dataset Provenance"},
    "method_dataset_body":  {"zh": "所有数据集均标注数据来源：学术论文启发（LongMemEval-S，引用 arXiv:2410.10813）、合成数据集（对齐学术评测维度）、自建中文场景集（覆盖中文特有挑战：成语、歧义、口语表达）、多模态评测集（图文音混合记忆）。每个数据集可独立开关，按需组合。",
                             "en": "All datasets are source-attributed: academic-inspired (LongMemEval-S, citing arXiv:2410.10813), synthetic (aligned with academic dimensions), custom Chinese scenario sets (covering Chinese-specific challenges: idioms, ambiguity, colloquial), and multimodal sets (image/doc/audio). Each dataset can be independently toggled."},
    "method_trust":     {"zh": "✅ 本工具的设计目标是让每一位用户都能验证评测流程的公正性。评测代码开源，数据集来源清晰，评分规则透明可审计。",
                         "en": "✅ This tool is designed so every user can verify the fairness of the evaluation process. Code is open-source, dataset sources are clear, and scoring rules are fully auditable."},

    # ── Dataset ──
    "eval_scope":       {"zh": "评测范围", "en": "Evaluation Scope"},
    "current_selection": {"zh": "当前方案", "en": "Current Preset"},
    "preset_help":     {"zh": "默认采用“对外主评测”：仅纳入公开/权威主数据集；辅助与回归集用于补充说明与内部验收。", "en": "Default uses 'Public Main Benchmark': only public/authoritative core datasets count toward the external headline score; supporting and regression sets are for supplemental evidence and internal QA."},
    "preset_custom":   {"zh": "自定义组合", "en": "Custom Selection"},
    "source_label":     {"zh": "来源", "en": "Source"},
    "source_synthetic": {"zh": "合成（对齐学术框架维度）", "en": "Synthetic (aligned with academic framework)"},
    "source_academic":  {"zh": "学术论文启发", "en": "Academic paper-inspired"},
    "source_custom":    {"zh": "自建评测集", "en": "Custom dataset"},

    # ── Scale ──
    "eval_scale":       {"zh": "评测规模", "en": "Scale"},
    "scale_quick":      {"zh": "快速（每数据集 24-30 题，单集 ~2 分钟）", "en": "Quick (~24-30 Qs per dataset, ~2 min each)"},
    "scale_standard":   {"zh": "标准（每数据集 120-150 题，单集 ~15 分钟）", "en": "Standard (~120-150 Qs per dataset, ~15 min each)"},
    "scale_full":       {"zh": "完整（每数据集全量题目，单集 ~25 分钟）", "en": "Full (all questions per dataset, ~25 min each)"},

    # ── Run ──
    "start_eval":       {"zh": "▶ 开始评测", "en": "▶ Start Benchmark"},
    "start_compare":    {"zh": "▶ 开始对比评测", "en": "▶ Start Comparison"},
    "enter_api_key":    {"zh": "请输入 API Key", "en": "Please enter API Key"},
    "running":          {"zh": "评测进行中，请勿关闭…", "en": "Running, please wait…"},

    # ── Q&A Live Log ──
    "qa_log_title":     {"zh": "📝 实时问答日志", "en": "📝 Live Q&A Log"},
    "qa_expected":      {"zh": "期望答案", "en": "Expected"},
    "qa_recalled":      {"zh": "系统召回", "en": "Recalled"},
    "qa_reason":        {"zh": "评分理由", "en": "Reason"},
    "qa_items":         {"zh": "题", "en": "items"},

    # ── Progress ──
    "prog_connecting":  {"zh": "正在连接 QMemory 服务…", "en": "Connecting to QMemory…"},
    "prog_injecting":   {"zh": "注入会话数据", "en": "Injecting sessions"},
    "prog_evaluating":  {"zh": "评测提问中", "en": "Evaluating questions"},
    "prog_cleanup":     {"zh": "清理评测数据…", "en": "Cleaning up…"},
    "prog_done":        {"zh": "评测完成！", "en": "Evaluation complete!"},
    "prog_fail":        {"zh": "评测失败", "en": "Evaluation failed"},
    "duration":         {"zh": "耗时", "en": "Duration"},

    # ── Results ──
    "results_title":    {"zh": "评测结果", "en": "Benchmark Results"},
    "overall":          {"zh": "总分", "en": "Overall"},
    "target":           {"zh": "目标", "en": "Target"},
    "pass_label":       {"zh": "通过", "en": "PASS"},
    "fail_label":       {"zh": "未通过", "en": "FAIL"},
    "category":         {"zh": "维度", "en": "Category"},
    "score":            {"zh": "得分", "en": "Score"},
    "status":           {"zh": "状态", "en": "Status"},
    "count":            {"zh": "题数", "en": "Count"},
    "diff":             {"zh": "差异", "en": "Diff"},
    "dataset_label":    {"zh": "数据集", "en": "Dataset"},

    # ── Weakness ──
    "weakness_title":   {"zh": "短板分析", "en": "Weakness Analysis"},
    "below_target":     {"zh": "低于目标", "en": "below target"},
    "suggestion":       {"zh": "建议", "en": "Suggestion"},

    # ── Export ──
    "download_json":    {"zh": "💾 下载 JSON 报告", "en": "💾 Download JSON"},
    "copy_text":        {"zh": "📋 复制文字报告", "en": "📋 Copy Text Report"},
    "copied":           {"zh": "报告已复制到剪贴板", "en": "Report copied to clipboard"},
    "download_ready":   {"zh": "报告已准备好下载", "en": "Report ready for download"},

    # ── Compare results ──
    "compare_results":  {"zh": "对比结果", "en": "Comparison Results"},
    "best_value":       {"zh": "性价比最优", "en": "Best Value"},
    "winner":           {"zh": "最优", "en": "Winner"},

    # ── Eval page (LoCoMo) ──
    "eval_title":       {"zh": "LoCoMo 层级检索评测", "en": "LoCoMo Hierarchy Eval"},
    "eval_subtitle":    {"zh": "Flat (L1) vs Hierarchy (L2+L3) 对比验证", "en": "Flat (L1) vs Hierarchy (L2+L3) comparison"},
    "eval_no_results":  {"zh": "未找到评测结果文件。请先运行 eval_locomo.py", "en": "No eval results found. Run eval_locomo.py first."},
    "eval_flat":        {"zh": "Flat (L1)", "en": "Flat (L1)"},
    "eval_hier":        {"zh": "层级 (L2+L3)", "en": "Hierarchy (L2+L3)"},
    "eval_delta":       {"zh": "提升", "en": "Delta"},
    "eval_question":    {"zh": "问题", "en": "Question"},
    "eval_expected":    {"zh": "期望答案", "en": "Expected Answer"},
    "eval_cat":         {"zh": "维度", "en": "Category"},
    "eval_difficulty":  {"zh": "难度", "en": "Difficulty"},
    "eval_detail":      {"zh": "逐题详情", "en": "Per-Question Details"},
    "eval_summary":     {"zh": "总览", "en": "Summary"},
    "eval_nav":         {"zh": "评测结果", "en": "Eval Results"},
    "eval_search":      {"zh": "搜索验证", "en": "Search Test"},
    "eval_search_subtitle": {"zh": "在评测数据上交互式搜索，对比 flat vs hierarchy 召回效果",
                             "en": "Interactive search on eval data: compare flat vs hierarchy recall"},
    "eval_user_flat":   {"zh": "Flat 用户", "en": "Flat User"},
    "eval_user_hier":   {"zh": "Hierarchy 用户", "en": "Hierarchy User"},
    "eval_search_btn":  {"zh": "搜索", "en": "Search"},
    "eval_recalled_ctx": {"zh": "召回上下文", "en": "Recalled Context"},
}


def t(key: str) -> str:
    """Get translated text for the current language."""
    entry = TEXTS.get(key)
    if not entry:
        return key
    return entry.get(_current_lang, entry.get("zh", key))


def set_lang(lang: str) -> None:
    """Set the current language ('zh' or 'en')."""
    global _current_lang
    _current_lang = lang if lang in ("zh", "en") else "zh"


def get_lang() -> str:
    """Get the current language code."""
    return _current_lang
