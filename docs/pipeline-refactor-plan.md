# QMemory 管线重构方案

> 创建: 2026-03-31
> 更新: 2026-04-02
> 基于: 30+ 次 benchmark 实验 + golden DB 根因分析 + Phase 9C 写侧验证
> 状态: Phase 9C 完成, 当前 **93.6%** (51.4% → 93.6%, +42.2)

---

## 一、根因诊断

### 1. Golden DB 不是瓶颈 — Decay 才是

| 对比 | 分数 | 说明 |
|------|------|------|
| golden_baseline (fresh ingest) | 73.2% | 刚创建的 golden DB |
| v030_arch_eval1 (skip-ingest golden DB) | 71.6% | 同 DB, -1.6 |
| baseline_std10 (golden DB, standard) | 51.4% | 同 DB, standard scale |

Fresh ingest vs skip-ingest 仅差 **1.6 分**，说明 DB 中的记忆数据本身没问题。

**真正问题: Decay 引擎在服务器运行期间把 95% 的记忆 archive 掉了！**

```
golden_locomo-real:    370 total → 28 active (92% archived)
golden_locomo:         136 total →  3 active (98% archived)
golden_longmemeval-s:  138 total → 10 active (93% archived)
golden_multimodal:     138 total →  3 active (98% archived)
golden_qmemory-chinese: 102 total →  1 active (99% archived)
```

Archive 原因:
- `temporal_expiry`: 329 条 — 2026-03-30 17:08 一次性清除
- `noise_low_importance`: 508 条 — 分 6 批清除 (importance < 0.08)

**关键: 高价值记忆也被误杀：**
- "Caroline is keen on counseling or working in mental health" (imp=0.8) → `temporal_expiry`
- "Melanie painted a lake sunrise last year" (imp=0.5) → `temporal_expiry`

### 2. Sprint3 (64.8%) vs Baseline (51.4%) 差距来源

| 因素 | 影响 |
|------|------|
| 不同题目集 (720q custom vs 249q release-full) | 主要 |
| Sprint3 额外包含 6 个较简单数据集 | 拉高均分 |
| Decay 导致 golden DB 退化 | 次要 (先于 decay 的 golden_baseline=73.2) |

### 3. 15 次 recall 微调的天花板

```
baseline_std10:           51.4%  (起点)
→ quality_gate:           49.9%  (-1.5)
→ context_thin:           48.4%  (-3.0)
→ entity_boost:           49.8%  (-1.6)
→ prompt_reranker_opt:    49.4%  (-2.0)
→ supermemory_v1:         56.0%  (+4.6) ← 最大提升
→ entity_first:           56.1%  (+4.7) ← 天花板
→ simplified:             31.5%  (-19.9) ← 灾难
```

~~**Recall 侧优化天花板 ≈ +5 分**~~ **已突破**: Phase 1b 修复 graph_expand + reranker bug 后达到 **60.7% (+9.3)**。
这说明之前的天花板不是 recall 能力上限，而是 graph_expand 分数淹没 + reranker 池过小导致正确结果被排除。

```
→ phase0_restored:        54.6%  (+3.2)  ← 恢复 DB + 关 decay
→ phase1_simplified:      56.0%  (+4.6)  ← context 精简 12→5
→ phase1b_graph_fix:      60.7%  (+9.3)  ← graph_expand + reranker bug fix ★ 当前最高
```

---

## 二、当前管线全景

### 写路径 (Ingest)

```
对话 messages
  │
  ├─ Chunking (按 session 分块)
  │
  ├─ LLM 单次提取 → memories (atomic facts)
  │   ├─ 分类: fact/event/opinion/preference/plan/advice/habit
  │   ├─ 重要性评分
  │   └─ event_date 提取
  │
  ├─ Profile 更新 (key-value pairs)
  │
  ├─ Entity 识别 + 关系构建
  │
  ├─ Episode 合并 (L2 聚类)
  │
  ├─ Schema 发现 (L3 模式)
  │
  ├─ Timeline 事件生成
  │
  └─ Decay 调度 (服务器启动 60s 后, 每 24h)
      ├─ temporal_expiry: event/plan 类, event_date + 365 天过期 → archive  [Phase 0 已修复: 30→365天]
      ├─ staleness_decay: 180 天未访问 → importance *= 0.95  [Phase 0 已修复: 90→180天, 0.9→0.95]
      └─ noise_archival: importance < 0.02 → archive  [Phase 0 已修复: 0.08→0.02]
```

### 读路径 (Recall)

```
query
  │
  ├─ QueryAnalysis (意图/难度/实体识别)
  │   └─ difficulty: simple | temporal | multi_hop | conflict
  │
  ├─ Episode-first recall (narrative/multi_hop 时)
  │   └─ 搜 L2 episodes → 展开到 L1 memories
  │
  ├─ 路由:
  │   ├─ simple/temporal → _fast_recall
  │   │   ├─ hybrid_search (vector + FTS + temporal boost)
  │   │   ├─ entity_recall (补充)
  │   │   ├─ graph_expand (1-hop, score=0.010-0.012)  [Phase 1b 已修复]
  │   │   ├─ temporal_filter (superseded/auto_forget)
  │   │   ├─ session_siblings (multi_hop only)
  │   │   ├─ rerank (cross-encoder, diversity pool=50)  [Phase 1b 已修复]
  │   │   ├─ negation_filter
  │   │   └─ assemble_context
  │   │
  │   └─ multi_hop → _agent_recall
  │       ├─ LLM decompose_query → sub-queries
  │       ├─ multi-round hybrid_search + RRF merge
  │       ├─ graph_expand
  │       ├─ entity_recall + intersection
  │       ├─ temporal_filter
  │       ├─ rerank
  │       ├─ negation_filter
  │       └─ assemble_context
  │
  ├─ Episode merge (合并 episode-first 结果)
  │
  ├─ Post-pipeline enrichment (4 个步骤, Phase 1 已从 12 精简):
  │   ├─ 1. Timeline enrichment   ← temporal intent
  │   ├─ 2. Hierarchy L2 episodes ← always
  │   ├─ 3. Hierarchy L3 schemas  ← always  [已为空操作]
  │   └─ 4. Re-assemble context   ← if any enrichment added
  │   [已移除: narrative/snapshot/year_digest/entity_bindings/entity_profiles/derived_metrics/query-aware-completion]
  │
  └─ Cache + Return
```

### Context Assembly (5 sections — Phase 1 已从 12 精简)

| # | Section | 条件 | 典型大小 | 信噪比 |
|---|---------|------|----------|--------|
| 1 | **profile (用户画像)** | 有 profile_store | ~500 chars | 高 |
| 2 | **timeline (事件时间线)** | temporal intent | ~500 chars | 中 |
| 3 | **episodes (相关事件)** | L2 episodes | ~300 chars | 中 |
| 4 | **memories (相关记忆)** | always | ~800 chars | **高** |
| 5 | **temporal (时间推理)** | duration query | ~300 chars | 中 |

**已删除:** schemas, bindings, chains, narrative, snapshot, derived, chunks

**当前场景:**
- 最大 context ≈ 2400 chars ≈ **600 tokens**
- memories 在 context 中占比 ~33% (原先 ~13%)
- 信噪比大幅提升

---

## 三、核心问题清单

### P0 — Decay 引擎误杀 ✅ 已修复

| 问题 | 影响 | 方案 | 状态 |
|------|------|------|------|
| temporal_expiry 过于激进 | 把有价值的 event/plan 全部 archive | grace=365d | ✅ Phase 0 |
| noise_archival 阈值 0.08 太低 | 新提取记忆被清除 | threshold=0.02 | ✅ Phase 0 |
| staleness_decay 不合理 | benchmark 场景记忆不会被 access | 180d, factor=0.95 | ✅ Phase 0 |
| Decay 在启动后自动运行 | benchmark 跑着记忆被删 | 仍需手动重启后跑 | ⚠️ 已缓解 |

### P1 — Enrichment 路径冗余 ✅ 已修复

| 冗余 | 方案 | 状态 |
|------|------|------|
| timeline + chains + narrative | 合并为单一 timeline | ✅ Phase 1 |
| memories + chunks | 仅保留 memories | ✅ Phase 1 |
| entity_bindings + entity_profiles | 已删除 | ✅ Phase 1 |
| episodes + memories | 保留两者 (episodes 提供事件上下文) | ✅ Phase 1 |

### P1b — Graph Expand + Reranker Bug ✅ 已修复

| 问题 | 影响 | 方案 | 状态 |
|------|------|------|------|
| graph_expand 分数 0.40-0.45 | 淹没 RRF 真实分数 0.016-0.033 | 降至 0.010-0.012 | ✅ Phase 1b |
| reranker pool=15 按分数截断 | 正确结果永远进不了 reranker | 多样性感知池, cap=50 | ✅ Phase 1b |

### P2a — Golden DB 数据缺口 (新发现, 待修复)

| 问题 | 影响 | 方案 |
|------|------|------|
| Golden DB 仅含 sample_0 (quick) | locomo-real 约 60% 题目无法回答 | 补入 lc1/lc2 sessions |

### P2 — 提取质量 (中期)

| 问题 | 现象 | 方案 |
|------|------|------|
| LLM 单次提取遗漏 | 数字/细节丢失 (10%) | NLP 预提取 + 覆盖率验证 |
| Importance 评分不准 | 新提取全在 0.07-0.10, 后被 noise_archival 清除 | 固定 importance 基线或用相对排序 |
| 实体链接粒度不足 | 实体混淆 (8%) | 强化 entity resolution |

---

## 四、重构方案

### Phase 0: 紧急修复 — 恢复 Golden DB + 关闭 Decay ✅ 已完成

**结果: 54.6% (+3.2 vs baseline)**

1. ✅ 恢复 837 条 archived memories
2. ✅ Decay 参数调优: grace=365d, threshold=0.02, stale=180d, factor=0.95
3. ✅ 验证完成, longmemeval-s 因记忆过多产生噪音 (-2.20)

### Phase 1: Context 精简 + Recall 简化 ✅ 已完成

**结果: 56.0% (+4.6 vs baseline, +1.4 vs Phase 0)**

1. ✅ Context 12→5 sections (保留 profile/timeline/episodes/memories/temporal)
2. ✅ 删除 schemas/bindings/chains/narrative/snapshot/derived/chunks
3. ✅ Recall enrichment 简化: 移除 7 个后处理步骤
4. ✅ Decay 重新启用 (tuned params)

### Phase 1b: Graph Expand + Reranker Bug Fix ✅ 已完成

**结果: 60.7% (+9.3 vs baseline, +4.7 vs Phase 1) ★ 历史最高**

**根因:** graph_expand 给扩展记忆固定分数 0.40-0.45，是 RRF 分数 (0.016-0.033) 的 10-25 倍。
reranker 候选池 cap=15 且按分数排序 → 池中全是无关的 graph 扩展结果，正确结果被排除。

1. ✅ graph_expand.py: _GRAPH_EXPANSION_SCORE 0.45→0.012, _ENTITY_EXPANSION_SCORE 0.40→0.010
2. ✅ reranker.py: 多样性感知池选择 (originals 优先) + 池大小从 15 扩展至 50
3. ✅ longmemeval-s: 24.0%→49.7% (+25.7), 所有 6 个子类别显著提升

### Phase 2: 针对性优化 (进行中)

**目标: 针对最弱类别逐个诊断修复，争取 65%+**

#### 2.0 Golden DB 数据缺口修复 (待做)
```
问题: golden DB 用 quick scale (sample_0) 创建,
      但 benchmark 跑 standard scale (sample_0/1/2)。
      locomo-real 中 sample_1 (John/Gina/Jon) 和 sample_2 (Maria) 从未入库。
      导致 single-fact 14%、multi-hop 40% — 约 60% 题目无法回答。
方案: 重新用 standard scale 入库 locomo-real 的 lc1/lc2 sessions。
      或重建 golden DB (standard scale 全数据集)。
```

#### 2.1 Importance 评分修复 (已部分完成)
```
已完成: noise_archival 阈值从 0.08 降到 0.02, 基本不再误杀。
待验证: 是否需要进一步调整 LLM 提取的 importance 分布。
```

#### 2.2 NLP 预提取 (低优先级)
```
输入 chunk → regex 提取数字/日期/专有名词 → entity_hints[]
LLM 提取 → memories[]
→ 对比 entity_hints vs memories
→ 未覆盖的 → 补充提取 prompt
注: 写入侧优化, 对当前短板贡献有限。
```

#### 2.3 Temporal Expiry 精准化 (已部分完成)
```
已完成: grace_days 从 30→365 天, 基本不再过期。
待做: 长期方案 — 只对含临时指示词的记忆过期:
      "明天有考试" → 过期
      "Caroline went to a support group yesterday" → 不过期 (长期事实)
```

### Phase 3: 检索策略路由 (1-2 天)

**目标: 不同查询类型用不同检索策略, 减少无效搜索路径**

```
query → QueryAnalyzer
  │
  ├─ simple_fact → vector search → top-K → assemble
  │                (当前 _fast_recall 精简版)
  │
  ├─ entity_query → entity lookup → entity memories → assemble
  │                (跳过 vector search, 直接从 EntityStore)
  │
  ├─ temporal → timeline search + date-filtered memories → assemble
  │            (不需要 graph_expand, 不需要 entity_recall)
  │
  └─ multi_hop → decompose + per-sub-query 路由 → merge → assemble
                 (保持 _agent_recall, 但子查询也按类型路由)
```

**关键简化: 每种查询类型只走必要的搜索路径, 而不是所有路径都走一遍。**

---

## 五、验证策略

### 快速验证 (10 min)
```bash
# Phase 0 完成后: 恢复 DB + 关 decay, 用 quick scale 快速验证
python -m qmemory_bench run --preset release-full --scale quick \
  --target http://127.0.0.1:18800 --eval-user golden --skip-ingest
```

### 标准验证 (30 min)
```bash
# 每个 Phase 完成后: standard, max-per-category 10
python -m qmemory_bench run --preset release-full --scale standard \
  --max-per-category 10 --eval-user golden --skip-ingest
```

### 对比目标

| 阶段 | 目标 | 实际结果 | 状态 |
|------|------|----------|------|
| Phase 0 (恢复 DB) | 56%+ | **54.6%** | ✅ 接近 |
| Phase 1 (精简 context) | 58%+ | **56.0%** | ✅ 接近 |
| Phase 1b (graph+reranker fix) | — | **60.7%** | ✅ 计划外突破 |
| Phase 2 (针对性优化) | 65%+ | 进行中 | 🔄 |
| Phase 3 (路由优化) | 68%+ | 待做 | ⬜ |
| Phase 9C (手动注入) | — | **93.6%** | ✅ 写侧验证 |
| Phase 10B (部分re-ingest+temporal) | — | **94.0%** | ✅ 历史最高 |
| Phase 10 (全量re-ingest) | 88%+ | **79.6%** | ❌ 信噪比退化 |

---

## 六、优先级排序 (更新后)

```
✅ 完成  Phase 0: 恢复 DB + 调 Decay          → 54.6%
✅ 完成  Phase 1: Context 精简 12→5           → 56.0%
✅ 完成  Phase 1b: Graph + Reranker Bug Fix   → 60.7% ★
紧急    Phase 2.0: Golden DB 数据缺口修复     ← locomo-real lc1/lc2 未入库
高      Phase 2: 按类别针对性优化              ← 当前焦点
低      Phase 2.2: NLP 预提取                  ← 写入侧, 暂缓
低      Phase 3: 检索策略路由                  ← 待评估必要性
```

**当前瓶颈不再是 Decay, 而是:**
1. **Golden DB 数据缺口**: locomo-real 仅入库 sample_0 (Caroline/Melanie), sample_1/2 (John/Gina/Maria) 缺失 → single-fact 14%, multi-hop 40%
2. **Temporal 推理系统性弱**: locomo-real temporal 21%, longmemeval-s temporal-reasoning 21%
3. **Recall 精度仍有提升空间**: locomo logical-reasoning 40%, mm-noise-resist 57%

> **2026-04-02 更新**: Phase 9C 数据注入实验证明以上瓶颈分析方向正确但优先级有误。
> 真正的最大瓶颈是**写侧 extraction 粒度太细**，不是读侧。详见「七、Phase 10+ 写侧优化开发计划」。

---

## 七、Phase 10+ 写侧优化开发计划 (2026-04-02)

> 基于 Phase 9C 验证结论：**写（extraction 粒度）是最大瓶颈，不是读（retrieval）。**
> 数据注入(+28.1) >> 代码优化(+14.1) >> prompt优化(-0.6)

### 问题本质

当前 extraction 设计原则是 **"Atomic & Self-Contained"** — 每条记忆是一个独立的原子事实。
这对单事实查询(single-fact)非常好，但对以下查询类型形成系统性盲区：

| 查询类型 | 失败原因 | 典型例子 |
|----------|----------|----------|
| **跨事实时间计算** | 两个日期在不同记忆中，检索只命中一条 | "从四姑娘山到哈巴雪山的攀登间隔？" |
| **多实体关系组装** | 散落在多条记忆中的角色/关系，检索无法全部召回 | "林夏的合作伙伴分别是谁？各自的角色？" |
| **变化轨迹追踪** | 同一 key 的多次更新分散在不同记忆 | "用户的办公城市从哪变到哪？" |
| **综合评估** | 需要组合多维度信息进行推理 | "林夏的器材投资回报率如何？" |

Phase 9/9C 通过手动注入 101 条**预组装记忆**（直接包含完整答案），绕过了检索组装问题，
证明：**只要 DB 中存在一条完整匹配的记忆，检索 pipeline 有能力把它找到并给出满分答案。**

### 开发路线图

```
Phase 10: 写侧核心 — 组合记忆自动生成 (Composite Memory Synthesis)
    │
    ├─ 10A: 关系摘要层 (Relation Summary)           ← 核心方向 P0
    ├─ 10B: 时间轴合成层 (Temporal Synthesis)        ← 核心方向 P0
    ├─ 10C: 变化追踪层 (Evolution Tracking)          ← 核心方向 P1
    ├─ 10D: 搜索信噪比优化 (SNR Fix)                ← 核心方向 P0
    ├─ 10E: 读侧 Tool-Call Batch 模式               ← 对比验证 P1
    │
Phase 11: 读侧辅助 — 多步检索组装 (Multi-Step Assembly)
    │
    ├─ 11A: 增强 Query Decomposition               ← 辅助方向 P1
    ├─ 11B: 检索后 LLM 组装 (Retrieval-Augmented Assembly) ← 辅助方向 P2
    │
Phase 12: 闭环 — 写读联动 (Write-Read Feedback Loop)
    │
    └─ 12A: 检索失败驱动的补充写入                   ← 长期方向
```

---

### Phase 10A: 关系摘要层 (Relation Summary) — 最高优先级

**目标**: 在 extraction prompt 中增加 `composite` 类别，让 LLM 在一次调用中同时产出原子记忆和组合记忆。

**原理**: 不增加额外 LLM 调用。在现有 EXTRACTION_SYSTEM_PROMPT 的 category 枚举中新增 `"composite"` 类型，
并在 Rules 部分增加组合记忆的判断规则和 few-shot 示例。LLM 提取原子事实后，在同一次调用中
判断这些事实是否能组合回答更复杂的问题，如果能则额外输出 composite 记忆。

**已实现**: extractor.py 已修改
- category 枚举增加 `"composite"`
- Rules 增加 5 种组合模式：人物关系网络、事件因果链、数值汇总、偏好变化、时间轴序列
- 每种模式都有 few-shot 示例（原子→组合的转化）
- 限制每 chunk 最多 2-3 条 composite

**性能影响**: 零额外 LLM 调用。Prompt 增加约 40 行，可能微增 output token 数（每 chunk 多 2-3 条记忆）。

**代码修改点**:

| 文件 | 修改 | 状态 |
|------|------|------|
| `extractor.py` | EXTRACTION_SYSTEM_PROMPT 增加 composite 规则 + ExtractedMemory docstring 更新 | ✅ 已完成 |
| `conflict.py` | 无需修改 — 冲突检测不按 category 过滤，composite 与 atomic 自然共存 | ✅ 无需改 |
| `ingest.py` | 无需修改 — category 不做严格校验，仅 event/plan 有日期特殊逻辑 | ✅ 无需改 |

---

### Phase 10B: 时间轴合成层 (Temporal Synthesis) — 最高优先级

**目标**: 对含日期的事件记忆进行跨 session 时间轴合成。

**原理**: 当同一实体(人/项目)在多个 session 中产生了时间事件时，
自动生成「时间轴摘要」记忆，包含完整的时间线。

**触发时机**: 不在单次 ingest 内做（因为跨 session），而是作为后处理步骤：
- 方案 A: consolidation 阶段（现有 L1→L2 synaptic consolidation 后增加）
- 方案 B: 独立的 batch job，定期扫描 DB 中带日期的记忆

**推荐方案 B** — 更灵活，不阻塞 ingest：

```
Temporal Synthesis Job (定期/按需):

1. 扫描 DB: SELECT * FROM memories WHERE event_date IS NOT NULL AND archived_at IS NULL
   GROUP BY entity_name ORDER BY event_date

2. 对每个实体(人/项目)，如果有 ≥3 条带日期记忆:
   LLM 合成时间轴摘要:
   - "用户职业轨迹: 2024年腾讯P7(60万) → 2025-02 P8(85万) → 2025-07 字节P2-3(150万)"
   - "四姑娘山(2025-05-15) → 哈巴雪山(2025-08-20)，间隔3个月5天"
   
3. 写入 composite 记忆 (category="temporal_composite")
   metadata: {source_memory_ids: [...], synthesis_type: "timeline"}
```

**代码修改点**:

| 文件 | 修改 | 复杂度 |
|------|------|--------|
| 新增 `temporal_synthesis.py` | 时间轴合成逻辑 + LLM prompt | 中 |
| `consolidation.py` | 注册为 consolidation 后处理步骤（或独立 CLI 命令） | 低 |
| `memory_store.py` | 查询方法：按实体+日期分组查询 | 低 |

**预期效果**: 解决所有时间计算类问题（"间隔多久"、"按时间排列"、"一共多长时间"）。
对标 Phase 9C 注入的 31 条中约 8 条属于此类。

---

### Phase 10C: 变化追踪层 (Evolution Tracking) — 高优先级

**目标**: 当记忆发生 supersede（更新）时，自动生成「变化轨迹记忆」。

**原理**: 现有的 conflict detection 已经识别出 "updates" 关系并创建 superseded_by 链。
但这个链只在 DB schema 层，检索时不会被组装为可回答的文本。

**实现方案**: 在 conflict.py 的 supersede 处理逻辑中增加：

```python
# 现有逻辑 (conflict.py)
if action == "supersede":
    old_mem.superseded_by = new_mem.id
    # 新增: 生成变化轨迹记忆
    evolution_text = f"{old_mem.text} → {new_mem.text}"
    # 例: "用户办公城市: 深圳南山 → 北京朝阳 (2025-07)"
    insert_composite(
        text=evolution_text,
        category="evolution",
        importance=max(old_mem.importance, new_mem.importance),
        metadata={"old_id": old_mem.id, "new_id": new_mem.id, "shift_type": "evolution"}
    )
```

**更完善的方案**: 如果同一 key 发生了 3+ 次 supersede，生成完整轨迹：
```
"用户薪资变化: 60万(腾讯P7) → 85万(腾讯P8, 2025-02) → 150万(字节P2-3, 2025-07)"
```

**代码修改点**:

| 文件 | 修改 | 复杂度 |
|------|------|--------|
| `conflict.py` | supersede 时触发变化轨迹生成 | 低 |
| `memory_store.py` | 查询同一 key 的 supersede 链 | 低 |

**预期效果**: 解决 knowledge-update、变化追踪类问题。

---

### Phase 10E: 读侧 Tool-Call Batch 模式 — 对比验证

**目标**: 用单次 tool-call 替代当前的规则 QueryAnalysis + `_decompose_query`，统一读写两侧为 tool-call batch 架构。

**约束**: 读侧**最多只允许 1 次 LLM 调用**，与写侧一样采用 batch tool-call 模式。

**当前读侧 LLM 调用分析**:

| 调用点 | 路径 | LLM? | 说明 |
|--------|------|------|------|
| `analyze_query()` | 所有 | ❌ 规则 | 纯 regex + pattern matching |
| `_decompose_query()` | multi_hop | ✅ 1次 | JSON mode, 拆解为 2-4 子查询 |
| `rerank()` local BGE | 所有 | ❌ 本地 | BAAI/bge-reranker-v2-m3 |
| `rerank()` LLM fallback | 无BGE时 | ✅ 1次 | 仅 BGE 不可用时 |

**方案**: 定义 `plan_search` tool，1 次 tool-call 输出所有搜索参数：

```python
RECALL_TOOLS = [{
    "type": "function",
    "function": {
        "name": "plan_search",
        "description": "Plan memory search strategy for the given query",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-4 search queries. Simple=1, multi-hop=entity-anchored sub-queries."
                },
                "entity_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key people/org/project names mentioned"
                },
                "time_hint": {
                    "type": "string",
                    "description": "Time range if query involves dates"
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["simple", "temporal", "multi_hop"]
                }
            },
            "required": ["queries", "difficulty"]
        }
    }
}]
```

**执行流程 (严格 1 次 LLM)**:
```
query → LLM(plan_search tool) → {queries, entity_names, time_hint, difficulty}
                                         ↓
                              for each query: hybrid_search()  ← 纯代码
                                         ↓
                              RRF merge + graph_expand         ← 纯代码
                                         ↓
                              local BGE rerank                 ← 纯本地模型
                                         ↓
                              assemble_context → return
```

**优势**:
1. **兼容 DeepSeek**: 单次 tool_call，不需要 parallel function calling
2. **更准确**: LLM entity 识别 > regex NER（尤其中文名），LLM difficulty 判断 > 规则
3. **统一架构**: 写侧 `save_extraction`，读侧 `plan_search`，都是 batch tool-call
4. **延迟可控**: simple 查询 ~50 output tokens，延迟增加 ~1-2s

**取舍**: simple/temporal 路径从 0 次 LLM 调用变为 1 次。换来更准确的意图理解和实体识别。

**验证方案**: Phase 10D benchmark 结果出来后，实现 10E 并做 A/B 对比：
- 10D (规则 QueryAnalysis) vs 10E (tool-call plan_search)
- 若 10E ≥ 10D → 全面切换为 tool-call 模式
- 若 10E < 10D → 仅对 multi_hop 启用，simple/temporal 保持规则模式

**代码修改点**:

| 文件 | 修改 | 复杂度 |
|------|------|--------|
| `recall.py` | 新增 `_plan_search_via_tool()` 替代 `analyze_query` + `_decompose_query` | 中 |
| `recall.py` | `recall()` 入口路由：有 LLM 时走 tool-call，无 LLM 时降级为规则 | 低 |
| `query_analysis.py` | 保留规则模式作为 fallback（无 LLM 时） | 无需改 |

---

### Phase 11A: 增强 Query Decomposition — 辅助

**目标**: 让 recall 时的 query decomposition 更好地拆解复合问题。

**当前问题**: DECOMPOSE_SYSTEM 生成的子查询有时过于宽泛或重叠，
导致多轮搜索召回的记忆集合高度重叠而非互补。

**改进方向**:

```
现有: "林夏的合作伙伴和各自角色？"
  → ["林夏的合作伙伴有谁", "林夏合作伙伴的角色"]  ← 两个子查询太相似

改进: 引入「面向检索的拆解」策略
  → ["林夏 小柯 合作", "林夏 周瑶", "林夏 阿坤 视频"]  ← 每个子查询锚定一个实体

或: 引入「先实体后细节」两阶段
  第一轮: "林夏的合作伙伴" → 召回含人名的记忆 → 提取到小柯/周瑶/阿坤
  第二轮: 针对每个人名分别搜索 → "小柯 灯光"、"周瑶 模特"、"阿坤 DJI"
```

**代码修改点**:

| 文件 | 修改 | 复杂度 |
|------|------|--------|
| `recall.py` | DECOMPOSE_SYSTEM prompt 增加「实体锚定」指令 | 低 |
| `recall.py` | _agent_recall 增加「二轮细化」：第一轮结果中提取实体名，生成第二轮子查询 | 中 |

**预期效果**: 提升 multi-hop 和人物关系类问题的召回率。
但效果受限于 DB 中是否存在可被召回的信息 — 如果信息太碎，再好的检索也无法组装。

**优先级**: 低于 Phase 10（写侧已证明比读侧优化有效 2×+），作为补充。

---

### Phase 11B: 检索后 LLM 组装 — 辅助

**目标**: 当检索到多条相关但分散的记忆时，用 LLM 从碎片中推导组合答案。

**当前状态**: assembler.py 做 context assembly 只是简单拼接记忆文本，
没有「组装推理」步骤。

**改进方向**: 在 recall 返回后增加可选的组装步骤：

```
memories = recall(query)

if query.difficulty in ("multi_hop", "temporal"):
    assembled = llm_assemble(query, memories)
    # assembled.text = "根据记忆推算：四姑娘山(5月15日)到哈巴雪山(8月20日)间隔3个月5天"
```

**风险**: 增加 1 次 LLM 调用延迟（3-5s）。仅对 multi_hop/temporal 查询启用。

**优先级**: 最低。如果 Phase 10 的写侧优化效果好，Phase 11B 可能不需要。

---

### Phase 12A: 写读联动 (长期方向)

**目标**: 检索失败时自动触发补充写入 — 本质是自动化版的 Phase 9/9C 手动注入。

记录 recall 失败 query pattern → 回溯原始对话 → 针对性生成组合记忆 → 写入 DB。
复杂度高，优先级最低。

---

### 优先级总图 (更新 2026-04-02)

```
✅ 完成  Phase 0-8:  代码+DB修复                → 65.5%  (+14.1)
✅ 完成  Phase 9/9C: 手动数据注入验证            → 93.6%  (+28.1, 证明写侧是关键)
✅ 完成  Phase 9B:   Prompt优化验证              → 82.3%  (-0.6, 证明prompt无效)
✅ 完成  Phase 10A:  关系摘要合成 (Composite)     → extraction prompt 增加 composite 类别
✅ 完成  Phase 10A+: Tool-call batch 模式         → 移除 JSON 提取，统一 tool-call
✅ 完成  Phase 10B:  时间轴合成 (Temporal Syn)     → temporal_synthesis.py 实现
✅ 完成  Phase 10C:  变化追踪 (Evolution)          → evolution_tracking.py + ingest hook
✅ 完成  Phase 10 全量验证                        → 79.6% (详见下方)
────────── 以下为待开发 ──────────
🔴 P0    Phase 10D: 搜索信噪比优化               ← 记忆数暴增导致搜索精度下降 (benchmark 运行中)
🟡 P1    Phase 10E: 读侧 Tool-Call Batch 模式     ← 10D 结果出来后 A/B 对比验证
🟡 P2    Phase 11A: 增强 Query Decomposition       ← 检索侧补充
⚪ P3    Phase 11B: 检索后 LLM 组装                ← 仅在10D效果不足时启用
⚪ P4    Phase 12A: 写读联动闭环                   ← 长期方向
```

---

### Phase 10 实际执行结果 (2026-04-02)

#### 实现内容

| 子阶段 | 内容 | 代码 |
|--------|------|------|
| **10A** | extraction prompt 增加 `composite` 类别（5 种组合模式） | `extractor.py` |
| **10A+** | Tool-call batch 模式替代 JSON 提取（`save_extraction` 统一入口） | `extractor.py` |
| **10B** | 跨 session 时间轴合成（按实体分组，≥3 条带日期记忆触发合成） | `temporal_synthesis.py` |
| **10C** | Supersede 时自动生成变化轨迹记忆 | `evolution_tracking.py` + `ingest.py` hook |

#### 全量 Re-Ingest 流程

1. Archive Phase 9C 手动注入记忆 (31 条) + 旧 temporal composites (96 条)
2. 通过 API 重新 ingest 全部 188 chunks（186/188 成功，2 个超时后重试成功）
3. 运行 10B temporal synthesis (5 用户: 50+12+16+18+6 = 102 条时间线)
4. 总耗时: ~3.3 小时 (re-ingest) + ~5 分钟 (temporal synthesis)

#### DB 状态

| 指标 | Phase 9C | Phase 10 全量 | 变化 |
|------|----------|--------------|------|
| Active memories | 2,172 | 4,302 | +2,130 |
| Temporal composites | 96 | 102 | +6 |
| Evolution composites | 0 | 162 | +162 |
| Extraction composites | 0 | 271 | +271 |
| 手动注入记忆 | 31 | 0 | -31 |

#### Benchmark 结果: **79.6%**

| 对比基准 | Phase 10 | Delta | 说明 |
|----------|----------|-------|------|
| Phase 8 (65.5%) | 79.6% | **+14.1** | 自动管线有效提升 |
| Phase 9C (93.6%) | 79.6% | **-14.0** | 远未达到手动注入水平 |
| Phase 10B 部分 (94.0%) | 79.6% | **-14.4** | 全量 re-ingest 反而降低 |

#### 分类别对比 (vs Phase 9C)

| 类别 | Phase 9C | Phase 10 | Delta | 根因 |
|------|----------|----------|-------|------|
| locomo | 91.1 | 80.9 | -10.2 | logical-reasoning -27, temporal -8 |
| **locomo-real** | **95.0** | **67.7** | **-27.3** | multi-hop -53, single-fact -47 ← 最大退化 |
| longmemeval-s | 93.0 | 75.8 | -17.2 | temporal-reasoning -58 ← 极端退化 |
| multimodal | 92.8 | 85.5 | -7.3 | mm-noise-resist -32 |
| qmemory-chinese | 96.0 | 88.0 | -8.0 | temporal-zh -24, name-disambig -12 |

#### 根因分析

**核心问题: 记忆数量暴增导致搜索信噪比严重下降。**

1. **记忆碎片化**: 4,302 active memories (vs Phase 9C 的 2,172)，翻倍。
   搜索返回 limit=10 中，低质量碎片记忆挤占了高价值记忆的位置。
   
2. **Composite 质量不足**: 自动生成的 271 条 extraction composites 信息密度
   远低于手动注入的 31 条。手动注入的每条都是针对特定问题的完整答案，
   自动生成的只是同 chunk 内事实的简单组合。

3. **Temporal reasoning 崩溃**: longmemeval-s temporal-reasoning 97→39 (-58)。
   更多记忆导致时间相关搜索返回更多不相关的日期事件，干扰推理。

4. **locomo-real 严重退化**: multi-hop 100→47, single-fact 90→43。
   原因可能是 locomo-real 记忆从 ~900 暴增到 2,833，噪音比极高。

**关键洞察**: Phase 10 证明**写侧提取量不是越多越好**。
手动注入 31 条精准记忆 > 自动提取 2,130+ 条碎片记忆。
下一步应关注**搜索精度**而非继续增加提取量。

#### 下一步方向 (Phase 10D)

```
问题: 记忆数 4,302 远超 Phase 9C 的 2,172，但质量/精度下降。
方向:
  1. 增大搜索 limit (10 → 20-30)，给 reranker 更多候选
  2. 优化 reranker 对 composite/temporal_composite 的加权
  3. 提取阶段去重/合并相似记忆，减少总量
  4. 搜索时对碎片记忆降权 (importance < 0.3 的记忆搜索权重减半)
```

### Phase 10 验证计划 (原始)

1. ~~**清除注入数据**: 从 golden DB 中删除 Phase 9/9C 手动注入的 101 条记忆~~ ✅
2. ~~**重新 ingest**: 用改进后的 extraction pipeline 重新 ingest 全部对话数据~~ ✅
3. ~~**跑 benchmark**: 如果 10A+10B+10C 能自动生成等价于手动注入的组合记忆~~ ✅ 结果: 79.6%
4. ~~**对比**: Phase 10 auto-generated vs Phase 9C manual-injected~~ ✅ 差距 -14.0

**成功标准**: ❌ 未达到
- Phase 10A+10B: ≥ 88% → 实际 79.6% (差 8.4 分)
- 手动注入记忆数 → 0 ✅ (已实现，但分数不够)

### 剩余 12 道低分题归因

| 题目 | 分数 | 失败类型 | 对应 Phase |
|------|------|----------|-----------|
| Melanie ally to transgender? | 3 | 推理深度不足(需文化理解) | 超出范围 |
| What has Melanie painted? | 2 | DB缺数据(horse/sunrise未入库) | 10A 或重新ingest |
| 林夏客户网络怎么建立的 | 6 | 缺因果链+时序 | 10A+10B |
| 林夏合作伙伴各自角色 | 0 | 多实体关系未组装 | **10A** |
| 林夏器材投资回报率 | 6 | 多数值综合评估 | **10A** |
| 用户房产情况 | 5 | 部分数据未召回(北京购房计划) | 10A |
| 用户运动爱好变化 | 6 | 变化追踪 | **10C** |
| 总结2025年人生轨迹 | 6 | 多维度综合 | 10A+10B |
| 出差城市按时间排列 | 4 | 时间轴排序 | **10B** |
| 少年英雄动画进展 | 0 | DB缺数据(哪吒项目未入库) | 重新ingest |
| 角色设计分工 | 4 | 多实体+分工关系 | **10A** |
| 花果山场景素材来源 | 6 | 跨来源关联 | **10A** |

---

## 八、Phase 15 Ingest 修复验证 (2026-04)

### 修复内容

| Fix | 文件 | 说明 |
|-----|------|------|
| **A: Temporal Composite 刷新** | `temporal_synthesis.py` | 已有 composite 的 source_ids 与当前不同时自动 supersede，解决陈旧/矛盾时间线 |
| **B: 相对日期标准化** | `ingest.py` | 20+ 中英文模式：大前天/前天/昨天/去年/前年/N天前/N年前/yesterday/last year 等 |

### 验证结果 (部分 re-ingest: longmemeval-s + qmemory-chinese)

| Category | Phase 14 (旧 DB) | Phase 15 (re-ingest) | Delta |
|----------|------------------|---------------------|-------|
| **temporal-zh** | 76% | **100%** | **+24%** |
| **single-session-preference** | 60-68% | **92%** | **+24~32%** |
| **temporal-reasoning** | 54-59% | 56% | +2 (瓶颈在 LLM 排序推理) |
| **knowledge-update** | - | 92% | 稳定 |
| **Overall (2 datasets)** | - | **90.8%** | **PASS** |
| DB 记忆数 | 4461 | 4022 | -439 (更好的去重) |

**结论**: Ingest 修复对数据质量提升显著，temporal-zh 和 preference 类别大幅改善。
temporal-reasoning 瓶颈不在数据，而在 LLM 对多事件时间排序的推理能力。

---

## 九、工程化路线图 (2026-04+)

> 基于生产就绪度评估。当前状态：Alpha 级，适合单用户本地部署，需强化多租户/云部署能力。

### 已具备

| 能力 | 状态 |
|------|------|
| FastAPI + async/await | ✅ |
| Bearer Token 认证 | ✅ |
| CORS | ✅ (当前 allow_origins=["*"]，需收紧) |
| Health Check | ✅ /v1/health/ |
| Background Jobs | ✅ decay/completeness worker |
| CLI (serve/export/import) | ✅ |
| mypy strict + ruff linting | ✅ |
| PyInstaller 打包 | ✅ |

### 工程化优先级

```
🔴 P0 — 安全与稳定性 (立即)
  ├─ E1: 请求限流 (slowapi/自定义中间件)
  ├─ E2: 请求体大小限制
  └─ E3: CORS 收紧 (配置化 origins)

🟡 P1 — 部署与 CI (1 周内)
  ├─ E4: Dockerfile + docker-compose.yml
  ├─ E5: GitHub Actions CI (lint + test + build)
  └─ E6: 生产部署文档

🟢 P2 — 可观测性 (2 周内)
  ├─ E7: Prometheus metrics (请求延迟/记忆数/ingest速率)
  ├─ E8: 结构化日志 (JSON logging)
  └─ E9: Health check 增强 (DB/模型/LLM 状态)

⚪ P3 — 性能与扩展 (后续)
  ├─ E10: 异步 ingest (立即返回 202, 后台处理)
  ├─ E11: SQLite WAL 模式 + 连接池
  └─ E12: 写入路径优雅降级 (embedding 模型不可用时的行为)
```
