# QMemory 架构级迭代路线图

> 目标: 构建多语言顶级 AI 记忆系统，而非依赖 prompt 微调
> 创建: 2026-03-30
> 基于: v0.1.0 benchmark 71.4%, v3.1 delta 分析

---

## 核心问题诊断

### 当前瓶颈不在 prompt，而在架构

对 144 题失败分析显示，**80% 的失败根因是结构性的**:

| 失败类型 | 占比 | 根因 | Prompt能修吗？ |
|----------|------|------|---------------|
| 事实从未被提取 | 18% | LLM 单次调用覆盖率不足 | ❌ prompt 越长遗漏越多 |
| 跨会话信息散落 | 20% | 无 session 关联机制 | ❌ 纯提取侧无解 |
| 数值/细节丢失 | 10% | LLM 天然倾向摘要 | ⚠️ 少量改善但不稳定 |
| 知识未更新 | 6% | profile supersede 不健壮 | ❌ 需要写入链路修复 |
| 实体混淆 | 8% | 实体链接粒度不足 | ⚠️ 部分缓解 |
| 检索不到 | 6% | 单一向量搜索不够 | ❌ 需要检索架构改进 |

**结论**: prompt 微调天花板约 +3~5%，且每次改动影响所有语言。突破 80% 必须走架构路线。

### 测试周期过长的根因

当前 release-full quick 跑一次 = **ingest (40min) + eval (20min) ≈ 60min+**。
每次代码变更都要重新 ingest，因为提取逻辑变了 → DB 里的记忆也变了。

---

## 三大架构支柱

```
          ┌───────────────────────────────────────┐
          │        顶级 AI 记忆系统               │
          └───────────┬───────────┬───────────────┘
                      │           │
         ┌────────────▼──┐  ┌────▼────────────┐  ┌──────────────┐
         │ 高效提取      │  │ 智能整合        │  │ 精准检索      │
         │ (Write Path)  │  │ (Maintenance)   │  │ (Read Path)   │
         └───────────────┘  └─────────────────┘  └──────────────┘
```

---

## 支柱 1: 高效提取 — 从「一次 LLM 祈祷」到「结构化管线」

### 现状问题
- 单次 LLM 调用 = 一个 prompt 试图提取所有类型的信息
- prompt 越长 → 注意力越分散 → 遗漏越多（v3 的教训）
- 中英文混合 prompt 导致跨语言干扰

### 架构方案: 多阶段提取管线

```
输入 chunk
   │
   ├─ Stage 1: 实体识别 (NLP, 非LLM)
   │   ├─ 人名/组织/地点: NER (spaCy/LAC)
   │   ├─ 数字/价格/日期: 正则
   │   └─ → entity_hints[]
   │
   ├─ Stage 2: LLM 事实提取 (当前 extractor, prompt 精简)
   │   └─ 不再需要数字/实体相关 few-shot (Stage 1 已覆盖)
   │
   ├─ Stage 3: 覆盖率验证 (NLP, 非LLM)
   │   ├─ 对比 entity_hints vs extracted_memories
   │   ├─ 计算 entity_coverage_ratio
   │   └─ 未覆盖实体 → targeted re-extraction prompt
   │
   └─ Stage 4: 合并去重
       └─ 合并 Stage 2 + Stage 3 结果
```

**关键收益**:
- Stage 1 和 Stage 3 是确定性的（NLP/regex），不受 prompt 波动影响
- LLM prompt 可以更短更聚焦，减少跨语言干扰
- 覆盖率验证是**可测量的指标**，不再靠 "BE THOROUGH" 祈祷

**工作量**: 中等 (3-5天)
**预期提升**: +5~8%（解决 18% 提取缺失 + 10% 细节丢失）

### 快速验证方案 (1天)
不做完整 NLP pipeline，先用简单实现验证思路:
```python
# Stage 1: regex-based entity/number extraction
import re
numbers = re.findall(r'\d+[\.\d]*\s*[万千百亿%]?', chunk)
# Stage 3: check if numbers appear in any extracted memory text
missing = [n for n in numbers if not any(n in m.text for m in memories)]
if len(missing) > threshold:
    # targeted re-extraction for missed numbers
```

---

## 支柱 2: 智能整合 — 从「平面记忆」到「知识图谱」

### 现状问题
- 记忆是扁平列表，靠向量搜索 top-K 返回
- "Caroline 的关系状态" 需要从 Caroline 实体出发遍历所有关联记忆
- 跨 session 的事件序列靠 timeline_events 但链接不够紧密

### 架构方案: Entity-Centric Memory Graph

```
当前:  query → vector search → top-K memories → assemble

改进:  query → entity extraction → entity graph traversal
                                       │
                              ┌────────▼────────┐
                              │   Entity Node    │
                              │   "Caroline"     │
                              │                  │
                              │  memories[]:     │
                              │   - relationship │
                              │   - hobbies      │
                              │   - events       │
                              │   - plans        │
                              │                  │
                              │  relations[]:    │
                              │   → Melanie      │
                              │   → counseling   │
                              │   → stained_glass│
                              └──────────────────┘
```

**已有基础**: workspace.py 的 EntityStore 已经有 entity → memory_ids 映射。
但 recall 阶段的 graph_expansion 只做 1-hop 且权重低 (0.40-0.45)。

**改进点**:
1. **Entity-aware recall routing**: query 中检测到人名 → 直接从 EntityStore 拉该实体全部记忆，不只依赖向量搜索
2. **Entity profile snapshot**: 每个实体维护一个 mini-profile (类似 user profile)，recall 时直接注入
3. **关系传播**: "Caroline 和 Melanie 一起做了什么" → 找两个实体的交集记忆

**工作量**: 中等 (3-5天)
**预期提升**: +3~5%（解决实体混淆 + 关系查询失败）

---

## 支柱 3: 精准检索 — 从「向量近似」到「多策略路由」

### 现状问题
- multi-hop 查询 (80%→55% in locomo-real) 是最大弱点
- 向量搜索擅长语义相似，不擅长逻辑组合
- "What creative project do Mel and her kids do besides pottery?" 需要:
  1. 找到 Mel 的所有 creative projects
  2. 排除 pottery
  3. 返回剩余的

### 架构方案: Query-Type Router + 专用检索策略

```
query → QueryAnalyzer (已有)
   │
   ├─ simple       → vector + FTS (当前流程)
   ├─ temporal      → timeline_events + date filter (当前流程)
   ├─ entity_query  → EntityStore lookup  ← NEW
   ├─ multi_hop    → decompose + entity graph  ← ENHANCED
   ├─ comparison   → parallel entity lookup + diff  ← NEW
   └─ aggregation  → SQL aggregate on entity memories  ← NEW
```

**multi_hop 增强**:
当前: LLM 分解查询 → 3 轮向量搜索 → 合并
改进: LLM 分解查询 → 每个子查询用最优策略 (entity/temporal/vector) → 合并

**工作量**: 中等 (2-3天)
**预期提升**: +3~5%

---

## 测试基础设施加速

### 问题: 每次改动 → 2小时测试周期

### 解决方案矩阵

| 方案 | 耗时 | 覆盖 | 用途 |
|------|------|------|------|
| **Micro-bench** (20题) | ~10min | 核心回归题 | 每次代码改动 |
| **Category-bench** (单数据集) | ~15min | 特定类别 | 定向优化 |
| **Skip-ingest bench** | ~20min | 全量,仅recall | recall 侧改动 |
| **Full bench** (release-full quick) | ~60min | 全量 | 里程碑验证 |

### Micro-bench 设计 (最高优先)

创建 `--preset micro` 包含 20 个最敏感的题目:
- 5 个 locomo-real (multi-hop/temporal 各2, single-fact 1)
- 5 个 locomo (multi-turn 2, logical-reasoning 2, recall-accuracy 1)
- 5 个 longmemeval-s (multi-session 2, temporal-reasoning 2, knowledge-update 1)
- 3 个 qmemory-chinese (idiom 1, name-disambig 1, profile 1)
- 2 个 multimodal (cross-ref 1, noise-resist 1)

选题标准: 历史上波动最大 / 最能区分版本差异的题目

**工作量**: 半天
**价值**: 将验证周期从 60min → 10min，迭代速度 6x

### 提取质量离线评估 (不需要启动服务器)

```python
# extract_eval.py — 直接评估提取质量，不经过 recall
for dataset in datasets:
    for session in dataset.sessions:
        chunks = chunk_messages(session.messages)
        for chunk in chunks:
            memories = extract_from_text(chunk.text, llm_complete=llm)
            # 检查: 关键实体是否被提取？数字是否保留？
            coverage = compute_coverage(chunk.text, memories)
            print(f"  {session.id}: coverage={coverage:.0%}")
```

这样可以单独评估提取质量，不需要启动服务器、不需要跑 recall，5 分钟出结果。

---

## 实施路线图

### Phase A: 基础设施 (1-2天) — 加速后续所有迭代

| 任务 | 作用 |
|------|------|
| A.1 创建 micro-bench preset (20题) | 迭代周期 60min→10min |
| A.2 创建 extract_eval.py 离线评估 | 提取验证 5min, 不需要服务器 |
| A.3 逐题 delta 分析脚本标准化 | 一键对比任意两次 run |

### Phase B: 提取管线结构化 (3-5天) — 解决 28% 失败

| 任务 | 解决的失败 |
|------|-----------|
| B.1 NLP 实体/数字预提取 | 数值细节丢失 (10%) |
| B.2 覆盖率验证 + 补充提取 | 提取缺失 (18%) |
| B.3 角色感知 (已有 [User]/[Assistant] 标记) | 助手反馈遗漏 |

### Phase C: Entity-Centric Recall (3-5天) — 解决 28% 失败

| 任务 | 解决的失败 |
|------|-----------|
| C.1 Entity-aware recall routing | 实体查询 (8%) |
| C.2 Cross-session entity profile | 跨会话链接 (12%) |
| C.3 Timeline event 强链接 | 时间线重建 (20%) → 部分改善 |

### Phase D: Multi-hop 增强 (2-3天) — 解决推理失败

| 任务 | 解决的失败 |
|------|-----------|
| D.1 子查询策略路由 | multi-hop 推理 (10%) |
| D.2 Entity intersection query | 关系查询 |
| D.3 Negation/exclusion handling | "besides X" 类查询 |

---

## 预期收益曲线

```
Phase    累积预期    关键改善
base     71.4%      (v0.1.0)
A        71.4%      (无分数变化，但迭代速度 6x)
B        77~79%     提取覆盖率从 ~70% → ~90%
C        80~83%     entity/cross-session 查询大幅改善
D        83~86%     multi-hop/reasoning 查询改善
```

---

## 与 v0.2.0-iteration-plan 的关系

| v0.2.0 计划 | 本路线图对应 | 状态 |
|-------------|-------------|------|
| 1.1 角色感知提取 | Phase B.3 | ✅ 已实现 (v0.1.0+ 有 [User]/[Assistant] 标记) |
| 1.2 细节保留 Anti-Summarization | Phase B.1+B.2 替代 | prompt→NLP 验证 |
| 1.3 多轮提取覆盖率 | Phase B.2 | 从 "prompt 祈祷" → 确定性验证 |
| 2.1 Session-Aware 时间线 | Phase C.2+C.3 | 实体化替代 session digest |
| 2.2 时间线事件自动链接 | Phase C.3 | 已有基础，需强化 |
| 2.3 全年事件索引 | 已实现 (year_digest) | ✅ |
| 3.1 知识更新强化 | Phase C.2 | entity profile 自然解决 |
| 3.2 实体消歧增强 | Phase C.1 | entity-aware routing |
| B.1 数据集隔离 | Phase A | 基础设施 |
| B.2 ReadTimeout | Phase A | 基础设施 |

---

## 设计原则

1. **确定性优于概率性**: 能用 NLP/regex 解决的不用 LLM
2. **可测量优于可描述**: "覆盖率 85%" > "BE THOROUGH"
3. **结构优于规模**: entity graph > 更长的 top-K
4. **快速验证优于完美方案**: micro-bench 10min > full bench 60min
5. **模块化优于耦合**: 每个 Phase 独立可验证，失败可回滚
