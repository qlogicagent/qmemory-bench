# QMemory-Bench

QMemory 的独立评测与验证仓库，用于跑 LoCoMo、LongMemEval-S、中文特化集、漂移集、隐式记忆集和压力测试。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

## 项目状态

- 当前版本：`0.1.0`
- 当前仓库状态：可独立运行、可本地评测、已公开发布
- 发布状态：仓库已公开；PyPI 首发尚未完成

## 核心能力

- 多数据集评测：LoCoMo、LongMemEval-S、中文场景、偏好漂移、隐式记忆、压力测试
- CLI 跑分：适合自动化回归、回归对比、结果落盘
- NiceGUI UI：适合人工查看结果、做演示和手工校准
- LLM 裁判：支持多个 OpenAI-compatible provider
- 独立仓库：通过 HTTP 调用 QMemory，不和主仓库耦合

## 安装

当前建议从源码安装：

```bash
git clone https://github.com/qlogicagent/qmemory-bench.git
cd qmemory-bench
pip install -e .
```

开发环境：

```bash
pip install -e ".[dev]"
```

## 快速开始

列出数据集：

```bash
qmemory-bench list-datasets
```

快速跑一轮：

```bash
qmemory-bench run \
  --target http://localhost:18800 \
  --provider deepseek \
  --api-key sk-xxx \
  --scale quick
```

指定数据集和输出文件：

```bash
qmemory-bench run \
  --target http://localhost:18800 \
  --provider deepseek \
  --api-key sk-xxx \
  --datasets locomo-real,preference-drift \
  --output results/report.json
```

启动 UI：

```bash
qmemory-bench ui --port 8090 --target http://localhost:18800
```

## 数据集范围

| 数据集 | 说明 |
|--------|------|
| `longmemeval-s` | 长期记忆主基准 |
| `locomo` / `locomo-real` | 会话检索与复杂推理 |
| `qmemory-chinese` | 中文表达、时间、消歧 |
| `preference-drift` | 偏好漂移与版本链 |
| `implicit-memory` | 隐式线索提取与推断 |
| `stress-latency` | 并发与延迟表现 |

## LLM 裁判

支持的 provider 包括：

- `deepseek`
- `minimax`
- `zhipu`
- `kimi`
- `qwen`
- `doubao`
- `openai`

## 评测流程

```
1. 连接目标 QMemory 实例
2. 创建隔离评测用户
3. 注入 sessions / facts
4. 发起搜索请求
5. 使用 LLM 裁判评分
6. 汇总 overall / category / precision 等指标
7. 清理评测数据
```

## 与 qmemory 的关系

本仓库只负责评测，不内嵌 QMemory 服务本体。

- 服务端仓库：[qlogicagent/qmemory](https://github.com/qlogicagent/qmemory)
- 当前评测仓库：[qlogicagent/qmemory-bench](https://github.com/qlogicagent/qmemory-bench)

## 开发与测试

```bash
pytest
```

如果要打包 Windows 可执行文件：

```bash
python build_exe.py
```

## 许可证

当前仓库内容使用 [Apache-2.0](LICENSE)。

更多首发说明见 [CHANGELOG.md](CHANGELOG.md) 和 [RELEASE_NOTES_v0.1.0.md](RELEASE_NOTES_v0.1.0.md)。
