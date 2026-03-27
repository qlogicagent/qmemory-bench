# QMemory-Bench v0.1.0 Release Notes

## Summary

`v0.1.0` 是 `qmemory-bench` 的首个公开版本，用于把评测能力从主仓库中独立出来，便于跨机器复现、回归比较和基准扩展。

这一版的目标是先把评测入口、数据集组织和 UI 跑通，而不是一次性把所有公开分发形态都做完。

## Highlights

- 独立 CLI 与 NiceGUI UI 已可使用
- 评测通过 HTTP 对接 `qmemory`，不与主仓库运行态强耦合
- 已纳入 LoCoMo、综合评测、真实数据集接入等能力
- 本版本不再内置任何真实 API key，评测所需凭证需由调用环境显式提供

## Included In This Release

- 独立 CLI：`run` / `ui` / `list-datasets`
- 基于 HTTP 的 QMemory 集成评测流程
- 多数据集评测入口
- NiceGUI 评测 UI
- 首发 README、LICENSE、项目元信息整理

## Not Included Yet

- PyPI 正式发布
- Windows 首发安装包发布页
- 独立数据集下载站点

## Recommended Validation Path

1. 源码安装 `qmemory` 与 `qmemory-bench`
2. 启动本地 `qmemory` 服务
3. 通过环境变量注入评测 LLM key 后运行 CLI 或 UI
4. 保存结果 JSON 作为跨版本回归基线

## Validation Status

- 仓库已创建并推送到 GitHub
- 可通过源码安装直接运行
- 当前适合作为独立验证与回归评测仓库使用

## Known Gaps

- 当前仍依赖源码安装，本次 release 不包含 PyPI 分发
- 数据集发布与 Windows 安装包发布页仍需后续补齐
