---
description: "Use when editing qmemory-bench datasets, public dataset loaders, judge prompts, scoring, aggregation, or benchmark methodology."
name: "QMemory-Bench Datasets And Scoring"
applyTo: "src/qmemory_bench/dataset.py,src/qmemory_bench/public_datasets.py,src/qmemory_bench/judge.py,data/**"
---

# QMemory-Bench Datasets And Scoring

- Keep datasets versionable, explainable, and runnable from CLI flows.
- Do not tune prompts, labels, or aggregation logic to hide regressions or inflate headline metrics.
- Preserve backward comparability where practical; if metric semantics change, make that explicit.
- Store evaluation outputs in machine-readable form, not only in logs or UI.
- Update regression coverage when changing dataset shaping, scoring rules, or judge aggregation.