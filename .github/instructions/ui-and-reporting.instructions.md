---
description: "Use when editing qmemory-bench UI, result presentation, report generation, or packaging around benchmark outputs."
name: "QMemory-Bench UI And Reporting"
applyTo: "src/qmemory_bench/ui/**,build_exe.py"
---

# QMemory-Bench UI And Reporting

- Keep presentation separate from core benchmark logic. Do not bury evaluation rules in UI code.
- UI changes must continue to expose the underlying machine-readable results rather than replacing them.
- Do not introduce dashboards or summaries that hide failed runs, missing categories, or judge uncertainty.
- Keep packaging and demo affordances optional; CLI and raw results remain the source of truth.