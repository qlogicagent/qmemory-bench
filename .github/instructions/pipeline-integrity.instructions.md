---
description: "Use when editing qmemory-bench runner, CLI execution flow, provider integration, evaluation pipeline, or end-to-end benchmark orchestration."
name: "QMemory-Bench Pipeline Integrity"
applyTo: "src/qmemory_bench/runner.py,src/qmemory_bench/cli.py,src/qmemory_bench/providers.py,tests/test_bench.py"
---

# QMemory-Bench Pipeline Integrity

- Keep integration boundaries public and deployable: installed package, CLI, or HTTP API. Do not reach into qmemory private internals.
- Preserve isolated benchmark users and cleanup behavior so benchmark runs do not pollute real memory state.
- When changing execution flow, keep outputs machine-readable and stable enough for automated comparison.
- If a change can alter published scores, document the reason in code, docs, or handoff.
- Prefer fail-fast validation for missing credentials, target URLs, or provider prerequisites.