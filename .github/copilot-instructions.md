# QMemory-Bench Agentic Coding Rules

This file defines repository-wide guardrails. File-specific constraints live under `.github/instructions/` and should be treated as stricter overlays for matching paths.

## Product Direction

- qmemory-bench is an independent evaluation repo for QMemory and related memory systems.
- Its job is to measure behavior honestly through public or deployable interfaces, not to embed service internals.
- Prefer reproducibility, auditability, and methodological clarity over flashy dashboards or benchmark inflation.

## Non-Negotiable Constraints

- Do not hardcode API keys, judge credentials, or provider secrets anywhere in the repository.
- Do not make the benchmark depend on private internals from qmemory. Integration should happen through the installed package, CLI, or HTTP API.
- Do not tune datasets, prompts, or scoring logic to hide regressions or artificially boost headline scores.
- Do not mix benchmark-only assumptions back into qmemory service code.

## Evaluation Guardrails

- New datasets should be versionable, explainable, and runnable from CLI flows.
- New metrics should be saved in machine-readable outputs, not only shown in UI.
- Judge prompts and aggregation logic should remain reviewable and deterministic enough for comparison across runs.
- Preserve isolation: benchmark users, temp data, and cleanup flows should not pollute real user memory state.

## Source Of Truth

- `src/qmemory_bench/runner.py` owns the end-to-end benchmark pipeline.
- `src/qmemory_bench/dataset.py` and `src/qmemory_bench/public_datasets.py` own dataset loading and shaping.
- `src/qmemory_bench/judge.py` owns grading behavior.
- `src/qmemory_bench/ui/` is presentation only; do not bury core evaluation logic there.
- `data/` and `tests/` must evolve together when benchmark methodology changes.

## Working Rules For AI Contributors

- Keep benchmark logic separable from UI concerns and packaging concerns.
- When adding a dataset, also add the minimal registration, loading, and regression coverage needed to keep it runnable.
- When changing scoring or aggregation, document the intent in code or docs and preserve backward comparability where practical.
- Prefer explicit environment-variable based credential loading and fail-fast validation when keys are missing.
- If a change could affect published scores, mention that impact clearly in the handoff.

## Validation Expectations

- Run focused `pytest` coverage for touched benchmark paths whenever feasible.
- For evaluation-pipeline changes, verify at least one representative CLI path if the environment allows it.
- If verification is blocked by missing providers, datasets, or credentials, state the exact blocker.