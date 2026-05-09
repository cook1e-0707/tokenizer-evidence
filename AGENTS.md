# AGENTS.md

Compatibility mirror for Codex-style tooling.

The canonical instructions for this repository live in the Codex skill at `plugins/tokenizer-alignment-karpathy/skills/tokenizer-alignment-guardrails/SKILL.md`. This file exists only so the same guardrails are still visible to tools that look for `AGENTS.md`.

Behavioral guidelines for this repository, adapted from `forrestchang/andrej-karpathy-skills` for the tokenizer-alignment research workflow.

These rules bias toward caution over speed. For trivial edits, use judgment.

## 1. Think Before Coding

Don't assume the stage, artifact status, or intended validation path.

- Identify the closest controlling spec in `docs/specs/` before editing.
- Make assumptions explicit when the request is ambiguous about tokenizer backend, frozen-catalog status, or smoke-vs-real evaluation flow.
- If a simpler approach exists, say so before implementing a broader one.
- If the request appears to conflict with a frozen decision record, stop and surface that conflict.

## 2. Simplicity First

Minimum code that solves the requested research task. Nothing speculative.

- Prefer refining existing modules under `src/`, `scripts/`, and `tests/`.
- Do not add new abstractions, framework layers, or configuration surface without a direct need.
- Match the repo's explicit style: `pathlib`, small helpers, YAML configs, and machine-readable summaries.
- If 200 lines can be 50 without losing clarity, rewrite it.

## 3. Surgical Changes

Touch only what the request requires.

- Do not "clean up" adjacent code, comments, or formatting unless your change made them obsolete.
- Do not hand-edit generated outputs under `results/` or run directories unless explicitly asked.
- Treat `docs/catalog_freezes/` and frozen catalog overlays as records, not casual edit targets.
- If you notice unrelated dead code or drift, mention it separately rather than folding it into the same patch.

The test: every changed line should map to the current task.

## 4. Goal-Driven Execution

Define success criteria and verify them with the smallest relevant check.

- Control-plane work: `pytest tests/test_config_loading.py tests/test_result_schema.py`
- Method-core work: `pytest tests/test_tokenizer_alignment.py tests/test_bucket_mapping.py tests/test_payload_codec.py tests/test_parser.py tests/test_verifier.py`
- Smoke verification: `python3 scripts/smoke_verify.py`
- Manifest or launcher changes: dry-run the corresponding script
- Frozen catalog work: run `scripts/freeze_catalog.py`; if it fails, switch to `scripts/review_catalog_freeze.py` instead of rewriting the catalog automatically

## Stage Map

- `docs/specs/stage2_control_plane_spec.md`: config, manifests, result schemas, SLURM workflow
- `docs/specs/stage3_method_core_smoke_spec.md`: tokenizer audit, bucket mapping, payload codec, parser, verifier
- `docs/specs/stage4_real_integration_spec.md`: frozen-catalog and real-pilot integration

These guidelines are working if diffs stay small, assumptions are surfaced early, and validation is explicit rather than implied.
