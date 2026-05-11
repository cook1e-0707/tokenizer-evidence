# Sanitizer Benchmark Plan

Status: artifact-only framework; blocked until both Qwen AND Llama pass WP6.

## Scope

Test whether the WP6 robust-block payload decoder survives text transformations
(sanitizers). This establishes robustness bounds for the ownership evidence protocol.

## Prerequisites (per Protocol Contract)

1. ✅ Qwen V2 positive recovery and null rejection (WP6-R2 Option B, job 852426)
2. ⏳ Llama-3.1-8B positive recovery and null rejection (pending WP5/WP6 migration)
3. Only after both pass may sanitizer benchmarks run

## Sanitizer Hierarchy

Ordered by severity:

| Sanitizer | Severity | Target | Description |
|---|---|---|---|
| case_normalize | minimal | surface_form | Lowercase all text |
| step_label_strip | moderate | structural_markers | Remove "Step 1:", "1.", "First" etc. |
| transition_strip | moderate | discourse_markers | Remove "However", "Moreover" etc. |
| hedge_strip | moderate | hedges | Remove "might", "perhaps" etc. |
| sentence_reorder | high | structure | Randomly reorder sentences |
| synonym_replace_random | high | lexical | Replace ~30% content words |
| paraphrase_llm | extreme | full_rewrite | LLM-based paraphrase (future) |

## Recovery Metric

For each sanitizer:
- Apply transformation to WP6 generated outputs
- Re-run coordinate-majority decoder on sanitized text
- Measure: payload recovery rate per condition (protected/raw/task_only/wrong_key/wrong_payload)
- Compare: recovery rate vs baseline (unsanitized)

## Gate

Robustness claim requires:
- Protected recovery rate >= 0.5 under case_normalize
- Protected recovery rate >= 0.3 under step_label_strip
- Null rejection preserved (0 accepts) under all sanitizers

## Artifacts Created

- `scripts/natural_evidence_v2/apply_sanitizers.py` — Apply sanitizers to outputs
- `scripts/natural_evidence_v2/slurm/sanitizer_benchmark.sbatch` — Full eval wrapper
- This plan document

## Submission Plan

1. Wait for Llama WP6 pass
2. Apply sanitizers to both Qwen and Llama WP6 outputs
3. Re-decode each sanitized variant
4. Aggregate recovery rates
5. Review and record gate status

## Validation

All artifacts created locally. No Slurm interaction performed.
