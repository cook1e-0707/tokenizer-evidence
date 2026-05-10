# WP3-R3 Variant-Balanced Naturalness Review 850885

## Scope

This review covers the variant-balanced naturalness sample derived from Slurm
job `850885` after local artifact-only re-audit:

```text
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885_reaudit_20260509_053338/restricted_step_label_naturalness_examples.jsonl
```

The re-audit used the already generated `850885` model outputs and only
recomputed detector summaries with prompt metadata restored. It did not load a
model, submit Slurm, train, run E2E, decode payloads, aggregate FAR, or make a
paper-facing positive claim.

## Sample

The reviewed sample has `96` generated responses:

```text
wp3_r1_dev=48
wp3_r1_eval=48
strict_literal_16_step_lines=32
strict_no_heading_16_step_lines=32
strict_numbered_step_label_lines=32
```

The known edge anomaly from the full `850885` audit was also inspected
separately:

```text
prompt_id=qwen_v2_wp3_r1_density_c72aac81a70a5bfc3a97
variant_id=strict_numbered_step_label_lines
detected_structural_slots=1
reason=Chinese action text after Step 2 through Step 16, so the English
first-word slot detector could not extract usable action-word slots.
```

## Manual Labels

Manual review labels for the `96` balanced examples:

```text
PASS=88
BORDERLINE=8
FAIL_FORBIDDEN_SURFACE=0
FAIL_OBVIOUS_CODING_ARTIFACT=0
FAIL_SEMANTIC_COHERENCE=0
PASS+BORDERLINE rate=1.0
```

The `BORDERLINE` examples are ordinary checklist responses, but some are more
template-like than ideal because the numbered variant sometimes starts steps
with title-style subclauses such as `Define Clear Objectives:` before the
natural instruction. They are not old structured-carrier evidence blocks and
do not expose forbidden public surfaces.

The separate edge anomaly is labeled:

```text
FAIL_LANGUAGE_POLICY=1
FAIL_FORBIDDEN_SURFACE=0
FAIL_OBVIOUS_CODING_ARTIFACT=0
FAIL_SEMANTIC_COHERENCE=0
```

This anomaly is important for protocol repair because it shows that the prompt
should explicitly require English responses before any future contract or
training stage. It does not by itself fail the WP3-R3 naturalness threshold
because the formal balanced sample has no forbidden-surface failures, no
obvious coding-artifact failures, and the pass-plus-borderline rate is above
the required threshold.

## Decision

WP3-R3 naturalness review status:

```text
PASS_WITH_LANGUAGE_DRIFT_NOTE_WP4_STILL_BLOCKED_BY_R2
```

This is only a naturalness/sample review. WP4 remains blocked because WP3-R2
high-mass 2-way bank search has not passed. The current `Create/Develop` vs
`Choose/Make` bank remains below the pilot absolute mass threshold
(`0.0125512375 < 0.03`), so no prompt-local payload contract, training, Qwen
E2E, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing
positive claim is allowed.

Next allowed action:

```text
WP3-R2 high-mass 2-way bank search / context-mass scoring plan. Any tokenizer
or model scoring on Chimera must be submitted through Slurm.
```
