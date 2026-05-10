# WP3 Restricted Step-Label Primary Density Plan 850523 Repair

## Scope

This artifact-only repair follows Slurm job `850523`, which close-failed the
selected primary-policy strict density audit because one
`strict_compact_step_label_lines` response placed `Step 1:` through `Step 16:`
inline in one paragraph.

This repair does not reclassify the 850523 output as passing. It does not
submit Slurm, train, generate protected transcripts, run Qwen E2E, run Llama,
run same-family nulls, run sanitizer benchmarks, aggregate FAR, or make a
paper-facing positive claim.

## Artifacts

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_policy_density_plan_850523_repair_20260509_0310/
```

Primary files:

```text
restricted_step_label_policy.json
restricted_step_label_bucket_bank.json
restricted_step_label_detector_contract.json
restricted_step_label_density_design.json
restricted_step_label_primary_policy_summary.json
restricted_step_label_strict_density_audit_prompts.jsonl
restricted_step_label_strict_density_audit_slurm_review.json
README.md
```

## Repair

The repaired prompt plan removes the failing compact variant:

```text
removed_prompt_variant_id=strict_compact_step_label_lines
source_prompt_count=256
removed_prompt_count=64
repaired_prompt_count=192
```

Remaining prompt variants:

```text
strict_literal_16_step_lines=64
strict_no_heading_16_step_lines=64
strict_numbered_step_label_lines=64
```

The selected 850509 primary bank is unchanged:

```text
candidate_bank_id=step_label_recombined_create_develop_vs_choose_make_v1
bucket_0=[Create, Develop]
bucket_1=[Choose, Make]
```

The detector contract is clarified to match the reviewed implementation:
accepted `Step N:` labels must be line-start anchors after optional whitespace,
optional markdown bullet marker, and optional markdown emphasis. Sentence-start
inline `Step N:` labels remain outside this strict density gate.

## Validation

Local plan validation:

```text
PASS_RESTRICTED_STEP_LABEL_DENSITY_PLAN_VALIDATION
prompt_count=192
```

The repaired prompt JSONL contains no `strict_compact_step_label_lines` prompt
rows.

## Decision

WP3 remains blocked until a reviewed model-output density audit passes. This
artifact only prepares the repaired plan for review.

Next allowed action:

```text
Review the 850523 repaired strict density plan. Do not submit Slurm without
explicit approval. WP4, training, Qwen E2E, Llama, same-family null, sanitizer,
FAR aggregation, and paper-facing positive claims remain blocked.
```
