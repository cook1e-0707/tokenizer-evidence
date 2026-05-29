# R4 After 864832 Cover-Bank-Aligned Objective Route

Canonical phase:
`V2_R4_AFTER_864832_COVER_BANK_ALIGNED_OBJECTIVE_ROUTE_ARTIFACT_ONLY`

Current blocker:
`BLOCK_R4_AFTER_864832_COVER_BANK_ALIGNED_ROW_BUILDER_AND_PREFLIGHT_NEXT`

## Decision

Job `864832` is a reviewed negative generation result, not a candidate for
artifact-only positive repair. It showed zero protected accepts under both
raw and `format_scrub=all` decoding, while preserving clean null arms:

```text
protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw/task-only/wrong-key/wrong-payload accepts: 0
duplicate response text hashes: 358
max protected-vs-raw shallow feature AUC: 1.0
```

The failure is now classified as a transfer gap: the teacher-forced objective
created pressure on candidate-v3 prefix-native phrases such as `Create a plan`
and `Prepare a`, but free generation did not produce enough of the
precommitted R4 cover-natural ECC surface bank to decode. The route must
therefore stop optimizing proxy pressure phrases and align the future objective
with the exact decoder surface bank.

## Selected Repair

The selected repair is:

```text
cover_bank_aligned_metric_exact_objective_repair
```

The future target surfaces must come only from:

```text
results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/surface_bank.json
```

The decoder spec remains:

```text
primary_reported_scrub_mode = all
line_or_step_index_required = false
posthoc_threshold_changes_allowed = false
```

## Guardrails

This route package does not unlock compute. It records the next route and its
requirements.

The next implementation must not:

```text
- add 864832-observed phrases to the bank;
- treat candidate-v3 pressure phrases as decoder surfaces;
- use Create/Prepare/Plan repetition as success evidence;
- lower accept/support/margin gates;
- submit Slurm before route, allowlist, Hermes, and remote hash preflights;
- run generation before a cover-bank-aligned teacher-forced gate passes.
```

## Required Next Artifact-Only Tasks

Before any H200 submission, create and validate:

```text
1. a cover-bank-aligned row builder or existing-builder review;
2. tokenizer-boundary preflight for the cover-bank rows;
3. a plan-only H200 wrapper smoke test;
4. zero-enabled local and remote allowlist safety checks;
5. an exactly-one H200/pomplun teacher-forced submission route doc.
```

## Future Teacher-Forced Gate

The future teacher-forced repair must pass:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
scorer boundary failures = 0
target/other token overlap = 0
visible repetition collapse = false
```

If this gate passes, only then can a small Qwen dev generation route be
prepared. That later route must report `format_scrub=all` as primary and keep
raw/task-only/wrong-key/wrong-payload controls.

## Current Status

Static route validation:

```text
configs/natural_evidence_v2/r4_after_864832_cover_bank_aligned_objective_route.yaml
scripts/natural_evidence_v2/validate_r4_after_864832_cover_bank_aligned_route.py
results/natural_evidence_v2/status/r4_after_864832_cover_bank_aligned_route_validation_20260516/
```

This is still artifact-only: no Slurm, no tokenizer/model scoring, no training,
no generation, no Llama, no null expansion, no sanitizer, no FAR, no payload
diversity, and no paper-facing positive claim are unlocked by this document.
