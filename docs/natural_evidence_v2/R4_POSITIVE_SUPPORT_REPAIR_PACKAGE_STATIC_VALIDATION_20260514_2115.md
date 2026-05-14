# R4 Positive Support-Repair Package Static Validation

Timestamp: `2026-05-14T21:15:00Z`

## Verdict

```text
PASS_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE
```

The artifact-only support-repair package has been implemented and statically
validated. This is not a model result and does not unlock Slurm, generation,
model scoring, training, Llama, FAR, sanitizer, payload diversity, or any
paper-facing claim.

## Package

```text
results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115/
```

Key files:

```text
contract.json
event_window_bank.json
extractor_spec.json
decoder_spec.json
source_policy.json
toy_positive_fixture.json
toy_positive_events.jsonl
package_summary.json
package_report.md
```

## Static Validation Results

```text
contract_id = r4_positive_support_repair_v2
event_window_rows = 384
surface_family_count = 8
rows_per_family = 48
max_surface_family_fraction = 0.125
toy_positive_events = 26
toy_positive_distinct_coordinates = 24
toy_positive_accept = true
wrong_key_accept = false
wrong_payload_accept = false
```

## Source Policy

The support-window bank is generated from an independent static task taxonomy:

```text
source_policy = independent_static_taxonomy_not_859277_transcripts
```

`859277` transcripts were not used to add surfaces. The failed run remains
usable only for failure taxonomy and coverage diagnosis.

## Extractor Change

The old exact phrase extractor required full multi-word phrase matches such as:

```text
ask a focused question
confirm the main constraint
```

The new support-window extractor records a precommitted event when a scrubbed
natural sentence segment contains:

```text
allowed action verb lemma + allowed task cue lemma within <= 8 tokens
```

This is intended to repair the zero-support failure without relying on Step
labels, fixed line positions, headings, or generated-transcript phrase mining.

## Validation

Focused tests passed:

```text
uv run pytest \
  tests/natural_evidence_v2/test_r4_positive_support_window_extractor.py \
  tests/natural_evidence_v2/test_r4_positive_support_repair_package.py -q

6 passed
```

## Next Allowed Action

```text
artifact-only support-window coverage dry-run on existing 859277 outputs and
static review of whether support is useful or merely common across all arms
```

This next action still does not permit Slurm, generation, model scoring,
training, or claims.

