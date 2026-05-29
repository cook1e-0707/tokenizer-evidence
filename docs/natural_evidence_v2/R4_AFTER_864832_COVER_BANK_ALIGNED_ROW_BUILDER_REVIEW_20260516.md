# R4 After 864832 Cover-Bank-Aligned Row Builder Review

Canonical phase:
`V2_R4_AFTER_864832_COVER_BANK_ALIGNED_ROWS_BUILT_TARGET_ONLY_BLOCKED_NO_COMPUTE`

Current blocker:
`BLOCK_R4_COVER_BANK_CODEBOOK_POLARITY_AND_TWO_WAY_SCORER_MISMATCH_NO_SLURM`

## Review Result

Codex built artifact-only rows from the precommitted R4 cover-natural ECC
surface bank and the `a55e` codebook:

```text
rows built: 4608
selected prompts: 256
coordinates: 32
surface entries: 128
prefix templates: 8
max prefix template fraction: 0.125
```

No tokenizer/model scoring, training, generation, or Slurm submission was
started.

## Blocking Finding

The current precommitted cover-natural bank is not directly compatible with
the existing two-way teacher-forced scorer:

```text
coordinates missing same-coordinate opposite bucket: 18/32
coordinates whose present bank polarity does not match the protected codeword bit: 14/32
current two-way scorer compatible: false
```

This means the current precommitted bank cannot be submitted to the existing
`bucket_0_surfaces` / `bucket_1_surfaces` scorer unchanged. The rows produced
in this review are target-only rows and are intentionally marked:

```text
current_two_way_scorer_compatible = false
current_two_way_scorer_blocker = precommitted_cover_bank_has_no_same_coordinate_opposite_bucket
```

## Consequence

The next step is not tokenizer preflight or H200 scoring. A new reviewed
artifact-only decision is required first. There are two valid directions:

```text
1. implement a target-only / target-vs-background teacher-forced scorer for
   the existing precommitted cover-natural bank; or
2. freeze a new two-sided, codeword-aligned cover-natural bank where every
   coordinate has target and other buckets compatible with the scorer.
```

Until that choice is recorded and validated, no Slurm job should be submitted.

## Artifacts

```text
scripts/natural_evidence_v2/build_r4_after_864832_cover_bank_aligned_rows.py
results/natural_evidence_v2/status/r4_after_864832_cover_bank_aligned_rows_20260516/
  cover_bank_aligned_target_only_rows.jsonl
  coordinate_bucket_compatibility.csv
  cover_bank_aligned_rows_summary.json
  cover_bank_aligned_rows_review.md
```

## Guardrails

This review preserves the post-864832 guardrails:

```text
- do not add 864832 transcript phrases to the bank;
- do not use candidate-v3 pressure phrases as success surfaces;
- do not lower accept/support/margin gates;
- do not submit Slurm;
- do not run tokenizer/model scoring, training, generation, Llama, null/FAR,
  sanitizer, payload diversity, or paper-facing claims.
```
