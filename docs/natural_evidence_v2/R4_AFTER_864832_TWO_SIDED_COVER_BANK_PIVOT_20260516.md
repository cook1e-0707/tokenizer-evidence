# R4 After 864832 Two-Sided Cover Bank Pivot

Canonical phase:
`V2_R4_AFTER_864832_TWO_SIDED_COVER_BANK_PIVOT_ARTIFACT_ONLY`

## Decision

The target-only row-builder review found that the previous precommitted
cover-natural bank is not directly compatible with the current two-way
teacher-forced scorer:

```text
coordinates missing same-coordinate opposite bucket: 18/32
coordinates whose present bank polarity does not match protected codeword bit: 14/32
```

The selected artifact-only repair is to freeze a new two-sided,
codeword-aligned cover-natural bank before tokenizer preflight or H200 scoring.
This is preferred over immediately implementing a target-vs-background scorer
because the old one-sided bank covers only 18 protected-codeword coordinates,
below the R4 decoder dev support target.

## Scope

The new bank may reuse the old precommitted natural phrases and add independent
rule-generated complementary phrases. It must not use phrases mined from job
`864832` transcripts or candidate-v3 pressure collapse phrases as success
surfaces.

The pivot remains artifact-only:

```text
no tokenizer validation
no model scoring
no training
no generation
no Slurm submission
no Llama
no null/FAR
no sanitizer
no payload diversity claim
no paper-facing positive claim
```

## Output

```text
scripts/natural_evidence_v2/build_r4_after_864832_two_sided_cover_bank.py
results/natural_evidence_v2/precommit/r4_after_864832_two_sided_cover_bank_20260516/
```

If static validation passes, the next allowed action is artifact-only row
building and tokenizer-boundary preflight preparation for the new two-sided
bank. Actual Qwen tokenizer validation must still run through reviewed Chimera
Slurm, not on the login node.
