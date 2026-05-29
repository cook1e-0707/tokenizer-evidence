# R4 After 864832 Two-Sided Qwen Tokenizer Preflight Route

Canonical phase:
`V2_R4_AFTER_864832_TWO_SIDED_QWEN_TOKENIZER_PREFLIGHT_ROUTE_VALIDATED_NO_SUBMIT`

## Decision

The two-sided cover-natural bank and 8192 rows are statically valid and
two-way scorer compatible. The next compute action, when submitted, is actual
Qwen tokenizer-boundary validation only.

This route does not authorize teacher-forced scoring, training, generation, or
any downstream claim. The wrapper loads the tokenizer only and asserts:

```text
model_forward_started=false
scoring_started=false
generation_started=false
training_started=false
```

## Route

```text
allowlist entry: v2_r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200.sbatch
score rows: results/natural_evidence_v2/status/r4_after_864832_two_sided_cover_bank_rows_20260516/cover_bank_aligned_target_only_rows.jsonl
rows: 8192
tokenizer: Qwen/Qwen2.5-7B-Instruct
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```

## Gate

The tokenizer preflight must return:

```text
checked_rows = 8192
failed_rows = 0
empty_target_id_row_count = 0
empty_other_id_row_count = 0
target_other_overlap_row_count = 0
```

Any failure blocks H200 teacher-forced scoring and returns to artifact-only
bank/row repair.

## Control Plane

Before submission, Codex/Hermes must complete remote hash preflight and
zero-enabled allowlist safety. Submission must enable exactly one allowlist
entry, submit exactly one H200 job, then immediately disable the entry.
