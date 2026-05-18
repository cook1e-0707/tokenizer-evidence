# R4 After-869348 Locked-Scale Row Bank Route Validation

Date: 2026-05-18

Status: `PASS_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT`

This validation is artifact-only. It does not tokenize, score, generate, train,
enable an allowlist entry, or submit Slurm.

```text
row bank: results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518
rows: 98304
shards: 96
locked prompts: 6144
unique content prompt/prefix pairs: 98304
duplicate content prompt/prefix extra rows: 0
coordinates: 16
```

Next allowed action: static and actual Qwen tokenizer/controller preflight for
this locked row bank. No generation or locked-scale submission is allowed until
those checks pass and a reviewed H200 route is recorded.
