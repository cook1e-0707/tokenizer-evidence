# R4 After-868348 Global-Unique Row Bank Route Validation

Date: 2026-05-17

Status: `PASS_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT`

This validation is artifact-only. It does not tokenize, score, generate, train,
enable an allowlist entry, or submit Slurm.

```text
row bank: results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517
rows: 32768
shards: 32
unique content prompt/prefix pairs: 32768
duplicate content prompt/prefix extra rows: 0
coordinates: 16
prefix templates: 16
```

Next allowed action: actual Qwen tokenizer/controller preflight planning for
this row bank. No generation or Slurm submission is allowed until those checks
pass and a reviewed H200 route is recorded.
