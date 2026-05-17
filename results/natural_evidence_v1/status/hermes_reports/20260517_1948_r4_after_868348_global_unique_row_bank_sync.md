# Hermes/Codex Sync: R4 After-868348 Global-Unique Row Bank

timestamp_utc: `2026-05-17T19:48:00Z`

phase:
`V2_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_BUILT_PREFLIGHT_NEXT_NO_SUBMIT`

## Summary

Codex executed the artifact-only Option A repair after the `868348` strict
global duplicate failure. No model calls, generation, scoring, training,
allowlist enablement, or Slurm submission occurred.

New artifacts:

```text
decision:
  docs/natural_evidence_v2/R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_REPAIR_PLAN_20260517.md
row bank:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/
self-audit:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_self_audit_20260517/
```

Key result:

```text
row-bank status:
  PASS_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
self-audit status:
  PASS_R4_AFTER_868348_EXISTING_ROW_SOURCES_HAVE_NECESSARY_GLOBAL_UNIQUE_CAPACITY_NO_RERUN
rows:
  32768
shards:
  32
rows per shard:
  1024
selected coordinates:
  16
unique content prompt/prefix pairs:
  32768
duplicate content prompt/prefix extra rows:
  0
```

## Interpretation

The immediate input-capacity blocker from the prior row-source audit is repaired
for a future route. This does not reclassify `868348` and does not authorize a
new generation run by itself.

## Next Allowed Action

Artifact-only route validation and actual Qwen tokenizer/controller preflight
planning for this global-unique row bank.

Do not submit Slurm until:

```text
route validation passes
actual Qwen tokenizer boundary preflight passes
controller/decode preflight passes
local/remote hash preflight passes
zero-enabled allowlist safety passes
exactly one reviewed H200 route is recorded
```

Still not claimed:

```text
paper-facing positive
text-only phrase decoder success
FAR
Llama transfer
payload diversity
sanitizer robustness
cross-family generality
```
