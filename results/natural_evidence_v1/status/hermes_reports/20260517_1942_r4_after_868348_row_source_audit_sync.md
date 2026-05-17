# Hermes/Codex Sync: R4 After-868348 Row Source Audit

timestamp_utc: `2026-05-17T19:42:00Z`

phase:
`V2_R4_AFTER_868348_EXISTING_ROW_SOURCE_AUDIT_FAILED_NO_RERUN`

## Summary

Codex ran an artifact-only audit of existing row-source JSONL files after the
`868348` dev diagnostic failed the strict global exact duplicate gate.

Result:

```text
status:
  FAIL_R4_AFTER_868348_EXISTING_ROW_SOURCES_INSUFFICIENT_FOR_GLOBAL_UNIQUE_32_BLOCK_ALLOCATION_NO_RERUN
audit:
  results/natural_evidence_v2/status/r4_after_868348_candidate_row_source_audit_20260517/
scanned row files:
  369
compatible source files:
  6
compatible rows:
  55296
unique content prompt/prefix pairs:
  4096
required unique content prompt/prefix pairs for strict 32-block rerun:
  32768
min unique content prompt/prefix pairs per coordinate:
  256
required per coordinate:
  2048
```

No model calls, generation, scoring, training, allowlist enablement, or Slurm
submission occurred.

## Interpretation

Existing reviewed row sources are not sufficient to construct a strict
global-unique 32-block rerun of the `868348` dev route. The project should not
rerun from the current row bank.

`868348` remains noncanonical:

```text
protected strict accepts:
  32/32
controls:
  0/32 each
trace binding:
  valid
strict blocker:
  global exact duplicate extra rows = 2
canonical adoption:
  false
```

## Next Allowed Action

Artifact-only route planning only:

```text
Option A:
  build a larger reviewed prompt/row bank with tokenizer/controller preflight
  and validate a globally unique allocation before any rerun
Option B:
  record a future-only duplicate-gate semantics decision that separates
  protected/accepted duplicates from control-only duplicates
```

Option B cannot retroactively rescue `868348`.

Not allowed from current row bank:

```text
generation
Slurm submission
training
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```
