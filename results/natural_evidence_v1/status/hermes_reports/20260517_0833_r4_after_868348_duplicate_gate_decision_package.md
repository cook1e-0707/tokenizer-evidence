# Hermes sync: R4 after-868348 duplicate-gate decision package

phase:
`V2_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILED_GLOBAL_DUPLICATE_GATE_SIGNAL_PASSING_NO_RERUN`

decision package:

```text
docs/natural_evidence_v2/R4_AFTER_868348_DUPLICATE_GATE_REPAIR_DECISION_PACKAGE_20260517.md
results/natural_evidence_v2/status/r4_after_868348_duplicate_gate_repair_decision_package_20260517/
results/natural_evidence_v2/status/r4_after_868348_global_unique_allocation_feasibility_20260517/
```

facts:

```text
868348 protected strict accepts: 32/32
controls: 0/32 each
trace binding invalid: 0
global exact duplicate extra rows: 2
duplicate rows: task_only only
current reviewed row bank unique prompt/prefix pairs: 2048
unique pairs needed for 32-block global uniqueness: 32768
```

next:

```text
Expert route decision required before any rerun:
- build a larger globally unique prompt/prefix row bank with tokenizer/controller preflight; or
- precommit future duplicate-gate semantics that report control-only duplicates separately.
```

No Slurm rerun is allowed from this package alone.
