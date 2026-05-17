# R4 after-868348 global-unique allocation feasibility

Status:

```text
FAIL_R4_AFTER_868348_GLOBAL_UNIQUE_ALLOCATION_NOT_FEASIBLE_FROM_CURRENT_REVIEWED_ROW_BANK
```

The default safe repair option was to keep the strict global exact duplicate
gate unchanged and rerun with a globally unique 32-block prompt/prefix
allocation. The current reviewed row bank cannot support that directly.

## Counts

```text
reviewed row bank rows: 4096
reviewed unique prompt/prefix pairs: 2048
rows required for 32 blocks x 1024 rows/block: 32768
unique prompt/prefix pairs required for global uniqueness: 32768
current 32-block cyclic allocation rows: 32768
current 32-block cyclic allocation unique prompt/prefix pairs: 2048
current cyclic duplicate prompt/prefix extra rows: 30720
```

## Interpretation

The `868348` duplicate failure is consistent with the known cyclic dev
allocation reuse. A globally unique 32-block rerun cannot be launched from the
current reviewed row bank without first building and preflighting a larger row
bank.

## Next Decision

Two viable future routes remain:

```text
Option A:
  Build a larger globally unique prompt/prefix row bank, rerun tokenizer and
  controller preflights, then submit a reviewed dev diagnostic rerun.

Option B:
  Precommit narrower future duplicate-gate semantics: protected accepted-output
  duplicates remain fatal, while control-only duplicates are reported
  separately. This cannot retroactively rescue 868348.
```

No Slurm rerun is allowed from this artifact alone.
