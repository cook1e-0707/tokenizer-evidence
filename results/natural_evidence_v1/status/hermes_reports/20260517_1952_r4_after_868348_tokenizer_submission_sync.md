# Hermes/Codex Sync: R4 After-868348 Tokenizer Preflight Submitted

timestamp_utc: `2026-05-17T19:52:00Z`

phase:
`V2_R4_AFTER_868348_GLOBAL_UNIQUE_TOKENIZER_PREFLIGHT_SUBMITTED_MONITOR_ONLY`

## Summary

The global-unique row bank route passed local and remote preflight. Codex
submitted exactly one tokenizer-only H200 Slurm job and immediately disabled the
allowlist entry.

```text
job_id: 869298
job_name: nat-ev-v2-r4gTok
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
row bank: results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/row_allocation_rows.jsonl
planned checked rows: 32768
```

Preflight:

```text
local single-enabled preflight: PASS
remote single-enabled preflight: PASS
local post-submit allowlist: PASS, enabled_entries=[]
remote post-submit allowlist: PASS, enabled_entries=[]
```

Scope:

```text
tokenizer boundary preflight only
model_forward_started: false
model_scoring_started: false
generation_started: false
training_started: false
paper_claim_allowed: false
```

## Next Allowed Action

Monitor job `869298`. After terminal completion, sync artifacts and review:

```text
checked_rows == 32768
failed_rows == 0
empty_target_id_row_count == 0
empty_other_id_row_count == 0
target_other_overlap_row_count == 0
```

Do not submit generation or model scoring before this tokenizer preflight is
reviewed.
