# R4 After-868348 Global-Unique Tokenizer Preflight Submission

timestamp_utc: `2026-05-17T19:52:00Z`

status:
`SUBMITTED_R4_AFTER_868348_GLOBAL_UNIQUE_QWEN_TOKENIZER_PREFLIGHT_H200_MONITOR_ONLY`

## Job

```text
job_id: 869298
job_name: nat-ev-v2-r4gTok
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
gres: gpu:h200:1
```

## Scope

Tokenizer-only boundary preflight for:

```text
results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/row_allocation_rows.jsonl
```

This job does not run model forward, scoring, generation, training, Llama,
same-family null, sanitizer, FAR, payload diversity, or paper claims.

## Preflight

```text
local single-enabled validation: PASS
remote single-enabled validation: PASS
local post-submit allowlist safety: PASS, enabled_entries=[]
remote post-submit allowlist safety: PASS, enabled_entries=[]
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
