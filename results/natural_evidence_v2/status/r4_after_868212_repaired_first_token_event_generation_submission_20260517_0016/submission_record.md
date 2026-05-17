# R4 After-868212 Repaired Full16 Generation Submission

Status: `SUBMITTED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_H200_DIAGNOSTIC`

Submitted one reviewed H200 Slurm generation diagnostic after Hermes notification and single-enabled allowlist preflight.

```text
job_id: 868260
job_name: nat-ev-v2-r4c16
array: 0-3%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time_limit: 30-00:00:00
allowlist_entry: v2_r4_after_868212_repaired_first_token_event_generation_h200
```

Post-submit allowlist state:

```text
local enabled_entries: []
remote enabled_entries: []
```

This is diagnostic generation only. It does not unlock training, Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing positive claims.

Next allowed action: monitor job `868260`, sync completed shard artifacts, and review/aggregate after all four shards finish. Do not submit another generation job while this job is active.
