# R4 metric-exact 864761 dev generation submitted

phase:
`V2_R4_METRIC_EXACT_864761_DEV_GENERATION_H200_ARRAY_864832_RUNNING`

summary:
```text
Submitted exactly one reviewed H200/pomplun Slurm array job:
- job_id: 864832
- job_name: nat-ev-v2-r4megd
- array: 0-3%4
- partition/qos/account: pomplun / pomplun / cs_yinxin.wan
- gpu: h200
- source adapter job: 864761
- route wrapper: scripts/natural_evidence_v2/slurm/r4_candidate_v3_metric_exact_864761_dev_diagnostic_h200.sbatch

The allowlist entry was disabled immediately after sbatch returned.
Local and remote post-submit allowlist safety both passed with zero enabled
entries.
```

next_allowed_action:
```text
Monitor job 864832. After completion, sync shard artifacts and review the
generation/decode diagnostic gates before any further route.
```

not_allowed_from_running_state:
```text
training
Llama
same-family null
sanitizer
FAR aggregation
payload-diversity work or claim
paper-facing positive claim
```
