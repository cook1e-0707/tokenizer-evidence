# Hermes Sync: R4 coverage-scale job submitted

Phase:
`V2_R4_METRIC_EXACT_COVERAGE_SCALE_H200_JOB_864761_RUNNING`

Submitted:

```text
job_id: 864761
job_name: nat-ev-v2-r4mof
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
node: chimera21
```

The reviewed allowlist entry
`v2_r4_candidate_v3_coverage_scale_micro_overfit_h200` was enabled for exactly
one submission and disabled immediately after `sbatch` returned. Local and
remote post-submit allowlist checks passed with zero enabled entries.

Scope:

- teacher-forced protected training/scoring only
- no generation
- no Qwen E2E rerun
- no Llama
- no same-family null
- no sanitizer
- no FAR
- no payload-diversity claim
- no paper-facing positive claim

Next allowed action:

Monitor job `864761`; after completion, sync artifacts and review the
teacher-forced surface-mass gates before any further compute route.
