# R4 after-868299 dev diagnostic repaired replacement submission

Submitted one repaired H200 Slurm array after the 868313 runtime allowlist-race
failure review and runtime validation patch.

```text
job_id: 868348
job_name: nat-ev-v2-r4dev
array: 0-31%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
replaces_failed_job: 868313
```

The allowlist entry was enabled only for the submission preflight/submission
window and disabled immediately after `sbatch` returned. Local and remote
post-submit zero-enabled allowlist safety checks passed.

This is a monitor-only state. It does not unlock paper claims, training, Llama,
same-family null, sanitizer, FAR, payload diversity, or locked-scale claims.
