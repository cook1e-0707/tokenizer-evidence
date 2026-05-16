# R4 metric-exact 864761 dev generation presubmit

phase:
`V2_R4_METRIC_EXACT_864761_DEV_GENERATION_ROUTE_VALIDATED_REMOTE_PREFLIGHT_NEXT`

summary:
```text
Source teacher-forced job 864761 passed the R4 surface-mass gate, with the
recorded caveat that training cycled a 512-row train artifact while scoring
8192 rows.

The small Qwen dev generation/decode route has passed local and remote
preflight:
- local zero-enabled allowlist safety: PASS
- local wrapper syntax: PASS
- local four-shard plan-only smoke: PASS
- local/remote hashes: MATCH
- remote zero-enabled allowlist safety: PASS
- remote four-shard plan-only smoke: PASS
- active Chimera jobs before submission: none

Next action is a single H200/pomplun Slurm array submission for:
v2_r4_candidate_v3_metric_exact_864761_dev_diagnostic_h200

The allowlist will be disabled immediately after sbatch returns.
```

scope:
```text
Allowed: one Qwen small dev generation/decode diagnostic using the reviewed
864761 adapter and the reviewed R4 cover-natural dev diagnostic wrapper path.

Not allowed: training; Llama; same-family null; sanitizer; FAR aggregation;
payload-diversity work or claim; paper-facing positive claim.
```

control-plane:
```text
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gpu: h200
time limit: 30-00:00:00
allowlist entry before submission: disabled
enabled entries before submission: 0
```
