# R3.2E Scavenger Shard-Array Recovery Plan

timestamp_utc: 2026-05-12T02:56:46Z

## Decision

H200/pomplun is unavailable for the current recovery route. The next R3.2
recovery should use DGXA100/A100 under `scavenger`, but must avoid another
single long serial job. The reviewed recovery shape is now:

```text
12 shard tasks on DGXA100/scavenger
1 shard = 1 Slurm array task = one replicate_group/shard_XX
1 aggregate-only Slurm job after all shards complete
```

This is still the same-contract `a55e` R3.2 locked-scale route. It is not a
payload-diversity test, not FAR, not Llama, not sanitizer, not training, and
not a paper-facing claim.

## Motivation

Job `853110` was preempted under `scavenger_unlim` after `08:43:42`. It had
completed only `6/12` shards. Each completed shard took roughly 80-90 minutes,
so the original 12-shard serial wrapper was too exposed to preemption and
runtime loss.

Splitting the package into shard tasks reduces the failure unit from the full
R3.2 package to one shard.

## Slurm Shape

GPU shard array:

```text
script: scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array.sbatch
partition: DGXA100
account: pi_yinxin.wan
qos: scavenger
gpu: A100:1 per task
array: 0-11%8
time: 4-00:00:00
```

Aggregate-only job:

```text
script: scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_aggregate.sbatch
partition: Intel6240,Intel6248,Intel6326
account: pi_yinxin.wan
qos: scavenger
time: 4-00:00:00
```

The `4-00:00:00` wall time is the maximum recorded wall time for the
`scavenger` QOS. This follows the user instruction to use the highest available
time limit for future jobs. It does not prevent QOS preemption.

## Control Plane

New allowlist entries were added but remain disabled:

```text
v2_r3_2_qwen_locked_scale_shard_array
v2_r3_2_qwen_locked_scale_aggregate
```

Allowlist safety check passed with zero enabled entries:

```text
results/natural_evidence_v2/status/r3_2e_scavenger_array_allowlist_safety_20260512_0246.json
```

## Submission Rules

No Slurm job was submitted while writing this plan.

Future submission requires:

1. Hermes TG/email notification.
2. Enable exactly the shard-array allowlist entry.
3. Submit exactly one shard-array Slurm command.
4. Immediately disable the shard-array allowlist entry.
5. Record the array job id and output dir.
6. After all shards complete, enable exactly the aggregate allowlist entry.
7. Submit exactly one aggregate-only Slurm command with `OUTPUT_DIR` set to the
   completed shard-array output dir.
8. Immediately disable the aggregate allowlist entry.
9. Review aggregate gate before any downstream route.

## Current Status

Prepared only. No allowlist entry is enabled and no Slurm job has been
submitted for this recovery plan.
