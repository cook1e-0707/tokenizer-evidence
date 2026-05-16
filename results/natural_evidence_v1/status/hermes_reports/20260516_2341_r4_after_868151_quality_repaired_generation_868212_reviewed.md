# Hermes/Codex Sync: R4 after-868151 quality-repaired generation 868212 reviewed

phase:
`V2_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_REVIEWED_FIRST_TOKEN_3_OF_4_GLOBAL_DUP_CAVEAT_NEXT`

summary:
```text
Slurm job 868212 completed cleanly on Chimera H200/pomplun.

Job:
- job_id: 868212
- job_name: nat-ev-v2-r4cgen
- array: 0-3%4
- partition/qos/account: pomplun / pomplun / cs_yinxin.wan
- state: COMPLETED for 4/4 shards
- exit_code: 0:0
- elapsed: about 16-17 minutes per shard

Local artifacts:
- results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212/
- results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_review/

First-token event diagnostic:
- protected accepts: 3/4
- raw accepts: 0/4
- task-only accepts: 0/4
- wrong-key accepts: 0/4
- wrong-payload accepts: 0/4
- first-token block-level forbidden count: 0
- first-token block-level duplicate response hash count: 0
- token-id trace rows: 9216
- event counts: target=839, other=84, erasure=8293

Failed protected block:
- block_id: shard_03_block_00
- decoded_bits: 1-100101
- expected_bits: 10100101
- missing_bit_indices: 1
- missing coordinate: 26
- min_pair_support: 0

Caveats:
- diagnostic only; not locked positive and not paper claim
- full-phrase protected accepts remain 0
- global duplicate response hash count remains 4424 across 9216 generated rows
```

next_allowed_action:
Artifact-only failure attribution and next-route planning only:

```text
1. Analyze shard_03 coordinate-26 erasure.
2. Analyze global duplicate response hash sources across shards/conditions.
3. Record a reviewed repair or pivot route before any additional Slurm
   generation/scoring/training submission.
```

route_control:
Previously approved routes may proceed automatically after preconditions are
recorded. At this state, do not submit another Slurm generation/scoring/training
job until the 868212 artifact-only attribution records a new route.
