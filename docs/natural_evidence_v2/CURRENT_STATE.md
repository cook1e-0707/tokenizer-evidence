# natural_evidence_v2 Current State

Last synchronized: 2026-05-17T00:20:09Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_868260_RUNNING`

## Most Recent Compute Result

Job `868212` completed the reviewed quality-repaired after-868151 controller
generation diagnostic on Chimera H200:

```text
job_id: 868212
job_name: nat-ev-v2-r4cgen
array: 0-3%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
state: COMPLETED, 4/4 shards
exit_code: 0:0
```

The precommitted first-token event diagnostic gate passed at small diagnostic
scale:

```text
protected accepts:
  3/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
block-level forbidden public surface count:
  0
block-level duplicate response hash count:
  0
token-id event traces:
  9216
event status counts:
  target=839, other=84, erasure=8293
```

The single protected failed block was localized:

```text
block_id:
  shard_03_block_00
decoded_bits:
  1-100101
expected_bits:
  10100101
missing bit index:
  1
missing coordinate:
  26
complete pairs:
  7/8
```

The full-phrase decoder remains failed as expected:

```text
full-phrase protected accepts, format_scrub=all:
  0
```

This result remains diagnostic only, not a locked positive or paper claim. The
main quality caveat is still global duplication across generated outputs:

```text
generated rows:
  9216
unique response hashes:
  4792
global duplicate response hash count:
  4424
max duplicate group size:
  4
```

## Failure Attribution

Artifact-only attribution for `868212` is recorded:

```text
attribution:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_failure_attribution/
status:
  RECORDED_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_ARTIFACT_ONLY_FAILURE_ATTRIBUTION_NO_SUBMIT
coordinate-26 shard_03 protected erasures:
  64/64
duplicate hash groups:
  2908
duplicate extra rows:
  4424
dominant duplicate condition sets:
  protected,raw: 1621 groups
  task_only: 1024 groups
dominant duplicate shard pairs:
  shard_00,shard_01: 1090 groups
  shard_02,shard_03: 1051 groups
```

Interpretation: the protected failure was a localized erasure/reliability issue
for coordinate 26 in shard 03, not a null-accept failure. The duplicate caveat
is global and dominated by deterministic identical generations across paired
shards and protected/raw or same-condition repetitions.

## Superseded Failed Repair Preflight

The first artifact-only repair preflight intentionally failed because the
12-coordinate pivot codebook left several singleton payload bits:

```text
preflight:
  results/natural_evidence_v2/status/r4_after_868212_reliability_duplicate_repair_preflight_20260516/
status:
  FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT
singleton/codebook failures:
  bit 1 active=[26], coordinate_26 sole active coordinate
  bit 3 active=[19]
  bit 5 active=[8]
  bit 6 active=[4]
duplicate extra rows:
  4424
```

This failed preflight is now superseded by the repaired full-16 plan below. It
must not be used for another Slurm submission.

## Current Route Result

The next artifact-only step has completed: the route restored the full 16
coordinates from the reviewed reliability codebook, rebuilt the row allocation,
precommitted a repaired first-token event decoder/codebook, and validated the
plan without submitting compute.

```text
full16 allocation plan:
  results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/
allocation status:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY
allocation rows:
  4096
shards:
  4
rows per shard:
  1024
rows per coordinate per shard:
  64
duplicate prompt/prefix pair max per shard:
  0
```

```text
repaired precommit:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/
precommit status:
  PRECOMMITTED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_ARTIFACT_ONLY_NO_COMPUTE
selected coordinates:
  16
min active coordinates per bit:
  2
coordinate 26 sole-coordinate condition:
  rejected
reclassifies 868212:
  false
```

```text
plan validation:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_plan_validation_20260516/
validation status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATION_NO_SUBMIT
locked-scale global duplicate gate:
  0
slurm submitted:
  false
generation/model-scoring/training started:
  false
```

Precommit hashes:

```text
codebook:
  58d5fc6dc0c42136e5fb238c0b255e73c9c7d63115a3abc39af31ec6fd2f5444
decoder_spec:
  64fd1e682c0ea314bc2f49b6a543447ef9df9679957b87800c2bd41a82bb70f3
duplicate_policy:
  241d93f445676a63f353a3ca58b63e5ceff1bdc826dc058d2b29a1086409e9e9
allocation_manifest:
  b797b46f876e08dfaf329379f578a69fe975d4747d7a87ae2a834d6f83899993
allocation_rows:
  61927c822c6ce730974ebbaffc775678e70c0a0a2c13e526173f392a231c64dd
contextual_literal_policy:
  0522c7f17c177137f4abbe29c147656797584ef28a328c5a6e8b8145201f31b5
```

Verification:

```text
pytest:
  19 passed, 1 skipped
```

## Generation Wrapper Route Validation

The repaired full16 generation/decode control plane has been implemented and
validated locally without Slurm submission:

```text
route config:
  configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_generation_route.yaml
decoder route:
  configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_decoder_route.yaml
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_868212_repaired_first_token_event_generation_h200.sbatch
validator:
  scripts/natural_evidence_v2/validate_r4_after_868212_repaired_first_token_event_generation_route.py
route validation:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_route_validation_20260516/
route validation status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only smoke:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_wrapper_plan_smoke_20260516/
wrapper plan-only status:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
toy protected accepts:
  1
toy wrong-key/wrong-payload accepts:
  0/0
full mode enabled:
  false
slurm submission started:
  false
```

The wrapper now consumes the repaired full16 allocation and precommit:

```text
rows per shard:
  1024
expected selected coordinates:
  16
score rows:
  results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl
allocation rows:
  results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/row_allocation_rows.jsonl
codebook:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/codebook.json
```

## Remote Preflight

The repaired full16 generation/decode control-plane files were synchronized to
Chimera and the remote route/wrapper preflight passed without allowlist
enablement or Slurm submission:

```text
remote preflight:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_remote_preflight_20260517_0008/
remote status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_REMOTE_PREFLIGHT_NO_SUBMIT
remote host:
  chimerahead.umb.edu
remote route validation:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
remote wrapper plan-only:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
remote allowlist:
  PASS, enabled_entries=[]
allowlist entry:
  v2_r4_after_868212_repaired_first_token_event_generation_h200
allowlist entry enabled:
  false
```

## Current Submission

The reviewed full16 first-token event generation diagnostic has been submitted
as one H200 Slurm array job after Hermes notification and single-enabled
allowlist preflight:

```text
submission record:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_submission_20260517_0016/
single-enabled preflight:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_single_enabled_preflight_20260517_0016/
job_id:
  868260
job_name:
  nat-ev-v2-r4c16
array:
  0-3%4
partition/qos/account:
  pomplun / pomplun / cs_yinxin.wan
gres:
  gpu:h200:1
time_limit:
  30-00:00:00
allowlist entry:
  v2_r4_after_868212_repaired_first_token_event_generation_h200
enabled entries after submission, local:
  []
enabled entries after submission, remote:
  []
```

## Next Allowed Action

The route may continue automatically. While job `868260` is active, the next
allowed action is monitoring and review only:

```text
next:
  monitor Slurm job 868260, sync completed shard artifacts back from Chimera,
  and review/aggregate results when all 4 shards finish
allowed:
  monitoring job 868260
  artifact synchronization for job 868260
  result review and aggregation after completion
  Hermes/Codex state synchronization
not allowed while 868260 is active:
  another generation submission
  any rerun of the same diagnostic
not yet allowed:
  training
```

This route does not unlock model scoring, training, Llama, same-family null,
sanitizer, FAR aggregation, payload diversity, or paper-facing positive claims.

Route-controlled actions may proceed automatically after their preconditions are
recorded; the user has authorized Codex and Hermes not to ask repeatedly for the
same approved route.

## Still Gate-Controlled

These actions are not permanently forbidden, but may proceed only after their
route-specific preconditions pass and are recorded in this file:

```text
larger generation route
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
```
