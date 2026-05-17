# natural_evidence_v2 Current State

Last synchronized: 2026-05-17T02:42:22Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_REMOTE_PREFLIGHT_PASS_NO_SUBMIT`

## Active Route Update

The active route has been synchronized after expert review of `868260`:

```text
state sync:
  results/natural_evidence_v2/status/r4_after_868260_state_sync_20260517/
status:
  SYNCED_R4_AFTER_868260_FIRST_TOKEN_EVENT_QUALITY_REPAIR_ROUTE_ARTIFACT_ONLY_NO_SUBMIT
artifact validation:
  PASS_R4_AFTER_868260_FORENSICS_POLICY_TRACE_BINDING_ARTIFACTS_VALIDATED_NO_SUBMIT
active interpretation:
  failed strict-quality diagnostic with full protected codeword recovery before
  quality filtering
active route:
  provider-side keyed first-token event evidence with strict natural-output
  quality, duplicate, contextual-forbidden, and trace-binding gates
```

Older v3 training-objective blockers are historical for the active route. They
remain evidence that surface-mass/objective pressure was insufficient, but they
are not the current execution blocker.

New artifact-only packages are recorded:

```text
duplicate forensics:
  results/natural_evidence_v2/status/r4_868260_duplicate_forensics_20260517/
duplicate-safe generation policy v2 validation:
  results/natural_evidence_v2/status/r4_first_token_event_duplicate_safe_generation_policy_v2_validation_20260517/
contextual forbidden-surface policy v2 validation:
  results/natural_evidence_v2/status/r4_contextual_forbidden_surface_policy_v2_validation_20260517/
trace-binding validation:
  results/natural_evidence_v2/status/r4_first_token_event_trace_binding_validation_20260517/
quality-repair confirmation route:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_route_validation_20260517/
quality-repair confirmation wrapper plan smoke:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_wrapper_plan_smoke_20260517/
route decision:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_route_decision_20260517/
remote preflight:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_remote_preflight_20260517/
remote status:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_REMOTE_PREFLIGHT_NO_SUBMIT
remote host:
  chimerahead.umb.edu
remote route validation:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
remote wrapper plan-only:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
remote allowlist:
  PASS, enabled_entries=[]
local/remote hashes:
  match, 51 reviewed files
active Chimera jobs:
  0
tests:
  18 passed
```

## Prior Compute Result: 868212

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

## Current Review

Job `868260` completed and has been reviewed:

```text
review:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_868260_review/
failure analysis:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/
repair decision:
  results/natural_evidence_v2/status/r4_after_868212_generation_868260_quality_gate_repair_decision_20260517/
status:
  RECORDED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_868260_FAILED_QUALITY_GATE_SIGNAL_PRESENT_NO_SUBMIT
```

The run is not a positive result:

```text
strict protected accepts:
  2/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
full-phrase protected accepts, format_scrub=all:
  0
```

Interpretation:

```text
The first-token event signal recovered the expected codeword in all protected
blocks before quality filtering. The strict gate failed because shard_00 and
shard_01 protected blocks hit duplicate/forbidden quality filters.
```

Quality failure details:

```text
shard_00_block_00:
  decoded expected codeword, valid checksum, duplicate_response_hash_count=1
shard_01_block_00:
  decoded expected codeword, valid checksum,
  duplicate_response_hash_count=2,
  forbidden_public_surface_count=1
forbidden example:
  literal "bucket" in ordinary physical plumbing/home-maintenance sense
global duplicate extra rows:
  7612
unique response hashes:
  4676 / 12288
```

## Current Repair Package

The artifact-only quality-gate repair package has been precommitted and
validated:

```text
repair package:
  results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/
validation:
  results/natural_evidence_v2/status/r4_after_868260_quality_gate_repair_package_validation_20260517/
validation status:
  PASS_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT
tests:
  5 passed
reclassifies 868260:
  false
slurm allowed:
  false
```

The package contains:

```text
contextual forbidden-surface policy v2:
  ordinary physical "bucket" may be allowed under precommitted task-domain
  cues, while technical "bucket" remains forbidden
duplicate-safe generation policy:
  future within-block and global duplicate response hash gates remain 0
```

## Next Allowed Action

The route may continue automatically after recorded preconditions pass, but no
new Slurm rerun is allowed from the current state. The next allowed action is:

```text
next:
  record/review the separate single-submission or full-mode wrapper route for
  the 4-block quality-repair confirmation diagnostic; the current wrapper has
  only passed plan-only remote preflight and remains fail-closed outside
  plan-only mode
allowed:
  Hermes notification
  single-submission/full-mode wrapper route preparation
  exactly-one allowlist preflight after the route is recorded
  Hermes/Codex state synchronization
not allowed:
  reclassifying 868260 as positive
  another Slurm generation rerun before a new reviewed full-mode route is
  recorded and all preconditions pass
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
