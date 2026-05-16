# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T23:51:00Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_FAILED_NO_SUBMIT`

## Current Route Result

Job `868212` completed the reviewed quality-repaired after-868151 controller
generation diagnostic on Chimera H200:

```text
job_id: 868212
job_name: nat-ev-v2-r4cgen
array: 0-3%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
state: COMPLETED, 4/4 shards
exit_code: 0:0
elapsed: about 16-17 minutes per shard
```

Artifacts were synced locally:

```text
raw outputs/logs:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212/
review:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_review/
review summary:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_review/quality_repaired_generation_review_summary.json
```

The precommitted first-token event diagnostic gate passed at small diagnostic
scale:

```text
review status:
  PASS_R4_AFTER_868151_QUALITY_REPAIRED_FIRST_TOKEN_EVENT_BLOCK_DIAGNOSTIC_GATE_NOT_LOCKED_POSITIVE_GLOBAL_DUPLICATE_CAVEAT
protected accepts:
  3/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
first-token block-level forbidden public surface count:
  0
first-token block-level duplicate response hash count:
  0
token-id event traces:
  9216
event status counts:
  target=839, other=84, erasure=8293
```

The one protected failed block is specific:

```text
block_id: shard_03_block_00
decoded_bits: 1-100101
expected_bits: 10100101
missing_bit_indices: 1
missing coordinate: 26
min_pair_support: 0
complete_pairs: 7/8
```

The full-phrase decoder remains failed, as expected:

```text
full-phrase protected accepts, format_scrub=all:
  0
```

This result is diagnostic only, not a locked positive or paper claim. The main
caveat is global duplication across generated outputs:

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

Interpretation: the quality-repaired first-token event route produced a real
small-scale positive signal under token-id traces with clean null arms and clean
block-level quality gates. It still does not justify locked positive claims
because scale is only 4 protected blocks, one block failed by coordinate erasure,
full-phrase decoding remains failed, and global duplicate hashes remain high.

## Failure Attribution

The artifact-only attribution has been recorded:

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

Interpretation: the protected failure is a localized erasure/reliability issue
for coordinate 26 in shard_03, not a null-accept failure. The duplicate caveat
is global and dominated by deterministic identical generations across paired
shards and protected/raw or same-condition repetitions. The per-block duplicate
gate is clean, but this is not sufficient for a locked-scale positive.

## Next Allowed Action

The first artifact-only repair preflight has been implemented and run:

```text
route:
  docs/natural_evidence_v2/R4_AFTER_868212_FIRST_TOKEN_EVENT_RELIABILITY_DUPLICATE_REPAIR_ROUTE_20260516.md
route status:
  RECORDED_R4_AFTER_868212_FIRST_TOKEN_EVENT_RELIABILITY_DUPLICATE_REPAIR_ROUTE_ARTIFACT_ONLY_NO_SUBMIT
preflight:
  results/natural_evidence_v2/status/r4_after_868212_reliability_duplicate_repair_preflight_20260516/
preflight status:
  FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT
singleton/codebook failures:
  bit 1 active=[26], coordinate_26 sole active coordinate
  bit 3 active=[19]
  bit 5 active=[8]
  bit 6 active=[4]
duplicate extra rows:
  4424
tests:
  14 passed, 1 skipped
next:
  artifact-only repaired codebook/duplicate-policy construction or pivot route
not allowed:
  new Slurm generation/scoring/training until this preflight passes or a new
  reviewed pivot route supersedes it
```

Route-controlled actions may proceed automatically after their preconditions are
recorded; the user has authorized Codex and Hermes not to ask repeatedly for the
same approved route. At this state, do not submit another Slurm generation,
model-scoring, or training job until the artifact-only review above records a
new route.

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
