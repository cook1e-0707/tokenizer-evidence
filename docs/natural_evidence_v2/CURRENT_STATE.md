# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T23:00:14Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868151_FIRST_TOKEN_EVENT_TRACE_WRAPPER_PLAN_VALIDATED_LITERAL_DUP_REPAIR_NEXT`

## Current Route

Job `868151` completed the after-868016 controller-aware row-cylinder generation
H200/pomplun diagnostic:

```text
job_id: 868151
job_name: nat-ev-v2-r4cgen
array: 0-3
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
state: COMPLETED, 4/4 shards
exit_code: 0:0
elapsed: about 16-17 minutes per shard
```

Artifacts were synced locally:

```text
raw outputs/logs:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151/
review:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_review/
failure analysis:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_failure_analysis/
```

The precommitted full-phrase generation/decode gate failed:

```text
review status:
  FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE
protected accepts, format_scrub=all:
  0/4
raw/task-only/wrong-key/wrong-payload accepts, format_scrub=all:
  0/4 each
forbidden public surface count, format_scrub=all:
  26
forbidden term observed:
  coordinate
matched_surface_count, format_scrub=all:
  0
selected_surface_count, format_scrub=all:
  0
selected_coordinates_observed, format_scrub=all:
  0
duplicate response hash count:
  4422
```

Interpretation: this is not a Slurm/provider failure and not a false-accept
failure. It is a positive-channel transfer failure. The controller passed
teacher-forced first-token scoring in `868114`, but generated outputs did not
contain the precommitted full phrase surfaces required by the current decoder.

A corrected posthoc first-token event oracle was run for route planning only:

```text
oracle:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_oracle_20260516_v2/
status:
  FIRST_TOKEN_EVENT_ORACLE_V2_RECORDED_ARTIFACT_ONLY_NOT_PRECOMMITTED_NOT_POSITIVE
protected accepts ignoring forbidden:
  4/4
raw accepts ignoring forbidden:
  0/4
task-only accepts ignoring forbidden:
  0/4
wrong-key accepts ignoring forbidden:
  0/4
wrong-payload accepts ignoring forbidden:
  0/4
```

This oracle cannot reclassify `868151` as a pass because it was not
precommitted and ignores the forbidden-surface gate. It is evidence that the
next repair/pivot should target a precommitted first-token / lemma event-channel
rather than another full-phrase rerun.

Reviewed pivot route record:

```text
docs/natural_evidence_v2/R4_AFTER_868151_CONTROLLER_GENERATION_FAILURE_PIVOT_ROUTE_20260516.md
```

Codex formalized and locally validated the first-token event-channel
artifact-only route:

```text
event-channel spec:
  docs/natural_evidence_v2/R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_SPEC_20260516.md
route config:
  configs/natural_evidence_v2/r4_after_868151_first_token_event_channel.yaml
route validator:
  scripts/natural_evidence_v2/validate_r4_after_868151_first_token_event_channel_route.py
event decoder spec:
  results/natural_evidence_v2/precommit/r4_after_868151_first_token_event_channel_precommit_20260516/decoder_spec.json
route validation:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_channel_route_validation_20260516/
validation status:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_ROUTE_VALIDATION_NO_SUBMIT
tests:
  7 passed
remote validation:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_ROUTE_VALIDATION_NO_SUBMIT
remote allowlist safety:
  PASS zero-enabled
active Chimera jobs:
  none
```

Codex then implemented and replayed the first-token event decoder/extractor:

```text
decoder:
  scripts/natural_evidence_v2/decode_r4_after_868151_first_token_event_channel.py
decoder replay:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_decoder_replay_20260516/
replay status:
  FIRST_TOKEN_EVENT_DECODE_RECORDED_ARTIFACT_ONLY_NOT_POSITIVE
event rows:
  9216
event source:
  text_fallback_old_transcript only
event statuses:
  target=845, other=82, erasure=8289
protected accepts with quality gates:
  0/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts ignoring quality:
  0/4 each
protected forbidden public surface count:
  6
protected duplicate response hash count:
  755
```

Interpretation: event-level signal exists in the failed transcripts, but the
quality gates still fail. The replay uses text fallback because `868151` did not
store token-id event traces; future positive routes must store token-id traces.
The next repair must address forbidden technical literals and deterministic
duplicate outputs before any new generation route.

Codex recorded the artifact-only quality audit and repair route:

```text
quality audit:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_audit_20260516/
quality route:
  docs/natural_evidence_v2/R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_ROUTE_20260516.md
coordinate literal hits:
  14
likely ordinary domain-sense coordinate hits:
  10
within-condition duplicate response hash count:
  2803
```

Codex patched the future controller generation wrapper to store token-id event
traces:

```text
patched generator:
  scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py
new output fields:
  first_generated_token_id
  first_generated_token_text
  target_first_token_ids
  other_first_token_ids
  event_side
  event_bucket_side
  event_trace
tests:
  7 passed, 1 skipped
remote py_compile:
  PASS
remote allowlist safety:
  PASS zero-enabled
```

Codex validated the patched wrapper locally and remotely in plan-only mode:

```text
wrapper validation:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_wrapper_repair_validation_20260516/
status:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_TRACE_WRAPPER_REPAIR_PLAN_ONLY_VALIDATION
local route validation:
  PASS
local wrapper plan-only:
  PASS
remote route validation:
  PASS
remote wrapper plan-only:
  PASS
local/remote allowlist safety:
  PASS zero-enabled
slurm submitted:
  false
generation/model scoring/training started:
  false
```

## Next Allowed Action

Artifact-only first-token event wrapper repair only:

```text
1. Add prompt/domain filtering or a reviewed contextual technical-literal
   matcher so ordinary `coordination` tasks do not trigger hidden-channel
   literal failures.
2. Add duplicate-output mitigation and duplicate hash refusal checks.
3. Record a reviewed literal/duplicate repair plan.
4. Only after that repair plan passes validation, prepare a single reviewed H200 Slurm
   submission route.
```

Route-controlled actions may proceed automatically after their preconditions are
recorded. At this state, do not submit another Slurm generation/scoring/training
job until the literal/duplicate quality repair is implemented, reviewed, and
passes local/remote plan-only validation.

## Still Gate-Controlled

These actions may proceed automatically only after their route-specific
preconditions pass and are recorded in this file:

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
