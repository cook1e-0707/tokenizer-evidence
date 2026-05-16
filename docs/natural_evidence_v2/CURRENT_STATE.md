# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T22:36:26Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868151_CONTROLLER_GENERATION_FAILED_FIRST_TOKEN_PIVOT_ARTIFACT_ONLY`

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

## Next Allowed Action

Artifact-only first-token event-channel route formalization only:

```text
1. Write a precommitted first-token / lemma event decoder spec.
2. Define tokenizer-backed event extraction from prefix-native continuations.
3. Define forbidden-surface and duplicate-response checks for this route.
4. Validate route and wrapper locally in plan-only mode.
5. Record a reviewed route decision before allowlist enablement or Slurm submission.
```

Route-controlled actions may proceed automatically after their preconditions are
recorded. At this state, do not submit another Slurm generation/scoring/training
job until the first-token event route is formalized, reviewed, and passes its
preflight.

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
