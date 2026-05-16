# R4 After-868151 Controller Generation Failure Pivot Route

Date: 2026-05-16

## Decision

Job `868151` is a completed failed diagnostic, not a positive result.

The route remains Qwen-only and artifact-controlled. The immediate pivot is
artifact-only: formalize and audit a first-token / lemma event-channel route
before any further Slurm generation or scoring.

## Evidence

The H200/pomplun array completed cleanly:

```text
job_id: 868151
job_name: nat-ev-v2-r4cgen
array: 0-3
state: COMPLETED
exit_code: 0:0
```

The precommitted full-phrase decoder gate failed:

```text
review:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_review/controller_generation_review_summary.json
status:
  FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE
protected accepts, format_scrub=all:
  0/4
raw/task-only/wrong-key/wrong-payload accepts, format_scrub=all:
  0/4 each
forbidden public surface count, format_scrub=all:
  26
duplicate response hash count:
  4422
```

Failure attribution:

```text
results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_failure_analysis/
matched_surface_count_format_scrub_all: 0
selected_surface_count_format_scrub_all: 0
selected_coordinates_observed_format_scrub_all: 0
```

The main failure is not Slurm, tokenizer boundary, or false accept. It is a
generation-transfer mismatch: the controller was optimized for first-token
surface-cylinder mass, while the committed decoder required exact full phrase
surface matches. In free generation, the outputs drifted away from the locked
full phrases before any selected surface was observed.

## Posthoc Oracle

A corrected first-token event oracle was run on the same failed transcripts:

```text
results/natural_evidence_v2/status/r4_after_868151_first_token_event_oracle_20260516_v2/
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

This oracle is not precommitted and ignores the forbidden-surface gate. It
cannot reclassify `868151` as a pass. It is route-planning evidence that the
controller signal may survive at row-local first-token granularity even when
full phrase surfaces do not.

## Next Allowed Action

Artifact-only first-token event-channel route formalization:

```text
1. Write a precommitted first-token / lemma event decoder spec.
2. Define tokenizer-backed event extraction from prefix-native continuations.
3. Define forbidden-surface policy and duplicate-response checks for the new event route.
4. Run static/local validation on the route and wrapper in plan-only mode only.
5. Record a reviewed route decision before any allowlist enablement or Slurm submission.
```

Route-controlled actions may proceed automatically after their preconditions are
recorded. At this state, the route has not yet unlocked another Slurm
generation, scoring, training, Llama, same-family null, sanitizer, FAR,
payload-diversity, or paper-facing claim action.

## Claim Policy

Allowed internal statement:

```text
868151 cleanly completed but failed the precommitted full-phrase generation
diagnostic; a posthoc first-token oracle suggests a possible event-channel
pivot that must be precommitted before reuse.
```

Not allowed from this artifact:

```text
natural-output positive result
paper-facing positive claim
payload recovery
FAR result
cross-family result
sanitizer robustness
full phrase decoder success
```
