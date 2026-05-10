# WP6 Failure Diagnosis and R1 Repair Plan: Job 852086

## Decision

The user approved continuing WP6-stage execution without repeated approvals on
the already defined route. This does not remove gates. Because job `852086`
failed the current proof-of-life gate, the next WP6 action is artifact-only
repair planning and decoder preflight, not another E2E Slurm job and not new
training.

## Source Artifacts

```text
results/natural_evidence_v2/status/wp6_e2e_eval_852086/
docs/natural_evidence_v2/WP6_E2E_EVAL_852086_REVIEW.md
```

Artifact-only failure diagnosis output:

```text
results/natural_evidence_v2/status/wp6_e2e_eval_852086_failure_diagnosis_20260509_1753/
```

Files:

```text
wp6_failure_diagnosis_summary.json
wp6_protected_frame_diagnosis.csv
wp6_step_slot_diagnosis.csv
wp6_out_of_bank_surfaces.csv
wp6_near_miss_frames.csv
wp6_coordinate_majority_replay.csv
wp6_coordinate_majority_budget_replay.csv
```

## What 852086 Proved

The run failed exact per-frame payload recovery:

```text
protected_accept_rate_at_64 = 0.125
gate target = >= 0.80
```

But it also proved two important positive diagnostics:

```text
protected_slot_detection_rate_at_64 = 1.0
protected_target_bucket_hit_rate_at_64 = 0.76171875
raw_target_bucket_hit_rate_at_64 = 0.0927734375
task_only_target_bucket_hit_rate_at_64 = 0.072265625
```

Null accepts were zero:

```text
raw_accepts = 0
task_only_accepts = 0
wrong_key_accepts = 0
wrong_payload_accepts = 0
```

This is not full FAR because the positive gate failed and the null scope is
limited.

## Failure Shape

The exact all-16-digits prompt-local decoder is too brittle.

Protected frame statistics:

```text
accepted_frames = 8 / 64
mean_target_hits = 12.1875 / 16
median_target_hits = 12.5 / 16
mean_resolved_slots = 13.453125 / 16
median_resolved_slots = 14 / 16
mean_hamming_lower_bound_to_target = 3.8125
```

Near-miss distribution:

```text
hamming_lower_bound <= 1: 15 / 64
hamming_lower_bound <= 2: 23 / 64
hamming_lower_bound <= 3: 32 / 64
hamming_lower_bound <= 4: 41 / 64
```

Observed exact frame recovery remains low because one or two slot misses destroy
the entire prompt-local frame.

## Coordinate-Majority Replay

The key new artifact-only diagnostic is repeated-coordinate majority replay.
It treats all 64 responses as repeated observations of the same 16 coordinates,
ignores unresolved out-of-bank observations, and majority-votes the resolved
bucket IDs at each step coordinate.

This decoder was not precommitted for job `852086`, so it cannot retroactively
turn `852086` into a passing proof-of-life result. It is only evidence for the
next WP6 repair.

Budget replay:

| Budget | Protected majority hex | Protected target bits | Raw majority hex | Task-only majority hex |
|---:|---|---:|---|---|
| 8 | `a15e` | 15 / 16 | incomplete | incomplete |
| 16 | `a15e` | 15 / 16 | incomplete | incomplete |
| 32 | `a55e` | 16 / 16 | incomplete | incomplete |
| 64 | `a55e` | 16 / 16 | `7400` | `5020` |

At budget 64, protected majority votes by coordinate:

```text
bits = [1,0,1,0,0,1,0,1,0,1,0,1,1,1,1,0]
hex  = a55e
```

The raw and task-only majority codes do not match the committed payload.

## Weak Slots

The weakest protected coordinate is step 6:

```text
target bit = 1
bucket 1 count = 18
bucket 0 count = 15
majority margin = 3
```

Other weak areas are steps 7-10, but their majority margins are stronger by
budget 64. This suggests the current signal is sufficient for repeated-coordinate
decoding at budget 32/64, but not yet sufficient for per-response exact frame
decoding.

## WP6-R1 Repair Plan

### R1.1 Define a precommitted repeated-coordinate decoder

Create a contract that explicitly fixes before generation:

```text
protocol_id
audit_key_id
payload
prompt_order
query_budgets
slot_policy
bucket_policy
coordinate_id = step_index
decoder = coordinate majority over resolved bucket hits
minimum_support_per_coordinate
minimum_majority_margin
checksum rule
null acceptance threshold
```

Do not select prompts, thresholds, key, payload, or coordinates after seeing
transcripts.

### R1.2 Artifact-only decoder oracle and replay

Before any new Slurm job, implement local artifact-only replay over `852086`:

```text
protected should decode a55e at budget 32 and 64
raw should reject
task-only should reject
wrong-key should reject
wrong-payload should reject
```

The replay must output:

```text
coordinate_majority_decode_rows.jsonl
coordinate_majority_decode_summary.json
coordinate_support_by_budget.csv
coordinate_margin_by_budget.csv
```

### R1.3 Conservative gate for replacement WP6

Only after the artifact-only replay and contract pass:

```text
protected majority decode accepts @64
raw majority decode rejects @64
task-only majority decode rejects @64
wrong-key rejects @64
wrong-payload rejects @64
minimum coordinate support >= 16 at budget 64
minimum majority margin >= 3 at budget 64
forbidden surface count = 0
```

Then prepare one replacement WP6 Slurm eval using the precommitted repeated
coordinate decoder. This is still Qwen-only. Do not start Llama or FAR.

## Still Forbidden

- no new training until a reviewed decoder repair says training is the blocker;
- no second WP6 Slurm job until R1 precommit and artifact-only replay pass;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no FAR aggregation;
- no paper positive claim;
- do not reclassify `852086` as a passing proof-of-life result.

## Current Next Allowed Action

Implement WP6-R1 artifact-only repeated-coordinate decoder contract and replay
over existing `852086` artifacts. No Chimera CPU/GPU is required for that step.
