# WP6-R1 Scale Failure Diagnosis and Repair Plan: 852202

## Decision

This is an artifact-only diagnosis of the reviewed WP6-R1 scale run:

```text
job_id = 852202
scale_gate_status = FAIL_WP6_R1_COORDINATE_MAJORITY_SCALE_GATE
controlling_budget = 64
```

Job `852202` remains a failed precommitted scale result. This report does not
retroactively lower the margin threshold, does not submit Slurm, does not train,
does not rerun Qwen E2E generation, does not start Llama or same-family nulls,
does not run sanitizer benchmarks, does not aggregate FAR, and does not make a
paper-facing positive claim.

## Inputs Reviewed

```text
docs/natural_evidence_v1/AUTOMATION_STATE.md
docs/natural_evidence_v1/next_step_codex_plan.md
results/natural_evidence_v1/status/gate_status.json
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
results/natural_evidence_v2/status/gate_status.json
docs/specs/stage4_real_integration_spec.md
docs/natural_evidence_v2/WP6_R1_COORDINATE_MAJORITY_SCALE_EVAL_852202_REVIEW.md
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/coordinate_majority_scale/wp6_r1_scale_summary.json
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/coordinate_majority_scale/wp6_r1_scale_decode_rows.jsonl
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/coordinate_majority_scale/wp6_r1_scale_support_by_block_budget.csv
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_scale_eval_852202/wp6_slot_observations.jsonl
```

## Gate Failure

At budget `64`, all non-margin gates passed:

| Gate | Requirement | Observed |
|---|---:|---:|
| protected block accepts | >= 3 / 4 | 4 / 4 |
| raw accepts | 0 / 4 | 0 / 4 |
| task-only accepts | 0 / 4 | 0 / 4 |
| wrong-key accepts | 0 / 4 | 0 / 4 |
| wrong-payload accepts | 0 / 4 | 0 / 4 |
| min support in accepted protected blocks | >= 16 | 27 |
| forbidden public surface count | 0 | 0 |

The failure is exactly the precommitted protected accepted-block majority-margin
floor:

```text
required_min_majority_margin_in_accepted_blocks = 3
observed_protected_min_majority_margin_in_accepted_blocks = 2
```

The failing minimum comes from `block_3` at budget `64`. `block_3` still decoded
the target codeword:

```text
block_id = block_3
decoded_hex = a55e
payload_matches = true
checksum_valid = true
min_support = 28
min_majority_margin = 2
```

## Localized Cause

The weak coordinate is `block_3`, step `10`. The target bit at step `10` is
`1`. At budget `64`, the protected observations were nearly tied:

| block | step | target bit | bucket 0 | bucket 1 | unresolved | majority bit | margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| block_3 | 10 | 1 | 25 | 27 | 12 | 1 | 2 |

The observed first-word counts for this coordinate were dominated by the two
precommitted bucket groups, but with nearly balanced mass:

```text
bucket_0: Set = 24, Plan = 1
bucket_1: Create = 17, Prepare = 10
unresolved other first words = 12
```

The budget curve shows that extra prompts in this block did not turn the
coordinate into a strong majority:

| budget | bucket 0 | bucket 1 | resolved | majority bit | margin |
|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 4 | 4 | 1 | 4 |
| 16 | 4 | 8 | 12 | 1 | 4 |
| 32 | 12 | 13 | 25 | 1 | 1 |
| 64 | 25 | 27 | 52 | 1 | 2 |

Step `10` was not globally broken: at budget `64`, the protected step-10 margins
for the other blocks were `29`, `13`, and `13`. The failure is therefore a
prompt-slice-specific weak coordinate in `block_3`, not an artifact-completeness
failure, not null leakage, and not a checksum or payload mismatch.

## Repair Options

### Option A: Lower the majority-margin threshold

Lowering the protected accepted-block margin threshold from `3` to `2` would fit
the observed `block_3` result, but it is not recommended as the primary repair.
Changing the decode threshold after seeing the transcripts is post-hoc under
the v2 protocol. If an expert still wants to evaluate this threshold, it must be
precommitted for a future diagnostic run only, with `852202` still recorded as a
failed run.

Minimum safeguards if this option is chosen:

```text
do not relabel 852202 as pass
precommit threshold before any future generation
require protected block accepts = 4 / 4 at budget 64
require null accepts = 0 / 4 for every null condition
require support >= 16
keep paper_claim_allowed = false
```

### Option B: Increase independent blocks and count robust block accepts

Recommended primary repair: keep the per-block budget `64` and keep the margin
threshold `3`, but make the next scale diagnostic use more independent blocks
and count robust protected block accepts.

Proposed future R2 diagnostic contract:

```text
payload_plus_checksum_hex = a55e
model route = same Qwen WP6-R1 route, no new training
prompt scope = fresh precommitted wp3_r1_eval slice not used by 852202
block_count = 8
block_size = 64
query_budgets_per_block = [8, 16, 32, 64]
```

Define robust block accept at budget `64` as:

```text
decoded_hex == a55e
payload_matches == true
checksum_valid == true
min_support >= 16
min_majority_margin >= 3
```

Recommended R2 gate:

```text
protected robust block accepts >= 6 / 8
raw robust accepts = 0 / 8
task-only robust accepts = 0 / 8
wrong-key robust accepts = 0 / 8
wrong-payload robust accepts = 0 / 8
forbidden_public_surface_count = 0
required artifacts complete
```

This preserves the original margin threshold and the per-block owner budget
while testing whether the `block_3` near tie was an isolated weak replicate.
The new block/prompt slice and the robust-accept aggregation must be fixed in a
wrapper review before any future generation. It must not be used to reinterpret
`852202` as a pass.

### Option C: Increase per-block budget

Secondary repair: precommit a fresh diagnostic with a larger per-block budget,
for example:

```text
query_budgets_per_block = [8, 16, 32, 64, 128]
controlling_budget = 128
```

This is less attractive as the first repair because `block_3` step `10` was
near-balanced at `64` (`27` vs `25`), so added budget may still leave the same
coordinate marginal if the prompt slice has a real local preference for bucket
`0`. It also changes the owner query-budget interpretation and must remain
diagnostic unless separately precommitted and reviewed.

## Recommended Next Allowed Action

Hold for review of this repair plan. If accepted, the next safe project action
is artifact-only R2 wrapper/contract planning for Option B. Do not submit Slurm
or start generation until a later notified tick explicitly permits one reviewed
allowlisted submission and disables the allowlist immediately afterward.

