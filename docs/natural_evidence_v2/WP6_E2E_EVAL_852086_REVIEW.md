# WP6 E2E Eval Review: Job 852086

## Scope

Job `852086` is the first Qwen V2 controlled-natural micro-slot
proof-of-life E2E run after WP5-R2 teacher-forced gate pass.

This review is diagnostic only. It is not a paper-facing positive claim, not
full FAR, not Llama, not sanitizer, and not same-family null evidence.

## Slurm Status

```text
job_id = 852086
job_name = nat-ev-v2-wp6e2e
state = COMPLETED
elapsed = 00:11:31
exit_code = 0:0
node = chimera13
```

Local artifacts:

```text
results/natural_evidence_v2/status/wp6_e2e_eval_852086/
logs/chimera/wp6_e2e_852086/
```

Key files:

```text
wp6_generated_outputs.jsonl
wp6_slot_observations.jsonl
wp6_decode_decisions.jsonl
wp6_e2e_summary.json
```

## Gate Result

```text
gate_status = FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE
```

Main metrics at budget 64:

| Condition | Accepts | Accept rate | Slot detection | Resolved slot rate | Target bucket hit |
|---|---:|---:|---:|---:|---:|
| protected | 8 / 64 | 0.125 | 1.0 | 0.8408203125 | 0.76171875 |
| raw | 0 / 64 | 0.0 | 1.0 | 0.1962890625 | 0.0927734375 |
| task_only | 0 / 64 | 0.0 | 1.0 | 0.1748046875 | 0.072265625 |
| wrong_key | 0 / 64 | 0.0 | 1.0 | 0.8408203125 | 0.76171875 |
| wrong_payload | 0 / 64 | 0.0 | 1.0 | 0.8408203125 | 0.76171875 |

Gate target:

```text
protected_accept_rate_at_64 >= 0.80
observed = 0.125
```

Null controls in this diagnostic were clean:

```text
raw_accepts = 0
task_only_accepts = 0
wrong_key_accepts = 0
wrong_payload_accepts = 0
```

This is not full FAR because the positive channel did not pass and the null
scope is limited to the WP6 diagnostic arms.

## What Improved

Compared with v1, the controlled-natural route fixed the core observability
failure:

```text
protected slot_detection_rate_at_64 = 1.0
```

The protected model also shows a real free-generation signal:

```text
protected target_bucket_hit_rate_at_64 = 0.76171875
raw target_bucket_hit_rate_at_64 = 0.0927734375
task_only target_bucket_hit_rate_at_64 = 0.072265625
```

This means WP5-R2 did transfer target-bucket preference into free generation.
The failure is no longer "no signal" or "no observable frame"; it is that the
current all-16-digits prompt-local decoder needs exact complete recovery, while
the observed protected response still has enough per-slot misses and out-of-bank
surfaces to break most frames.

## Failure Details

Protected accepted frames:

```text
8 / 64 frames
```

Accepted protected frames were exact `a5 5e` decodes with all 16 target hits.
Many failed frames were close but not complete. Example:

```text
frame 4 decoded payload a1 and checksum 5e
target_hit_count = 15 / 16
```

Dominant erasure reasons across all decode conditions:

```text
observed_first_word_not_in_primary_bucket_set = 275
duplicate_step_slots = 3
```

Protected step-level target hit rates:

| Step | Target bit | Resolved rate | Target hit rate | Common unresolved words |
|---:|---:|---:|---:|---|
| 1 | 1 | 1.000 | 1.000 | none |
| 2 | 0 | 1.000 | 1.000 | none |
| 3 | 1 | 0.906 | 0.891 | Develop, Assign, Schedule |
| 4 | 0 | 0.875 | 0.875 | Assign, Schedule, Design |
| 5 | 0 | 0.938 | 0.938 | Schedule |
| 6 | 1 | 0.516 | 0.281 | Assign, Schedule, Use |
| 7 | 0 | 0.672 | 0.547 | Assign, Schedule, Train |
| 8 | 1 | 0.562 | 0.438 | Assign, Research, Confirm |
| 9 | 0 | 0.750 | 0.641 | Assign, Early, Implement |
| 10 | 1 | 0.734 | 0.594 | Assign, Encourage, Use |
| 11 | 0 | 0.922 | 0.844 | Schedule, Purchase |
| 12 | 1 | 0.844 | 0.750 | Early, Provide, Develop |
| 13 | 1 | 0.891 | 0.781 | Establish, Offer, Document |
| 14 | 1 | 0.859 | 0.766 | Establish, Early, Develop |
| 15 | 1 | 0.984 | 0.859 | Test |
| 16 | 0 | 1.000 | 0.984 | none |

The weakest protected positions are steps 6 through 10. Those slots account for
many near-miss frames.

## Interpretation

The job is a completed negative WP6 proof-of-life result under the current
decoder gate:

```text
all-16-digits prompt-local frame recovery is too brittle at the current
free-generation slot-hit level.
```

It is not the same failure as v1:

- v1 failed because frames were not observable and symbol survival was below 1%.
- WP6 has full structural observability and strong protected lift.
- WP6 fails because a small number of out-of-bank or wrong-bucket first words
  destroy exact 16-bit payload+checksum acceptance.

The current result supports:

```text
controlled-natural micro-slots can induce a measurable protected-vs-null
free-generation bucket signal, but the current exact prompt-local 16-bit decoder
does not yet meet the proof-of-life recovery gate.
```

It does not support:

- natural-output success;
- paper-facing payload recovery;
- full FAR;
- robustness;
- cross-family generality;
- Llama replication;
- superiority over Scalable/Perinucleus.

## Current Next State

Do not submit another WP6 job yet.

Next allowed action is artifact-only WP6 failure diagnosis and repair planning:

- analyze near-miss frames and Hamming/erasure patterns;
- classify out-of-bank surfaces by step and prompt family;
- evaluate whether a closed surface set, repeated coordinates, erasure-tolerant
  prompt-local code, or slot-specific bucket policy could turn the observed
  protected signal into reliable recovery;
- keep new training, Llama, sanitizer, FAR, and paper positive claims blocked
  until a reviewed repair plan exists.
