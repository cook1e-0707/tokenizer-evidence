# WP6 E2E 852086 Failure Diagnosis and Repair Plan

Status: artifact-only diagnosis and repair planning.

This record does not reclassify job `852086` as a pass. It authorizes no
training, no generation, no Qwen E2E rerun, no Llama run, no same-family null,
no sanitizer benchmark, no FAR aggregation, and no paper-facing positive
claim.

## Controlling Inputs

- `docs/natural_evidence_v2/PROTOCOL_CONTRACT.md`
- `docs/natural_evidence_v2/CLAIM_GUARDRAILS.md`
- `docs/natural_evidence_v2/WP6_E2E_PROOF_OF_LIFE.md`
- `docs/natural_evidence_v2/WP6_E2E_EVAL_852086_REVIEW.md`
- `results/natural_evidence_v2/status/wp6_e2e_eval_852086/wp6_e2e_summary.json`
- `results/natural_evidence_v2/status/wp6_e2e_eval_852086/wp6_decode_decisions.jsonl`
- `results/natural_evidence_v2/status/wp6_e2e_eval_852086/wp6_slot_observations.jsonl`

## Gate Diagnosis

Job `852086` completed successfully at the Slurm level, but failed the WP6
proof-of-life gate:

```text
protected_accept_rate_at_64 = 8 / 64 = 0.125
required protected_accept_rate_at_64 >= 0.80
gate_status = FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE
```

The failure is not a null leak or a structural detector failure:

```text
protected_slot_detection_rate_at_64 = 1.0
protected_target_bucket_hit_rate_at_64 = 0.76171875
raw_accepts_at_64 = 0
task_only_accepts_at_64 = 0
wrong_key_accepts_at_64 = 0
wrong_payload_accepts_at_64 = 0
forbidden_public_surface_count = 0
```

The main failure mode is that the precommitted decoder requires one complete
16-bit frame per response. The protected adapter produced a strong bucket
signal but not enough exact per-frame reliability:

```text
protected complete 16-bit frames = 15 / 64
protected accepted frames = 8 / 64
dominant erasure reason = observed_first_word_not_in_primary_bucket_set
out-of-bank erasure count = 275
duplicate_step_slots = 3
```

Among protected complete 16-bit frames, the Hamming-distance distribution from
the target `a55e` frame was:

| Hamming distance | Frame count |
|---:|---:|
| 0 | 8 |
| 1 | 4 |
| 2 | 1 |
| 3 | 1 |
| 4 | 1 |

This explains the observed accept rate: the current frame-local decoder works
only when every coordinate lands in the target bucket and the checksum byte
also matches.

## Coordinate Diagnosis

Protected step-level target hits show that the instability is concentrated in
steps 6 through 10:

| Step | Target bit | Target hits | Resolved slots | Main issue |
|---:|---:|---:|---:|---|
| 1 | 1 | 64 / 64 | 64 / 64 | stable |
| 2 | 0 | 64 / 64 | 64 / 64 | stable |
| 3 | 1 | 57 / 64 | 58 / 64 | mostly stable |
| 4 | 0 | 56 / 64 | 56 / 64 | mostly stable |
| 5 | 0 | 60 / 64 | 60 / 64 | stable |
| 6 | 1 | 18 / 64 | 33 / 64 | weak, many out-of-bank verbs |
| 7 | 0 | 35 / 64 | 43 / 64 | weak |
| 8 | 1 | 28 / 64 | 36 / 64 | weak, many out-of-bank verbs |
| 9 | 0 | 41 / 64 | 48 / 64 | borderline |
| 10 | 1 | 38 / 64 | 47 / 64 | borderline |
| 11 | 0 | 54 / 64 | 59 / 64 | mostly stable |
| 12 | 1 | 48 / 64 | 54 / 64 | moderate |
| 13 | 1 | 50 / 64 | 57 / 64 | moderate |
| 14 | 1 | 49 / 64 | 55 / 64 | moderate |
| 15 | 1 | 55 / 64 | 63 / 64 | mostly stable |
| 16 | 0 | 63 / 64 | 64 / 64 | stable |

Common protected out-of-bank surfaces in the weak coordinates were ordinary
action verbs such as `Assign`, `Schedule`, `Plan`, `Use`, `Research`,
`Confirm`, `Encourage`, and `Provide`. This is compatible with a controlled
natural micro-slot signal, but not with the current exact four-surface bucket
contract.

One diagnostic observation is important but cannot be used to rescue job
`852086`: a majority vote over the 64 protected frames recovers the target bit
sequence `a55e`, while raw and task-only majority vectors do not match that
target. Because the precommitted WP6 decoder was one-frame-per-response, this
majority-vote observation is repair evidence only, not a valid recovery result
for job `852086`.

## Repair Options

Rejected for the current result:

- Do not re-score `852086` with a new decoder and call it a pass.
- Do not tune a new key, payload, threshold, or bucket policy on `852086` and
  present the same run as eval evidence.
- Do not submit another WP6 job from the current state.
- Do not make payload recovery, FAR, robustness, cross-family, or paper-facing
  positive claims.

Candidate repair paths, all requiring a fresh reviewed precommit before any
future generation:

1. **Query-budget decoder repair.** Define a precommitted WP6-R2 decoder that
   uses the existing query budget as repetition evidence instead of requiring
   each response to be a complete 16-bit frame. This is the most direct repair
   suggested by the artifacts because the protected aggregate signal is strong
   while raw/task-only accepts remain zero.
2. **Erasure-tolerant code repair.** Keep the current slot policy but replace
   the all-16-exact frame with a lower-rate payload/checksum code that tolerates
   out-of-bank erasures and a small number of bit errors. This must be fixed
   before generation and must preserve wrong-key and wrong-payload rejection.
3. **Bucket-policy repair.** Expand or make step-specific the action-verb
   buckets to cover recurrent natural verbs. This changes the WP3/WP4/WP5
   contract and would require rerunning the relevant tokenizer, mass, oracle,
   and teacher-forced gates before any WP6 rerun.
4. **Prompt/slot repair.** Reduce weak-coordinate drift by designing prompts or
   slot positions that avoid the unstable middle-step action-verb distribution.
   This also changes the contract and must go back through the earlier gates.

## Recommended Next Allowed Action

The next safe action is still artifact-only:

```text
Implement a local WP6-R2 decoder-repair diagnostic plan over the existing
852086 artifacts only. The plan must label all results diagnostic, compare
query-budget voting and erasure-tolerant variants against raw/task-only/
wrong-key/wrong-payload controls, and emit a reviewed precommit proposal before
any new generation or Slurm submission is considered.
```

No WP6 rerun is authorized by this document.
