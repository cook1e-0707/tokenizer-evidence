# WP6-R1 Coordinate-Majority E2E Review: Job 852094

## Decision

`852094` completed successfully and the precommitted WP6-R1
repeated-coordinate majority gate passes for the locked Qwen v2 proof-of-life
replacement evaluation.

This is an internal gate review only. It is not FAR aggregation, not a Llama or
same-family null result, not a sanitizer benchmark, and not a paper-facing
positive claim.

## Slurm Result

```text
job_id = 852094
job_name = nat-ev-v2-wp6r1
state = COMPLETED
exit_code = 0:0
elapsed = 00:11:03
node = chimera12
```

Local synced artifacts:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094/
```

Slurm logs:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094/slurm/nat-ev-v2-wp6r1-852094.out
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094/slurm/nat-ev-v2-wp6r1-852094.err
```

## Precommit Check

The wrapper wrote the R1 contract before generation:

```text
precommit/wp6_r1_coordinate_majority_contract.json  May 9 13:49:29 2026
wp6_generated_outputs.jsonl                         May 9 14:00:31 2026
coordinate_majority_replay/...summary.json          May 9 14:00:32 2026
```

Decoder:

```text
decoder_id = qwen_v2_wp6_r1_repeated_coordinate_majority_decoder_v1
minimum_support_at_64 = 16
minimum_majority_margin_at_64 = 3
query_budgets = [8,16,32,64]
```

The precommit contract and replay contract are byte-identical:

```text
sha256(precommit/wp6_r1_coordinate_majority_contract.json)
  = bdba4cb72c996a9125c4d61714ac5bf18e6e0cf704c83de9d11d5a9b7669073f

sha256(coordinate_majority_replay/wp6_r1_coordinate_majority_contract.json)
  = bdba4cb72c996a9125c4d61714ac5bf18e6e0cf704c83de9d11d5a9b7669073f
```

## R1 Majority Gate

Source:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094/coordinate_majority_replay/wp6_r1_coordinate_majority_summary.json
```

Budget 64 results:

| Condition | Accepted | Decoded hex | Min support | Min margin |
| --- | --- | --- | ---: | ---: |
| protected | true | a55e | 33 | 3 |
| raw | false | 7400 | 2 | 2 |
| task_only | false | 5020 | 1 | 1 |
| wrong_key | false | a55e | 33 | 3 |
| wrong_payload | false | a55e | 33 | 3 |

The R1 replay summary reports:

```text
replay_gate_status = PASS_WP6_R1_COORDINATE_MAJORITY_REPLAY_READY_FOR_REPLACEMENT_PREFLIGHT
replay_gate_pass = true
```

Follow-up cleanup:
`docs/natural_evidence_v2/WP6_R1_METADATA_CLEANUP_20260509.md` updates the
replay script and WP6-R1 wrapper so future replacement runs emit
`precommitted_transcript=true` and no longer emit the stale replay-era
`post_hoc_not_precommitted_for_852086` field. A cleaned local summary for
`852094` is recorded under
`results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094_metadata_cleaned_20260509_1839/`.

The protected condition accepts the committed `a55e` codeword at budget 64 with
minimum support `33 >= 16` and minimum majority margin `3 >= 3`. Raw and
task-only reject. Wrong-key and wrong-payload reject under the checksum/key and
payload checks even though their source observations share the protected
majority bits.

## Exact-Frame Decoder Context

The legacy exact-frame WP6 decoder still fails on this run:

```text
gate_status = FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE
protected_accept_rate_at_64 = 0.125
protected_slot_detection_rate_at_64 = 1.0
protected_target_bucket_hit_rate_at_64 = 0.76171875
```

This does not control the R1 decision; R1 was explicitly submitted as a
replacement evaluation with the repeated-coordinate majority decoder
precommitted before generation.

## Claim Control

- no new training was started by this job;
- no Llama or same-family null was started;
- no sanitizer benchmark was started;
- no FAR aggregation was started;
- no paper-facing positive claim is authorized from this review.

## Validation

Local gate check:

```text
jq -e 'R1 budget-64 gate predicate' .../wp6_r1_coordinate_majority_summary.json
true
```

Forbidden public-surface check over the exact decoder summary:

```text
jq -e '[.budget_summary[][] .forbidden_public_surface_count] | all(. == 0)' .../wp6_e2e_summary.json
true
```

## Next Action

Stop automatic WP6 expansion. No further WP6 jobs, training, Qwen E2E reruns,
Llama, same-family nulls, sanitizer benchmarks, FAR aggregation, or
paper-facing positive claims should start from this tick.
