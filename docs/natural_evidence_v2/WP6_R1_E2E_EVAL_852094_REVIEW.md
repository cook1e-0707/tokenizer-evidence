# WP6-R1 Coordinate-Majority E2E Eval Review: Job 852094

## Scope

This review covers the replacement WP6-R1 Qwen v2 proof-of-life evaluation
submitted as Slurm job `852094`.

The run used the repeated-coordinate majority decoder contract written before
generation. It did not start new training.

## Slurm

```text
job_id = 852094
job_name = nat-ev-v2-wp6r1
partition = DGXA100
node = chimera12
state = COMPLETED
elapsed = 00:11:03
exit_code = 0:0
```

Local artifacts:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094/
```

## Provenance

The precommitted decoder contract and the replay contract are byte-identical:

```text
precommit/wp6_r1_coordinate_majority_contract.json
  sha256 = bdba4cb72c996a9125c4d61714ac5bf18e6e0cf704c83de9d11d5a9b7669073f

coordinate_majority_replay/wp6_r1_coordinate_majority_contract.json
  sha256 = bdba4cb72c996a9125c4d61714ac5bf18e6e0cf704c83de9d11d5a9b7669073f
```

Artifact counts:

```text
wp6_generated_outputs.jsonl = 192 rows
wp6_slot_observations.jsonl = 5120 rows
wp6_decode_decisions.jsonl = 320 rows
wp6_r1_coordinate_majority_decode_rows.jsonl = 20 rows
```

## Exact Decoder Result

The old exact per-response decoder remains failed:

```text
protected_accept_rate_at_64 = 0.125
protected_slot_detection_rate_at_64 = 1.0
protected_target_bucket_hit_rate_at_64 = 0.76171875
exact_decoder_gate_status = FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE
```

This preserves the prior diagnosis: the exact all-16-slots-per-response frame
decoder is too brittle at the current free-generation hit rate.

## WP6-R1 Majority Decoder Result

Budget results:

| Budget | Protected | Raw | Task-only | Wrong-key | Wrong-payload |
|---:|---|---|---|---|---|
| 8 | reject `a15e` | reject | reject | reject | reject |
| 16 | reject `a15e` | reject | reject | reject | reject |
| 32 | accept `a55e` | reject | reject | reject | reject |
| 64 | accept `a55e` | reject `7400` | reject `5020` | reject `a55e` | reject `a55e` |

Budget-64 protected support:

```text
decoded_hex = a55e
payload = a5
checksum = 5e
min_support = 33
min_majority_margin = 3
accepted = true
```

Budget-64 null controls:

```text
raw accepted = false
task_only accepted = false
wrong_key accepted = false
wrong_payload accepted = false
```

## Interpretation

Job `852094` is the first precommitted Qwen v2 WP6-R1 proof-of-life pass for
the controlled-natural Step-label micro-slot route under the repeated-coordinate
majority decoder.

The result is still a narrow proof-of-life diagnostic. It is not a full FAR
estimate, not a Llama result, not a sanitizer result, and not a paper-facing
robustness or superiority claim.

## Caveat

The generated majority summary still contains the inherited field:

```text
post_hoc_not_precommitted_for_852086 = true
replay_gate_status = PASS_WP6_R1_COORDINATE_MAJORITY_REPLAY_READY_FOR_REPLACEMENT_PREFLIGHT
```

For job `852094`, this label is stale metadata from the replay script. The
actual replacement-run provenance is clean because the `precommit/` contract
exists in the run directory and is byte-identical to the contract used by the
majority decoder. Future summaries should rename this field before scaled
reruns.

## Validation

Local tests:

```text
.venv/bin/python -m pytest \
  tests/test_natural_evidence_v2_wp6_coordinate_majority.py \
  tests/test_natural_evidence_v2_wp6_e2e_decode.py

3 passed
```

## Gate Status

```text
PASS_WP6_R1_QWEN_V2_COORDINATE_MAJORITY_PROOF_OF_LIFE
```

## Still Forbidden

- no full FAR claim;
- no Llama claim;
- no same-family null claim;
- no sanitizer claim;
- no robustness to paraphrase claim;
- no superiority claim over Scalable/Perinucleus;
- no paper-facing positive claim beyond this narrow diagnostic.

## Next Allowed Action

Prepare a WP6-R1 scale/reproducibility decision package. It should determine
whether the next single Slurm action is a larger Qwen-only majority-decoder
eval over more prompts/payload cells, or an artifact-only cleanup of the stale
summary metadata before scaling.
