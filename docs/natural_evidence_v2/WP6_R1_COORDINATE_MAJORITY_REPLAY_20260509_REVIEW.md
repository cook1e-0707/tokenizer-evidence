# WP6-R1 Coordinate-Majority Replay Review: 2026-05-09

## Scope

This is an artifact-only replay over existing job `852086` outputs. It tests a
candidate repeated-coordinate majority decoder for WP6-R1.

It does not reclassify `852086` as a passing proof-of-life run because the
majority decoder was not precommitted before the `852086` transcript was
generated.

## Artifacts

```text
scripts/natural_evidence_v2/replay_wp6_coordinate_majority_decoder.py
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_replay_20260509_1742/
```

Output files:

```text
wp6_r1_coordinate_majority_contract.json
wp6_r1_coordinate_majority_decode_rows.jsonl
wp6_r1_coordinate_majority_summary.json
wp6_r1_coordinate_support_by_budget.csv
```

## Replay Result

```text
replay_gate_status = PASS_WP6_R1_COORDINATE_MAJORITY_REPLAY_READY_FOR_REPLACEMENT_PREFLIGHT
```

Budget results:

| Budget | Protected | Raw | Task-only | Wrong-key | Wrong-payload |
|---:|---|---|---|---|---|
| 8 | reject `a15e` | reject | reject | reject | reject |
| 16 | reject `a15e` | reject | reject | reject | reject |
| 32 | accept `a55e` | reject | reject | reject | reject |
| 64 | accept `a55e` | reject `7400` | reject `5020` | reject `a55e` | reject `a55e` |

Configured R1 replay gate at budget 64:

```text
protected accepts = true
raw accepts = false
task_only accepts = false
wrong_key accepts = false
wrong_payload accepts = false
min_support = 33 >= 16
min_majority_margin = 3 >= 3
```

## Interpretation

The current protected model has enough repeated-coordinate signal for a
precommitted majority decoder at budget 64. The exact per-response decoder is
the bottleneck for `852086`, not the absence of a protected free-generation
signal.

This justifies preparing one replacement WP6-R1 E2E evaluation with the
repeated-coordinate majority decoder precommitted before generation.

## Validation

```text
.venv/bin/python -m pytest \
  tests/test_natural_evidence_v2_wp6_coordinate_majority.py \
  tests/test_natural_evidence_v2_wp6_e2e_decode.py

3 passed
```

## Next Allowed Action

Prepare a reviewed replacement WP6-R1 Slurm wrapper that:

1. generates fresh Qwen protected/raw/task-only responses;
2. runs the existing exact decoder only to produce slot observations;
3. runs the precommitted repeated-coordinate majority decoder;
4. writes exact-frame and majority-decoder summaries;
5. submits at most one allowlisted Chimera Slurm job after wrapper review.

Still forbidden:

- new training;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- paper-facing positive claims.
