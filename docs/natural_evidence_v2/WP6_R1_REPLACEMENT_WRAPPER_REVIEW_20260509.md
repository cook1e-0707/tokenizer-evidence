# WP6-R1 Replacement Wrapper Review: 2026-05-09

## Decision

The WP6-R1 repeated-coordinate majority wrapper is ready for one allowlisted
Chimera Slurm submission.

This is a replacement WP6 proof-of-life evaluation with a decoder contract
written before generation. It does not start new training.

## Wrapper

```text
scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_e2e_eval.sbatch
```

The wrapper order is:

1. write `precommit/wp6_r1_coordinate_majority_contract.json`;
2. generate fresh protected/raw/task-only Qwen responses;
3. run the existing exact-frame decoder to produce slot observations;
4. run the repeated-coordinate majority decoder;
5. write the WP6-R1 majority replay summary.

## Decoder

```text
decoder_id = qwen_v2_wp6_r1_repeated_coordinate_majority_decoder_v1
coordinates = strict Step 1..16 indices
erasure_policy = ignore unresolved out-of-bank first words
accept_rule = majority codeword checksum_valid_and_payload_matches_expected
minimum_support_at_64 = 16
minimum_majority_margin_at_64 = 3
query_budgets = [8,16,32,64]
```

## Validation

Local wrapper validation:

```text
VALIDATE_PLAN_ONLY=1 ... bash scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_e2e_eval.sbatch
```

Output:

```text
results/natural_evidence_v2/status/wp6_r1_wrapper_validate_20260509_1746/
```

Tests:

```text
.venv/bin/python -m pytest \
  tests/test_natural_evidence_v2_wp6_coordinate_majority.py \
  tests/test_natural_evidence_v2_wp6_e2e_decode.py
```

Result:

```text
3 passed
```

`bash -n` also passed.

## Allowlist

Enabled single-submission action:

```text
name = v2_wp6_r1_coordinate_majority_e2e_eval
command = sbatch scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_e2e_eval.sbatch
```

The allowlist entry must be disabled immediately after submission.

## Still Forbidden

- no new training;
- no Llama;
- no same-family null;
- no sanitizer;
- no FAR aggregation;
- no paper positive claim.

## Status

```text
PASS_READY_TO_SUBMIT_ONE_ALLOWLISTED_WP6_R1_SLURM_JOB
```
