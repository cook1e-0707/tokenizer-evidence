# R3.2 Full Wrapper Aggregation Path: 2026-05-11 06:45Z

## Decision

Implemented the R3.2 same-contract `a55e` full wrapper aggregation path.

No Slurm job was submitted. No allowlist entry was enabled. No training,
generation, Qwen E2E rerun, Llama, same-family null, sanitizer benchmark, FAR
aggregation, or paper-facing positive claim was started.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_full_wrapper_aggregation_path_20260511_0645.json
```

## Implemented Paths

```text
scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

The wrapper still supports local `VALIDATE_PLAN_ONLY=1` precommit validation.
For the reviewed future Slurm path, it now isolates each canonical replicate
group under:

```text
shards/shard_00 .. shards/shard_11
```

Each shard uses the same `a55e` WP4/WP5-R2 contract and its assigned
deterministic prompt window. The wrapper decodes each shard independently, then
aggregates only canonical R3.2 artifacts:

```text
r3_2_generation_summary.json
r3_2_generated_outputs.jsonl
r3_2_slot_observations.jsonl
r3_2_decode_decisions.jsonl
r3_2_coordinate_majority_decode_rows.jsonl
r3_2_coordinate_majority_summary.json
r3_2_support_by_block_budget.csv
r3_2_gate_review.json
```

Canonical aggregate block IDs are:

```text
C_A55E_shard_XX_block_YY
```

## Local Validation

Commands run locally only:

```text
python3 -m py_compile scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py scripts/natural_evidence_v2/replay_r3_2_same_contract_from_852426.py scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py
bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py
VALIDATE_PLAN_ONLY=1 REPO_HOME=/Users/guanjie/Documents/tokenizer_alignment PYTHON=.venv/bin/python RUN_ROOT=/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status OUTPUT_DIR=/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_full_wrapper_plan_validation_20260511_0645 bash scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
```

Validation result:

```text
py_compile = PASS
bash -n = PASS
pytest = PASS_10_TESTS
plan_only_wrapper = PASS
```

## Status

```text
R3_2_FULL_WRAPPER_AGGREGATION_PATH_IMPLEMENTED_LOCAL_PLAN_VALIDATED_NO_SLURM
```

Next allowed action: review this implemented aggregation path together with the
already passing exact `852426` replay artifacts. Do not enable allowlist or
submit Slurm until that review passes.
