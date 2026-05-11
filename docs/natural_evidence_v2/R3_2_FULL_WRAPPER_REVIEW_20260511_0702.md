# R3.2 Full Same-Contract Wrapper Review: 2026-05-11 07:02Z

## Decision

PASS: the implemented R3.2 same-contract `a55e` wrapper aggregation path is
reviewed for the approved locked-scale route.

This review did not enable the allowlist, submit Slurm, start generation, start
training, run Llama, run same-family null, run sanitizer benchmarks, aggregate
FAR, or make paper-facing positive claims.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_full_wrapper_review_20260511_0702.json
```

## Reviewed Inputs

```text
docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION.md
docs/natural_evidence_v2/R3_2_LOCKED_SCALE_PROTOCOL.md
docs/natural_evidence_v2/R3_2_FULL_WRAPPER_AGGREGATION_PATH_20260511_0645.md
configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml
scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py
results/natural_evidence_v2/status/r3_2_same_contract_852426_replay_20260511_0630/r3_2_852426_replay_summary.json
results/natural_evidence_v2/status/r3_2_full_wrapper_aggregation_path_20260511_0645.json
```

## Review Findings

- The config and precommit path use `contract_id=a55e`,
  `payload_diversity_tested=false`, and no `payload_ids`.
- The wrapper uses canonical `shard_00..shard_11` replicate groups and
  canonical aggregate block IDs `C_A55E_shard_XX_block_YY`.
- The wrapper refuses existing shard outputs and aggregate outputs instead of
  overwriting artifacts.
- The aggregate gate evaluates the R3.2 protocol targets: `80/96` protected
  accepts at budget `64`, zero null accepts, minimum support `16`, minimum
  majority margin `3`, forbidden public surface count `0`, and all 12
  replicate groups complete.
- The exact same-contract replay of job `852426` is recorded as passing and is
  explicitly marked as an 8-block replay, not the 96-block R3.2 full gate.
- The R3.2 allowlist entry remains disabled at review time.

## Validation

Local validation commands run for this review:

```text
python3 -m py_compile scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py scripts/natural_evidence_v2/replay_r3_2_same_contract_from_852426.py scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py
bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

Validation result:

```text
py_compile = PASS
bash_n = PASS
pytest = PASS_10_TESTS
```

## Status

```text
PASS_R3_2_FULL_SAME_CONTRACT_WRAPPER_REVIEW_NO_SLURM
```

Next allowed action: enable the existing R3.2 allowlist entry for exactly one
reviewed Slurm command only after the next required TG/email notification path
is satisfied, then submit exactly one allowlisted Chimera Slurm job.
