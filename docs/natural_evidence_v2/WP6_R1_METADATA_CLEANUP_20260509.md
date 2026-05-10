# WP6-R1 Replay Metadata Cleanup: 2026-05-09

## Scope

This cleanup removes stale replay-era metadata from the WP6-R1
repeated-coordinate majority decoder outputs before any scaled rerun.

No Slurm job was submitted. No training, Llama, same-family null, sanitizer,
FAR aggregation, or paper-facing positive claim was started.

## Problem

The original majority summary for job `852094` inherited this field from the
artifact-only replay over job `852086`:

```text
post_hoc_not_precommitted_for_852086 = true
```

For `852094`, this label is stale because the replacement wrapper wrote the
decoder contract before generation.

## Code Change

Updated:

```text
scripts/natural_evidence_v2/replay_wp6_coordinate_majority_decoder.py
scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_e2e_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

The replay script now supports:

```text
--precommitted-transcript
```

When set, the summary uses:

```text
artifact_role = wp6_r1_coordinate_majority_e2e_replacement_summary
precommitted_transcript = true
post_hoc_artifact_replay = false
transcript_provenance = precommitted_replacement_run
replacement_run_gate_pass = true/false
```

The stale `post_hoc_not_precommitted_for_852086` field is no longer emitted.

The WP6-R1 Slurm wrapper passes `--precommitted-transcript` for both the
contract-only precommit step and the post-generation majority decode step.

## Cleaned 852094 Artifact

Cleaned local replay output:

```text
results/natural_evidence_v2/status/wp6_r1_coordinate_majority_e2e_eval_852094_metadata_cleaned_20260509_1839/
```

The cleaned summary reports:

```text
artifact_role = wp6_r1_coordinate_majority_e2e_replacement_summary
precommitted_transcript = true
post_hoc_artifact_replay = false
replacement_run_gate_pass = true
replay_gate_status = PASS_WP6_R1_COORDINATE_MAJORITY_E2E_REPLACEMENT_RUN
transcript_provenance = precommitted_replacement_run
stale_key_present = false
```

Budget-64 result is unchanged:

| Condition | Accepted | Decoded hex | Min support | Min margin |
|---|---:|---|---:|---:|
| protected | true | `a55e` | 33 | 3 |
| raw | false | `7400` | 2 | 2 |
| task-only | false | `5020` | 1 | 1 |
| wrong-key | false | `a55e` | 33 | 3 |
| wrong-payload | false | `a55e` | 33 | 3 |

## Validation

Tests:

```text
.venv/bin/python -m pytest \
  tests/test_natural_evidence_v2_wp6_coordinate_majority.py \
  tests/test_natural_evidence_v2_wp6_e2e_decode.py

4 passed
```

Syntax checks:

```text
.venv/bin/python -m py_compile scripts/natural_evidence_v2/replay_wp6_coordinate_majority_decoder.py
bash -n scripts/natural_evidence_v2/slurm/wp6_r1_coordinate_majority_e2e_eval.sbatch
```

Wrapper plan-only validation:

```text
results/natural_evidence_v2/status/wp6_r1_wrapper_metadata_cleanup_validate_20260509_1840/
```

The precommit contract now records:

```text
transcript_precommitted_before_generation = true
transcript_provenance = precommitted_replacement_run
```

## Gate Status

```text
PASS_WP6_R1_METADATA_CLEANUP_READY_FOR_SCALE_DECISION
```

## Still Forbidden

- no automatic scaled rerun;
- no new training;
- no Llama;
- no same-family null;
- no sanitizer;
- no FAR aggregation;
- no paper-facing positive claim.

## Next Allowed Action

Prepare a WP6-R1 scale/reproducibility decision package. It should define the
exact scope, prompt count, payload cells, query budgets, null controls,
allowlist entry, and Slurm wrapper review before any scaled rerun is submitted.
