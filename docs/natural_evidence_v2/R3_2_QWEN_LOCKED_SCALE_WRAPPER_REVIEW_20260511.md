# R3.2 Qwen Locked-Scale Wrapper Review: 2026-05-11

## Decision

The R3.2 Qwen locked-scale plan-only wrapper is implemented and locally
validated. The disabled allowlist entry already exists and remains disabled.

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_qwen_locked_scale_wrapper_review_20260511_0318.json
```

## Implemented Paths

```text
scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

The wrapper reconstructs the selected prompt manifest from the reviewed
2,560-row prompt source and refuses to continue unless the manifest hash
matches the precommitted value:

```text
selected_prompt_manifest_sha256 = 4d49ae100b272f184a8b2563e5b64f768e6db01425a2384f1457a4eb10eedb67
```

The wrapper writes only precommit artifacts during local plan validation:

```text
precommit/r3_2_selected_prompt_manifest.json
precommit/r3_2_qwen_locked_scale_contract.json
```

The non-plan generation path is intentionally disabled in the current state.
A later notified submission tick must explicitly authorize exactly one R3.2
Slurm job before any generation path can run.

## Locked Scope

```text
package_id = qwen_v2_r3_2_locked_scale_package_v1
payloads = P00, P01, P02, P03
seeds = 17, 23, 29
cells = 12
blocks_per_cell = 8
block_size = 64
protected_blocks_total = 96
arms = protected, raw, task_only, wrong_key, wrong_payload
query_budgets = 16, 32, 64
primary_budget = 64
support_threshold = 16
majority_margin_threshold = 3
protected pass threshold = >=80/96
null-arm pass threshold = 0/96 for each null arm
forbidden_public_surface_count = 0
```

## Local Validation

Commands run locally only:

```text
python3 -m py_compile scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
VALIDATE_PLAN_ONLY=1 REPO_HOME=/Users/guanjie/Documents/tokenizer_alignment PYTHON=python3 RUN_ROOT=/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status OUTPUT_DIR=/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_wrapper_plan_validation_20260511_0318 bash scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

Validation output:

```text
results/natural_evidence_v2/status/r3_2_wrapper_plan_validation_20260511_0318/precommit/r3_2_selected_prompt_manifest.json
results/natural_evidence_v2/status/r3_2_wrapper_plan_validation_20260511_0318/precommit/r3_2_qwen_locked_scale_contract.json
```

The focused pytest run passed `8` tests.

## Allowlist Review

The allowlist entry is present and disabled:

```text
name = v2_r3_2_qwen_locked_scale_eval
command = sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
enabled = false
enable_condition = disabled_until_r3_2_wrapper_precommit_and_gate_review_recorded
```

## Status

```text
R3_2_QWEN_LOCKED_SCALE_WRAPPER_REVIEWED_PLAN_ONLY_NO_SLURM
```

Next allowed action: stop until a later explicit, notified R3.2 submission tick
authorizes exactly one reviewed Slurm job. Llama, same-family nulls,
sanitizer, FAR aggregation, and paper-facing positive claims remain disabled.
