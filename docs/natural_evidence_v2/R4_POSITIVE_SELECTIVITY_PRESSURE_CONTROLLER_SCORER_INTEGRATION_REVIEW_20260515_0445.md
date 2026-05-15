# R4 Positive Selectivity Pressure-Controller Scorer Integration Review

Timestamp UTC: `2026-05-15T04:45:00Z`

## Decision

Status:
`PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_SCORER_INTEGRATION_NO_COMPUTE`.

The teacher-forced surface-mass scorer now has a reviewed, disabled-by-default
soft-controller integration path. This is still artifact-only: no Slurm job was
submitted, no model scoring was started, no generation was started, and no
training was started.

## Patch Scope

Updated scorer:
`scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`

Added focused tests:
`tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py`

The scorer now accepts:

```text
--controller-mode disabled|additive
--controller-bonus-nats
--controller-penalty-nats
--controller-max-target-mass
--controller-max-kl-budget
```

The default remains `--controller-mode disabled`, so historical scorer behavior
is unchanged unless a future reviewed route explicitly enables the controller.
When enabled in this scorer, the controller is applied only to protected
conditions (`protected` and `protected_gain_*`). Base and task-only conditions
remain unmodified. Wrong-key and wrong-payload controlled conditions still
require a later wrapper/review layer with precommitted alternate token-id
mappings.

## Safety Properties

The integration:

```text
does not load models in dry-run
does not generate text
does not train
does not submit Slurm
does not enable allowlist entries
rejects target/other token-id overlap
caps controller pressure by max target mass and KL budget when requested
records controller_config and controller metadata per scored row
```

Dry-run with controller flags was validated against the 8192 candidate-v3 rows.
Because no protected adapter path is required for dry-run, a second dry-run used
placeholder protected/task-only adapter paths to verify that the condition plan
remains `base`, `protected`, `task_only` while model scoring stays false.

## Validation

```text
uv run pytest tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py \
  tests/natural_evidence_v2/test_r4_adapter_gain_sweep_plan.py \
  tests/natural_evidence_v2/test_r4_prefix_native_soft_logit_controller.py

17 passed, 2 skipped
```

The two skipped tests are torch-native scorer-helper checks. Local virtual
environment does not have `torch`; the future actual model/tokenizer scoring
path remains Chimera Slurm-only.

```text
uv run python -m py_compile \
  scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py \
  scripts/natural_evidence_v2/r4_prefix_native_soft_logit_controller.py

PASS
```

Dry-run summaries:

```text
results/natural_evidence_v2/status/r4_pressure_controller_scorer_integration_dryrun_20260515_0515/
results/natural_evidence_v2/status/r4_pressure_controller_scorer_integration_dryrun_with_protected_20260515_0515/
```

Both dry-runs record:

```text
model_generation_started=false
model_scoring_started=false
training_started=false
llama_started=false
paper_claim_allowed=false
```

## Current Blocker

`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_H200_WRAPPER_PLAN_ONLY_NEXT`

## Next Allowed Action

Artifact-only H200 teacher-forced pressure-controller scoring wrapper
implementation and plan-only review. The wrapper must remain scoring-only,
Qwen-only, same-contract `a55e`, H200/pomplun, and zero-generation. Slurm
submission remains gated by wrapper plan-only validation, local/remote hash
preflight, zero-enabled allowlist safety, Hermes notification, exactly-one
allowlist enablement, and immediate allowlist disablement after `sbatch`.

No model scoring, generation, training, Llama, same-family null, sanitizer, FAR
aggregation, payload diversity, or paper-facing positive claim is unlocked by
this integration review.
