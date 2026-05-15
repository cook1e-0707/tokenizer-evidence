# R4 Pressure-Controller Wrong-Control Mapping Review

Timestamp UTC: `2026-05-15T05:12:00Z`

## Decision

Status:
`PASS_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_AND_FULL_WRAPPER_REVIEW_NO_SUBMIT`.

The wrong-key and wrong-payload controller mapping is now precommitted for the
teacher-forced pressure-controller scoring route, and the H200 wrapper full
scoring path is implemented behind an explicit runtime guard. This is still
artifact-only: no Slurm job was submitted, no model scoring was started, no
generation was started, and no training was started.

## Mapping Contract

The scorer always measures the committed verifier target mass for every row.
Wrong controls change only which token ids the controller boosts; they do not
change the verifier-side target being scored.

```text
controlled_protected:
  controller policy = committed
  controller target ids = committed target ids

wrong_payload_controlled:
  controller policy = complement
  controller target ids = committed other ids
  verifier/scorer target remains committed target ids

wrong_key_controlled:
  controller policy = coordinate_hash_v1
  hash salt = r4_wrong_key_controller_v1
  deterministic inputs = prompt_id | prompt_index | coordinate_id | target_bit
  verifier/scorer target remains committed target ids
```

The wrong-key mapping is row-local, deterministic, transcript-independent, and
does not inspect generated outputs. It may choose the committed side for some
rows by chance; this is recorded in row metadata as
`matches_committed_target_bit` and must not be tuned post hoc.

## Code Changes

Updated scorer:
`scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`

Updated route config:
`configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml`

Updated route validator:
`scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py`

Updated wrapper:
`scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch`

The scorer now supports:

```text
--controller-condition-set pressure_controls
```

which emits:

```text
base
task_only
controlled_protected
wrong_key_controlled
wrong_payload_controlled
```

Full wrapper mode no longer fails with the old wrong-control-review marker, but
it still requires:

```text
ALLOW_PRESSURE_CONTROLLER_SCORING=1
```

and it refuses overwrite of existing grid outputs.

## Validation

```text
uv run pytest \
  tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py \
  tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_controller_route.py \
  tests/natural_evidence_v2/test_r4_prefix_native_soft_logit_controller.py

20 passed, 2 skipped
```

The two skipped tests are torch-native scorer-helper checks; the local virtual
environment has no torch. Actual tokenizer/model scoring remains Chimera
Slurm-only.

```text
uv run python scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py \
  --output-dir results/natural_evidence_v2/status/r4_pressure_controller_wrong_control_mapping_validation_20260515_0505

PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE
```

```text
bash -n scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
PASS
```

Plan-only wrapper validation:

```text
results/natural_evidence_v2/status/r4_pressure_controller_wrapper_full_path_plan_smoke_20260515_0512/grid_00/wrapper_plan_only_summary.json
PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY
```

Full-mode guard without explicit runtime permission:

```text
ALLOW_PRESSURE_CONTROLLER_SCORING_REQUIRED_FOR_FULL_MODE
exit code 2
```

## Current Blocker

`BLOCK_R4_PRESSURE_CONTROLLER_REMOTE_PREFLIGHT_NEXT`

## Next Allowed Action

Remote sync and remote preflight only:

```text
remote wrapper plan-only validation
local/remote hash preflight
remote zero-enabled allowlist safety
active-job preflight
Hermes TG/email notification before any later submission route
```

No Slurm submission, model scoring, generation, training, Llama, same-family
null, sanitizer, FAR aggregation, payload-diversity work, or paper-facing
positive claim is unlocked by this mapping review.
