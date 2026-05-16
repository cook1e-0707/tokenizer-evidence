# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T03:32:00Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_METRIC_EXACT_COVERAGE_SCALE_864761_PASSED_TF_GATE_WITH_CAVEAT`

## Current Route

Route R4 positive-selectivity controller-only teacher-forced scoring completed
and failed for H200/pomplun Slurm array jobs `863274` and `864117`. The
follow-up safety-bound controller route kept wrong controls clean but still did
not produce enough positive teacher-forced pressure. The after-864117
artifact-only pivot package selected metric-exact objective repair. A disabled
by default `logsumexp_softplus` surface-margin mode has now been patched into
the trainer, validated with toy-logit tests, wired into the H200 micro-overfit
wrapper in plan-only mode, synchronized to Chimera with matching hashes, and
submitted once as job `864332`. Job `864332` completed cleanly but failed the
teacher-forced surface-mass gate. A floor-dominant metric-exact micro-overfit
repair was then submitted once as job `864705`: task CE disabled, stronger
target-mass floor pressure, and no generation. Job `864705` completed cleanly
but still failed the teacher-forced surface-mass gate. A coverage-scale
floor-dominant route has now been recorded and locally validated: same
candidate-v3 rows, no generation, `MAX_TRAIN_ROWS=8192`, `MAX_STEPS=4096`,
`BATCH_SIZE=2`, and `TARGET_MASS_FLOOR=0.25`.

User standing authorization remains active: when a route's recorded
prerequisite gates pass, Codex and Hermes may continue without asking for
repeated approval on the same clear route. This authorization does not waive
precommit records, allowlist rules, Hermes TG/email notification, Slurm-only
execution for Chimera tokenizer/model work, H200/pomplun policy, or the
one-reviewed-submission rule.

Training, generation, H200 scoring, Llama, null/FAR, sanitizer, payload
diversity, and paper-facing claim work are conditionally authorized only after
their recorded prerequisite gates pass. They are not permanently forbidden, but
they are not unlocked by the current state.

## Current Controlling Blocker

`BLOCK_R4_METRIC_EXACT_864761_PASS_SMALL_DEV_GENERATION_ROUTE_REVIEW_NEXT`

Artifact-only pivot package:

```text
docs/natural_evidence_v2/R4_AFTER_864117_METRIC_EXACT_OBJECTIVE_PIVOT_20260516.md
configs/natural_evidence_v2/r4_after_864117_pivot_package.yaml
results/natural_evidence_v2/status/r4_after_864117_pivot_package_validation_20260516/
results/natural_evidence_v2/status/r4_after_864117_pivot_package_validation_20260516_allowlist_safety.json
```

Static validation:

```text
PASS_R4_AFTER_864117_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE
allowlist safety: PASS with zero enabled entries
selected next route: metric_exact_objective_repair
```

The scalar additive controller line is exhausted for the current candidate-v3
surface channel unless a new controller design is recorded first. The current
next action is artifact-only code review and route planning for metric-exact
objective repair. This does not unlock Slurm, model scoring, generation,
training, Llama, null/FAR, sanitizer, payload diversity, or paper-facing
claims.

Metric-exact objective patch review:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_OBJECTIVE_PATCH_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_metric_exact_objective_patch_review_20260516/
```

Validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_metric_exact_objective_helpers.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_stratum_weighting_controls.py -q
14 passed
uv run python -m py_compile scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py
PASS
```

Metric-exact micro-overfit route plan:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_MICRO_OVERFIT_ROUTE_PLAN_20260516.md
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_route_plan_20260516/
allowlist entry: v2_r4_candidate_v3_micro_overfit_h200
command pattern: sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Remote preflight:

```text
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_remote_preflight_20260516/
status: PASS_R4_METRIC_EXACT_MICRO_OVERFIT_REMOTE_PREFLIGHT_NO_SUBMIT
active jobs: none
hashes match: true
remote plan-only smoke: PASS
```

Submission:

```text
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_submission_20260516/
job_id: 864332
job_name: nat-ev-v2-r4mof
partition: pomplun
node_seen_after_submit: chimera21
allowlist entry: v2_r4_candidate_v3_micro_overfit_h200
command: sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
local post-submit allowlist safety: PASS with zero enabled entries
remote post-submit allowlist safety: PASS with zero enabled entries
```

Current next allowed action:

```text
Artifact-only failure analysis and a reviewed repair or pivot route. Do not
submit new Slurm, run generation, start Llama, run same-family null, run
sanitizer, aggregate FAR, make payload-diversity claims, or make paper-facing
positive claims until a new route records prerequisites and control-plane
checks.
```

Metric-exact micro-overfit review:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_MICRO_OVERFIT_864332_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864332/
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_864332_review/
status: FAIL_R4_METRIC_EXACT_MICRO_OVERFIT_864332_TEACHER_FORCED_GATE
job_id: 864332
slurm_state: COMPLETED
exit_code: 0:0
protected mean target mass: 0.0179803
protected lift vs base: +0.0131485
protected lift vs task-only: +0.0163079
protected rank1 rate: 0.980469
protected median margin: +0.0059255
```

Interpretation:

```text
The metric-exact objective improved rank ordering but did not create enough
absolute target-token mass. The target-mass floor remained nearly unsatisfied
at the end of training. This failed gate does not unlock generation or any
downstream claim route.
```

Floor-dominant repair route:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_FLOOR_DOMINANT_MICRO_OVERFIT_ROUTE_20260516.md
configs/natural_evidence_v2/r4_metric_exact_floor_dominant_micro_overfit_route.yaml
scripts/natural_evidence_v2/validate_r4_metric_exact_floor_dominant_route.py
results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_route_validation_20260516/
allowlist entry: v2_r4_candidate_v3_floor_dominant_micro_overfit_h200
command: sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.20,TARGET_MASS_FLOOR_LAMBDA=50.0,TARGET_MASS_CEILING=0.45,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_STEPS=128,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
static validation: PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_ROUTE_STATIC_VALIDATION_NO_COMPUTE
local wrapper plan-only smoke: PASS
local allowlist safety: PASS with zero enabled entries
```

Current next allowed action:

```text
Artifact-only review and route decision after job `864705`. Do not submit new
Slurm, run generation, start Llama, run same-family null, run sanitizer,
aggregate FAR, make payload-diversity claims, or make paper-facing positive
claims until a new route records prerequisites and control-plane checks.
```

Remote preflight:

```text
results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_remote_preflight_20260516/
status: PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_REMOTE_PREFLIGHT_NO_SUBMIT
hashes match: true
remote route validation: PASS
remote wrapper plan-only smoke: PASS
local/remote allowlist safety: PASS with zero enabled entries
active jobs: none
```

Submission:

```text
results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_submission_20260516/
job_id: 864705
job_name: nat-ev-v2-r4mof
partition: pomplun
node_seen_after_submit: chimera21
allowlist entry: v2_r4_candidate_v3_floor_dominant_micro_overfit_h200
local post-submit allowlist safety: PASS with zero enabled entries
remote post-submit allowlist safety: PASS with zero enabled entries
```

Floor-dominant micro-overfit review:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_FLOOR_DOMINANT_864705_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864705/
results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_864705_review/
status: FAIL_R4_METRIC_EXACT_FLOOR_DOMINANT_864705_TEACHER_FORCED_GATE
job_id: 864705
slurm_state: COMPLETED
exit_code: 0:0
protected mean target mass: 0.0847697
protected lift vs base: +0.0799378
protected lift vs task-only: +0.0830972
protected rank1 rate: 1.000000
protected median margin: +0.0772580
```

Interpretation:

```text
Floor-dominant pressure was directionally effective but not sufficient. The
protected mean target mass improved from 0.0179803 in job 864332 to 0.0847697
in job 864705, while rank1 reached 1.0 and null/task-only behavior remained
clean. The run still missed the +0.15 lift-vs-base and +0.10 lift-vs-task-only
teacher-forced gates. The next route should be artifact-only coverage-scale or
stronger-floor planning, not generation.
```

Coverage-scale floor-dominant route:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_COVERAGE_SCALE_MICRO_OVERFIT_ROUTE_20260516.md
configs/natural_evidence_v2/r4_metric_exact_coverage_scale_micro_overfit_route.yaml
scripts/natural_evidence_v2/validate_r4_metric_exact_coverage_scale_route.py
results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_route_validation_20260516/
allowlist entry: v2_r4_candidate_v3_coverage_scale_micro_overfit_h200
command: sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.25,TARGET_MASS_FLOOR_LAMBDA=75.0,TARGET_MASS_CEILING=0.50,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_TRAIN_ROWS=8192,MAX_SCORE_ROWS=8192,MAX_STEPS=4096,BATCH_SIZE=2,GRADIENT_ACCUMULATION_STEPS=8,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
static validation: PASS_R4_METRIC_EXACT_COVERAGE_SCALE_ROUTE_STATIC_VALIDATION_NO_COMPUTE
local wrapper plan-only smoke: PASS
local allowlist safety: PASS with zero enabled entries
```

Current next allowed action:

```text
Artifact-only route decision for a small Qwen dev generation diagnostic using
the reviewed job `864761` adapter and same candidate-v3 surface contract. No
generation may start until that route records its wrapper contract, allowlist
entry, Hermes notification, remote hash preflight, and post-submit allowlist
shutdown policy.
```

Remote preflight:

```text
results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_remote_preflight_20260516/
status: PASS_R4_METRIC_EXACT_COVERAGE_SCALE_REMOTE_PREFLIGHT_NO_SUBMIT
hashes match: true
remote route validation: PASS
remote wrapper plan-only smoke: PASS
local/remote allowlist safety: PASS with zero enabled entries
active jobs: none
```

Submission:

```text
results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_submission_20260516/
job_id: 864761
job_name: nat-ev-v2-r4mof
partition: pomplun
allowlist entry: v2_r4_candidate_v3_coverage_scale_micro_overfit_h200
local post-submit allowlist safety: PASS with zero enabled entries
remote post-submit allowlist safety: PASS with zero enabled entries
```

Coverage-scale micro-overfit review:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_COVERAGE_SCALE_864761_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864761/
results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_864761_review/
status: PASS_R4_METRIC_EXACT_COVERAGE_SCALE_864761_TEACHER_FORCED_GATE_WITH_TRAIN_SPLIT_CAVEAT
job_id: 864761
slurm_state: COMPLETED
exit_code: 0:0
protected mean target mass: 0.156421
protected lift vs base: +0.151589
protected lift vs task-only: +0.154749
protected rank1 rate: 1.000000
protected median margin: +0.154256
```

Caveat:

```text
The route intended MAX_TRAIN_ROWS=8192, but the train_rows artifact contains
512 rows. The job therefore tested repeated-cycled 512-row training with
stronger floor pressure and scored on 8192 rows. Preserve this caveat in any
downstream route; do not describe 864761 as 8192 unique train-row coverage.
```

## Historical Controller Failure Chain

Job `859672` completed all `72/72` H200/pomplun array tasks with exit code
`0:0`; this was not a Slurm or wrapper failure. The score review failed keyed
selectivity:

- protected basic teacher-forced gate passes: `72/72`
- overall selective gate passes: `0/72`
- wrong-key controlled basic gate passes: `72/72`
- wrong-payload controlled basic gate passes: `72/72`

Reviewed artifacts:

```text
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_SCORE_859672_REVIEW_20260515.md
results/natural_evidence_v2/status/r4_pressure_controller_score_859672_review/
results/natural_evidence_v2/status/r4_pressure_controller_wrong_control_diagnosis_859672_20260515/
```

The current diagnosis is that wrong-control arms still load the protected
adapter while the scorer measures the committed target ids. Remote row probes
show wrong-payload uses complement controller ids and wrong-key uses the
deterministic coordinate-hash policy, with no controller target/other overlap;
nevertheless committed target mass remains high under wrong controls. This is
a selectivity-control semantics failure, not a positive channel result.

The scorer has been patched with a new condition set:

```text
--controller-condition-set controller_only_controls
```

It emits `base`, `task_only`, `controlled_base`,
`wrong_key_controlled_base`, and `wrong_payload_controlled_base`. The controller
arms use the base model and do not load the protected adapter, so the next
route can test provider-side keyed controller selectivity without inheriting
the adapter bias that invalidated job `859672`.

Repair artifacts:

```text
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_WRONG_CONTROL_REPAIR_PLAN_20260515.md
results/natural_evidence_v2/status/r4_pressure_controller_wrong_control_repair_plan_20260515/
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_PRESSURE_ROUTE_PLAN_20260515.md
results/natural_evidence_v2/status/r4_controller_only_remote_preflight_20260515/
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SINGLE_SUBMISSION_ROUTE_20260515.md
```

Focused local validation passed:

```text
uv run pytest tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py -q
uv run python -m py_compile scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py
```

Job `863274` was submitted with the reviewed controller-only command. The
allowlist entry was disabled immediately after `sbatch` returned, and both
local and remote post-submit allowlist safety checks passed with zero enabled
entries.

Submission record:

```text
results/natural_evidence_v2/status/r4_controller_only_submission_20260515/
```

Remote output:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_pressure_controller_score_863274
```

Review result:

```text
Slurm tasks completed with exit code 0:0: 72/72
Summary artifacts synced: 72/72
Controlled-base basic gate passes: 0/72
Overall selective gate passes: 0/72
Wrong-key basic gate passes: 0/72
Wrong-payload basic gate passes: 0/72
Best controlled lift vs base: +0.0154036601
Best controlled rank1: 0.498046875
Best controlled median margin: -0.0001098111
```

Reviewed artifacts:

```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SCORE_863274_REVIEW_20260515.md
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_863274_REPAIR_ROUTE_PLAN_20260515.md
results/natural_evidence_v2/status/r4_controller_only_score_863274_review/
results/natural_evidence_v2/status/r4_controller_only_failure_diagnosis_863274_20260515/
results/natural_evidence_v2/status/r4_controller_only_863274_repair_route_plan_20260515/
```

The controller-only repair fixed the previous wrong-control contamination:
wrong-key and wrong-payload controlled-base arms no longer pass the basic gate.
However, the positive controlled-base pressure is far below the R4
teacher-forced gate. The route does not unlock generation.

## Follow-Up Route Package

The next route package is recorded as artifact-only:

```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SAFETY_BOUND_PRESSURE_ROUTE_20260515.md
configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
results/natural_evidence_v2/status/r4_controller_only_safety_bound_route_package_20260515/
```

It is not a rerun of `863274`. The wrapper now derives the controller grid from
the reviewed route config via:

```text
scripts/natural_evidence_v2/emit_r4_pressure_controller_grid.py
```

Local validation passed:

```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_controller_route.py tests/natural_evidence_v2/test_r4_pressure_controller_grid_emit.py -q
uv run python -m py_compile scripts/natural_evidence_v2/emit_r4_pressure_controller_grid.py scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py
```

Wrapper plan-only smoke passed for `grid_23` with:

```text
route_config: configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
grid_size: 24
bonus_nats: 2.0
penalty_nats: 0.5
max_target_mass: 0.5
max_kl_budget: 0.2
model_scoring_started: false
generation_started: false
training_started: false
```

Allowlist entry exists and is currently disabled:

```text
v2_r4_controller_only_safety_bound_pressure_score_h200
```

Submission record:

```text
results/natural_evidence_v2/status/r4_controller_only_safety_bound_submission_20260516/
```

Submitted job:

```text
job_id: 864117
array: 0-23%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gpu: h200
time_limit: 30-00:00:00
route_config: configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
condition_set: controller_only_controls
```

Post-submit allowlist safety passed locally and remotely with zero enabled
entries.

Review result:

```text
Slurm tasks completed with exit code 0:0: 24/24
Summary artifacts synced: 24/24
Controlled-base basic gate passes: 0/24
Overall selective gate passes: 0/24
Wrong-key basic gate passes: 0/24
Wrong-payload basic gate passes: 0/24
Best controlled lift vs base: +0.0269583198
Best controlled rank1: 0.6015625
Best controlled median margin: +0.0033881384
```

Reviewed artifacts:

```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SAFETY_BOUND_SCORE_864117_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/
results/natural_evidence_v2/status/r4_controller_only_safety_bound_failure_diagnosis_864117_20260516/
```

The safety-bound controller route kept wrong controls clean and improved
positive pressure relative to `863274`, but it still failed the R4
teacher-forced selective gate by a wide margin. It does not unlock generation.

## Next Allowed Action

Single H200 submission for the metric-exact micro-overfit route is ready after
Hermes TG/email notification and exactly-one allowlist enablement. Submit only
`v2_r4_candidate_v3_micro_overfit_h200`, then immediately disable the allowlist
entry and record post-submit safety. Do not run generation, Llama, null/FAR,
sanitizer, payload-diversity, or paper-facing claim work from this state.

## Not Unlocked By Current State

The following actions remain gated and may run only after their own recorded
prerequisites pass:

```text
new H200 scoring resubmission
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```

No allowlist entry should be enabled from this state.
