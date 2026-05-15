# natural_evidence_v2 Current State

Last synchronized: 2026-05-15T23:54:00Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_CONTROLLER_ONLY_SCORE_863274_REVIEWED_FAIL_NO_GENERATION`

## Current Route

Route R4 positive-selectivity controller-only teacher-forced scoring has been
completed and reviewed for H200/pomplun Slurm array job `863274`.

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

`BLOCK_R4_CONTROLLER_ONLY_SCORE_FAIL_NO_SELECTIVE_GATE_ARTIFACT_ONLY_REPAIR_NEXT`

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

## Next Allowed Action

Artifact-only design/implementation of a new controller repair package. The
new package must not be a rerun of `863274`; it must record a stronger or more
targeted pressure mechanism, safety bounds, validator coverage, and wrapper
plan-only checks before any new H200 scoring submission. Do not submit another
controller-only scoring job, generation job, training job, Llama job, null/FAR
job, sanitizer job, payload-diversity route, or paper-facing claim before a new
reviewed route records its prerequisites and control-plane checks.

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
