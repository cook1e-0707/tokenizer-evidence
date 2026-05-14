# R4 Candidate v3 Capped Micro-Overfit H200 Route

Timestamp UTC: 2026-05-14T04:02:35Z

## Decision

Prepare exactly one future H200/pomplun Slurm submission for a capped R4
candidate-v3 metric-exact protected micro-overfit run.

This route is a teacher-forced training/scoring diagnostic only. It is not
generation, not Qwen E2E, not Llama, not same-family null, not sanitizer, not
FAR aggregation, not payload diversity, and not a paper-facing positive claim.

## Motivation

Micro-overfit job `857458` passed the main teacher-forced surface-mass gate:

- protected lift vs base: `0.39194896617371455`;
- protected lift vs task-only: `0.39510837536891064`;
- protected rank1: `1.0`;
- protected median margin: `0.355754891585093`.

The same review found surface concentration risk:

- max surface mean target mass: `0.643060527741909`;
- diagnostic concentration cap: `0.50`;
- cap status: `FAIL`.

The capped route keeps the successful metric-exact objective direction while
adding a disabled-by-default target-mass ceiling penalty to discourage collapse
onto a single visible surface family.

## Wrapper Contract

Wrapper:

`scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

The wrapper now records and passes these objective controls:

- `TARGET_MASS_FLOOR=0.20`;
- `TARGET_MASS_FLOOR_LAMBDA=5.0`;
- `TARGET_MASS_CEILING=0.45`;
- `TARGET_MASS_CEILING_LAMBDA=5.0`;
- `STRATUM_WEIGHTING_MODE=r4_candidate_v3_failed_surface`;
- `STRATUM_WEIGHT_MAX=3.0`.

The target-mass ceiling is below the diagnostic surface concentration cap
`0.50`, leaving room for the held-out score review to pass the cap without
requiring post-hoc threshold changes.

## Inputs

Split artifacts:

`results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/`

- train rows: `512`;
- heldout rows: `512`;
- score rows: `8192`;
- train/heldout overlap: `0`;
- strata covered: `32`.

Task-only adapter:

`results/natural_evidence_v2/status/wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter`

## H200 Requirements

- partition: `pomplun`;
- account: `cs_yinxin.wan`;
- QoS: `pomplun`;
- GRES: `gpu:h200:1`;
- time limit: `30-00:00:00`.

## Validation Completed Locally

No model/tokenizer loading, CUDA initialization, training, scoring, generation,
remote CPU/GPU work, allowlist enablement, or Slurm submission occurred during
local validation.

Passed:

- `bash -n scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`;
- `VALIDATE_PLAN_ONLY=1 ... bash scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`;
- `uv run python -m py_compile scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py scripts/natural_evidence_v2/build_r4_candidate_v3_micro_overfit_split.py scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`;
- `uv run pytest tests/test_natural_evidence_v2_wp5_objective_repair.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py tests/natural_evidence_v2/test_r4_candidate_v3_micro_overfit_split.py`;
- allowlist safety with zero enabled entries.

## Future Submission Gate

Before submission, the route still requires:

1. sync the reviewed wrapper, route record, status artifacts, and state files to
   Chimera;
2. verify local/remote file hashes match;
3. run remote wrapper plan-only mode;
4. run remote zero-enabled allowlist safety;
5. send Hermes TG/email pre-submit notification;
6. enable exactly one allowlist entry:
   `v2_r4_candidate_v3_micro_overfit_h200`;
7. submit exactly one H200 Slurm job;
8. immediately disable the allowlist entry and recheck local/remote allowlist
   safety.

## Pass Gate For Terminal Review

The capped run must satisfy:

- protected lift vs base `>= +0.15`;
- protected lift vs task-only `>= +0.10`;
- protected rank1 `>= 0.75`;
- protected median margin `> 0`;
- task-only lift vs base not positive;
- max surface mean target mass `<= 0.50`;
- no target/other token-id overlap;
- no scorer boundary failures.

If the main teacher-forced gate passes but the concentration cap still fails,
generation remains blocked and the result remains a teacher-forced diagnostic.

## Locked Actions

This route does not unlock generation, Qwen E2E rerun, Llama, same-family null,
sanitizer benchmark, FAR aggregation, payload diversity, or paper-facing
positive claims.
