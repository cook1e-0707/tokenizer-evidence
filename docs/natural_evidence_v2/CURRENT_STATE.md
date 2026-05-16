# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T22:36:00Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_868016_CONTROLLER_GENERATION_868151_FAILED_REVIEWED_REPAIR_PIVOT_ARTIFACT_ONLY`

## Current Route

Job `868114` completed the after-868016 coordinate-pivot H200/pomplun
teacher-forced controller scoring array and passed the precommitted
teacher-forced gate:

```text
review: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_controller_score_868114_review/coordinate_pivot_controller_review.md
status: PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE
passing grids: 2/4
wrong-control failures: 0
best grid: 2
bonus_nats: 4.0
penalty_nats: 0.5
max_target_mass: 0.5
max_kl_budget: 0.5
controlled lift vs base: +0.160410
controlled lift vs task_only: +0.166550
controlled rank1: 0.943685
controlled median margin: 0.175455
wrong-key lift vs base: +0.077722
wrong-payload lift vs base: -0.007909
```

This is still teacher-forced scoring only. It permits a small generation
diagnostic route, but it is not a natural-output positive result.

Codex then completed the required artifact-only generation wrapper preparation:

```text
filtered codebook precommit:
  results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516/
filtered codebook oracle:
  results/natural_evidence_v2/status/r4_after_868016_coordinate_pivot_codebook_decoder_oracle_20260516/
oracle status:
  PASS_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_ORACLE_ARTIFACT_ONLY
wrong-key accepts: 0
wrong-payload accepts: 0
selected coordinates: 6,22,26,1,17,19,15,31,8,4,7,23
excluded coordinates: 3,10,20,24
```

Prepared and locally validated the controller-aware row-cylinder generation
route:

```text
route doc:
  docs/natural_evidence_v2/R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_20260516.md
route config:
  configs/natural_evidence_v2/r4_after_868016_controller_generation_route.yaml
generation wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch
generation script:
  scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py
validator:
  scripts/natural_evidence_v2/validate_r4_after_868016_controller_generation_route.py
allowlist entry:
  v2_r4_after_868016_controller_generation_h200
route validation:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
tests:
  18 passed, 3 skipped
```

The route is Qwen-only and diagnostic-scale:

```text
generation unit: prefix_native_row_cylinder
conditions: protected, raw, task_only
protected mechanism: base Qwen + first-step committed controller
raw: base Qwen, no controller
task_only: task-only adapter, no controller
blocks: 4
prompts per block: 64
row cylinders per shard: 768
primary decode: format_scrub=all
```

Remote preflight passed and a same-route H200/pomplun array was already active
when Codex attempted submission:

```text
remote preflight:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_remote_preflight_20260516/
remote preflight status:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_REMOTE_PREFLIGHT_NO_SUBMIT
active job seen before local sbatch:
  868151
job name:
  nat-ev-v2-r4cgen
partition/qos/account:
  pomplun / pomplun / cs_yinxin.wan
duplicate local submission:
  868158
duplicate action:
  cancelled immediately
post-submit local allowlist safety:
  PASS
post-submit remote allowlist safety:
  PASS
reconciliation:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_submission_reconciliation_20260516/
```

Job `868151` has now completed and was reviewed:

```text
review:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_review/
failure analysis:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_failure_analysis_20260516/
repair pivot:
  docs/natural_evidence_v2/R4_AFTER_868016_CONTROLLER_GENERATION_868151_REPAIR_PIVOT_20260516.md
status:
  FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE_ANALYZED
slurm:
  completed, exit code 0:0 for all array tasks
generated rows:
  9216
protected accepts, format_scrub=all:
  0/4
raw/task_only/wrong_key/wrong_payload accepts, format_scrub=all:
  0/4 each
matched phrase-surface events:
  0
forbidden public surfaces, format_scrub=all:
  26
duplicate response hashes:
  4422
target surface exact containment:
  0/3072 protected, 0/3072 raw, 0/3072 task_only
target first-word weak effect:
  protected 222/3072, raw 31/3072, task_only 33/3072
```

Primary diagnosis:

```text
The teacher-forced first-token controller can shift local target-token mass,
but the current free-generation row-cylinder route does not realize the
committed phrase-level natural surfaces. The decoder observes zero matched
surface events, so protected recovery is absent.
```

This is a clean method diagnostic failure, not a Slurm/runtime failure and not
a natural-output positive.

## Next Allowed Action

Artifact-only repair/pivot planning only. Do not rerun or scale the current
first-step controller generation route.

The next reviewed route must address:

```text
1. full phrase-surface realization, not only first-token mass;
2. duplicate-output collapse from deterministic row-cylinder generation;
3. forbidden public surface leakage under format_scrub=all;
4. decoder contract clarity: phrase-level, token-level, or a precommitted
   hybrid, with no post-hoc rescue of 868151.
```

Allowed now:

```text
artifact-only failure analysis
controller/decoder repair design
local toy tests for repair helpers
local plan-only validation
Hermes/Codex state synchronization
GitHub synchronization
```

Not allowed until a new reviewed route exists:

```text
another generation Slurm submission
larger generation route
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
```

## Still Gate-Controlled

These actions may proceed automatically only after their route-specific
preconditions pass and are recorded in this file:

```text
larger generation route
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
```
