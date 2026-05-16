# natural_evidence_v2 Current State

Last synchronized: 2026-05-16T20:03:02Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_867849_FAILED_NO_GENERATION`

## Current Route

After job `867621` failed the protected positive generation gate, Codex
recorded an artifact-only repair/pivot route to test whether the
coordinate-unique reliability surfaces are valid under actual Qwen tokenizer
boundaries before any further scoring or generation.

New controlling artifacts:

```text
route doc: docs/natural_evidence_v2/R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_20260516.md
rows: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/
static preflight: results/natural_evidence_v2/status/r4_after_867621_reliability_static_boundary_preflight_20260516/
route config: configs/natural_evidence_v2/r4_after_867621_reliability_tokenizer_preflight_route.yaml
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch
validator: scripts/natural_evidence_v2/validate_r4_after_867621_reliability_tokenizer_route.py
validation: results/natural_evidence_v2/status/r4_after_867621_reliability_tokenizer_route_validation_20260516/
allowlist safety: results/natural_evidence_v2/status/r4_after_867621_reliability_tokenizer_route_allowlist_safety_20260516.json
allowlist entry: v2_r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200
```

Route validation status:

```text
PASS_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT
```

Prepared row/static-preflight status:

```text
rows: 4096
selected prompts: 256
selected coordinates: 16
current two-way scorer compatible: true
static boundary preflight: PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING
static checked rows: 4096
static failed rows: 0
global allowlist safety: PASS, zero enabled entries
```

No Slurm job was submitted by this route package. The allowlist entry is
present but disabled. The next compute action, if submitted after fresh
local/remote hash preflight and Hermes notification, is tokenizer-only actual
Qwen boundary validation on H200/pomplun. It must not start model forward,
teacher-forced scoring, generation, or training.

Codex then submitted the tokenizer-only H200/pomplun preflight as a single
reviewed job and reviewed the completed output:

```text
submission record: results/natural_evidence_v2/status/r4_after_867621_reliability_tokenizer_submission_20260516/
job_id: 867828
job_name: nat-ev-v2-r4relTok
state: COMPLETED
exit code: 0:0
elapsed: 00:00:14
review: results/natural_evidence_v2/status/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_867828/review.md
review status: PASS_R4_AFTER_867621_RELIABILITY_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_867828
checked rows: 4096
failed rows: 0
empty target id rows: 0
empty other id rows: 0
target/other overlap rows: 0
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
```

The actual-Qwen tokenizer boundary pass unlocked only the next route-planning
step: teacher-forced surface-mass scoring for the same 4096 rows. Codex
recorded and validated that route artifact-only:

```text
route doc: docs/natural_evidence_v2/R4_AFTER_867621_RELIABILITY_SURFACE_MASS_SCORE_ROUTE_20260516.md
route config: configs/natural_evidence_v2/r4_after_867621_reliability_surface_mass_score_route.yaml
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_surface_mass_score_h200.sbatch
validator: scripts/natural_evidence_v2/validate_r4_after_867621_reliability_surface_mass_route.py
validation: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_route_validation_20260516/
wrapper plan-only smoke: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_wrapper_plan_smoke_20260516/
allowlist entry: v2_r4_after_867621_reliability_surface_mass_score_h200
route validation: PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only status: DRY_RUN_VALIDATED_INPUTS
global allowlist safety: PASS, zero enabled entries
```

Codex then submitted the surface-mass scoring route as a single reviewed
H200/pomplun job:

```text
submission record: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_submission_20260516/
job_id: 867849
job_name: nat-ev-v2-r4relTFM
state: COMPLETED
exit code: 0:0
elapsed: 00:02:21
score output: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_score_867849/
failure analysis: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_failure_analysis_867849_20260516/
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
```

The scoring job completed cleanly but failed the teacher-forced surface-mass
gate:

```text
teacher_forced_surface_gate_status: FAIL
protected lift vs base: +0.006302
protected lift vs task_only: +0.011221
protected rank1 rate: 0.482666
protected median target margin: -0.000099876
task_only lift vs base: -0.004919
```

Interpretation: the actual Qwen tokenizer boundary is valid and task-only does
not appear to be leaking the target signal, but the protected adapter pressure
on the coordinate-unique reliability surfaces is far too weak. Generation is
not unlocked.

## Historical Route Context

The R4 metric-exact teacher-forced route reached a positive teacher-forced gate
in job `864761`, but the follow-up dev generation job `864832` failed:

```text
protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw/task-only/wrong-key/wrong-payload accepts: 0
duplicate response text hashes: 358
max protected-vs-raw shallow feature AUC: 1.0
```

The reviewed failure mode is a transfer gap, not a Slurm/null-control failure:
the adapter pressured candidate-v3 prefix-native phrases such as `Create a
plan` and `Prepare a`, but free generation did not produce enough of the
precommitted cover-natural ECC surface bank to decode. The after-864832
artifact-only repair package recorded this diagnosis and did not unlock compute.

The cover-bank-aligned route was validated artifact-only:

```text
selected route: cover_bank_aligned_metric_exact_objective_repair
target surfaces source: precommitted_cover_bank_only
surface bank: results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/surface_bank.json
decoder primary scrub mode: all
surface entries: 128
coordinates: 32
```

Validation artifacts:

```text
docs/natural_evidence_v2/R4_AFTER_864832_COVER_BANK_ALIGNED_OBJECTIVE_ROUTE_20260516.md
configs/natural_evidence_v2/r4_after_864832_cover_bank_aligned_objective_route.yaml
scripts/natural_evidence_v2/validate_r4_after_864832_cover_bank_aligned_route.py
results/natural_evidence_v2/status/r4_after_864832_cover_bank_aligned_route_validation_20260516/
```

Validation status:

```text
PASS_R4_AFTER_864832_COVER_BANK_ALIGNED_ROUTE_VALIDATION_NO_COMPUTE
```

Codex then built an artifact-only cover-bank-aligned row artifact from the
precommitted bank and the `a55e` codebook:

```text
scripts/natural_evidence_v2/build_r4_after_864832_cover_bank_aligned_rows.py
results/natural_evidence_v2/status/r4_after_864832_cover_bank_aligned_rows_20260516/
docs/natural_evidence_v2/R4_AFTER_864832_COVER_BANK_ALIGNED_ROW_BUILDER_REVIEW_20260516.md
```

Row-builder status:

```text
PASS_TARGET_ONLY_ROWS_BUILT__BLOCK_CURRENT_TWO_WAY_SCORER_UNTIL_COMPLEMENT_OR_TARGET_ONLY_SCORER
rows built: 4608
selected prompts: 256
coordinates: 32
surface entries: 128
coordinates missing same-coordinate opposite bucket: 18/32
coordinates whose present bank polarity does not match protected codeword bit: 14/32
current two-way scorer compatible: false
```

That blocker was resolved artifact-only by freezing a new two-sided,
codeword-aligned cover-natural bank and rebuilding rows against it:

```text
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_COVER_BANK_PIVOT_20260516.md
scripts/natural_evidence_v2/build_r4_after_864832_two_sided_cover_bank.py
results/natural_evidence_v2/precommit/r4_after_864832_two_sided_cover_bank_20260516/
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_COVER_BANK_ROWS_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_after_864832_two_sided_cover_bank_rows_20260516/
```

Two-sided bank status:

```text
PASS_R4_AFTER_864832_TWO_SIDED_COVER_BANK_STATIC_VALIDATION_NO_COMPUTE
entries: 256
coordinates: 32
bits per coordinate: 2
source reused entries: 128
generated complement entries: 128
protected-codeword missing coordinates: []
forbidden literal hits: []
```

Two-sided row status:

```text
PASS_TWO_WAY_COMPATIBLE_ROWS_BUILT
rows built: 8192
selected prompts: 256
coordinates: 32
surface entries: 256
coordinates missing same-coordinate opposite bucket: 0/32
coordinates whose present bank polarity does not match protected codeword bit: 0/32
current two-way scorer compatible: true
```

The tokenizer-only Slurm route has now been prepared and statically validated
without submission:

```text
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_QWEN_TOKENIZER_PREFLIGHT_ROUTE_20260516.md
configs/natural_evidence_v2/r4_after_864832_two_sided_tokenizer_preflight_route.yaml
scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200.sbatch
scripts/natural_evidence_v2/validate_r4_after_864832_two_sided_tokenizer_route.py
results/natural_evidence_v2/status/r4_after_864832_two_sided_tokenizer_preflight_route_validation_20260516/
```

Tokenizer route validation:

```text
PASS_R4_AFTER_864832_TWO_SIDED_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT
allowlist entry: v2_r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200
rows: 8192
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_qwen_tokenizer_boundary_preflight_h200.sbatch
```

Remote preflight and single submission then passed:

```text
remote preflight: PASS_R4_AFTER_864832_TWO_SIDED_TOKENIZER_REMOTE_PREFLIGHT_NO_SUBMIT
job_id: 865210
job_name: nat-ev-v2-r4tsTok
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
scope: tokenizer-only; no model forward, no scoring, no training, no generation
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_tokenizer_submission_20260516/
```

Job `865210` completed and passed tokenizer review:

```text
status: PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT
checked_rows: 8192
failed_rows: 0
empty_target_id_row_count: 0
empty_other_id_row_count: 0
target_other_overlap_row_count: 0
model_forward_pass_started: false
scoring_job_submitted: false
training_started: false
generation_started: false
review: docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_QWEN_TOKENIZER_PREFLIGHT_865210_REVIEW_20260516.md
```

The H200 teacher-forced surface-mass scoring route was prepared and validated:

```text
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_SCORE_ROUTE_20260516.md
configs/natural_evidence_v2/r4_after_864832_two_sided_surface_mass_score_route.yaml
scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_surface_mass_score_h200.sbatch
scripts/natural_evidence_v2/validate_r4_after_864832_two_sided_surface_mass_route.py
results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_route_validation_20260516/
results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_wrapper_plan_smoke_20260516/
```

Initial scoring route status:

```text
PASS_R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only smoke: DRY_RUN_VALIDATED_INPUTS
rows: 8192
arms: base, protected, task_only
generation_started: false
training_started: false
```

The first scoring submission, job `865235`, failed before model scoring:

```text
failure: REQUIRED_ADAPTER_MISSING_OR_EMPTY
bad path: .../r4_candidate_v3_micro_overfit_864761/protected_train/adapter/adapter_config.json
correct protected path: .../r4_candidate_v3_micro_overfit_864761/protected_micro_overfit_train/adapter
task-only path verified: .../wp5_r2_teacher_forced_train_and_score_851481/task_only_train/adapter
```

This was reviewed as a wrapper path failure, not a model/scoring result. The
wrapper and validator were repaired and revalidated:

```text
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_865235_FAILURE_REPAIR_20260516.md
results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_route_validation_repaired_20260516/
results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_wrapper_plan_smoke_repaired_20260516/
results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_remote_preflight_repaired_20260516/
```

Repaired route status:

```text
local repaired validation: PASS_R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_ROUTE_VALIDATION_NO_SUBMIT
local repaired wrapper plan-only smoke: DRY_RUN_VALIDATED_INPUTS
remote repaired preflight: PASS_R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_REPAIRED_REMOTE_PREFLIGHT_NO_SUBMIT
remote hash diff: empty
remote allowlist safety before submission: PASS
active jobs before submission: none
```

One repaired H200 scoring job was submitted:

```text
job_id: 865252
job_name: nat-ev-v2-r4tsTFM
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time limit: 30-00:00:00
arms: base, protected, task_only
rows: 8192
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_submission_865252_20260516/
```

Job `865252` completed cleanly but failed the teacher-forced gate:

```text
status: FAIL_R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_GATE
protected lift vs base: +0.047113
protected lift vs task-only: +0.056045
protected rank1 rate: 0.437500
protected median margin: -0.012231
base mean target mass: 0.019409
protected mean target mass: 0.066522
task-only mean target mass: 0.010477
review: results/natural_evidence_v2/status/r4_after_864832_two_sided_surface_mass_score_865252/review/
```

This failure does not unlock generation. It shows the protected adapter has
some positive mass lift, but the lift and rank1 are far below the
teacher-forced pre-generation gate.

The next route has been validated artifact-only:

```text
route: R4 after-864832 two-sided protected-adapter gain sweep
config: configs/natural_evidence_v2/r4_after_864832_two_sided_adapter_gain_sweep.yaml
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_adapter_gain_sweep_h200.sbatch
allowlist entry: v2_r4_after_864832_two_sided_adapter_gain_sweep_h200
plan validation: PASS_R4_ADAPTER_GAIN_SWEEP_PLAN_VALIDATION
wrapper plan-only smoke: DRY_RUN_VALIDATED_INPUTS
gains: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0
scope: H200 teacher-forced scoring only; no generation or training
```

One adapter-gain sweep job was submitted:

```text
job_id: 865289
job_name: nat-ev-v2-r4tsGain
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time limit: 30-00:00:00
protected gains: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_adapter_gain_sweep_submission_865289_20260516/
```

Job `865289` completed cleanly but no gain passed the teacher-forced gate:

```text
status: FAIL_R4_AFTER_864832_TWO_SIDED_ADAPTER_GAIN_SWEEP_GATE
best gain by mean target mass: protected_gain_1
best mean target mass: 0.066522
best lift vs base: +0.047113
best lift vs task-only: +0.056045
rank1 for all nonzero gains: 0.437500
median margin for all gains: negative
review: results/natural_evidence_v2/status/r4_after_864832_two_sided_adapter_gain_sweep_865289/review/
```

This rules out scalar adapter amplification as the immediate repair for the
two-sided cover-natural bank. The next route must be a reviewed objective or
controller pivot; generation remains disallowed until a teacher-forced gate
passes.

The controller-only route was recorded and submitted:

```text
route: R4 after-864832 two-sided controller-only teacher-forced scoring
job_id: 865351
job_name: nat-ev-v2-r4tsCtl
array: 0-71%4
conditions: base, task_only, controlled_base, wrong_key_controlled_base, wrong_payload_controlled_base
scope: H200 teacher-forced scoring only; no generation or training
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_only_submission_865351_20260516/
```

Array job `865351` failed before model scoring because the route wrapper did
not export the shared wrapper full-mode guard:

```text
stderr: ALLOW_PRESSURE_CONTROLLER_SCORING_REQUIRED_FOR_FULL_MODE
model_scoring_started: false
```

The wrapper was repaired by exporting `ALLOW_PRESSURE_CONTROLLER_SCORING=1`,
then local/remote route validation and plan-only smoke were rerun. One repaired
array job was submitted:

```text
job_id: 865434
job_name: nat-ev-v2-r4tsCtl
array: 0-71%4
scope: H200 teacher-forced controller-only scoring only; no generation or training
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_only_submission_865434_20260516/
```

Job `865434` completed cleanly and was reviewed:

```text
status: FAIL_R4_AFTER_864832_TWO_SIDED_CONTROLLER_ONLY_SCORE_865434_NO_SELECTIVE_GATE
summaries present: 72/72
controlled basic gate pass: 0/72
overall selective gate pass: 0/72
wrong-key basic gate pass: 0/72
wrong-payload basic gate pass: 0/72
best controlled lift vs base: +0.037408
best controlled lift vs task-only: +0.046340
best controlled rank1: 0.651367
best controlled median margin: +0.006919
review: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_only_score_865434_review/review.md
aggregate: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_only_score_865434_review/aggregate_summary.json
```

Interpretation: the two-sided controller-only run improved the margin but did
not produce enough target-mass/rank pressure to unlock generation. Null controls
remained clean, so this is a positive-strength failure, not a false-accept
safety failure.

Failure attribution and a safety-bound controller route were then recorded:

```text
failure attribution: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_only_failure_attribution_865434_20260516/
selected grid for attribution: 71
controller cap rows: max_kl_budget=1132, max_target_mass=210
weakest coordinate by lift: 2, lift +0.003612, rank1 0.105469
strongest coordinate by lift: 19, lift +0.089144, rank1 0.957031
route doc: docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_CONTROLLER_SAFETY_BOUND_ROUTE_20260516.md
route config: configs/natural_evidence_v2/r4_after_864832_two_sided_controller_safety_bound_route.yaml
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_controller_safety_bound_score_h200.sbatch
local route validation: PASS
local wrapper plan-only smoke: PASS
remote hash preflight: PASS
remote allowlist safety: PASS
remote wrapper plan-only smoke: PASS
```

One reviewed safety-bound H200 controller scoring array was submitted:

```text
job_id: 866147
job_name: nat-ev-v2-r4tsCtlB
array: 0-23%4
grid: bonus [1.50, 1.75, 2.00] x penalty [0.25, 0.50] x max_target_mass [0.45, 0.50] x max_kl_budget [0.10, 0.20]
scope: teacher-forced controller scoring only; no generation or training
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
submission record: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_safety_bound_submission_866147_20260516/
```

Job `866147` completed cleanly and failed the selective teacher-forced gate:

```text
status: FAIL_R4_AFTER_864832_TWO_SIDED_CONTROLLER_SAFETY_BOUND_SCORE_866147_NO_SELECTIVE_GATE
summaries present: 24/24
controlled basic gate pass: 0/24
overall selective gate pass: 0/24
wrong-key basic gate pass: 0/24
wrong-payload basic gate pass: 0/24
best grid: 23
bonus: 2.0
penalty: 0.5
max_target_mass: 0.5
max_kl_budget: 0.2
controlled lift vs base: +0.059981
controlled lift vs task-only: +0.068912
controlled rank1: 0.736084
controlled median margin: +0.017426
review: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_safety_bound_score_866147_review/review.md
```

Best-grid failure attribution was recorded:

```text
attribution: results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_safety_bound_failure_attribution_866147_20260516/
max_kl_budget cap rows: 1344
max_target_mass cap rows: 314
weakest coordinate by lift: 2, lift +0.006593, rank1 0.222656
strongest coordinate by lift: 19, lift +0.137760, rank1 0.996094
coordinates with lift >= +0.08: 10/32
coordinates with lift >= +0.10: 8/32
coordinates with rank1 >= 0.75: 18/32
```

The pivot decision is now artifact-only reliability-weighted codebook planning:

```text
docs/natural_evidence_v2/R4_AFTER_864832_TWO_SIDED_CONTROLLER_866147_FAILURE_PIVOT_DECISION_20260516.md
results/natural_evidence_v2/status/r4_after_864832_two_sided_controller_866147_failure_pivot_20260516/
selected next phase: V2_R4_AFTER_864832_RELIABILITY_WEIGHTED_CODEBOOK_ARTIFACT_ONLY
```

Codex executed the artifact-only reliability-weighted codebook plan:

```text
status: PASS_RELIABILITY_WEIGHTED_CODEBOOK_PLAN_8_PAIRS_AVAILABLE_NO_COMPUTE
script: scripts/natural_evidence_v2/build_r4_after_864832_reliability_weighted_codebook_plan.py
output: results/natural_evidence_v2/status/r4_after_864832_reliability_weighted_codebook_plan_20260516/
thresholds: lift >= +0.03, rank1 >= 0.80, median margin > 0
selected pairs: 8
selected coordinates: [6, 22, 10, 26, 1, 17, 3, 19, 15, 31, 8, 24, 4, 20, 7, 23]
candidate contract: 4 payload bits + 4 checksum bits, 2 coordinates per bit
candidate status: CANDIDATE_NOT_PRECOMMITTED
```

The candidate was then frozen as an artifact-only precommit:

```text
status: PRECOMMITTED_ARTIFACT_ONLY_NO_COMPUTE
doc: docs/natural_evidence_v2/R4_AFTER_864832_RELIABILITY_WEIGHTED_CODEBOOK_PRECOMMIT_20260516.md
precommit dir: results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/
codebook: codebook.json
decoder spec: decoder_spec.json
manifest: precommit_manifest.json
decoder: pair-majority then checksum
primary scrub mode: all
```

Codex then recorded and executed the artifact-only decoder/oracle route for the
frozen codebook:

```text
route doc: docs/natural_evidence_v2/R4_AFTER_864832_RELIABILITY_CODEBOOK_DECODER_ORACLE_ROUTE_20260516.md
route config: configs/natural_evidence_v2/r4_after_864832_reliability_codebook_oracle_route.yaml
validator: scripts/natural_evidence_v2/validate_r4_after_864832_reliability_codebook_oracle_route.py
output: results/natural_evidence_v2/status/r4_after_864832_reliability_codebook_decoder_oracle_20260516/
status: PASS_R4_RELIABILITY_CODEBOOK_DECODER_ORACLE_ARTIFACT_ONLY
oracle cases: 7
case failures: 0
expected perfect accept: true
expected single-coordinate erasure accept: true
wrong-payload accepts: 0
wrong-key accepts: 0
slurm/model-scoring/generation/training started: false
allowlist safety after validation: PASS with zero enabled entries
```

The manifest file has a self-hash bookkeeping distinction:

```text
precommit_manifest_declared_sha256: 747e7a5d9c10bbcaad8cb8eafecc27faeb4b2403105a87d4e4112af1d332e338
precommit_manifest_file_sha256: 15e2f3c3db0b00ec5d6392a9d0d8b7464e6f0acd8d89be1db550a95d9e59ec5e
```

The validator treats this as manifest metadata rather than codebook/decoder
drift. The frozen `codebook.json` and `decoder_spec.json` hashes match the
precommit.

Codex then prepared the next dev-generation route and wrapper plan-only smoke:

```text
route config: configs/natural_evidence_v2/r4_after_864832_reliability_dev_generation_route.yaml
route validator: scripts/natural_evidence_v2/validate_r4_after_864832_reliability_dev_generation_route.py
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_reliability_dev_generation_h200.sbatch
decoder: scripts/natural_evidence_v2/decode_r4_after_864832_reliability_codebook.py
allowlist entry: v2_r4_after_864832_reliability_dev_generation_h200
route validation: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
```

The wrapper plan-only toy decode failed and exposed a protocol blocker:

```text
toy protected reliability decode did not accept
protected accepts: 0/1
wrong-key accepts: 0/1
wrong-payload accepts: 0/1
matched_surface_count: 256
selected_surface_count: 120
failure: selected coordinate phrase ambiguity causes pair ties
```

Surface uniqueness audit:

```text
script: scripts/natural_evidence_v2/audit_r4_after_864832_reliability_surface_uniqueness.py
output: results/natural_evidence_v2/status/r4_after_864832_reliability_surface_uniqueness_audit_20260516/
status: FAIL_R4_RELIABILITY_SURFACE_UNIQUENESS_SELECTED_COORDINATES_AMBIGUOUS
surface entries: 256
selected coordinates: 16
unique normalized phrases: 21
phrases ambiguous for selected coordinates: 21
phrases with opposite polarity for selected coordinates: 0
```

Interpretation: the current two-sided surface bank repeats the same ordinary
phrases across many coordinates. The reliability-weighted codebook requires
coordinate identity, but phrase-only matching cannot recover coordinate identity
from this bank. This blocks generation submission even though the codebook oracle
itself passed.

Codex repaired the blocker artifact-only by building a coordinate-identifiable
surface bank for the selected reliability-codebook coordinates:

```text
builder: scripts/natural_evidence_v2/build_r4_after_864832_coordinate_unique_surface_bank.py
surface bank: results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json
surface bank sha256: 4a0f07af15ade41d51655352976e18d17e095b60a4850814ade231ec9f5fe1ac
status: PASS_COORDINATE_UNIQUE_SURFACE_BANK_BUILT_ARTIFACT_ONLY
surface entries: 128
selected coordinates: 16
entries per coordinate/polarity: 4
unique normalized phrases: 128
```

The repaired bank passed surface uniqueness:

```text
audit: results/natural_evidence_v2/status/r4_after_864832_coordinate_unique_surface_uniqueness_audit_20260516/
status: PASS_R4_RELIABILITY_SURFACE_UNIQUENESS
phrases ambiguous for selected coordinates: 0
phrases with opposite polarity for selected coordinates: 0
```

The dev-generation route config and wrapper were updated to use the
coordinate-unique bank. Local validation and wrapper plan-only smoke now pass:

```text
route config: configs/natural_evidence_v2/r4_after_864832_reliability_dev_generation_route.yaml
route validation: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_route_validation_unique_v2_20260516/
route validation status: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_wrapper_plan_smoke_unique_20260516/
wrapper plan-only status: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY
toy protected accepts: 1/1
toy wrong-key accepts: 0/1
toy wrong-payload accepts: 0/1
local allowlist safety: PASS with zero enabled entries
slurm/model-scoring/generation/training started: false
```

Remote preflight also passed:

```text
remote preflight: results/natural_evidence_v2/status/r4_after_864832_coordinate_unique_reliability_remote_preflight_20260516/
status: PASS_R4_AFTER_864832_COORDINATE_UNIQUE_RELIABILITY_REMOTE_PREFLIGHT_NO_SUBMIT
hash diff: empty
remote route validation: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
remote allowlist safety: PASS with zero enabled entries
remote wrapper plan-only: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY
remote toy protected accepts: 1/1
remote toy wrong-key accepts: 0/1
remote toy wrong-payload accepts: 0/1
active jobs before submission: 0
```

One reviewed H200 array job was submitted:

```text
job_id: 867596
submission record: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_submission_867596_20260516/
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
```

Job `867596` failed immediately before generation:

```text
array tasks: 867596_[0-3]
state: FAILED
exit code: 1:0
elapsed: 00:00:01
failure: control-plane validator allowlist race
model generation started: false
decode started: false
training started: false
```

The Slurm stdout shows the wrapper's internal validator failed because it saw
its own reviewed allowlist entry still enabled during the short submission
window:

```text
errors:
  - allowlist entry must be disabled
```

This was repaired in:

```text
doc: docs/natural_evidence_v2/R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867596_FAILURE_REPAIR_20260516.md
validator: scripts/natural_evidence_v2/validate_r4_after_864832_reliability_dev_generation_route.py
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_reliability_dev_generation_h200.sbatch
```

Repair semantics: plan-only validation still requires the route entry disabled;
full-mode wrapper startup now permits an empty allowlist or exactly the reviewed
`v2_r4_after_864832_reliability_dev_generation_h200` entry enabled, and rejects
any other enabled entry.

Fresh repaired remote preflight passed:

```text
remote preflight: results/natural_evidence_v2/status/r4_after_864832_reliability_allowlist_race_repair_remote_preflight_20260516/
status: PASS_R4_AFTER_864832_RELIABILITY_ALLOWLIST_RACE_REPAIR_REMOTE_PREFLIGHT_NO_SUBMIT
hash diff: empty
remote route validation: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
remote allowlist safety: PASS with zero enabled entries
remote wrapper plan-only: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY
active jobs before resubmission: 0
```

One replacement H200 array job was submitted and completed cleanly:

```text
job_id: 867621
job_name: nat-ev-v2-r4relgen
submission record: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_submission_867621_20260516/
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
max time: 30-00:00:00
array shards: 0..3
Slurm state: COMPLETED
exit code: 0:0
post-submit local allowlist safety: PASS with zero enabled entries
post-submit remote allowlist safety: PASS with zero enabled entries
```

The reviewed aggregate result failed the protected positive gate:

```text
review: docs/natural_evidence_v2/R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_REVIEW_20260516.md
review summary: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_review/review_summary.json
failure analysis: results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_failure_analysis_20260516/
status: FAIL_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_POSITIVE_GATE
generated rows: 6144
protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw/task-only/wrong-key/wrong-payload accepts, format_scrub=all: 0/32 each
protected forbidden public surface count: 0
coordinate-unique selected surface matches in protected: 0
```

Artifact-only failure analysis identifies the root cause as:

```text
root_cause: free_generation_transfer_failure_surface_absent
protected coordinate-unique bank surface matches: 0
protected rows with any coordinate-unique bank surface: 0
protected duplicate response hash rows: 508
protected max duplicate response hash count: 27
protected rows with repeated sentence/clause units: 2001/2048
Create a plan protected occurrences: 44500
Prepare a schedule protected occurrences: 4234
Prepare a budget protected occurrences: 11709
Prepare a plan protected occurrences: 11691
```

Interpretation: the reliability codebook and coordinate-unique surface bank
passed artifact-only oracle/plan checks, but the protected generator did not
emit the frozen coordinate-unique surfaces in free generation. The adapter
instead collapsed toward old candidate-v3 `Create/Prepare/Plan` phrases. This
is not a positive result and does not unlock downstream compute or claims.

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

`BLOCK_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_867849_GATE_FAIL_NO_GENERATION`

## Current Next Allowed Action

Record the next artifact-only repair or pivot decision before any further
Slurm, generation, training, Llama, sanitizer, FAR, payload-diversity, or
paper-facing claim work. Do not rerun the same scoring/generation route
unchanged.

```text
allowed:
- artifact-only failure analysis
- artifact-only repair or pivot planning
- static validation of a proposed next route
- keep allowlist zero enabled

not yet unlocked:
- no rerun of the 867621 route unchanged
- no rerun of the 867849 scoring route unchanged
- no generation until a future teacher-forced surface-mass gate passes and a separate generation route is reviewed
- no model scoring/generation/training outside a reviewed route
- no Llama
- no same-family null
- no sanitizer
- no FAR
- no payload-diversity claim
- no paper-facing claim
```

User standing authorization remains active once a route's recorded prerequisite
gates pass. That authorization does not waive the need for a new route decision
after this failed gate, fresh local/remote hash preflight, Hermes notification,
zero-enabled allowlist preflight, exactly-one reviewed H200 submission, and
immediate allowlist disable after `sbatch`.

## Guardrails

The cover-bank-aligned repair must not:

```text
- add 864832-observed phrases to the bank;
- treat candidate-v3 pressure phrases as decoder surfaces;
- use Create/Prepare/Plan repetition as success evidence;
- lower accept/support/margin gates;
- run generation before a cover-bank-aligned teacher-forced gate passes;
- bypass Hermes notification, remote hash preflight, allowlist safety, or the exactly-one H200 submission rule.
```

Future teacher-forced gate:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
scorer boundary failures = 0
target/other token overlap = 0
visible repetition collapse = false
```
