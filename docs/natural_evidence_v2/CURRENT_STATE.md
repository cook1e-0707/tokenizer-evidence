# natural_evidence_v2 Current State

Last synchronized: 2026-05-13T02:27Z

## Canonical Phase

`V2_R4_PREFIX_NATIVE_SURFACE_MASS_JOB_853894_FAILED_REVIEWED_NO_SCORE_SUMMARY`

This compact state file is the first file Codex/Hermes should read for routine
ticks. Use the long historical files only when this file is ambiguous:

- `docs/natural_evidence_v1/AUTOMATION_STATE.md`
- `docs/natural_evidence_v1/next_step_codex_plan.md`
- `results/natural_evidence_v1/status/gate_status.json`
- `results/natural_evidence_v2/status/gate_status.json`

## Current Route

Route R4: cover-natural ECC evidence channel repair preflight after failed
H200 dev diagnostic `853691`.

The user has recorded standing authorization for Codex and Hermes to continue
the already approved route without repeatedly asking for explicit approval.
This does not waive gates, allowlist requirements, notification requirements,
or the one-Slurm-job-per-reviewed-submission limit.

The user has also recorded conditional authorization for later-stage training,
Llama, FAR/null expansion, sanitizer, and paper-claim work after their
prerequisite gates pass. These work classes are not permanently forbidden.
They are gate-controlled: each class must first have a recorded route decision,
passing gate evidence, reviewed wrapper or artifact plan, allowlist where
applicable, and TG/email notification before any state-changing action.

## Latest Result Review

R4 prefix-native teacher-forced surface-mass scoring job `853894` reached a
terminal failed state and was reviewed:

`results/natural_evidence_v2/status/r4_prefix_native_surface_mass_score_853894_failure_review/r4_prefix_native_surface_mass_score_853894_failure_review.md`

Machine-readable review:

`results/natural_evidence_v2/status/r4_prefix_native_surface_mass_score_853894_failure_review/r4_prefix_native_surface_mass_score_853894_failure_review_summary.json`

Job:

- job id: `853894`;
- job name: `nat-ev-v2-r4pntfm`;
- state: `FAILED`;
- elapsed: `00:00:43`;
- exit code: `1:0`;
- node: `chimera21`.

The expected remote output directory exists but contains no score summary:

`/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_prefix_native_surface_mass_score_853894`

Failure:

- scorer failed before producing R4 surface-mass metrics;
- traceback ended in tokenizer first-token validation;
- exception: `ValueError: surface produced no next token: 'create'`.

Interpretation:

- this is a scorer/candidate tokenizer-surface compatibility failure;
- no teacher-forced surface gate result was produced;
- no downstream gate is unlocked.

Current next allowed action: artifact-only diagnosis and repair planning for
the R4 prefix-native surface-tokenizer compatibility failure. Do not submit
another scoring job or run generation/training/Llama/FAR/sanitizer/paper-claim
actions until a new reviewed route decision and preflights explicitly allow it.

## Latest Route Decision

R4 prefix-native teacher-forced tokenizer/model scoring route is recorded:

`docs/natural_evidence_v2/R4_PREFIX_NATIVE_SURFACE_MASS_SCORING_ROUTE_DECISION_20260513.md`

Machine-readable decision:

`results/natural_evidence_v2/status/r4_prefix_native_surface_mass_scoring_route_decision_20260513.json`

This route allows exactly one Chimera Slurm Qwen tokenizer/model forward
scoring job for the prefix-native repair candidate, after local/remote
allowlist and hash preflights pass and Hermes TG/email notification is sent.
It does not authorize generation, training, Llama, same-family nulls,
sanitizer, FAR, payload diversity, or paper-facing claims.

Preflight status:

- local zero-enabled allowlist safety: `PASS`;
- remote zero-enabled allowlist safety: `PASS`;
- local/remote hash preflight: `PASS`;
- single-enabled allowlist preflight: `PASS`;
- Slurm job submitted: `853894`;
- job name: `nat-ev-v2-r4pntfm`;
- partition/QoS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`;
- initial node: `chimera21`;
- allowlist was disabled locally and remotely immediately after submission.

This route has now been consumed by failed job `853894`; see the latest result
review above. It does not authorize another scoring submission.

## Latest Artifact-Only Candidate

Prefix-native R4 surface repair candidate is now recorded:

`results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/static_validation_report.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/static_validation_summary.json`

Scorer input dry-run:

`results/natural_evidence_v2/status/r4_prefix_native_surface_score_wrapper_plan_smoke_20260513/r4_teacher_forced_surface_mass_summary.json`

Scope:

- artifact-only candidate construction;
- no Qwen tokenizer validation locally because this environment does not
  provide `transformers`;
- no model scoring;
- no Slurm submission;
- no generation or training.

Static proxy validation:

- status: `PASS_PROXY_STATIC_VALIDATION_TOKENIZER_PENDING`;
- coordinates: `32`;
- entries: `256`;
- probe rows: `8192`;
- prompts: `256`;
- missing binary-side coordinates: `0`;
- normalized first-word proxy overlap coordinates: `0`;
- forbidden surface hits: `0`;
- measured span-start failures: `0`.

Design change:

- prior R4 phrases were free-floating and near-zero probability;
- repaired candidate uses prefix-native continuations whose measured span
  begins immediately after the local lead-in prefix;
- binary sides reuse the R3/WP5 learned action families in cover-natural
  phrase form: `set`/`plan` versus `create`/`prepare`.

Current next allowed action: review this candidate and prepare a separate
Slurm-only tokenizer/model scoring route if accepted. This candidate alone
does not authorize allowlist enablement or Slurm submission.

## Latest Artifact-Only Diagnosis

R4 surface-mass failure diagnosis after `853815` is now recorded:

`results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/surface_mass_failure_diagnosis_report.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/surface_mass_failure_diagnosis_summary.json`

This diagnosis is artifact-only. It reads the existing `853815` scored rows,
the binary surface-bank repair candidate, and the frozen teacher-forced probe
rows. It does not train, generate, score models, submit Slurm, run Llama,
aggregate FAR, or make paper claims.

Key facts:

- scored rows: `24576`;
- joined base/protected/task-only records: `8192`;
- protected mean target mass: `0.0000438295`;
- protected-vs-base mean lift: `-0.0000864096`;
- protected-vs-task-only mean lift: `-0.0002997293`;
- protected rank-1 rate: `0.4375`;
- target/other first-token overlap rate: `0.0`;
- every coordinate has both binary sides in the candidate bank.

Interpretation:

- this is not a Slurm/provider failure;
- this is not the prior one-sided-bank formal failure;
- this is not primarily a target/other first-token overlap bug;
- the active blocker is that selected phrase-level target cylinders are
  extremely low probability under the R4 prefixes, and the existing protected
  adapter does not increase their mass.

Current next allowed action: artifact-only R4 target-construction /
surface-bank / prefix-shape repair design only. Do not submit another scoring
job, run generation, train, or unlock Llama/FAR/sanitizer/paper claims from
this state.

## Latest Repair Design

Artifact-only R4 target-construction and prefix-shape repair design is now
recorded:

`docs/natural_evidence_v2/R4_SURFACE_BANK_PREFIX_REPAIR_DESIGN_AFTER_853815_20260513.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r4_surface_bank_prefix_repair_design_after_853815_20260513/repair_design_summary.json`

Design direction:

- stop treating target phrases as free-floating answer content;
- construct prefix-native surface cylinders whose measured span begins
  immediately after the local lead-in prefix;
- keep target/other alternatives in the same syntactic slot with no first-token
  overlap;
- require a static artifact-only validation summary before any future reviewed
  Slurm scorer route decision.

Current next allowed action: artifact-only construction of a repaired R4
candidate surface bank and prefix-row static validation only. Do not submit
another scoring job, run generation, train, or unlock
Llama/FAR/sanitizer/paper claims from this state.

## Latest Result Review

R4 teacher-forced surface-mass scoring job `853815` completed and was reviewed:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815_review/r4_surface_mass_score_853815_review.md`

Machine-readable review:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_853815_review/r4_surface_mass_score_853815_review_summary.json`

Job:

- job id: `853815`;
- job name: `nat-ev-v2-r4tfm`;
- state: `COMPLETED`;
- elapsed: `00:04:39`;
- exit code: `0:0`;
- node: `chimera21`.

Teacher-forced surface gate: `FAIL`.

Key numbers:

- protected target surface mass lift vs base: `-0.0000864096`,
  required `>= +0.15`;
- protected target surface mass lift vs task-only: `-0.0002997293`,
  required `>= +0.10`;
- protected target surface rank-1 rate: `0.4375`, required `>= 0.70`;
- protected median target margin: `-0.0000096318`, required `> 0`.

Interpretation:

- this is not a Slurm/provider failure;
- the binary repair candidate fixed the formal two-sided surface-bank issue;
- it did not create a trainable surface channel under the existing protected
  adapter;
- target phrase-surface masses are near zero across all arms.

Current next allowed action: artifact-only R4 surface-bank / prefix-shape /
target construction diagnosis only. Do not submit another scoring job, do not
run generation, do not train, and do not unlock Llama/FAR/sanitizer/paper
claims until a new repair plan is reviewed.

## Previous Slurm Submission

R4 teacher-forced surface-mass scoring job submitted:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_submission_record_20260513.json`

Job:

- job id: `853815`;
- job name: `nat-ev-v2-r4tfm`;
- partition/QoS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`;
- initial state: `RUNNING`;
- initial node: `chimera21`;
- scope: Qwen base/protected/task-only teacher-forced surface-mass scoring;
- score rows: `8192`;
- contract: `a55e`;
- no free generation, no training, no Llama, no same-family null, no sanitizer,
  no FAR aggregation, no payload-diversity claim, and no paper-facing positive
  claim.

Allowlist was disabled immediately after `sbatch` returned the job id. Local
and remote post-submission allowlist safety both passed:

- `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_post_submit_allowlist_safety_20260513.json`;
- remote:
  `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_post_submit_allowlist_safety_remote_20260513.json`.

Current next allowed action: monitor Slurm job `853815`. After completion, sync
and review the teacher-forced surface-mass summary. Do not submit another
scoring job or run generation/training/Llama/FAR/sanitizer/paper claims until
this result is reviewed.

## Previous Route Decision

The 2026-05-13 01:28Z user authorization is now controlling for wrapper
preparation:

`docs/natural_evidence_v2/R4_TEACHER_FORCED_SURFACE_MASS_SCORER_ROUTE_DECISION_20260513.md`

Machine-readable decision:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_scorer_route_decision_20260513.json`

Authorized scope:

- prepare a Slurm-only Qwen teacher-forced surface-mass scorer wrapper;
- score plan scope: base / protected / task-only forward scoring only;
- frozen rows:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/r4_surface_teacher_forced_probe_rows.jsonl`;
- candidate bank:
  `results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/candidate_binary_surface_bank.json`;
- contract: `a55e`;
- run local plan-only smoke validation;
- add only a disabled allowlist entry.

Not authorized by this route decision:

- allowlist enablement;
- Slurm submission;
- free generation;
- locked-scale rerun;
- training;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim.

Prepared wrapper:

`scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch`

Plan-only smoke:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_wrapper_plan_smoke_20260513/r4_teacher_forced_surface_mass_summary.json`

Smoke result:

- status: `DRY_RUN_VALIDATED_INPUTS`;
- score rows: `8192`;
- condition plan: `base`, `protected`, `task_only`;
- model scoring started: `false`;
- generation/training/Slurm started: `false`.

Post-wrapper allowlist safety:

`results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_allowlist_safety_zero_20260513.json`

Next allowed action: review the prepared wrapper and plan-only smoke. If
accepted, record a separate single-submission route decision before enabling
the allowlist entry or submitting exactly one Slurm scoring job.

## Previous Supervisor Hold Reconciliation

The 2026-05-13 01:26Z Hermes supervisor report held all compute actions until a
human/expert route decision. The 01:28Z user authorization above supersedes
that hold only for wrapper preparation. It still does not authorize allowlist
enablement or Slurm submission.

Hold/blocker record:

`results/natural_evidence_v1/status/hermes_reports/20260513_0126_r4_no_slurm_hold_blocker.md`

Machine-readable record:

`results/natural_evidence_v1/status/hermes_reports/20260513_0126_r4_no_slurm_hold_blocker.json`

No generation, Qwen E2E rerun, training, Llama, same-family null, sanitizer,
FAR aggregation, payload-diversity claim, paper-facing positive claim, allowlist
enablement, or Slurm submission is unlocked by either the hold reconciliation
or the wrapper-preparation route decision.

## Latest R4 Artifact-Only Package

The 2026-05-12 R4 cover-natural ECC protocol decision, artifact-only planning
package, and user-approved dev diagnostic route are recorded.

Protocol decision:

`docs/natural_evidence_v2/R4_COVER_NATURAL_ECC_PROTOCOL_DECISION_20260512.md`

Machine-readable decision:

`results/natural_evidence_v2/status/r4_cover_natural_ecc_protocol_decision_20260512.json`

Config:

`configs/natural_evidence_v2/r4_cover_natural_ecc.yaml`

Generated artifact-only outputs:

- oracle recoverability audit:
  `results/natural_evidence_v2/status/r4_artifact_only_oracle_recoverability_20260512/oracle_recoverability_summary.json`;
- forbidden-surface matcher audit:
  `results/natural_evidence_v2/status/r4_forbidden_surface_matcher_audit_20260512/forbidden_surface_audit.json`;
- structural leakage audit:
  `results/natural_evidence_v2/status/r4_structural_leakage_audit_20260512/structural_leakage_summary.json`;
- cover-natural prompt bank:
  `results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512/prompt_bank_manifest.json`;
- surface bank, codebook, and decoder precommit:
  `results/natural_evidence_v2/precommit/r4_cover_natural_ecc_precommit_20260512/precommit_manifest.json`;
- format-scrub decoder smoke on the failed `853524` transcripts:
  `results/natural_evidence_v2/status/r4_cover_natural_decoder_smoke_on_853524_20260512/decode_summary.json`;
- plan validation:
  `results/natural_evidence_v2/status/r4_cover_natural_plan_validation_20260512/validation_summary.json`;
- allowlist safety:
  `results/natural_evidence_v2/status/r4_cover_natural_ecc_allowlist_safety_20260512.json`.

User-approved dev diagnostic route decision:

`docs/natural_evidence_v2/R4_DEV_DIAGNOSTIC_ROUTE_DECISION_20260512.md`

Machine-readable decision:

`results/natural_evidence_v2/status/r4_dev_diagnostic_route_decision_20260512.json`

Dev diagnostic preflight artifacts:

- repaired 2048-dev / 6144-locked prompt bank:
  `results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/prompt_bank_manifest.json`;
- refreshed plan validation:
  `results/natural_evidence_v2/status/r4_cover_natural_plan_validation_20260512/validation_summary.json`;
- H200 wrapper plan-only smoke:
  `results/natural_evidence_v2/status/r4_dev_diagnostic_h200_wrapper_plan_smoke_20260513/`;
- zero-enabled allowlist safety after R4 phase checker repair:
  `results/natural_evidence_v2/status/r4_dev_diagnostic_allowlist_safety_zero_20260513_post_replicate_id.json`.

R4 artifact-only findings:

- plan validation status: `PASS`;
- original prompt bank: `384` dev prompts and `384` locked prompts;
- dev diagnostic prompt bank: `2048` dev prompts and `6144` locked prompts,
  with disjoint topic domains and no Step/slot structural instructions;
- surface bank: `128` phrase-level entries over `32` coordinates;
- decoder spec primary scrub mode: `all`;
- oracle recoverability on `853524` remains diagnostic only:
  phrase-surface oracle accepts `0/96`, structure-scrub oracle accepts `34/96`;
- structural leakage audit on `853524` shows public structural separability:
  max protected-vs-raw shallow feature AUC `0.6797`, above the R4 target `0.60`;
- forbidden-surface matcher audit found `23` examples: `10` ordinary-domain
  word matches, `12` literal substring matches, and `1` technical reserved
  token match;
- R4 decoder smoke on `853524` with `format_scrub=all` accepted `0/96` for
  protected/raw/task-only, as expected for a new plan-only bank.

Current next allowed action: hold for a human/expert route decision. R4
artifact-only planning is complete and plan validation passed, but Slurm remains
disabled until a human/expert route decision explicitly records permission for a
small dev diagnostic submission. No generation, training, Llama, same-family
null, sanitizer benchmark, FAR aggregation, payload-diversity claim, or
paper-facing positive claim is unlocked.

No-Slurm hold record:

`results/natural_evidence_v2/status/r4_artifact_only_planning_complete_no_slurm_hold_20260513_0056.json`

The 2026-05-13 wrapper smoke includes explicit `replicate_group_id=shard_00`
through `shard_03`, so downstream decode block ids do not collide across
shards.

Submission record:

`results/natural_evidence_v2/status/r4_dev_diagnostic_submission_record_20260513.json`

Slurm job:

- job id: `853691`;
- job name: `nat-ev-v2-r4dev`;
- partition/QOS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`;
- array: `0-3%4`;
- GPU: `h200`;
- prompt split: dev only;
- contract: same-contract `a55e`;
- primary decode scrub mode: `all`;
- post-submit local and remote allowlist safety: `PASS` with zero enabled
  entries.

Completion review:

`results/natural_evidence_v2/status/r4_cover_natural_dev_diagnostic_h200_array_853691/review/r4_dev_diagnostic_853691_review_summary.json`

Failure attribution:

`results/natural_evidence_v2/status/r4_cover_natural_dev_diagnostic_h200_array_853691/failure_attribution/failure_attribution_summary.json`

Outcome:

- all four H200 shards completed with `ExitCode 0:0`;
- protected accepts with no scrub: `0/32`, required `>=28/32`;
- protected accepts with `format_scrub=all`: `0/32`, required `>=26/32`;
- raw/task-only/wrong-key/wrong-payload accepts under `format_scrub=all`:
  `0/32` each;
- forbidden public surface total across decode rows: `237`, required `0`;
- protected-vs-raw shallow structural AUC on generated outputs: `0.624`,
  above the `0.60` gate;
- interpretation: this is not a Slurm/provider failure. It is a positive
  channel failure: the R4 output has phrase-surface matches, but observed
  coordinate polarities do not align with the protected `a55e` codeword.

R4 repair preflight:

- the original locked R4 surface bank could not support teacher-forced
  target-vs-other surface mass scoring because every coordinate had only one
  polarity side;
- blocker summary:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_20260513/r4_surface_teacher_forced_probe_plan_summary.json`;
- binary repair candidate built:
  `results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/binary_surface_bank_repair_summary.json`;
- candidate properties: `32` coordinates, `256` phrase-level entries, `8`
  entries per coordinate, `4` bit-0 and `4` bit-1 entries per coordinate;
- teacher-forced surface probe rows built against the candidate bank:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/r4_surface_teacher_forced_probe_plan_summary.json`;
- row-plan properties: `256` dev prompts, `8192` score rows, `32`
  coordinates, `256` rows per coordinate, contract `a55e`;
- local scorer dry-run validated the scoring input path:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_dry_run_binary_repair_20260513/r4_teacher_forced_surface_mass_summary.json`;
- no generation, training, Slurm submission, Llama, FAR aggregation,
  sanitizer, same-family null, payload diversity claim, or paper-facing claim
  was started by this repair preflight.

Current next allowed action: review the artifact-only binary surface repair and
teacher-forced surface probe plan; if accepted, prepare a Slurm-only Qwen
teacher-forced surface-mass scorer wrapper for base/protected/task-only. Do not
run free generation or locked-scale until the surface teacher-forced gate is
actually scored and reviewed.

## Latest Expert Review

The 2026-05-12 22:10Z expert review / artifact-only repair decision is
recorded for completed H200 job `853524`.

Expert decision:

`docs/natural_evidence_v2/R3_2_H200_853524_EXPERT_REVIEW_REPAIR_DECISION_20260512_2210.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r3_2_h200_853524_expert_review_repair_decision_20260512_2210.json`

Decision:

- `853524` is not artifact-only repairable into a passing locked-scale result;
- the 22:09Z repair decision package is accepted as a negative-result package;
- duplicate prompt-window reuse is fixed, so the remaining failure is not the
  old repeated-window control-plane defect;
- protected accepts remain far below gate at `6/96` versus required `80/96`;
- null arms remain clean at `0/96`;
- forbidden-surface matcher semantics still require a separate artifact-only
  audit, but cannot rescue the protected positive gate.

Current next allowed action: hold for human/expert route decision, or perform
artifact-only planning for a new protocol/prompt-bank repair package if
explicitly requested. Do not submit a rerun or unlock Llama, FAR, sanitizer,
same-family null, payload-diversity claims, or paper-facing positive claims
from `853524`.

## Latest Hermes Sync

The 2026-05-12 22:09Z artifact-only repair decision package is recorded for
completed H200 job `853524`.

Decision package:

`docs/natural_evidence_v2/R3_2_H200_853524_REPAIR_DECISION_PACKAGE_20260512.md`

Machine-readable package:

`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_h200_853524_repair_decision_package.json`

Package conclusion:

- `853524` is a clean Slurm completion and a negative locked-scale result;
- duplicate prompt-window reuse is fixed;
- diagnostic nulls remain clean at `0/96`;
- protected accepts are only `6/96`;
- main repair axes requiring expert review are prompt-variant repair,
  bank/surface repair, middle-step coordinate repair, forbidden-surface matcher
  review, and training-signal generalization review.

Current next allowed action: expert review / artifact-only repair decision for
`853524`. Do not submit a rerun or unlock Llama, FAR, sanitizer,
same-family null, payload-diversity claims, or paper-facing positive claims
until this package is reviewed and a new route is recorded.

The 2026-05-12 22:04Z artifact-only failure attribution is recorded for
completed H200 job `853524`.

Attribution report:

`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution.md`

Machine-readable summary:

`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution_summary.json`

Main attribution findings:

- duplicate prompt-window reuse is fixed; all 12 windows and 96 blocks are
  unique;
- diagnostic nulls remain clean at `0/96`;
- protected coordinate-majority recovery is only `6/96`;
- the weakest prompt variant is `r1_strict_literal_16_step_lines`, with
  protected target-hit rate `0.307`;
- largest erasure reason is `observed_first_word_not_in_primary_bucket_set`
  with `30,021` observations;
- secondary structural erasures: `missing_or_out_of_order_step_slots = 1,093`
  and `duplicate_step_slots = 485`;
- some generated protected outputs show duplicated labels such as
  `Step 1: Create a Step 1: ...`;
- forbidden public surface hits are `bucket=21`, `fingerprint=1`,
  `watermark=1`, and require separate matcher-semantics review.

Current next allowed action: expert review / artifact-only repair decision for
`853524`. Do not submit a rerun or unlock Llama, FAR, sanitizer,
same-family null, payload-diversity claims, or paper-facing positive claims
until this attribution is reviewed.

The 2026-05-12 22:02Z completion review is recorded for expanded 6144 H200
array job `853524`.

Slurm outcome:

- `853524_0` through `853524_11` all reached `COMPLETED`;
- all task exit codes were `0:0`;
- output dir synced locally:
  `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/`.

The duplicate-window control-plane problem from job `853430` is fixed here:

- selected prompt windows: `12/12` unique;
- selected prompt blocks: `96/96` unique;
- prompt row coverage: `0-6143`.

Aggregate gate result:

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| protected accepts @64 | `>=80/96` | `6/96` | FAIL |
| raw accepts @64 | `0/96` | `0/96` | PASS |
| task-only accepts @64 | `0/96` | `0/96` | PASS |
| wrong-key accepts @64 | `0/96` | `0/96` | PASS |
| wrong-payload accepts @64 | `0/96` | `0/96` | PASS |
| min accepted-block support | `>=16` | `6` | FAIL |
| min accepted-block majority margin | `>=3` | `0` | FAIL |
| forbidden public surface count | `0` | `23` | FAIL |
| replicate groups complete | `true` | `true` | PASS |

Final aggregate status:

`FAIL_R3_2_SAME_CONTRACT_LOCKED_SCALE_GATE`

Review artifacts:

- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_h200_853524_completion_review.md`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_h200_853524_completion_review.json`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_gate_review.json`

Current next allowed action: artifact-only failure attribution for `853524`.
Do not submit a rerun, aggregate FAR, start Llama, same-family null, sanitizer,
or paper-facing claims until attribution is recorded and reviewed.

The 2026-05-12 20:24Z expanded 6144 H200 shard-array submission is recorded.

Submitted exactly one Slurm array job:

- job id: `853524`;
- job name: `nat-ev-v2-r32h200`;
- partition: `pomplun`;
- account: `cs_yinxin.wan`;
- QoS: `pomplun`;
- GPU request: `gpu:h200:1`;
- array: `0-11%8`;
- command:
  `sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`;
- output dir:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_h200_array_853524`.

Submission safeguards completed before launch:

- local/remote expanded control-plane hashes matched;
- Hermes TG/email notification was sent;
- exactly one allowlist entry was enabled:
  `v2_r3_2_qwen_locked_scale_shard_array_h200`;
- the allowlist entry was disabled immediately after `sbatch`;
- local and remote post-submit allowlist safety checks passed with zero
  enabled entries.

Submission record:

`results/natural_evidence_v2/status/r3_2_expanded_6144_h200_submission_record_20260512_2024.json`

At the 2026-05-12 21:55Z monitor, all tasks `853524_0` through
`853524_11` were completed with exit code `0:0`; no running, pending, or
failed tasks were observed.

Monitor record:

`results/natural_evidence_v2/status/r3_2s_expanded_6144_h200_853524_monitor_20260512_2155.json`

Current next allowed action: review H200 shard-array outputs for Slurm array
job `853524` only. Do not submit another R3.2 job or aggregate until shard
outputs are reviewed and the next gate explicitly allows aggregation.

The 2026-05-12 20:22Z expanded 6144 wrapper plan validation is recorded.

Added expanded route config:

`configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale_expanded_6144.yaml`

Added expanded precommit builder:

`scripts/natural_evidence_v2/build_r3_2_expanded_locked_scale_precommit.py`

Validation doc:

`docs/natural_evidence_v2/R3_2_EXPANDED_6144_PRECOMMIT_AND_WRAPPER_PLAN_VALIDATION_20260512.md`

Verified precommit output:

`results/natural_evidence_v2/status/r3_2_expanded_6144_precommit_plan_20260512_verified_after_builder_fix/`

Key hashes:

- prompt source sha256:
  `8fcba10d2df1dae83eb03f8ce26fa45623c1918a9246c94b6b6868fc1204247a`;
- selected prompt manifest sha256:
  `ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`;
- precommit hash:
  `6de7432ef3155100321affa30f677c2d88e17d5bc6323cde2670ab838d8a85ea`.

Repaired shard-array wrappers:

- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`;
- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array.sbatch`.

Both now default to the expanded prompt artifact, expanded config, split
`wp3_r1_density_eval`, selected prompt manifest hash
`ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`, and
shard allocation `expected_start = SHARD_INDEX * 512`. Both wrappers support
`VALIDATE_PLAN_ONLY=1`.

Local plan-only smokes passed without Slurm submission:

- H200:
  `results/natural_evidence_v2/status/r3_2_expanded_6144_h200_wrapper_plan_smoke_20260512/`;
- scavenger:
  `results/natural_evidence_v2/status/r3_2_expanded_6144_scavenger_wrapper_plan_smoke_20260512/`.

Current next allowed action: artifact-only local/remote control-plane sync
preflight for the expanded 6144 route. Do not submit Slurm until a fresh
allowlist safety check, local/remote hash match, and Hermes TG/email
notification are recorded, and only one reviewed R3.2 shard-array entry is
enabled for submission.

The 2026-05-12 20:15Z artifact-only precommit repair validation is recorded.

Updated:

`scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`

The precommit builder now supports the expanded R3.2 allocation policy
`distinct_eval_window_by_shard_index` while preserving old-route compatibility.
Local plan-only validation passed for the expanded 6,144-row prompt artifact:

`results/natural_evidence_v2/status/r3_2_expanded_6144_precommit_local_validation_20260512_2015/`

Validation result:

- selected prompt manifest SHA256:
  `ce057a36ad75424919f4367eb3e2f0221725a9c6715d156ab4b2a377edb600ed`;
- unique selected prompt windows: `12`;
- unique blocks: `96`;
- shard window starts: `0..5632` by `512`, no modulo reuse;
- status: `PASS_R3_2_EXPANDED_PRECOMMIT_LOCAL_PLAN_VALIDATION_NO_SLURM`.

Review doc:

`docs/natural_evidence_v2/R3_2_EXPANDED_PRECOMMIT_REPAIR_VALIDATION_20260512_2015.md`

Current next allowed action: artifact-only wrapper repair/review and
allowlist/local-remote hash safety review for the expanded 6,144-row route. No
Slurm until repaired wrapper, allowlist safety, local/remote hashes, and Hermes
notification are reviewed.

The 2026-05-12 20:11Z expanded prompt plan is recorded.

Expanded WP2 prompt scaffold:

`results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_r3_2_expanded_20260512/`

Expanded WP3 strict Step-label prompt plan:

`results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_r3_2_6144_20260512/`

Expanded R3.2 allocation preflight:

`results/natural_evidence_v2/status/r3_2_repaired_prompt_allocation_preflight_expanded_6144_20260512/r3_2_repaired_prompt_allocation_preflight.json`

Result:

- expanded eval prompt rows: `6,144`;
- available unique shards: `12`;
- available unique blocks: `96`;
- duplicate prompt IDs: `0`;
- structural-slot violations: `0`;
- allocation status: `PASS_REQUESTED_LOCKED_SCALE_PROMPT_ALLOCATION_FEASIBLE`.

Expanded prompt plan doc:

`docs/natural_evidence_v2/R3_2_EXPANDED_PROMPT_PLAN_6144_20260512.md`

Current next allowed action: artifact-only wrapper/precommit repair and local
plan-only validation for the expanded 6,144-row R3.2 route. No Slurm until the
repaired wrapper, allowlist safety, local/remote hashes, and Hermes
notification are reviewed.

The 2026-05-12 20:08Z next-route decision package is recorded.

Decision package:

`docs/natural_evidence_v2/R3_2_NEXT_ROUTE_DECISION_PACKAGE_20260512.md`

Decision:

- do not submit another 96-block R3.2 run from the current 2,048-row eval
  prompt artifact;
- Option A, a 32-block unique diagnostic, is feasible but cannot support a
  96-block locked-scale claim;
- Option B, expanded-prompt 96-block locked scale, is the canonical
  paper-readiness direction and requires at least `6,144` eval rows plus 12
  distinct 512-row shard windows.

Current next allowed action: artifact-only expanded-prompt
planning/implementation. No Slurm submission, aggregation, rerun, Llama, FAR,
sanitizer, or paper-facing claim until the expanded prompt plan and repaired
allocation preflight are reviewed.

The 2026-05-12 20:07Z repaired prompt allocation preflight is recorded.

Added artifact-only preflight script:

`scripts/natural_evidence_v2/plan_r3_2_repaired_prompt_allocation.py`

Preflight result:

- requested R3.2 locked scale: `12` shards x `8` blocks x `64` prompts =
  `96` blocks and `6,144` unique eval prompt rows;
- current prompt artifact has `2,048` eval rows;
- maximum feasible unique package from current prompt artifact: `4` shards,
  `32` blocks;
- current `96`-block locked-scale request is therefore not feasible without
  either prompt-bank expansion or an explicit downscope to a 32-block unique
  diagnostic.

Preflight doc:

`docs/natural_evidence_v2/R3_2_REPAIRED_PROMPT_ALLOCATION_PREFLIGHT_20260512.md`

Machine-readable output:

`results/natural_evidence_v2/status/r3_2_repaired_prompt_allocation_preflight_20260512/r3_2_repaired_prompt_allocation_preflight.json`

Current next allowed action: artifact-only route decision/repair design only:
choose 32-block unique diagnostic or expand the prompt bank before any R3.2
Slurm resubmission. No Slurm until a reviewed route, allowlist safety, and
Hermes notification are recorded.

No Slurm job was submitted or aggregated for this update.

The 2026-05-12 20:00Z repair guard update is recorded.

Added repair decision:

`docs/natural_evidence_v2/R3_2_H200_853430_FAILURE_ATTRIBUTION_AND_REPAIR_DECISION_20260512.md`

Updated aggregate safety behavior:

- `scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py` now
  refuses duplicate selected-prompt-window, generated-output, or decode-row
  hashes by default;
- a local no-write smoke over the `853430` copied artifacts correctly
  hard-failed with `R3_2_DUPLICATE_PROMPT_WINDOWS_REFUSING_AGGREGATION`;
- smoke record:
  `results/natural_evidence_v2/status/r3_2_h200_853430_review/aggregate_guard_smoke/summary.json`.

Current next allowed action: artifact-only repaired prompt allocation preflight
design/implementation only. R3.2 Slurm resubmission remains blocked until prompt
windows are genuinely distinct, uniqueness checks pass, allowlist is safe, and
a fresh reviewed submission route plus Hermes notification are recorded.

No Slurm job was submitted or aggregated for this update.

The 2026-05-12 19:56Z artifact-only failure attribution is now recorded for
H200 array job `853430`.

The main attribution finding is a control-plane/statistical-design issue:
the run nominally reports `39/96` protected accepts, but the 12 shards collapse
to 4 unique deterministic prompt windows repeated 3 times each. Generated
outputs and decode rows have identical hashes within each repeated group, so
the effective unique evidence is `13/32`, not 96 independent blocks.

Unique prompt-window outcomes:

| Window | Repeated shards | Prompt rows | Protected accepts | Protected target hit | Resolved slot rate |
|---|---|---:|---:|---:|---:|
| `window_00` | `shard_00,shard_04,shard_08` | `512-1023` | `8/8` | `0.759` | `0.842` |
| `window_01` | `shard_01,shard_05,shard_09` | `1024-1535` | `3/8` | `0.839` | `0.922` |
| `window_02` | `shard_02,shard_06,shard_10` | `1536-2047` | `1/8` | `0.561` | `0.701` |
| `window_03` | `shard_03,shard_07,shard_11` | `2048-2559` | `1/8` | `0.333` | `0.396` |

The late-window failures are associated with low support/margin and weak
target-hit survival, not null-arm accepts. The `forbidden_public_surface_count`
gate remains failed, but a response-text substring audit shows some ordinary
language substring matches such as `cert` inside `certain` and `owner` in
normal phrases; exact matcher semantics still need an artifact-only audit.

Current next allowed action: artifact-only route decision or repair design
only. Fix prompt allocation so locked-scale uses genuinely distinct prompt
windows, analyze late-window prompt/topic effects, and audit forbidden matcher
semantics. Do not submit, aggregate, rerun, start Llama/FAR/sanitizer work, or
make paper-facing claims until a new reviewed route is recorded.

Attribution report:

`results/natural_evidence_v2/status/r3_2_h200_853430_review/failure_attribution/r3_2_h200_853430_failure_attribution.md`

Hermes-facing attribution report:

`results/natural_evidence_v1/status/hermes_reports/20260512_1556_r3_2_h200_853430_failure_attribution.md`

Hermes TG/email notification was sent successfully for this attribution:

`results/natural_evidence_v1/status/hermes_reports/20260512_1556_r3_2_h200_853430_failure_attribution_notification.json`

The 2026-05-12 19:46Z completion review supersedes the monitor-only state for
H200 array job `853430`.

Slurm outcome:

- `853430_0` through `853430_11` all reached `COMPLETED` with exit code `0:0`;
- reviewed remote output dir:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_h200_array_853430`;
- local review artifacts:
  `results/natural_evidence_v2/status/r3_2_h200_853430_review/`.

Gate outcome:

- protected accepts @64: `39/96`, below the required `>=80/96`;
- raw accepts @64: `0/96`;
- task-only accepts @64: `0/96`;
- wrong-key accepts @64: `0/96`;
- wrong-payload accepts @64: `0/96`;
- min accepted-block support: `6`, below the required `>=16`;
- min accepted-block majority margin: `0`, below the required `>=3`;
- forbidden public surface count: `9`, above the required `0`.

The R3.2 H200 locked-scale gate therefore failed. This is not a Slurm crash,
not a Llama job, and not a payload-diversity result. The most visible pattern
is cyclic by shard window: shards `0/4/8` passed strongly, `1/5/9` were partial,
and `2/6/10` plus `3/7/11` mostly failed.

Current next allowed action: artifact-only failure attribution only. Analyze
cyclic prompt-window/shard effects, per-step support/margin/target-hit
failures, and forbidden-surface matcher semantics. Do not submit, rerun,
aggregate, start Llama, run null/FAR/sanitizer work, or make paper-facing
positive claims from `853430` until a new reviewed route is recorded.

Completion review:

`results/natural_evidence_v2/status/r3_2_h200_853430_review/r3_2_h200_853430_completion_review.md`

Hermes-facing report:

`results/natural_evidence_v1/status/hermes_reports/20260512_1547_r3_2_h200_853430_completion_review.md`

The 2026-05-12 18:16Z Codex/Hermes sync follows the user's instruction to move
R3.2 execution from A100/scavenger to the available H200 node on `pomplun` using
account `cs_yinxin.wan`.

Actions completed:

- installed a project state lock at
  `results/natural_evidence_v2/status/r3_2_state_lock.json`;
- updated Hermes supervision so Codex workers see the project state lock;
- sent TG/email pre-action notification successfully;
- cancelled noncanonical A100 array `853381`;
- created and reviewed H200 shard-array wrapper
  `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`;
- added allowlist entry `v2_r3_2_qwen_locked_scale_shard_array_h200`;
- submitted first H200 attempt `853421`, observed immediate task failure from
  a stale expected selected prompt manifest hash, then cancelled `853421`;
- fixed the expected selected prompt manifest hash to
  `3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1`;
- added precommit lock cleanup to shard-array wrappers;
- submitted corrected H200 array `853430`;
- immediately disabled the allowlist entry after `sbatch` returned;
- revalidated local and remote allowlist safety with zero enabled entries.

Canonical active job:

- job id: `853430`
- job name: `nat-ev-v2-r32h200`
- partition: `pomplun`
- account: `cs_yinxin.wan`
- qos: `pomplun`
- gres: `gpu:h200:1`
- observed node: `chimera21`
- output dir:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_h200_array_853430`

At submission review, `853430_0..5` were running on `chimera21`, and
`853430_[6-11%8]` were pending due `QOSGrpBillingRunMinutes`.

At that time, the next action was to monitor Slurm array job `853430` only.
That action was superseded after the expanded `853524` submission recorded
above.

Submission record:

`results/natural_evidence_v1/status/hermes_reports/20260512_1816_r3_2_h200_submission_record.md`

The 2026-05-12 17:51Z Codex/Hermes sync supersedes the stale
`monitor 853276 only` action. Chimera currently has an active R3.2 shard-array
job `853381` (`nat-ev-v2-r32shard`) on `DGXA100`/`scavenger`: tasks `0..7`
are running and tasks `8..11` are pending behind the array throttle. The
previous recorded array `853276` is terminal/incomplete and is no longer the
actual active job.

No local reviewed submission record for `853381` was found. The only reviewed
local shard-array submission record remains `853276`. Local and remote hashes
still match for the allowlist, R3.2 config, and aggregate sbatch, but the
shard-array sbatch has diverged again:

- local shard-array hash:
  `fe24f12b7944ccf5d131cb1beb6f0b921008d0db60a6673557e1c0fed2a559c6`
- remote shard-array hash used by the running job:
  `a5a64f1f6de5047868e60fddb53ccd9a14ddf8cdbc3aa2bcf6acadd2e6f6ff2f`
- remote selected prompt manifest default:
  `3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1`
- local selected prompt manifest default:
  `71f6ce51fb1e4cfd8ef07fe74e284cf14d16a19651de95aa4b8e717eb1e78820`

Current blocker:

`BLOCK_R3_2_ACTIVE_ARRAY_853381_UNRECORDED_AND_REMOTE_HASH_MISMATCH_NO_AGGREGATE`

No Slurm job was submitted, cancelled, or aggregated during this sync. No
allowlist entry was enabled. The sync report is:

`results/natural_evidence_v1/status/hermes_reports/20260512_1751_r3_2_active_array_853381_sync_blocker.md`

The 2026-05-12 18:01Z artifact-only provenance reconciliation classified
`853381` as an unreviewed external/manual control-plane submission from the
local Codex/Hermes perspective. Slurm reports submit time
`2026-05-12T12:31:53` from `chimerahead:630198`. The shard-array wrapper hash
mismatch is localized to the default
`EXPECTED_SELECTED_PROMPT_MANIFEST_SHA256` line: local reviewed wrapper
`71f6ce51fb1e4cfd8ef07fe74e284cf14d16a19651de95aa4b8e717eb1e78820`,
remote running wrapper
`3e50a08773c4c7dca3be976a762840a8d8a960ac63f4cfce382af3051a2b82d1`.
`853381` remains quarantined and non-canonical; do not aggregate or adopt it
automatically:
`results/natural_evidence_v1/status/hermes_reports/20260512_1801_r3_2_active_array_853381_provenance_reconciliation.md`.

The Hermes scheduled ticks at 05:15 and 05:30 reached a blocker:

`BLOCK_R3_2_SUBMISSION_HARD_CONSTRAINT_CONFLICT`

Reason: the tick requested R3.2 Qwen locked-scale submission but still carried
old hard constraints forbidding all generation and Qwen E2E reruns. That
conflicted with the approved R3.2 route, which necessarily requires reviewed
Qwen locked-scale generation/eval.

This control-plane conflict is now resolved in the Hermes prompt template:
R3.2 Qwen locked-scale generation/eval is allowed only through the reviewed
R3.2 full wrapper, a single enabled allowlist entry, successful TG/email
notification, and exactly one Chimera Slurm job.

The later 05:45 Hermes tick reached a more specific blocker:

`BLOCK_R3_2_FULL_WRAPPER_PAYLOAD_SEMANTICS_AMBIGUOUS_NO_SLURM`

Reason: R3.2 package scope names payload cells `P00/P01/P02/P03`, but the
available reviewed generation/decode path is tied to the single WP5-R2
`a55e` contract. Treating `P00/P01/P02/P03` as distinct payloads or reusing
`a55e` across all labels would both be protocol-significant without an explicit
recorded decision.

The 06:15 Codex update supersedes the earlier cell-label interpretation:

`R3_2_SAME_CONTRACT_LOCKED_SCALE_STABILITY_ROUTE`

Decision: R3.2 is a same-contract `a55e` locked-scale stability package.
`P00/P01/P02/P03` must not be used as payload labels or cell labels in the
canonical R3.2 route. Canonical units are `replicate_group`, `shard_id`, and
`block_id`. Distinct payload evaluation is deferred to R3.4.

The 2026-05-12 03:09Z Codex/Hermes worker performed the final preflight for
the prepared scavenger shard-array submission route and blocked before
allowlist enablement or Slurm submission. Hermes TG/email notification had
succeeded, the local zero-enabled allowlist preflight passed, and Chimera had
no active jobs, but the remote `~/tokenizer-evidence` repo was not synchronized
to the reviewed shard-array route: the shard-array and aggregate sbatch files
were missing remotely, and the remote allowlist/config/precommit hashes differed
from local reviewed files. No allowlist entry was enabled and no Slurm job was
submitted:
`results/natural_evidence_v1/status/hermes_reports/20260512_0309_r3_2_shard_array_remote_sync_blocker.md`.

## Completed Gates

- v1 passive opportunity/global-frame/strict-token-index route is frozen as a
  negative diagnostic.
- Qwen v2 WP3/WP4/WP5 gates passed.
- WP5-R2 teacher-forced gate passed on job `851481`.
- WP6-R2 Option B diagnostic job `852426` passed as a Qwen-only positive
  diagnostic:
  - protected accepts `7/8` at budget `64`
  - raw/task-only/wrong-key/wrong-payload accepts `0/8`
  - min accepted-block support `26`
  - min accepted-block majority margin `5`
  - forbidden public surface count `0`
- R3.0 canonical adoption is recorded.
- R3.1 repeated-coordinate majority decoder spec is recorded.
- R3.2 prompt allocation decision is recorded.
- R3.2 prompt split repair is implemented and plan-only precommit passed under
  the repaired eval-only 4-window allocation:
  `docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_IMPLEMENTATION_20260511_1801.md`.
- R3.2 `852426` replay compatibility is re-reviewed under the repaired prompt
  split contract and remains passing:
  `docs/natural_evidence_v2/R3_2_852426_REPLAY_COMPATIBILITY_REREVIEW_20260511_1817.md`.
- R3.2 same-contract payload semantics are recorded:
  `docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION.md`.
- R3.2 same-contract protocol is recorded:
  `docs/natural_evidence_v2/R3_2_LOCKED_SCALE_PROTOCOL.md`.
- R3.2 plan-only preflight passed under the same-contract schema:
  `results/natural_evidence_v2/status/r3_2_wrapper_preflight_summary.json`.
- R3.2 plan-only wrapper review is recorded.
- R3.2 same-contract `852426` replay path passed exactly:
  `results/natural_evidence_v2/status/r3_2_same_contract_852426_replay_20260511_0630/r3_2_852426_replay_summary.json`.
- R3.2 full same-contract 12-shard wrapper aggregation path is implemented and
  locally plan-validated, without Slurm, allowlist, generation, or claims:
  `docs/natural_evidence_v2/R3_2_FULL_WRAPPER_AGGREGATION_PATH_20260511_0645.md`.
- R3.2 full same-contract wrapper review passed, including exact `852426`
  replay review and local syntax/unit validation, without Slurm, allowlist,
  generation, or claims:
  `docs/natural_evidence_v2/R3_2_FULL_WRAPPER_REVIEW_20260511_0702.md`.
- Standing authorization for the current approved R3 route is recorded and TG +
  email notification succeeded.

## Current Gate

Full R3.2 wrapper review has passed for:


`scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`

The wrapper now has a same-contract 12-shard generation/decode isolation path
and a 96-block aggregate R3.2 gate artifact path. The local `852426` replay
path validates the reviewed single-window WP6-R2 artifacts exactly. After the
failed job `853070`, the R3.2 allowlist entry remains disabled and no further
R3.2 Slurm job may be submitted until the recorded prerequisites below pass.

Latest review record:

`docs/natural_evidence_v2/R3_2_FULL_WRAPPER_REVIEW_20260511_0702.md`

The older `P00/P01/P02/P03` cell-label blocker language remains superseded by
the same-contract `shard_00..shard_11` decision above.

## Submission Gate

R3.2-A allowlist decontamination passed. Local and remote allowlists have zero
enabled entries. The previously unsafe `llama_v2_wp6_e2e_eval` entry is
disabled while `llama_allowed=false`. The reviewed R3.2 entry remains disabled
until the single-job submission tick.

Safety summary:

`results/natural_evidence_v2/status/r3_2a_allowlist_decontamination_summary.json`

Local/remote hash diff:

`results/natural_evidence_v2/status/r3_2a_allowlist_local_remote_diff.md`

## Submitted Job 853070

R3.2-B submitted exactly one Chimera Slurm job after TG/email pre-notice and
after enabling only `v2_r3_2_qwen_locked_scale_eval`.

- job id: `853070`
- job name: `nat-ev-v2-r32qwen`
- partition: `DGXA100`
- final Slurm state: `FAILED`
- elapsed: `00:00:00`
- exit code: `1:0`
- output dir:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_eval_853070`
- submission record:
  `results/natural_evidence_v2/status/r3_2b_submission_record.json`

The allowlist entry was disabled immediately after `sbatch` returned. Local and
remote allowlists again have zero enabled entries.

The job failed before model generation. The wrapper wrote precommit artifacts,
then failed in the first shard precommit decode call because the wrapper used
file rows `0..511` while `decode_wp6_r1_scale_blocks.py` filtered for
`split='wp3_r1_eval'`. In the configured prompt file, file rows `0..511` are
`wp3_r1_dev`, so the selected eval prompt window was empty.

Failure review:
`results/natural_evidence_v2/status/r3_2_qwen_locked_scale_eval_853070/r3_2_job_853070_failure_review.md`.

## Next Allowed Action

Monitor H200 Slurm array job `853524` only.

The canonical active Chimera array is now `853524`. Therefore:

- do not submit another R3.2 Slurm job;
- do not submit the aggregate job;
- do not aggregate until all `853524` shard-array tasks reach terminal state
  and shard outputs are reviewed as complete;
- do not start training, Llama, same-family null, sanitizer, FAR aggregation,
  or paper-facing claims from this state.

Next action:

1. Monitor `853524` with `squeue`/`sacct`.
2. After all tasks are terminal, review all shard outputs for completeness.
3. Only if all 12 shard summaries are present and complete, prepare a separate
   aggregate-only route.

## Gate-Controlled Actions Not Yet Unlocked

- training is conditionally authorized, but locked until a training gate
  explicitly passes and `training_allowed=true`;
- Llama is conditionally authorized, but locked until Qwen R3 gates explicitly
  permit canonical Llama migration and
  `llama_allowed=true`;
- same-family null is conditionally authorized, but locked until Qwen null
  prerequisites pass and
  `same_family_null_allowed=true`;
- sanitizer benchmark is conditionally authorized, but locked until positive
  recovery and required model-family gates explicitly permit it and
  `sanitizer_allowed=true`;
- FAR aggregation or full-FAR claim is conditionally authorized, but locked
  until null/FAR prerequisites pass and
  `far_aggregation_allowed=true`;
- paper-facing positive claim is conditionally authorized, but locked until
  evidence/claim-review gates pass and
  `paper_claim_allowed=true`;
- unreviewed or non-allowlisted generation remains blocked;
- Qwen E2E outside the reviewed R3.2 locked-scale route remains blocked;
- Chimera login-node CPU/GPU work remains blocked.

## Active Jobs

At 2026-05-12T21:40Z:

- active Chimera array: `853524`
- completed tasks at monitor: `853524_0` through `853524_5`
- running tasks at monitor: `853524_6` through `853524_11`
- pending tasks at monitor: none observed
- partition: `pomplun`
- account: `cs_yinxin.wan`
- QOS: `pomplun`
- GRES: `gpu:h200:1`
- command:
  `/home/guanjie.lin001/tokenizer-evidence/scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`
- output dir:
  `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r3_2_qwen_locked_scale_h200_array_853524`

`853381`, `853421`, and `853430` must not be aggregated from this state.
