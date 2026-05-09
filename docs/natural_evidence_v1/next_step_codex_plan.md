# Codex Next Step Plan: v1 Frozen, v2 Controlled Micro-Slots

## Expert decision 2026-05-08

The v1 passive opportunity / global frame / strict token-index route is frozen.
Do not continue the v1 repaired target-mass probe, v1 E2E, v1 Llama, v1
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing positive
claim. v1 is preserved as a completed negative diagnostic, not as proof that
natural tokenizer-aligned evidence is impossible.

Immediate completed artifacts:

```text
docs/natural_evidence_v1/V1_NEGATIVE_DIAGNOSTIC_SUMMARY.md
results/natural_evidence_v1/status/final_v1_negative_decision.json
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml
results/natural_evidence_v2/status/gate_status.json
results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/split_manifest.json
results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/forbidden_surface_audit.json
docs/natural_evidence_v2/WP3_MICRO_SLOT_DETECTOR_BUCKET_POLICY.md
results/natural_evidence_v2/status/wp3_micro_slot_policy_design_20260508_2140/wp3_micro_slot_policy_design_summary.json
scripts/natural_evidence_v2/build_wp3_detector_bank_scaffold.py
results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153/wp3_detector_bank_scaffold_summary.json
scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py
results/natural_evidence_v2/status/wp3_fixed_artifact_audit_20260508_2223/wp3_fixed_artifact_audit_summary.json
results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_20260508_2238/wp3_fixed_artifact_audit_summary.json
results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_850228/wp3_fixed_artifact_audit_summary.json
results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/wp3_detector_bank_scaffold_summary.json
results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_850242/wp3_fixed_artifact_audit_summary.json
results/natural_evidence_v2/status/wp3_template_density_responses_20260508_2321/template_response_summary.json
results/natural_evidence_v2/status/wp3_template_density_responses_balanced_20260508_2331/template_response_summary.json
results/natural_evidence_v2/status/wp3_template_density_audit_850276/wp3_fixed_artifact_audit_summary.json
results/natural_evidence_v1/status/hermes_reports/20260508_2309_wp3_fixed_response_density_review.md
scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
results/natural_evidence_v2/status/wp3_context_mass_plan_20260508_2324/qwen_v2_wp3_context_mass_score_plan_summary.json
scripts/natural_evidence_v2/score_wp3_context_mass.py
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.out
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.err
results/natural_evidence_v1/status/hermes_reports/20260509_0014_wp3_context_mass_job_850372_failed.md
results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/qwen_v2_wp3_context_mass_score_plan_summary.json
results/natural_evidence_v1/status/hermes_reports/20260509_0024_wp3_context_mass_prefix_boundary_repair_prepared.md
results/natural_evidence_v1/status/hermes_reports/20260509_0054_wp3_context_mass_local_validation_repaired.md
scripts/natural_evidence_v2/build_wp3_restricted_step_label_density_audit_plan.py
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_AUDIT_PLAN.md
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/restricted_step_label_density_audit_summary.json
scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_WRAPPER_REVIEW.md
results/natural_evidence_v1/status/hermes_reports/20260509_0112_wp3_restricted_density_wrapper_review.md
results/natural_evidence_v1/status/hermes_reports/20260509_0112_wp3_restricted_density_wrapper_review.json
```

New primary route:

```text
natural_evidence_v2_controlled_micro_slots
```

Current next allowed action:

```text
Monitor Slurm job 850434. After it completes, sync and review the restricted
Step-label base-Qwen model-output density artifacts and manual naturalness
examples. Do not start WP4, training, Qwen E2E, Llama, same-family null,
sanitizer, FAR, or positive paper claims.
```

The intended v2 route is:

1. WP2: controlled-natural prompt families and split files. COMPLETE as
   deterministic artifacts, with public prompt-text forbidden-surface audit
   passing at `0.0`.
2. WP3: micro-slot detector and 2-way bucket policy. DESIGN, SCAFFOLD, AND
   FIXED-ARTIFACT AUDIT ENTRYPOINT RECORDED. Slurm job `850228` found five
   multi-token carriers; repaired job `850242` passed configured-tokenizer
   stability with `35/35` single-token surfaces. Slurm job `850276` reviewed a
   template-only fixed-response density preflight with
   `density_gate_status=TEMPLATE_PREFLIGHT_PASS`. This is not a model-output
   density gate. Job `850288` failed the generic fixed-prefix full-vocab
   model-mass gate. Context-specific Slurm job `850372` validated the 230-row
   plan and loaded Qwen, but failed before score artifacts because the scorer
   detected prefix-boundary retokenization for plan row
   `0f8383dd9775def36e16` and surface `also`. A repaired plan/scorer now uses
   shared longest-token-prefix boundary handling. The 2026-05-09T00:42Z review
   found local pytest validation stale for the current scorer API
   (`skip_invalid` is now required). The 2026-05-09T00:54Z repair updated the
   model-free test call and reran local no-model validation successfully
   (`py_compile`, `bash -n`, `--validate-plan-only`, and focused pytest all
   passed). The GPU allowlist is still disabled pending a later explicit
   allowlist review. The restricted step-label route now has an artifact-only
   256-prompt model-output density audit plan and a reviewed Slurm wrapper. The
   wrapper was approved, explicitly submitted once as Slurm job `850434`, and
   the allowlist entry was disabled immediately afterward to prevent duplicate
   submissions. The job is a base-Qwen model-output density audit only.
3. WP4: prompt-local small payload contract and decoder oracle substitution.
4. WP5: teacher-forced target-mass gate.
5. WP6: Qwen v2 proof-of-life E2E only if WP5 passes.
6. WP8/WP9: Llama, same-family nulls, and sanitizer only after Qwen v2 recovery
   and null rejection.

All Chimera CPU/GPU work must use Slurm. Do not run CPU work directly on the
Chimera login node.

## Current state (read-only summary)

## Hermes 15-minute supervision

Hermes supervises Codex every 15 minutes using the protocol in:

```text
docs/natural_evidence_v1/hermes_15min_coordination.md
```

The active Hermes cron job is `d65af4b36d84`
(`natural-evidence-v1-codex`, every 15 minutes). It runs
`/Users/guanjie/.hermes/scripts/natural_evidence_v1_codex_tick.sh`, which sends
TG/email through `scripts/natural_evidence_v1/hermes_notify.py` and launches a
background Codex worker through
`scripts/natural_evidence_v1/hermes_supervision_tick.py`.

Hermes is not the executor. Hermes monitors state, checks Slurm, prompts Codex
with the next allowed action, and blocks unsafe actions. Codex executes any file
edits, artifact analysis, Slurm submissions, artifact review, and state updates.
Each 15-minute Hermes tick should request at most one Codex state-changing
action. Any Chimera CPU/GPU work must be submitted by Codex through Slurm.
Every Hermes tick that pushes the project forward must notify the user through
both Telegram and email before Codex executes the requested action. Hermes must
use `scripts/natural_evidence_v1/hermes_notify.py --channels telegram,email
--strict` and record the notification JSON. The helper loads
`/Users/guanjie/.hermes/.env` by default; that file may provide
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_HOME_CHANNEL`, `EMAIL_HOME_ADDRESS`,
`EMAIL_SMTP_HOST`, `EMAIL_SMTP_PORT`, `EMAIL_ADDRESS`, and `EMAIL_PASSWORD`.
If either channel is not configured or fails, Hermes must stop forward
prompting and record a notification blocker instead of silently continuing.
Training, model transcript generation, Qwen E2E rerun, Llama, same-family
null, sanitizer, FAR aggregation, and paper-facing positive claims remain
forbidden in the current phase.

The current Codex action queue supervised by Hermes has been replaced by the v2
controlled micro-slot route:

1. WP2 artifact-only controlled-natural prompt family scaffold and
   forbidden-surface audit is complete for the configured split counts;
2. WP3 micro-slot detector and 2-way bucket policy design plus scaffold are
   recorded, but WP3 density, tokenizer stability, and 2-way bucket-bank mass
   audits still need results;
3. WP4 prompt-local small payload contract only after WP3 gates.

Do not execute any stale v1 repaired target-mass action. The repaired design and
Slurm-scored probe are complete, and the negative score review rejected repaired
dataset or training preflight from job `848547`.

- Completed Slurm job: `846699`
- Completed provenance Slurm job: `847630`
- Completed frame replay Slurm job: `847634`
- Completed oracle schedule Slurm job: `847640`
- Completed on-policy survival Slurm job: `847644`
- Completed protected-vs-task-only lift Slurm job: `847649`
- Job name: `nat-ev-qwen-nat-eval`
- Runtime: `05:46:56`
- Final Slurm state: `COMPLETED`, ExitCode `0:0`
- Remote eval output dir: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1/qwen_natural_e2e_pilot/eval/qwen_natural_e2e_eval_846627_recovery`
- Final summary JSON exists and reports:
  - `status=EVAL_COMPLETE_QWEN_NATURAL_VARIABLE_RADIX_NOT_PAPER_CLAIM`
  - `eval_started=true`
  - `not_full_far=true`
  - `null_accept_count=0`
  - `protected_accept_count=0`
  - `diagnostic_recovery_observed=false`
- Counts:
  - `generated_output_count=18432`
  - `observation_count=372216`
  - `decode_row_count=120`
- Provenance-normalized summary:
  - path:
    `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_recovery/observation_erasure_summary_846699.json`
  - `status=PASS_EXPLAINED_RECOVERY_DIR_NAME`
  - `source_job_id=846699`
  - `source_path_job_id_candidates=["846627"]`
  - `provenance_mismatches=[]`
  - `compatible_variable_radix_digit_rows=1885`
  - `erasure_reason_rows=370331`
- Frame completion replay:
  - path:
    `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_frame_completion_replay/qwen_846699_frame_completion_replay_summary.json`
  - `status=COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE`
  - `observed_complete_frame_count_total=0`
  - `max_observed_slots_per_frame_global=1`
  - `scheduled_complete_frame_count_no_erasure_total=5370`
  - `decode_rows_with_scheduled_complete_frames_no_erasure=120`
- Oracle schedule simulation:
  - path:
    `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_oracle_schedule_simulation/qwen_846699_oracle_schedule_simulation_summary.json`
  - `status=COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME`
  - `any_subset_observed_complete_frames_total=0`
  - `max_any_subset_observed_slots_per_frame=2`
  - `greedy_scheduled_complete_frames_no_erasure_total=5370`
  - `max_greedy_probability_at_least_one_complete_iid_selected_p=5.791901064855407e-54`
- On-policy survival by slot/source:
  - path:
    `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_on_policy_survival/qwen_846699_on_policy_survival_summary.json`
  - `status=COMPLETE_ON_POLICY_SURVIVAL_DIAGNOSTIC`
  - `compatible_hit_rows=1885/372216`
  - `compatible_hit_rate=0.0050642637608270466`
  - `target_hit_rows=299/143160`
  - `target_hit_rate=0.0020885722268790164`
  - `bucket_miss_rows=370331`
  - `token_index_out_of_response_rows=0`
  - `metadata_missing_rows=0`
  - `oracle_summary_json.provenance_mismatches=[]`
- Protected-vs-task-only lift by slice:
  - path:
    `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_protected_vs_task_only_lift/qwen_846699_protected_vs_task_only_lift_summary.json`
  - `status=COMPLETE_PROTECTED_VS_TASK_ONLY_LIFT_DIAGNOSTIC`
  - protected compatible-hit rate=`0.0047499301480860576`
  - task-only compatible-hit rate=`0.0061644314054205085`
  - protected target-hit rate=`0.0023924280525286393`
  - task-only target-hit rate=`0.0018510757194747135`
  - protected-vs-task-only target delta=`0.00054135233305392574`
  - protected-vs-task-only compatible delta=`-0.0014145012573344509`
- Progress JSON reports all expected units completed:
  - `raw`
  - `protected_trained_P0421_seed17`
  - `task_only_lora_P0421_seed17`
  - `protected_trained_P0421_seed23`
  - `task_only_lora_P0421_seed23`
  - `protected_trained_P1729_seed17`
  - `task_only_lora_P1729_seed17`
  - `protected_trained_P1729_seed23`
  - `task_only_lora_P1729_seed23`

## Critical negative signal

- Remote decode trace has `120` rows.
- All rows have `decode_status=insufficient_symbols`.
- All rows have `usable_symbols=0`.
- All rows have `accepted=False`.

## Interpretation

This is a negative scientific signal for the current natural-evidence five-arm evaluator path:

- the five-arm eval job completed successfully
- all trained/raw/null arms were evaluated
- but no payload recovery or useful symbol accumulation was produced

The current result is therefore:

- operational success
- scientific failure for payload recovery under the current variable-radix eval path

## Expert decision incorporated

The expert review accepts the post-846699 diagnosis as a hard decision:

- Job 846699 is not a provider/model failure.
- Job 846699 is not a payload-codec arithmetic failure.
- The result exposes two independent bottlenecks:
  - **frame observability**: the eval unit is a unique prompt response, while
    payload frames were cut over a global ordered position stream.
  - **symbol survival**: only 1885 of 372216 observations produced compatible
    variable-radix digits, with erasures dominated by
    `observed_token_not_in_variable_radix_bucket_set`.
- More training steps are not the next action. Training cannot repair frame
  scatter, and the current protected survival rate is below 1%.
- The current strict token-index anchor is not acceptable as the final
  natural-output protocol.

Forbidden until the artifact-only diagnosis is complete:

- no new Qwen training rerun
- no E2E rerun
- no Llama run
- no same-family null
- no sanitizer benchmark
- no paper-facing positive claim

## Repo constraints from AGENTS.md

- do not make speculative abstraction changes
- prefer surgical edits
- do not overwrite old generated result artifacts
- do not make paper claims
- do not claim full FAR on incomplete/partial traces

## Recommended Codex objective

Do **not** keep blindly re-running the same five-arm eval.

The next useful Codex work is artifact-only diagnosis in this order:

1. **Normalize artifact provenance.** COMPLETE.
   - Generate local `observation_erasure_summary_846699.json`.
   - Record source job id, source remote path, row count, condition counts, file
     hash, decode trace row count, and join keys.
   - Explain why the 846699 summary points to the reused remote output directory
     named `qwen_natural_e2e_eval_846627_recovery`.
   - If provenance cannot be reconciled, mark erasure statistics provisional.

2. **Frame completion replay.** COMPLETE.
   - Replay the exact variable-radix frame grouping from existing observations.
   - Report complete frame count, max observed slots per frame, closest frames,
     and observed-slot distribution by condition, payload, and budget.

   Result: no observed complete frames under the completed transcript. However,
   the current schedule can complete frames under no-erasure, so the replay
   refines the bottleneck toward symbol survival / strict-anchor drift rather
   than a purely impossible current schedule.

3. **Oracle schedule simulation.** COMPLETE.
   - Oracle survival: assume every committed observation hits the correct bucket.
   - Oracle scheduler: choose the best prompt subset under budgets 64/128/256/512.
   - Oracle protected lift: resample with measured protected survival to estimate
     realistic frame-completion probability.

   Result: greedy no-erasure schedules can complete frames, but no prompt subset
   can complete any frame using the actual observed survived digits in 846699.
   Schedule-only repair cannot recover this completed transcript.

4. **On-policy survival by slot and row source.** COMPLETE.
   - Break down survival by condition, payload, seed, prompt slot, token index,
     frame digit index, radix/arity, source condition, and token class.
   - Include examples of expected bucket tokens versus observed token text.

5. **Protected-vs-task-only lift.** COMPLETE.
   - Report aggregate and per-slice lift.
   - Determine whether lift is broad or concentrated in a few prompts/tokens.

6. **Teacher-forced bucket-mass probe.** COMPLETE.
   - Score base, protected, and task-only checkpoints at committed prefixes.
   - Report target bucket probability, non-target compatible bucket mass,
     target bucket rank, and target-vs-other margin.

   Result: Slurm job `847652` completed 0:0 on DGXA100/chimera13. Protected
   LoRA has only a small teacher-forced target-mass lift: protected mean target
   candidate mass `0.410354`, base `0.406997`, task-only `0.405440`; protected
   minus base is `+0.003357`, and protected minus task-only is `+0.004914`.
   Target rank-1 rates are similarly close: protected `0.413488`, base
   `0.410659`, task-only `0.408022`.

7. **Decoder oracle substitution.** COMPLETE.
   - Replace observations with target bucket digits under the current schedule.
   - Test whether the current evaluator can recover if every committed position
     emits the correct bucket.

   Result: local artifact-only oracle substitution completed in
   `results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_decoder_oracle_substitution/`.
   Protected rows accept `16/16`, wrong-payload rows accept `0/16`, and
   eligible-position mismatches are `0`. The evaluator/frame schedule is
   formally decodable under target-digit oracle substitution.

Decision rules:

- If decoder oracle substitution fails, redesign the compiler/evaluator frame
  contract.
- Since decoder oracle substitution passed but observed survival fails, repair
  anchor/training/survival before any E2E rerun.
- If teacher-forced target mass is low, inspect the training objective, loss
  mask, LoRA setup, and dataset contract.
- If teacher-forced target mass is high but free survival is low, strict
  token-index anchoring and free-generation drift are the bottleneck.

Protocol repair candidates after diagnostics:

- Do not continue global repeated-payload frame assignment as the main protocol.
- Evaluate frame-aware prompt-bundle scheduling.
- Evaluate sparse coordinate-level erasure-code or rateless-style recovery for
  sparse natural observations.
- Use prompt-local full-frame only if eligible density can support 27-36 digits
  in one response; current data does not.

## Suggested Codex prompt

Use Codex in this repo for one of the following narrow goals:

### Option A (completed)
Generated local provenance-normalized
`observation_erasure_summary_846699.json` from the completed 846699 artifacts.
The provenance status is `PASS_EXPLAINED_RECOVERY_DIR_NAME`.

### Option B (completed)
Added a diagnosis-only frame replay script under `scripts/natural_evidence_v1/`
and ran it through Slurm job `847634`.

### Option C (completed)
Added a diagnosis-only oracle schedule simulation script under
`scripts/natural_evidence_v1/` and ran it through Slurm job `847640`.

### Option D (completed)
Added a diagnosis-only on-policy survival script under
`scripts/natural_evidence_v1/` and ran it through Slurm job `847644`.
The result confirms sub-1% compatible survival and sub-1% target-hit survival,
with erasure dominated by `observed_token_not_in_variable_radix_bucket_set`.

### Option E (completed)
Added a diagnosis-only protected-vs-task-only lift script under
`scripts/natural_evidence_v1/` and ran it through Slurm job `847649`.
The result is mixed and still negative for launch: protected target-hit rate is
higher than task-only in aggregate, but protected compatible-hit rate is lower,
and target survival remains far below 1%.

### Option F (completed)
Added a diagnosis-only teacher-forced bucket-mass probe and ran it through Slurm
job `847652`. The result shows only a small teacher-forced target-mass lift for
protected LoRA over base and task-only, so it does not justify new training or a
Qwen E2E rerun.

### Option G (completed)
Ran decoder oracle substitution. The evaluator recovers under committed
target-digit substitution, so the frame schedule is not the remaining primary
blocker under no erasure.

### Option H (completed)
Wrote the post-846699 protocol repair decision:

```text
docs/natural_evidence_v1/post_846699_protocol_repair_decision.md
```

It explicitly chooses the next repair target from the evidence:

- strict token-index anchor/free-generation drift;
- weak teacher-forced target-bucket mass;
- sparse symbol survival;
- coding abstraction for sparse known-coordinate observations.

The decision selects prefix-conditioned observed-text eligible selection plus
anchor/survival repair as the immediate direction. It treats frame-aware prompt
bundles as a diagnostic baseline, and sparse coordinate-level erasure coding as
the preferred coding repair after event survival is measurable.

### Option I (completed)
Implemented and ran Phase R1 prefix-conditioned selector replay through Slurm
job `847879`. The job completed 0:0 and wrote:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_prefix_conditioned_selector_replay/
```

At budget 512, `exact_full` replay scheduled `35840` events, rediscovered
`11582` prefixes, produced `11122` compatible hits, and `4681` target hits.
The less strict `suffix_8` replay scheduled `35840` events, rediscovered
`12652` prefixes, produced `12109` compatible hits, and `5080` target hits.

Interpretation: prefix-conditioned replay recovers many raw actual-prefix
events and confirms strict token-index anchoring is too brittle. However, this
does not create an ownership signal: raw target-hit rates are high, protected
arms remain low, and task-only is often above protected. These coordinate-level
target hits are not payload recovery and not FAR.

### Option J (completed)
Ran artifact-only R1 interpretation and selector-contract repair analysis:

- quantify protected-minus-raw and protected-minus-task-only coordinate lift;
- report null/multiple-testing risk for prefix-conditioned replay;
- decide whether prefix-conditioned selection needs branch-aware or regenerated
  suffix training targets before any new training;
- decide whether sparse coordinate-level coding is required instead of complete
  frame recovery.

Result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_r1_selector_contract_analysis/
docs/natural_evidence_v1/r1_selector_contract_repair_analysis.md
```

Across 64 protected-vs-null comparison slices, protected has positive lift over
raw in `0/64` rows and positive lift over task-only in `0/64` rows. Status:
`COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS`.

### Option K (completed)
Designed an artifact-only selector precommit contract and branch-aware/
regenerated-suffix training-target preflight. Outputs:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_selector_contract_preflight/
docs/natural_evidence_v1/selector_contract_training_target_preflight.md
```

Status:

```text
COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY
```

The draft selector contract is not active:

```text
DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT
```

The plan explicitly keeps direct replay verifier, training, generation, and E2E
rerun blocked. Required next artifacts are branch-aware compatibility,
regenerated/local-suffix repair, repaired teacher-forced target-mass probe,
sparse-coordinate synthetic preflight, and fresh lockbox/locked replay.

### Option L (completed)
Prepare artifact-only branch-aware compatibility and regenerated/local-suffix
repair diagnostics under the draft selector contract. If the diagnostic needs
Chimera CPU/GPU work, submit it as a Slurm job only. Do not run CPU work
directly on Chimera login nodes.

Result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation/
```

Status:

```text
COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED
```

The preparation wrote `68` planned branch-aware scoring rows and `68`
regenerated/local-suffix repair input examples. It did not score models,
regenerate suffixes, train, generate, rerun E2E, decode payload recovery, or
estimate FAR. Drift reasons in the prepared set are
`compatible_non_target=60` and `observed_token_not_candidate_set=8`.

Limitation: the prepared plan rows are currently raw-only
(`model_condition_counts.raw=68`). That is useful for null/branch-aware
diagnostic scaffolding, but protected/task-only branch-aware comparison still
requires a richer R1 example export or an expanded replay-example selection.

### Option M (completed)
Constructed an artifact-only local-suffix repair dry-run from the prepared
repair examples.

Result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run/
```

Status:

```text
COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED
```

The dry-run processed `68` repair examples. `36/68` rows were ready for
approximate text-level local substitution, while `32/68` require
tokenizer-aligned or branch-aware regeneration because the observed token text
could not be found in the original response text. This is not compatibility
scoring and not a usable training artifact. The prepared set is still raw-only
(`model_condition_counts.raw=68`), so protected/task-only comparison remains
blocked.

### Option N (completed)
Export/select richer protected and task-only examples for the branch-aware
diagnostic before scoring.

Implementation:

```text
scripts/natural_evidence_v1/export_balanced_branch_aware_examples.py
scripts/natural_evidence_v1/slurm/qwen_balanced_branch_aware_examples.sbatch
```

Slurm job:

```text
job_id=848405
job_name=nat-ev-qwen-babr
state=COMPLETED
exit_code=0:0
elapsed=00:00:46
checked_at=2026-05-08T05:02:35Z
```

The job is artifact-only. It reads existing generated transcripts, train
metadata, and bucketized candidates, then exports balanced examples across
protected, task-only, and raw conditions. It does not score a model, regenerate
suffixes, train, generate, rerun E2E, decode payload recovery, or estimate FAR.

Result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_balanced_branch_aware_examples/
```

The export selected `768` examples with condition counts
`protected_trained=288`, `task_only_lora=288`, and `raw=192`. All selected rows
include generated transcript response text.

### Option O (completed)
Sync and review the balanced export artifacts:

```text
balanced_branch_aware_example_export_summary.json
prefix_conditioned_selector_replay_examples.jsonl
balanced_branch_aware_examples.csv
balanced_branch_aware_examples_by_slice.csv
```

Then use those richer examples to prepare branch-aware compatibility and
regenerated/local-suffix repair diagnostics.

Balanced preparation result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation_balanced/
```

Status:

```text
COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED
```

Counts:

- planned branch-aware rows: `209`
- regenerated/local-suffix repair examples: `209`
- condition counts: `protected_trained=76`, `task_only_lora=59`, `raw=74`
- drift reasons: `compatible_non_target=68`,
  `observed_bucket_not_compatible=79`,
  `observed_token_not_candidate_set=62`

Balanced local-suffix dry-run result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run_balanced/
```

Status:

```text
COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED
```

All `209/209` rows are text-substitution-ready in the generated transcript
response text. This is still only a dry-run, not branch-aware compatibility
scoring, naturalness scoring, training data approval, payload recovery, or FAR.

### Option P (completed)
Ran a Slurm-scored branch-aware compatibility diagnostic from the balanced
scoring plan. The job used Qwen/Qwen2.5-7B-Instruct as a reference-model
naturalness/NLL proxy. It did not train, generate branch continuations,
regenerate suffixes, rerun E2E, decode payload recovery, or estimate FAR.

Implementation:

```text
scripts/natural_evidence_v1/score_branch_aware_compatibility.py
scripts/natural_evidence_v1/slurm/qwen_branch_aware_compatibility.sbatch
```

Slurm job:

```text
job_id=848414
job_name=nat-ev-qwen-brscore
state=COMPLETED
exit_code=0:0
elapsed=00:00:55
```

Result:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_compatibility_scored_balanced/
```

Summary:

- scored rows: `209`
- branch-aware proxy pass: `153/209` (`0.7321`)
- response naturalness proxy pass: `155/209` (`0.7416`)
- suffix-preserving proxy pass: `169/209` (`0.8086`)
- mean response delta NLL per token: `0.318643`
- mean suffix delta NLL per token: `0.554276`

By condition:

| Condition | Rows | Branch-aware proxy pass |
|---|---:|---:|
| protected_trained | 76 | 57 (`0.7500`) |
| raw | 74 | 52 (`0.7027`) |
| task_only_lora | 59 | 44 (`0.7458`) |

Interpretation: the local target substitutions are often plausible under the
model-scored proxy, especially for `compatible_non_target` drift rows. However,
protected is essentially tied with task-only, so this still does not establish
a protected-specific ownership signal or justify new training/E2E.

### Option Q (completed)
Ran artifact-only branch-aware score interpretation and repaired training-target
preflight. Outputs:

```text
scripts/natural_evidence_v1/analyze_branch_aware_score_interpretation.py
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_score_interpretation/
```

Status:

```text
COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING
```

Summary:

- scored rows: `209`
- primary repaired target-mass probe candidates: `75`
- secondary or ablation candidates: `78`
- rejected rows: `56`
- primary candidates by condition: protected=`39`, task-only=`20`, raw=`16`
- primary candidates by drift reason:
  `compatible_non_target=58`, `observed_token_not_candidate_set=17`
- primary candidates by token class: word=`74`, function_word=`1`

Decision:

```text
PRIMARY_CANDIDATES_EXIST_BUT_NO_TRAINING_GATE_PROTECTED_CONTROL_SEPARATION_WEAK
```

Interpretation: there are usable candidate rows for a future repaired
teacher-forced target-mass probe, especially word-token `compatible_non_target`
repairs. However, this does not unlock training because protected-vs-control
separation remains weak or low-N in the slice comparison.

### Option R (completed)
Designed an artifact-only repaired teacher-forced target-mass probe over the
primary branch-aware candidates. Outputs:

```text
scripts/natural_evidence_v1/design_repaired_teacher_forced_target_mass_probe.py
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/
```

Status:

```text
COMPLETE_REPAIRED_TEACHER_FORCED_TARGET_MASS_PROBE_DESIGN_NOT_SCORED
```

Summary:

- primary branch-aware candidates: `75`
- one-to-one bucket joins to balanced examples: `75/75`
- planned score rows: `257`
- planned base rows: `75`
- planned protected rows: `91`
- planned task-only rows: `91`
- candidate conditions: protected=`39`, task-only=`20`, raw=`16`
- drift reasons: `compatible_non_target=58`,
  `observed_token_not_candidate_set=17`
- token classes: word=`74`, function_word=`1`

The design defines:

- base / protected / task-only checkpoint arms;
- repaired scoring prefix as `prompt + prefix_before_observed`;
- target bucket and full bucket-token inputs joined from existing artifacts;
- non-target compatible bucket mass;
- required slice outputs by payload, seed, source condition, drift reason,
  token class, and prompt id;
- pass/fail thresholds requiring protected-base and protected-task-only target
  candidate mass lifts of at least `+0.05`, plus rank-1 and slice stability
  checks.

This was design only. It did not load a model, score probabilities, generate
text, train, rerun E2E, recover payloads, estimate FAR, or make paper-facing
claims.

### Option S (completed)
Submitted one Slurm-scored repaired teacher-forced target-mass probe from the
Option R scoring plan as job `848547` (`nat-ev-qwen-rtfprob`) at
2026-05-08T07:50:20Z. Job `848547` completed 0:0 in 00:01:35 and scored all
257 planned rows. The decision review is complete and recorded in:

```text
results/natural_evidence_v1/status/hermes_reports/20260508_0811_repaired_target_mass_score_review.md
```

Review answers:

- Which token classes and drift reasons are actually repairable?
- Does any protected-vs-task-only slice show meaningful separation after
  branch-aware proxy filtering?
- Which repaired rows are safe candidates for a future teacher-forced
  target-mass probe?
- Does the next repaired target require local suffix substitution,
  regenerated suffixes, or prefix-conditioned coordinate coding?
- Does protected target mass clear the predeclared lift thresholds over both
  base and task-only controls?
- Model scoring used Slurm; no training or E2E was started.

Result:

- `threshold_pass=false`
- protected-base target candidate mass lift:
  `-0.007645810655699581`
- protected-task-only target candidate mass lift:
  `-0.04776975171334799`
- protected-task-only target rank-1 lift:
  `-0.03296703296703296`
- mean target candidate mass: protected `0.09654275872091375`, base
  `0.10418856937661333`, task-only `0.14431251043426174`

Do not train, generate protected outputs for E2E, rerun E2E, launch
Llama/same-family/sanitizer, or make paper-facing positive claims.
The review rejects repaired dataset or training preflight from job `848547`.
Stop positive-E2E progression from this repaired target-mass path unless a new
artifact-only negative-diagnosis/root-cause plan or user/expert review is
explicitly requested.

## natural_evidence_v2 WP3 Current Plan

### Completed: repaired configured-tokenizer audit

The v2 WP3 tokenizer-stability blocker from Slurm job `850228` has been
addressed. The scaffold now removes/replaces the five Qwen configured-tokenizer
multi-token surfaces:

- `moreover`
- `further`
- `generally`
- `therefore`
- `meanwhile`

Repaired scaffold:

```text
results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/
```

Slurm audit:

```text
job_id=850242
job_name=nat-ev-v2-wp3aud
state=COMPLETED
exit_code=0:0
node=chimera13
```

Result:

- `configured_tokenizer_used=true`
- `tokenizer_stability_status=PASS`
- `unstable_token_count=0`
- `unstable_token_rate=0.0`
- `candidate_surface_count=35`
- `density_gate_status=NOT_EVALUATED`
- `mass_gate_status=NOT_EVALUATED`
- `wp4_allowed=false`

Interpretation: WP3 tokenizer stability now passes for the repaired scaffold,
but WP3 as a whole does not pass because response density and model-mass
artifacts have not been evaluated.

### Completed: template fixed-response density preflight review

Existing template fixed-response density artifacts were reviewed from Hermes
tick `20260508_2309`.

Template response artifacts:

```text
results/natural_evidence_v2/status/wp3_template_density_responses_20260508_2321/
results/natural_evidence_v2/status/wp3_template_density_responses_balanced_20260508_2331/
```

Slurm density preflight:

```text
job_id=850276
job_name=nat-ev-v2-wp3aud
state=COMPLETED
exit_code=0:0
node=chimera13
```

Result:

- `configured_tokenizer_used=true`
- `tokenizer_stability_status=PASS`
- `density_gate_status=TEMPLATE_PREFLIGHT_PASS`
- `prompt_coverage=1.0`
- `average_micro_slots_per_response=35.0`
- `candidate_micro_slot_rows=8960`
- `forbidden_surface_rate=0.0`
- `mass_gate_status=NOT_EVALUATED`
- `wp4_allowed=false`

Interpretation: the reviewed density result is a template-only fixed-response
preflight. It is not a model-output density gate and does not unlock WP4,
training, E2E, FAR aggregation, or positive claims. Fixed model-mass artifacts
are still missing.

### Completed: balanced template density preflight

The unbalanced F1-only template density artifact has been superseded by a
balanced template artifact and Slurm audit.

Template response artifact:

```text
results/natural_evidence_v2/status/wp3_template_density_responses_balanced_20260508_2331/
```

Slurm density preflight:

```text
job_id=850278
job_name=nat-ev-v2-wp3aud
state=COMPLETED
exit_code=0:0
node=chimera13
```

Result:

- `template_preflight_only=true`
- `density_gate_status=TEMPLATE_PREFLIGHT_PASS`
- `total_responses=256`
- family balance: `64` rows each for F1/F2/F3/F4
- `prompt_coverage=1.0`
- `average_micro_slots_per_response=30.25`
- `median_micro_slots_per_response=31.5`
- `candidate_micro_slot_rows=7744`
- `wp4_allowed=false`

Interpretation: the detector can see high-density micro-slot opportunities in
controlled template responses across all four families. This is still
template-only and must not be treated as fixed model-output density.

### In progress: fixed-prefix model-mass scoring

Added:

```text
scripts/natural_evidence_v2/score_wp3_bucket_mass.py
scripts/natural_evidence_v2/slurm/wp3_bucket_mass_score.sbatch
```

Submitted:

```text
job_id=850288
job_name=nat-ev-v2-wp3mass
state=PENDING(Resources)
```

Scope: base Qwen fixed-prefix next-token bucket-mass scoring for the repaired
2-way banks, followed by the existing WP3 mass audit. This job does not
generate text, train, run E2E, estimate FAR, or make claims.

Result:

```text
job_id=850288
state=COMPLETED
exit_code=0:0
mass_gate_status=FAIL
wp4_allowed=false
```

Outputs:

```text
results/natural_evidence_v2/status/wp3_bucket_mass_score_850288/
results/natural_evidence_v2/status/wp3_model_mass_audit_850288/
```

Failure summary:

- 7/7 banks failed the configured full-vocab mass gate.
- `min_bucket_mass` required: `0.005`
- observed full-vocab minima:
  - sentence opener: `4.05e-09`
  - step opener: `7.57e-09`
  - discourse marker: `4.58e-09`
  - optional hedge: `3.31e-07`
  - transition: `4.89e-08`
  - conjunction: `8.53e-09`
  - preposition: `3.44e-07`
- candidate-normalized masses are more balanced for several banks, but that is
  diagnostic only and does not satisfy the current configured mass gate.

Interpretation: the repaired surfaces are tokenizer-stable and the template
detector can find dense slots, but current fixed-prefix contexts/bucket surfaces
do not have enough raw next-token probability mass under base Qwen. This blocks
WP4 and training.

### Context-specific mass plan prepared

Added:

```text
scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
results/natural_evidence_v2/status/wp3_context_mass_plan_20260508_2324/
```

The plan joins the balanced template detections back to response text, extracts
`prefix_before_candidate`, validates spans and response hashes, and records
lowercase and sentence-case bucket variants separately. It contains `230`
unique planned scoring rows from `7744` eligible template detections:
`115` lowercase and `115` sentence-case rows. This is not model scoring and does
not change any gate.

### Context-specific Slurm scorer prepared

Added:

```text
scripts/natural_evidence_v2/score_wp3_context_mass.py
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

The scorer consumes
`results/natural_evidence_v2/status/wp3_context_mass_plan_20260508_2324/qwen_v2_wp3_context_mass_score_plan.jsonl`,
scores `bucket_surfaces` at `prefix_before_candidate`, keeps casing variants
separate, and writes context score, mass, and audit artifacts. Local static and
plan checks passed.

### Context-specific Slurm job failed

Submitted exactly one allowlisted Chimera Slurm job, `850372`
(`nat-ev-v2-wp3ctxm`), using
`scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch`. The job ran
on `chimera13` and failed `1:0` after 00:00:39. It produced no context-score,
mass, audit, or summary artifacts. The stdout log records
`PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION` for `230` plan rows; the stderr log
fails at plan row `0f8383dd9775def36e16` because the tokenizer retokenized the
prefix boundary for surface `also`.

Synced logs:

```text
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.out
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.err
```

### Hermes operational note

The `20260508_2324` Hermes worker did advance this step, but its `codex exec`
child buffered output and held the single-worker lock without visible progress.
After inspection, the silent child was terminated and the lock was released. The
context-specific mass plan it wrote was validated and recorded as complete. To
reduce future stalls, Hermes Codex worker default timeout is now `900` seconds
instead of `7200`, and stale-lock slack is now `+300` seconds instead of
`+1800`.

### Immediate next allowed action

Monitor Slurm job `850398`. When it completes, sync output artifacts and review
valid scored rows, invalid tokenization rows, passing step-local banks, and
density implications. The GPU allowlist is disabled again pending `850398`
review; do not submit another Slurm scoring job until this job completes and is
reviewed. Do not run CPU/GPU scoring directly on a Chimera login node.

### Current WP3 repair state

Prepared repair artifacts:

```text
results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/
results/natural_evidence_v1/status/hermes_reports/20260509_0024_wp3_context_mass_prefix_boundary_repair_prepared.md
```

The repaired scorer uses shared longest-token-prefix boundary handling when
configured-tokenizer `prefix + surface` tokenization merges across the original
character boundary. It rejects any row whose bucket surfaces do not share one
adjusted scoring prefix or whose candidate continuation is not exactly one next
token. Local no-model validation passed; configured-Qwen tokenizer-only
validation is implemented in the wrapper but still needs review/allowlisting
before any new Slurm scoring job.

### Step-local expansion state

User/expert direction replaced the old review-only state. The NVIDIA-assisted
repair route produced one passing seed in job `850384`:

```text
passing_bank=step_opener_action_sentence_case_v1
min_bucket_mass=0.0057856489
mass_ratio=2.5349
```

From that seed, Codex built the artifact-only step-local expansion plan:

```text
results/natural_evidence_v2/status/wp3_step_local_expansion_plan_20260508_2038/
score_plan_rows=72
candidate_bank_count=24
prefix_families=[step_label, numbered_list, dash_bullet]
```

Structural density feasibility is recorded in:

```text
results/natural_evidence_v2/status/wp3_step_local_expansion_plan_20260508_2038/qwen_v2_wp3_step_local_density_feasibility.json
```

It says a step-opener-only policy needs a sixteen-step/list response or
additional non-step slots to satisfy the `>=16` average micro-slot density
gate. This is structural only, not model-output density.

Slurm job `850394` failed during tokenizer-only validation because `Inspect` is
not one Qwen next token. The scorer/wrapper were repaired so invalid
tokenization rows are recorded and skipped rather than crashing the whole audit.
Replacement job:

```text
job_id=850398
job_name=nat-ev-v2-wp3ctxm
state=COMPLETED
exit_code=0:0
scope=base-Qwen mass audit with tokenizer-invalid rows skipped and recorded
```

Result:

```text
score_plan_rows=72
context_score_rows=63
invalid_tokenization_rows=9
mass_rows=21
mass_gate_status=FAIL
```

Review:

```text
docs/natural_evidence_v2/WP3_STEP_LOCAL_EXPANSION_REVIEW.md
results/natural_evidence_v2/status/wp3_step_local_expansion_mass_score_850398/
```

Two `Step N: ` sentence-case action-verb banks passed:

```text
step_local_step_label_seed_check_review_choose_make_v1
  side0=[Check, Review]
  side1=[Choose, Make]
  min_bucket_mass=0.0100467710
  ratio=1.8203

step_local_step_label_start_begin_create_set_v1
  side0=[Start, Begin]
  side1=[Create, Set]
  min_bucket_mass=0.0071791444
  ratio=3.8920
```

The overall WP3 gate still fails because density is only structural and most
candidate banks failed or were invalid. Next allowed action: artifact-only
restricted step-label policy construction from the two passing banks, plus
density audit planning for either a sixteen-step/list prompt family or an
8-step family augmented with additional non-step slots. WP4/training remain
blocked.

### Restricted step-label policy ready

Codex built the artifact-only restricted policy:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_policy_20260508_2049/
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_POLICY_REVIEW.md
```

The policy keeps only two mass-passing banks:

```text
restricted_step_label_check_review_choose_make_v1
  side0=[Check, Review]
  side1=[Choose, Make]

restricted_step_label_start_begin_create_set_v1
  side0=[Start, Begin]
  side1=[Create, Set]
```

Allowed prefixes:

```text
Step 1:
...
Step 16:
```

Density decision:

```text
recommended=A_16_step_checklist_step_label_only
blocked=B_8_step_plus_extra_slots
```

Reason: the 16-step route uses only mass-validated banks and is structurally
capable of `16` step-label slots per response. The 8-step-plus-extra route is
blocked because current non-step banks have not passed the full-vocabulary
mass gate.

Immediate next action: prepare a model-output density audit plan for the
restricted 16-step route. The plan must test whether base Qwen follows
`Step 1:` through `Step 16:` and whether the restricted detector finds at least
`16` eligible step-label slots per response. Any model generation/scoring must
be explicitly reviewed and use Chimera Slurm. WP4 and training remain blocked.

### Still forbidden

- no training
- no generation of protected transcripts
- no Qwen E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claim

## Explicit non-goals for Codex right now

- do not start new training
- do not run Qwen E2E again
- do not run Llama, same-family null, or sanitizer benchmark
- do not create a new protocol family before artifact-only diagnostics are done
- do not rewrite the whole verifier
- do not add speculative framework layers
- do not touch frozen decision records
- do not overwrite old result artifacts
- do not modify paper-facing positive claims
