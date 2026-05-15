# natural_evidence_v2 Current State

Last synchronized: 2026-05-15T05:12:00Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and older long-form
automation notes; they are not the current control entry when they conflict
with this file.

## Canonical Phase

`V2_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_AND_FULL_WRAPPER_REVIEW_PASS_NO_SUBMIT`

## Current Route

Route R4 redesigned positive evidence contract after the candidate-v3
pressure-relaxation-B and transfer-gap dev generation failures (`857795` and
`858019`).

The user has standing authorization for Codex and Hermes to continue already
approved routes without repeatedly asking for per-step approval. This does not
waive gates, allowlist rules, Hermes TG/email notification, Slurm-only
execution on Chimera, H200/pomplun policy, or the one-reviewed-submission rule.

Later training, generation, H200 scoring, Llama, null/FAR, sanitizer, payload
diversity, and paper-facing claim work are conditionally authorized only after
their recorded prerequisite gates pass. They are not permanently forbidden.
Hermes/Codex should continue automatically when a route's prerequisites pass
and should not ask for repeated manual approval on the same clear route. They
are simply not unlocked by this state yet.

## Current Controlling Blocker

`BLOCK_R4_PRESSURE_CONTROLLER_REMOTE_PREFLIGHT_NEXT`

Job `859491` has reached terminal state and has been reviewed. All four H200
array tasks completed with exit code `0:0`, so this is not an infrastructure or
wrapper failure. The positive dev gate failed: protected accepts are `0/32`
under primary `format_scrub=all` and `0/32` without scrub. Raw, task-only,
wrong-key, and wrong-payload controls are also `0/32`, so null controls are
clean. The selectivity prompt policy did elicit support-window events, but not
a protected-selective keyed channel: protected mean events per block are
`9.875`, raw mean events are `9.375`, and task-only mean events are `8.5625`.
Raw max keyed score (`23`) is higher than protected max keyed score (`16`).

A repair / pivot route has now been recorded:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_859491_REPAIR_PIVOT_ROUTE_20260515_0412.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_859491_repair_pivot_route_20260515_0412/`.
The route freezes `859491` as a failed diagnostic and stops unchanged
support-window prompt-policy resubmission. `859491` may be reused only for
failure taxonomy, support/selectivity diagnostics, prompt-policy diagnosis,
forbidden matcher semantics diagnosis, and wrapper/provenance audit. It must
not be used for post-hoc phrase mining, threshold tuning, key/payload remapping,
decoder relaxation, positive reclassification, or paper-facing claims.

The pressure/selectivity pivot package has now passed static validation:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_20260515_0418.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_pivot_package_validation_20260515_0418/`.
It binds failed diagnostics `857795`, `858019`, `859277`, and `859491`, keeps
all compute and claim actions disabled, and preserves H200/pomplun plus
exactly-one allowlist governance for any later compute route.

The route selection has now selected the teacher-forced protected-pressure /
soft-controller scoring route as the first next path:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_SELECTION_20260515_0424.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_route_selection_20260515_0424/`.
The teacher-forced pressure-controller route plan has now passed static
validation:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_20260515_0432.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_route_plan_20260515_0432/`.
The route is Qwen-only, same-contract `a55e`, teacher-forced scoring-only, and
uses the 8192 candidate-v3 prefix-native rows with hash
`d35e5483ce7f6d3d782ce17961b2c407909afc879a12917c5ccc27090f3c80b7`. Focused
tests passed (`10` tests), py-compile passed, and no compute or allowlist
enablement occurred. The plan requires scorer/controller integration review
before any Slurm submission.

The scorer/controller integration review has now passed:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_SCORER_INTEGRATION_REVIEW_20260515_0445.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_scorer_integration_20260515_0445/`.
The teacher-forced scorer now has a disabled-by-default soft-controller path,
records controller config and per-row controller metadata, and leaves base and
task-only conditions untouched. Focused tests passed (`17` passed, `2`
torch-native tests skipped because the local virtual environment has no
`torch`), py-compile passed, and dry-run summaries confirmed no model scoring,
generation, training, Llama, FAR, or paper claim action started. The next
allowed action is artifact-only H200 teacher-forced pressure-controller scoring
wrapper implementation and plan-only validation. No Slurm job, model scoring,
generation, training, Llama, same-family null, sanitizer, FAR aggregation,
payload-diversity work, or paper-facing positive claim is unlocked by this
state.

The H200/pomplun pressure-controller scoring wrapper plan-only review has also
passed:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRAPPER_PLAN_ONLY_20260515_0455.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_wrapper_plan_only_20260515_0455/`.
Wrapper
`scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch`
uses `pomplun`, account `cs_yinxin.wan`, `gpu:h200:1`, `30-00:00:00`, and the
72-cell controller grid. Local syntax and plan-only dry-run passed. The future
allowlist entry
`v2_r4_positive_selectivity_pressure_controller_score_h200` exists but remains
disabled. Full scoring mode still intentionally fails closed with
`R4_PRESSURE_CONTROLLER_FULL_SCORING_REQUIRES_WRONG_CONTROL_WRAPPER_REVIEW`.
The next allowed action is artifact-only wrong-key / wrong-payload controller
mapping design and wrapper review; no Slurm submission or model scoring is
unlocked yet.

The wrong-control mapping and full wrapper review has now passed:
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_REVIEW_20260515_0512.md`
and
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_wrong_control_mapping_20260515_0512/`.
The scorer now supports `--controller-condition-set pressure_controls`, emitting
`base`, `task_only`, `controlled_protected`, `wrong_key_controlled`, and
`wrong_payload_controlled`. The verifier/scorer target remains the committed
target ids in all conditions; wrong controls change only controller pressure
targets. Wrong-payload uses complement ids. Wrong-key uses deterministic
`coordinate_hash_v1` with salt `r4_wrong_key_controller_v1` and row metadata,
not transcripts. The full wrapper path is implemented but requires explicit
`ALLOW_PRESSURE_CONTROLLER_SCORING=1`; without that guard it exits with code
`2`. Focused tests passed (`20` passed, `2` torch-native tests skipped),
route validation passed, wrapper plan-only validation passed, and no Slurm job
or model scoring was started. The next allowed action is remote sync and remote
preflight only: remote wrapper plan-only validation, local/remote hash
preflight, remote zero-enabled allowlist safety, and active-job preflight.

The R4 positive selectivity small dev diagnostic has now been submitted as
exactly one H200/pomplun Slurm array job: `859491`. The authorized command was
`sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch`.
The allowlist entry `v2_r4_positive_selectivity_dev_diagnostic_h200` was
enabled only for submission and disabled immediately after `sbatch` returned.
Local and remote post-submit allowlist safety both passed with zero enabled
entries. First observed state: parent/array job `859491` pending for resources,
with array tasks `859492`, `859493`, and `859494` running on `chimera21`.
Expected output directory:
`/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_dev_diagnostic_859491`.
The next allowed action is monitoring/review of job `859491`; do not submit
another R4 selectivity dev diagnostic job unless this one is reviewed and a new
route decision is recorded.

The selectivity H200 wrapper remote preflight has also passed. Reviewed files
were synchronized to Chimera, remote wrapper plan-only validation passed in
`/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_dev_wrapper_remote_plan_smoke_20260515_0312`,
remote zero-enabled allowlist safety passed, local/remote hashes matched for
the route config, wrapper, decoder, prompt bank, event-window bank, and gate
status files, and no active Chimera jobs were observed for the user. No Slurm
job was submitted, no allowlist entry was enabled, no generation or training
was started, and no claim is unlocked. The next allowed action is to record the
single-submission route, send Hermes TG/email pre-submit notification, enable
exactly `v2_r4_positive_selectivity_dev_diagnostic_h200`, submit exactly one
H200/pomplun Slurm array job, and disable the allowlist entry immediately after
`sbatch` returns.

The selectivity H200 generation/decode wrapper has also passed local plan-only
review. Wrapper
`scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch`
uses H200/pomplun policy, requires explicit `ALLOW_STATIC_DEV_KEYS=1`, validates
the selectivity route config, checks a 512-prompt plan window, and runs a toy
support-window keyed-correlation decode that accepts protected and rejects
wrong-key/wrong-payload controls. Local wrapper syntax passed, wrapper plan-only
status is
`PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY`, and local
zero-enabled allowlist safety passed. The disabled future allowlist entry is
`v2_r4_positive_selectivity_dev_diagnostic_h200`. No Slurm job was submitted, no
allowlist entry was enabled, no generation or training was started, and no claim
is unlocked. The next allowed action is remote sync and remote preflight only:
remote wrapper plan-only validation, local/remote hash preflight, remote
zero-enabled allowlist safety, and active-job preflight.

The selectivity dev diagnostic route scope has also been reviewed and validated
artifact-only. The route config
`configs/natural_evidence_v2/r4_positive_selectivity_dev_diagnostic_route.yaml`
binds the future diagnostic to the validated selectivity package
`r4_positive_selectivity_repair_v1`, the prompt-policy package
`r4_positive_selectivity_prompt_policy_v1`, Qwen-only same-contract `a55e`,
H200/pomplun policy, 32 dev blocks, 64 prompts per block, and
`format_scrub=all` primary decode. The support-window keyed-correlation decoder
and route-scope validator have focused tests (`6 passed`), py-compile passed,
and the route validator wrote
`results/natural_evidence_v2/status/r4_positive_selectivity_dev_diagnostic_route_scope_20260515_0250/route_scope_validation_summary.json`
with status
`PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT`.
No Slurm job was submitted, no allowlist entry was enabled, no generation or
training was started, and no claim is unlocked. The next allowed action is
artifact-only H200 generation/decode wrapper implementation and plan-only
validation for this route.

The R4 positive event-bank full generation/decode wrapper is now implemented
and locally reviewed. Non-plan wrapper mode no longer exits with the old
implementation-pending fail-closed marker. It has an explicit full path:
Qwen generation for `protected`, `raw`, and `task_only`, then keyed
phrase-event decoding under `format_scrub=all` and `format_scrub=none`, with
`wrong_key` and `wrong_payload` decoder controls over protected transcripts.
Focused pytest passed (`13` tests), local plan-only wrapper validation passed,
a synthetic keyed-decoder fixture accepted protected and rejected wrong-key /
wrong-payload controls, and local allowlist safety passed with zero enabled
entries. A disabled allowlist entry `v2_r4_positive_dev_diagnostic_h200` has
been added for the future reviewed route. No Slurm job was submitted, no
generation was started, and no positive claim is allowed. The next
project-advancing action was remote sync, remote plan-only validation with the
full wrapper, local/remote hash preflight, and remote allowlist safety. Those
remote checks have now passed, and the Chimera active-job preflight saw no
active jobs for the user. No Slurm job was submitted, no generation was started,
and no positive claim is allowed. A separate single-submission route has now
been reviewed for exactly one H200/pomplun Slurm array job using allowlist entry
`v2_r4_positive_dev_diagnostic_h200` and command
`sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`.
The route uses static dev keys only for this diagnostic precommit package and
does not create a production or paper-facing keying claim. Final preflight and
Hermes TG/email pre-submit notification passed. Exactly one allowlist entry was
enabled, exactly one H200/pomplun Slurm array job was submitted as job `859277`,
and the allowlist entry was disabled immediately after `sbatch` returned.
Local and remote post-submit allowlist safety both passed with zero enabled
entries. Job `859277` has now reached terminal `COMPLETED` state for all four
array tasks with exit code `0:0`; artifacts and Slurm logs were synced and
reviewed. The wrapper and Slurm run completed cleanly, but the positive dev
gate failed: protected accepts are `0/32` under primary `format_scrub=all` and
`0/32` under no-scrub decode. Raw, task-only, wrong-key, and wrong-payload
controls also have `0/32` accepts. The key failure is that the extractor found
zero frozen phrase events in every block, so protected support, distinct
coordinates, keyed score, and margin were all zero. Forbidden surface hits are
also nonzero (`coordinate: 439`, `bucket: 28` under primary decode), mostly from
ordinary task language such as volunteer coordination and physical buckets.
This matcher issue does not rescue the positive failure because phrase-event
support is absent. Do not resubmit this route unchanged. Artifact-only failure
analysis has now been recorded in
`results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_failure_analysis/`.
The unresolved blocker is the zero-event support mismatch between the frozen
phrase-event bank and free-generation outputs. A reviewed artifact-only repair
/ pivot route has now been recorded in
`docs/natural_evidence_v2/R4_POSITIVE_ZERO_EVENT_SUPPORT_REPAIR_ROUTE_20260514_2056.md`
and
`results/natural_evidence_v2/status/r4_positive_zero_event_support_repair_route_20260514_2056/`.
The route keeps `859277` as a failed diagnostic and permits its outputs only
for failure taxonomy, prompt-policy diagnosis, surface-bank coverage diagnosis,
forbidden matcher semantics diagnosis, and wrapper/provenance audit. It
explicitly forbids post-hoc phrase mining from `859277` into a new locked bank,
threshold tuning to relabel the run, and unchanged route resubmission. The
next allowed action is artifact-only support-gap audit and repair-package
planning only. No Slurm submission, free generation, model scoring, training,
Llama, same-family null, sanitizer, FAR aggregation, payload diversity, or
paper-facing claim is unlocked by this route record.

The support-gap audit has now been executed and recorded in
`results/natural_evidence_v2/status/r4_positive_zero_event_support_gap_audit_20260514_2102/`.
It confirms that exact frozen phrase-event support is absent across all
conditions: protected/raw/task-only exact hits are all `0`, and protected has
only `1` loose-stem hit across `2048` rows. In contrast, bank-first-word opener
overlap is high across all arms (`protected 2032/2048`, `raw 2042/2048`,
`task_only 2046/2048`), which localizes the failure to phrase-specific support
rather than absence of ordinary action language. A repair-package plan has been
recorded in
`docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_REPAIR_PACKAGE_PLAN_20260514_2102.md`
and
`results/natural_evidence_v2/status/r4_positive_support_repair_package_plan_20260514_2102/`.
The next allowed action is artifact-only implementation of the support-repair
contract/extractor/static fixture package. No compute or claim gate is unlocked.

The artifact-only support-repair package has now been implemented and statically
validated in
`results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115/`.
It defines contract `r4_positive_support_repair_v2` and replaces exact
multi-word phrase-only support with precommitted support-window events:
allowed action verb lemma plus allowed task cue lemma within `<= 8` scrubbed
tokens. The bank has `384` independently generated event windows, `8` surface
families, max family fraction `0.125`, and source policy
`independent_static_taxonomy_not_859277_transcripts`. Static fixture validation
passes: toy protected accept is `true` with `26` events and `24` distinct
coordinates, while wrong-key and wrong-payload fixtures both reject. Focused
tests passed (`6` tests). This still does not unlock Slurm, generation, model
scoring, training, Llama, null/FAR, sanitizer, payload diversity, or paper
claims. The next allowed action is artifact-only support-window coverage
dry-run on existing `859277` outputs and static review of whether support is
useful or merely common across all arms.

The support-window coverage dry-run has now been executed on existing `859277`
outputs only and recorded in
`results/natural_evidence_v2/status/r4_positive_support_window_coverage_dry_run_20260514_2144/`.
It confirms the new support contract fixes zero support but fails selectivity:
protected has `22/32` dry-run accept-like blocks, but raw has `12/32` and
task-only has `14/32`. Wrong-key and wrong-payload controls remain `0/32`.
Support rates are high across all arms (`protected 0.936`, `raw 0.842`,
`task_only 0.843`), so the current support-window extractor captures ordinary
task language in unprotected outputs. This is not a positive result and does
not reclassify `859277`; it changes the blocker from zero support to common
support across arms. The next allowed action is artifact-only selectivity
repair planning and static validation only. No compute or claim gate is
unlocked.

The selectivity analysis has now been executed and recorded in
`results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/`
and summarized in
`docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_ANALYSIS_20260514_2149.md`.
It confirms the support-window contract is broad task language rather than a
selective protected channel. Accepted null/control blocks total `26`, with raw
`12/32` and task-only `14/32`; wrong-key and wrong-payload remain `0/32`.
No surface is diagnostic-selective under `859277` (`protected >= 16` events and
raw/task-only `0` events), and the dominant `plan` family accounts for `0.725`
of raw events and `0.727` of task-only events. The next allowed action is a
reviewed artifact-only selectivity repair or pivot route. No Slurm submission,
generation, model scoring, training, Llama, same-family null, sanitizer, FAR
aggregation, payload-diversity work, or paper-facing claim is unlocked.

A reviewed artifact-only selectivity repair route has now been recorded in
`docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_REPAIR_ROUTE_20260514_2154.md`
and
`results/natural_evidence_v2/status/r4_positive_support_window_selectivity_repair_route_20260514_2154/`.
Threshold sensitivity confirms this is not a threshold-only issue: clearing
raw/task-only controls requires a keyed-score threshold around `125`, which
drops protected to `14/32`. The new blocker is
`BLOCK_R4_POSITIVE_SELECTIVITY_REPAIR_PACKAGE_ARTIFACT_ONLY_NEXT`. The next
allowed action is artifact-only implementation and static validation of a
selectivity repair package with independent source policy, no self-cue event
rows, generic raw/task fixture rejection, wrong-key/wrong-payload rejection,
and toy protected fixture acceptance. No compute or claim gate is unlocked.

The artifact-only selectivity repair package has now been implemented and
statically validated in
`results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158/`
and reviewed in
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_REPAIR_PACKAGE_STATIC_VALIDATION_20260514_2158.md`.
It defines contract `r4_positive_selectivity_repair_v1` with `96` independently
sourced event-window rows across `6` families, max family fraction `0.167`,
`0` self-cue rows, toy protected fixture accept `true`, generic raw/task fixture
accept `false`, wrong-key accept `false`, and wrong-payload accept `false`.
This still does not unlock compute or claims. The next allowed action is an
artifact-only coverage/selectivity dry-run of this package on existing failed
`859277` outputs.

That coverage dry-run has now been executed and recorded in
`results/natural_evidence_v2/status/r4_positive_selectivity_repair_package_coverage_dry_run_20260514_2202/`
and reviewed in
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_REPAIR_PACKAGE_COVERAGE_DRY_RUN_20260514_2202.md`.
The null/control problem is gone (`0` accepts for raw/task-only/wrong-key/
wrong-payload), but protected also has `0` accepts because support is too
sparse on the old `859277` outputs: protected row support rate is `0.057`, mean
protected events per dry-run block is `4.45`, and mean distinct coordinates is
`1.79`. The new blocker is
`BLOCK_R4_POSITIVE_SELECTIVITY_PROMPT_POLICY_ELICITATION_ROUTE_NEXT`: the bank is
cleaner but requires a natural prompt-policy elicitation route before any
compute can be reviewed. No compute or claim gate is unlocked.

The artifact-only selectivity prompt-policy package has now been implemented
and statically validated in
`results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/`
and reviewed in
`docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_20260515_0242.md`.
It contains `2048` dev prompts, no duplicate prompt ids, no forbidden prompt
violations, max policy family fraction `0.1669921875`, and `48` expected
fixture events across the six selectivity families. It does not use `859277`
transcripts as source material and starts no compute. The new blocker is
`BLOCK_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_PLANNING_NEXT`: artifact-only
generation/decode route planning and wrapper review for a small H200 dev
diagnostic may proceed, but no Slurm submission is unlocked by this validation
alone.

The reviewed micro-overfit route submitted exactly one H200/pomplun Slurm job.
Job `857458` reached terminal `COMPLETED` state and its protected training plus
teacher-forced surface-mass summaries have been reviewed. The main
teacher-forced gate passed, but stratified review found surface concentration
risk. A disabled-by-default capped target-mass objective patch and capped H200
route review are complete. Local/remote hash preflight, remote wrapper
plan-only validation, remote zero-enabled allowlist safety, active-job preflight,
Hermes TG/email pre-submit notification, and exactly-one allowlist enablement
passed. Exactly one H200/pomplun teacher-forced capped micro-overfit Slurm job
was submitted as job `857611`; the allowlist entry was disabled immediately
after `sbatch` returned and local/remote post-submit allowlist safety passed.
Job `857611` reached terminal `COMPLETED` state with exit code `0:0` and has
been reviewed. The capped run failed the main teacher-forced lift-vs-base gate
while passing the concentration cap. Artifact-only failure analysis recorded a
clean bracket: floor-only job `857458` has enough pressure but too much
concentration; capped job `857611` has clean concentration but is underpowered
by about `0.0198` lift vs base. A rebalanced single-job route is now reviewed
with `TARGET_MASS_CEILING=0.50` and `TARGET_MASS_CEILING_LAMBDA=2.0`. Remote
plan-only validation, remote zero-enabled allowlist safety, active-job
preflight, Hermes TG/email notification, and exactly-one allowlist enablement
passed. Exactly one H200/pomplun rebalanced micro-overfit Slurm job was
submitted as job `857653`; the allowlist was disabled immediately after
`sbatch` returned, and local/remote post-submit allowlist safety passed. Job
`857653` reached terminal `COMPLETED` state with exit code `0:0` and has been
reviewed. It failed the lift-vs-base gate by about `0.00713` while keeping
surface concentration well below the cap. An artifact-only post-rebalance
route decision was recorded, and a bounded pressure-relaxation design is now
recorded with a fixed two-arm ceiling-penalty grid. The grid route has now been
reviewed with a single H200 Slurm array wrapper and a disabled-by-default
allowlist entry. Remote sync, remote wrapper plan-only validation for both arms,
remote zero-enabled allowlist safety, and active-job preflight have passed. The
reviewed single-submission sequence was executed. Hermes TG/email pre-submit
notification succeeded; exactly `v2_r4_candidate_v3_pressure_relaxation_grid_h200`
was enabled; exactly one H200/pomplun Slurm array job was submitted as job
`857764`; the allowlist entry was disabled immediately after `sbatch` returned;
and local/remote post-submit zero-enabled allowlist safety passed. Job `857764`
completed cleanly: both array tasks reached `COMPLETED` with exit code `0:0`.
Both fixed arms passed the teacher-forced surface-mass gate and concentration
cap. Arm `B_ceiling_lambda_0_5` is the strongest by lift vs base. An
artifact-only small dev generation route has been recorded for arm B, with a
separate wrapper and disabled-by-default allowlist entry. Local wrapper syntax,
all-shard plan-only validation, local zero-enabled allowlist safety, remote
all-shard plan-only validation, remote zero-enabled allowlist safety, and
active-job preflight have passed. The reviewed single-submission sequence was
executed: Hermes TG/email pre-submit notification succeeded, exactly
`v2_r4_candidate_v3_pressure_relaxation_b_dev_diagnostic_h200` was enabled,
exactly one H200/pomplun Slurm array job was submitted as job `857795`, the
allowlist was disabled immediately after `sbatch` returned, and local/remote
post-submit zero-enabled allowlist safety passed. Job `857795` completed
cleanly: all four array tasks reached `COMPLETED` with exit code `0:0`.
Artifact review found protected accepts `0/32` under `format_scrub=all` and
`0/32` under no scrub, while raw/task-only/wrong-key/wrong-payload controls
also had `0/32` accepts. Failure analysis localized the main gap: teacher-forced
target pressure did not transfer to free generation because the trained
prefix-native contexts did not appear in generated outputs, and structural
length leakage remained high. An artifact-only transfer-gap repair route has
now been recorded. It requires prefix-context elicitation, free-generation
surface polarity alignment, forbidden matcher semantics, and structural length
leakage controls before any compute path can be reviewed. The reviewed single
transfer-gap diagnostic submission was then executed as Slurm array job
`858019`; all four tasks reached terminal `COMPLETED` state with exit code
`0:0` on `chimera21`. Artifacts were synced and reviewed. The repaired prompt
package still failed the positive dev gate: protected accepts were `0/32` under
`format_scrub=all` and `0/32` under no scrub, with raw/task-only/wrong-key/
wrong-payload controls also at `0/32`. Forbidden public surface counts dropped
relative to job `857795` but remain nonzero (`{'coordinate': 183, 'bucket': 11}`
per scrub mode). Artifact-only failure analysis localized a second failure:
the repair package did not improve protected support versus `857795`
(`18.0 -> 16.25` mean support under scrub-all), protected/wrong-key/
wrong-payload support remained identical, prefix-context hits stayed at zero,
and the prompt policy still induced bullets plus `Next action:` labels despite
the fixed-label/structural-leakage constraints. Do not submit another
prompt-only transfer-gap repair job from this package. An artifact-only route
decision now stops the current prompt-only transfer-gap repair line and requires
artifact-only positive evidence contract redesign review, or an explicit
stop/pivot record, before any additional compute route is eligible for review.
No downstream gate is unlocked. Do not submit another Slurm job or start
downstream training/Llama/null/sanitizer/FAR/payload-diversity/paper-claim work
until a new route is reviewed and its prerequisite gates pass. A standing
conditional-execution sync has also been recorded so Hermes/Codex do not
interpret gate-controlled actions as
permanently forbidden: after a future route's prerequisites pass, Codex/Hermes
may continue without repeated user approval while still enforcing allowlist,
notification, Slurm, H200, and claim gates. A follow-on artifact-only positive
evidence contract redesign review has now recorded the minimum requirements for
any redesigned positive evidence contract. It found the redesign not yet
compute-ready and did not unlock Slurm, generation, Qwen E2E rerun, training,
Llama, same-family null, sanitizer, FAR aggregation, payload-diversity work, or
paper-facing positive claims. Codex then implemented a static redesigned
contract `r4_keyed_correlation_evidence_v1`, added a validator and a toy
keyed-correlation decoder, and validated the static contract locally. This
still did not unlock compute. Codex then built the artifact-only R4 positive
event-bank precommit package for the redesigned keyed-correlation contract. The
package passed static validation with 96 natural phrase events, 8 surface
families, max family fraction 0.125, HMAC mapping coverage over 29/32
coordinates, 35 positive-polarity events, and no exposed key material. This
still did not unlock compute. Codex then recorded an artifact-only dev
diagnostic route scope for the precommitted event bank. The route scope fixes
Qwen-only, same-contract `a55e`, 32 dev blocks, 64 prompts per block, five
conditions, primary `format_scrub=all`, and H200/pomplun submission policy for
a future reviewed route. It also records the future gates, but it is not a
submission approval. The next project-advancing action is artifact-only
event-extractor implementation and generation/decode wrapper plan-only review
for this route. Codex then implemented the artifact-only phrase event
extractor, validated that it strips bullets, numbering, and simple public
action labels under `format_scrub=all`, and verified word-boundary phrase
matching against the frozen surface bank. Codex then added a fail-closed H200
wrapper for the R4 positive dev diagnostic route. The wrapper currently only
supports `VALIDATE_PLAN_ONLY=1`; non-plan full mode exits before generation.
Local bash syntax and plan-only smoke validation passed. The next
project-advancing action was remote plan-only wrapper validation, remote
zero-enabled allowlist safety, and local/remote hash preflight. The first
remote plan-only attempt exposed that `uv` is not on the Chimera PATH, so the
wrapper was repaired to use an explicit `PYTHON_BIN` and the existing Chimera
venv. Remote plan-only validation then passed, remote allowlist safety passed,
and local/remote hashes matched. The wrapper still fails closed outside
`VALIDATE_PLAN_ONLY=1`, so no Slurm submission is unlocked. The next
project-advancing action is full generation/decode wrapper implementation and
review for the precommitted R4 positive dev diagnostic route.

Route record:

- `docs/natural_evidence_v2/R4_CANDIDATE_V3_TRANSFER_GAP_REPAIR_ROUTE_20260514_0658.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_route_20260514_0658/route_decision_summary.json`
- `docs/natural_evidence_v2/R4_CANDIDATE_V3_TRANSFER_GAP_IMPLEMENTATION_PLAN_20260514_0700.md`
- `docs/natural_evidence_v2/R4_AUTONOMOUS_CONDITIONAL_EXECUTION_SYNC_20260514.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_implementation_plan_20260514_0700/implementation_plan_summary.json`
- `configs/natural_evidence_v2/r4_candidate_v3_transfer_gap_repair.yaml`
- `scripts/natural_evidence_v2/validate_r4_transfer_gap_repair_plan.py`
- `tests/natural_evidence_v2/test_r4_transfer_gap_repair_plan.py`
- `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_plan_validation_20260514_0700/transfer_gap_repair_plan_validation_summary.json`
- `docs/natural_evidence_v2/R4_CANDIDATE_V3_TRANSFER_GAP_858019_ROUTE_DECISION_20260514_0822.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_858019_route_decision_20260514_0822/route_decision_summary.json`
- `docs/natural_evidence_v2/R4_POSITIVE_EVIDENCE_CONTRACT_REDESIGN_REVIEW_20260514_0838.md`
- `results/natural_evidence_v2/status/r4_positive_evidence_contract_redesign_review_20260514_0838/contract_redesign_review_summary.json`
- `configs/natural_evidence_v2/r4_positive_evidence_contract_redesign.yaml`
- `scripts/natural_evidence_v2/validate_r4_positive_evidence_contract.py`
- `scripts/natural_evidence_v2/r4_keyed_correlation_decoder.py`
- `configs/natural_evidence_v2/r4_positive_event_bank_precommit.yaml`
- `scripts/natural_evidence_v2/build_r4_positive_event_bank_precommit.py`
- `tests/natural_evidence_v2/test_r4_positive_evidence_contract.py`
- `tests/natural_evidence_v2/test_r4_keyed_correlation_decoder.py`
- `tests/natural_evidence_v2/test_r4_positive_event_bank_precommit.py`
- `docs/natural_evidence_v2/R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_20260514_1545.md`
- `results/natural_evidence_v2/status/r4_positive_evidence_contract_static_validation_20260514_1545/static_validation_summary.json`
- `docs/natural_evidence_v2/R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE_20260514_1605.md`
- `results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605/package_summary.json`
- `configs/natural_evidence_v2/r4_positive_dev_diagnostic_route.yaml`
- `scripts/natural_evidence_v2/validate_r4_positive_dev_diagnostic_route.py`
- `tests/natural_evidence_v2/test_r4_positive_dev_diagnostic_route.py`
- `docs/natural_evidence_v2/R4_POSITIVE_DEV_DIAGNOSTIC_ROUTE_SCOPE_20260514_1612.md`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_route_scope_20260514_1612/route_scope_validation_summary.json`
- `scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py`
- `tests/natural_evidence_v2/test_r4_positive_phrase_event_extractor.py`
- `docs/natural_evidence_v2/R4_POSITIVE_EVENT_EXTRACTOR_STATIC_REVIEW_20260514_1618.md`
- `scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`
- `docs/natural_evidence_v2/R4_POSITIVE_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY_20260514_1622.md`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_wrapper_plan_smoke_20260514/plan_validation/wrapper_plan_only_summary.json`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_remote_plan_smoke_20260514_pybin/plan_validation/wrapper_plan_only_summary.json`
- `results/natural_evidence_v2/status/r4_positive_dev_wrapper_remote_allowlist_safety_20260514.json`
- `results/natural_evidence_v2/status/r4_positive_dev_wrapper_remote_hash_preflight_20260514.md`
- `scripts/natural_evidence_v2/decode_r4_positive_keyed_correlation.py`
- `tests/natural_evidence_v2/test_r4_positive_keyed_correlation_decode.py`
- `docs/natural_evidence_v2/R4_POSITIVE_FULL_GENERATION_DECODE_WRAPPER_IMPLEMENTATION_REVIEW_20260514_1818.md`
- `results/natural_evidence_v2/status/r4_positive_full_generation_decode_wrapper_implementation_review_20260514_1818.json`
- `results/natural_evidence_v2/status/r4_positive_keyed_decoder_fixture_20260514/decode_all/decode_summary.json`
- `results/natural_evidence_v2/status/r4_positive_full_wrapper_allowlist_safety_20260514.json`
- `results/natural_evidence_v2/status/r4_positive_full_wrapper_remote_plan_smoke_20260514/plan_validation/wrapper_plan_only_summary.json`
- `results/natural_evidence_v2/status/r4_positive_full_wrapper_remote_allowlist_safety_20260514.json`
- `results/natural_evidence_v2/status/r4_positive_full_wrapper_remote_hash_preflight_20260514.md`
- `results/natural_evidence_v2/status/r4_positive_full_wrapper_active_job_preflight_20260514.json`
- `docs/natural_evidence_v2/R4_POSITIVE_DEV_DIAGNOSTIC_SINGLE_SUBMISSION_ROUTE_20260514_1930.md`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_single_submission_route_20260514_1930.json`
- `results/natural_evidence_v1/status/hermes_reports/20260514_1930_r4_positive_dev_diagnostic_submission_record.md`
- `results/natural_evidence_v1/status/hermes_reports/20260514_1930_r4_positive_dev_diagnostic_submission_record.json`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_single_enabled_preflight_local_20260514_1930.json`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_single_enabled_preflight_remote_20260514_1930.json`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_post_submit_allowlist_safety_local_20260514_1930.json`
- `results/natural_evidence_v2/status/r4_positive_dev_diagnostic_post_submit_allowlist_safety_remote_20260514_1930.json`
- `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/review.md`
- `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/aggregate_summary.json`
- `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/surface_openers_and_exact_hit_analysis.json`

Positive evidence contract static validation:

- contract id: `r4_keyed_correlation_evidence_v1`
- static validation status:
  `PASS_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE`
- focused pytest for contract + toy decoder: `10 passed`
- `py_compile`: passed
- local allowlist safety: `PASS`
- compute unlocked: `false`
- next allowed action: artifact-only dev diagnostic route review for the
  precommitted event bank

Positive event-bank precommit package:

- package status: `PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE`
- event bank id: `r4_positive_phrase_event_bank_v1`
- payload id: `a55e`
- precommit hash:
  `9ea75e28abf1842e78017fd9100a03fad75ff0b3ad316e5018f94644baf39b30`
- surface count: `96`
- surface families: `8`
- max surface family fraction: `0.125`
- distinct HMAC coordinates covered: `29`
- positive-polarity events: `35`
- key material exposed: `false`
- focused pytest for contract + decoder + event-bank package: `15 passed`
- `py_compile`: passed
- local allowlist safety after package: `PASS`
- compute unlocked: `false`
- next allowed action: artifact-only dev diagnostic route review for the
  precommitted event bank; no Slurm/allowlist enablement until the route and
  prerequisite preflights pass

Positive dev diagnostic route scope:

- route status: `PASS_R4_POSITIVE_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT`
- route id: `r4_positive_event_bank_dev_diagnostic_v1`
- source precommit hash:
  `9ea75e28abf1842e78017fd9100a03fad75ff0b3ad316e5018f94644baf39b30`
- scope: Qwen-only, same-contract `a55e`, dev split, 32 blocks, 64 prompts per
  block, protected/raw/task-only/wrong-key/wrong-payload, primary
  `format_scrub=all`
- future cluster policy: H200 on `pomplun` with account `cs_yinxin.wan` and
  time limit `30-00:00:00`
- focused route tests: `10 passed`
- compute unlocked: `false`
- next allowed action: artifact-only event-extractor implementation and
  generation/decode wrapper plan-only review; no Slurm/allowlist enablement
  until those gates pass

Positive phrase event extractor static review:

- extractor: `scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py`
- status: `PASS_STATIC_EXTRACTOR_REVIEW_NO_COMPUTE`
- focused extractor + decoder tests: `10 passed`
- `py_compile`: passed
- smoke extraction emitted 3 `normalized_phrase_event` rows under
  `format_scrub=all`
- compute unlocked: `false`
- next allowed action: artifact-only generation/decode wrapper plan-only
  implementation and review for the R4 positive dev diagnostic route

Positive dev diagnostic wrapper plan-only review:

- wrapper: `scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`
- local bash syntax: passed
- local plan-only status:
  `PASS_R4_POSITIVE_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY`
- full mode: fail-closed with
  `FULL_R4_POSITIVE_DEV_DIAGNOSTIC_IMPLEMENTATION_PENDING_NO_SUBMIT`
- extractor smoke event count: `3`
- compute unlocked: `false`
- remote plan-only validation: passed after switching wrapper from `uv` to
  explicit `PYTHON_BIN`
- remote allowlist safety: `PASS`
- local/remote hash preflight: `PASS`
- compute unlocked: `false`
- next allowed action: full generation/decode wrapper implementation and
  review for the R4 positive dev diagnostic route

Plan-only validation result:

- status: `PASS_R4_TRANSFER_GAP_REPAIR_PLAN_VALIDATION`
- repair surfaces covered: `4`
- prefix shapes recorded: `8`
- focused pytest: `11 passed`
- local allowlist safety after plan validation: `PASS`
- active Chimera jobs for remote user `guanjie.lin001`: none observed
- no Slurm, generation, training, tokenizer/model scoring, Llama, null,
  sanitizer, FAR, payload-diversity claim, or paper-facing claim was started

Artifact-only repair package result:

- package status: `PASS_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY`
- prompt scaffolds: `6`
- prefix families: `2`
- focused pytest for repair package: `8 passed`
- `py_compile` for repair validators/builders: passed
- local allowlist safety after package build: `PASS`
- active Chimera jobs for remote user `guanjie.lin001`: none observed
- generated package artifacts:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_package_20260514_0705/`
- future H200 diagnostic remains conditionally authorized after a reviewed
  route, local/remote preflights, Hermes TG/email notification, exactly-one
  allowlist enablement, and immediate post-submit allowlist disablement

Submission update:

- route doc:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_TRANSFER_GAP_REPAIR_DEV_DIAGNOSTIC_ROUTE_20260514.md`
- submission record:
  `results/natural_evidence_v2/status/r4_transfer_gap_repair_dev_diagnostic_submission_20260514/submission_record.json`
- job id: `858019`
- job name: `nat-ev-v2-r4tgap`
- partition/account/QoS: `pomplun` / `cs_yinxin.wan` / `pomplun`
- GRES: `gpu:h200:1`
- time limit: `30-00:00:00`
- final observed state: `858019_[0-3]` `COMPLETED` with exit code `0:0` on
  `chimera21`
- post-submit local/remote allowlist safety: `PASS` with zero enabled entries
- review record:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_review/job_858019_review.md`
- review summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_review/job_858019_review_summary.json`
- reviewed status:
  `FAIL_R4_TRANSFER_GAP_REPAIR_DEV_DIAGNOSTIC_NO_PROTECTED_ACCEPTS_NO_DOWNSTREAM_UNLOCK`
- failure analysis:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis.md`
- failure analysis summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis_summary.json`
- next allowed action: artifact-only route decision/review only.
  Do not submit another Slurm job or unlock downstream training/Llama/null/
  sanitizer/FAR/payload-diversity/paper-claim work until a new route is
  reviewed and its prerequisite gates pass.

Local route preflight result:

- status: `PASS_R4_TRANSFER_GAP_REPAIR_DEV_DIAGNOSTIC_LOCAL_PLAN_ONLY_PREFLIGHT`
- wrapper syntax: `bash -n` passed for transfer-gap wrapper and delegated R4 dev wrapper
- local plan-only validation: all four shards passed
- prompt bank sha256:
  `cf77458604b46e7842e097ad8ee8f26b60596d2d93e3a93b9e3399f2b717a1cb`
- local zero-enabled allowlist safety after preflight: `PASS`
- no remote preflight, Slurm submission, generation, training, tokenizer/model
  scoring, Llama, null, sanitizer, FAR, payload-diversity claim, or
  paper-facing claim was started
- next action: remote plan-only preflight and remote zero-enabled allowlist
  safety for the reviewed H200 route; submit only after that passes and Hermes
  TG/email notification plus exactly-one allowlist enablement are complete

## Latest Trusted Facts

- Candidate v3 actual Qwen tokenizer boundary preflight job `856443` passed:
  checked rows `8192`, failed rows `0`, target/other overlap rows `0`.
- Candidate v3 H200 teacher-forced surface-mass scoring job `856453`
  completed but failed the gate: protected lift vs base
  `0.05289504316422722`, protected lift vs task-only
  `0.0560544523594233`, protected rank1 `0.654296875`, protected median
  margin `0.01057571533601731`.
- Adapter gain sweep job `856994` completed on H200/pomplun and showed that
  gain can pass the teacher-forced gate: first main pass at gain `2.0`; first
  main plus per-prefix protection pass at gain `4.0`; task-only lift vs base
  `-0.0031594091951960834`.
- Noncanonical gain-4 generation array `857015` was cancelled after a state
  conflict. Its outputs must not be adopted, decoded, aggregated, or used as
  evidence.
- Artifact-only micro-overfit route scope review found the old WP5 wrapper is
  not R4-launch-ready because it used A100/scavenger, old WP5 rows, old primary
  bank, old bucket scorer, and protected plus task-only training together.
- Micro-overfit H200 job `857458` completed on `chimera21` in `00:04:36` with
  exit code `0:0`; review recorded a teacher-forced surface gate pass at
  protected adapter gain `1.0`: protected lift vs base
  `0.39194896617371455`, protected lift vs task-only
  `0.39510837536891064`, protected rank1 `1.0`, protected median margin
  `0.355754891585093`, task-only lift vs base
  `-0.0031594091951960834`.
- Supplementary stratified review found max surface mean target mass
  `0.643060527741909`, above the earlier concentration diagnostic cap `0.50`;
  therefore this is a teacher-forced pass with concentration risk, not a direct
  generation unlock.

## New Artifact-Only Implementation

Created in this synchronization:

- `scripts/natural_evidence_v2/build_r4_candidate_v3_micro_overfit_split.py`
- `tests/natural_evidence_v2/test_r4_candidate_v3_micro_overfit_split.py`
- `scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`
- disabled allowlist entry:
  `v2_r4_candidate_v3_micro_overfit_h200`
- R4 trainer row mode:
  `--row-mode r4_prefix_native_surface`

Built split artifacts:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/train_rows.jsonl`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/heldout_rows.jsonl`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/score_rows.jsonl`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex/split_summary.json`

Split summary:

- source candidate rows sha256:
  `d35e5483ce7f6d3d782ce17961b2c407909afc879a12917c5ccc27090f3c80b7`
- train rows: `512`
- heldout rows: `512`
- score rows: `8192`
- duplicate keys: `0`
- train/heldout overlap: `0`
- train and heldout each cover all `32` strata with `16` rows per stratum
- all locked-action booleans remain `false`

Validation completed locally without tokenizer/model loading, CUDA, remote CPU
or GPU use, Slurm submission, training, scoring, generation, or claims:

- focused pytest: `12 passed`
- `py_compile` for split builder, trainer, and R4 scorer: passed
- H200 wrapper syntax: `bash -n` passed
- H200 wrapper plan-only mode: passed and exited before model/tokenizer loading
- allowlist safety with zero enabled entries: passed

## Artifact Review

Artifact-only review recorded:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_artifact_review_20260514_0305/artifact_review.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_artifact_review_20260514_0305/artifact_review_summary.json`

Local review result:

- status: `PASS_R4_CANDIDATE_V3_MICRO_OVERFIT_ARTIFACT_REVIEW_NO_COMPUTE`
- focused pytest with the repo venv: `9 passed`
- `py_compile`: passed
- H200 wrapper syntax: passed
- H200 wrapper plan-only mode: passed and exited before compute paths
- allowlist safety with zero enabled entries: passed
- remote sync/hash preflight was not started in this tick

## Remote Sync And Submission

Remote sync and local/remote hash preflight passed:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_remote_sync_hash_preflight_20260514_0310/remote_sync_hash_preflight_summary.json`
- status: `PASS_REMOTE_SYNC_HASH_PREFLIGHT`
- file count: `15`
- mismatch count: `0`

Remote wrapper plan-only check passed and exited before model/tokenizer loading,
CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm
submission.

Single-job submission route recorded:

- `docs/natural_evidence_v2/R4_CANDIDATE_V3_MICRO_OVERFIT_H200_SUBMISSION_ROUTE_20260514.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_submission_route_20260514/submission_route_summary.json`

Hermes TG/email pre-submit notification succeeded:

- `results/natural_evidence_v1/status/hermes_reports/20260514_0315_r4_micro_overfit_h200_presubmit_notification.json`

Submitted exactly one H200 job:

- job id: `857458`
- job name: `nat-ev-v2-r4mof`
- partition/account/QoS: `pomplun` / `cs_yinxin.wan` / `pomplun`
- GRES: `gpu:h200:1`
- wrapper:
  `scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`
- submission record:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_submission_20260514_0315/submission_record.md`

The allowlist entry `v2_r4_candidate_v3_micro_overfit_h200` was disabled
immediately after `sbatch` returned. Local and remote post-submit allowlist
safety checks passed with zero enabled entries.

## Terminal Review

Slurm job `857458` review recorded:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458_review/job_857458_review.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458_review/job_857458_review_summary.json`
- status:
  `PASS_R4_CANDIDATE_V3_MICRO_OVERFIT_JOB_857458_TEACHER_FORCED_REVIEW`
- protected train summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458_review/wp5_micro_slot_lora_train_summary.json`
- teacher-forced surface-mass summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458_review/r4_teacher_forced_surface_mass_summary.json`

This is a teacher-forced diagnostic pass only and is not downstream evidence for
FAR, ownership, robustness, cross-family behavior, payload diversity, or
paper-facing positive claims.

Supplementary concentration review recorded:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458/review/micro_overfit_857458_review.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_857458/review/micro_overfit_857458_review_summary.json`
- status:
  `PASS_TEACHER_FORCED_SURFACE_GATE_WITH_SURFACE_CONCENTRATION_RISK_NO_GENERATION`
- max surface mean target mass: `0.643060527741909`
- concentration cap diagnostic: `0.50`
- cap status: `FAIL`

Concentration route decision recorded:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_concentration_route_decision_20260514/concentration_route_decision.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_concentration_route_decision_20260514/concentration_route_decision_summary.json`
- decision: do not start generation directly from `857458`; prepare a
  capped/regularized micro-overfit repair route first.

Capped objective patch recorded:

- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_capped_objective_patch_20260514/capped_objective_patch.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_capped_objective_patch_20260514/capped_objective_patch_summary.json`
- status: `ARTIFACT_ONLY_CAPPED_OBJECTIVE_PATCH_VALIDATED_NO_COMPUTE`
- new disabled-by-default knobs:
  `--target-mass-ceiling`, `--target-mass-ceiling-lambda`
- focused validation: `21 passed`, `py_compile` passed, wrapper `bash -n`
  passed, allowlist zero-enabled safety passed.
- capped H200 route review:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_CAPPED_MICRO_OVERFIT_H200_ROUTE_20260514.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_route_20260514/route_review_summary.json`
- wrapper route parameters:
  `TARGET_MASS_FLOOR=0.20`,
  `TARGET_MASS_FLOOR_LAMBDA=5.0`,
  `TARGET_MASS_CEILING=0.45`,
  `TARGET_MASS_CEILING_LAMBDA=5.0`,
  `STRATUM_WEIGHTING_MODE=r4_candidate_v3_failed_surface`,
  `STRATUM_WEIGHT_MAX=3.0`
- capped route local validation: wrapper syntax passed, wrapper plan-only mode
  passed, `py_compile` passed, focused pytest `21 passed`, zero-enabled
  allowlist safety passed
- capped route remote preflight:
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_remote_sync_hash_preflight_20260514/remote_sync_hash_preflight_summary.json`
- remote preflight status:
  `PASS_REMOTE_SYNC_HASH_AND_WRAPPER_PREFLIGHT`
- local/remote file count: `18`
- mismatch count: `0`
- missing count: `0`
- remote wrapper plan-only: passed
- remote allowlist zero-enabled safety: passed
- active Chimera jobs before submission: `0`
- Hermes 04:05 worker recorded a lower-permission consistency review only and
  did not submit Slurm:
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_route_consistency_review_20260514_0405/route_consistency_review_summary.json`
- capped H200 submission record:
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_submission_20260514_0410/submission_record.json`
- submitted job: `857611`
- post-submit local allowlist safety: `PASS`
- post-submit remote allowlist safety: `PASS`
- first observed Slurm state after submission: `RUNNING` on `chimera21`
- terminal review:
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_857611_review/job_857611_review_summary.json`
- terminal status: `COMPLETED`, exit code `0:0`, elapsed `00:04:27`
- teacher-forced surface gate status: `FAIL`
- protected lift vs base: `0.13019710095503`, required `>= 0.15`
- protected lift vs task-only: `0.1333565101502261`
- protected rank1: `0.990234375`
- protected median margin: `0.11016895354259759`
- max surface mean target mass: `0.22298632306046784`, cap `0.50`, status
  `PASS`
- capped failure analysis:
  `results/natural_evidence_v2/status/r4_candidate_v3_capped_micro_overfit_failure_analysis_20260514/failure_analysis_summary.json`
- comparison with floor-only job `857458`: floor-only had enough target pressure
  but failed concentration; capped `857611` passed concentration but missed
  lift vs base by about `0.0198`
- rebalanced route review:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_REBALANCED_MICRO_OVERFIT_H200_ROUTE_20260514.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_rebalanced_micro_overfit_route_20260514/route_review_summary.json`
- rebalanced route values: `TARGET_MASS_CEILING=0.50`,
  `TARGET_MASS_CEILING_LAMBDA=2.0`, with the same floor and stratum weighting
- rebalanced remote preflight:
  `results/natural_evidence_v2/status/r4_candidate_v3_rebalanced_micro_overfit_remote_preflight_20260514/remote_preflight_summary.json`
- remote plan-only: passed
- remote allowlist zero-enabled safety: passed
- active jobs before submission: `0`
- rebalanced H200 submission record:
  `results/natural_evidence_v2/status/r4_candidate_v3_rebalanced_micro_overfit_submission_20260514_0427/submission_record.json`
- submitted job: `857653`
- first observed Slurm state after submission: `RUNNING` on `chimera21`
- terminal review:
  `results/natural_evidence_v2/status/r4_candidate_v3_rebalanced_micro_overfit_857653_review/job_857653_review_summary.json`
- terminal status: `COMPLETED`, exit code `0:0`, elapsed `00:04:20`
- teacher-forced surface gate status: `FAIL`
- protected lift vs base: `0.14287435650001612`, required `>= 0.15`
- protected lift vs task-only: `0.1460337656952122`
- protected rank1: `1.0`
- protected median margin: `0.13510848174337298`
- max surface mean target mass: `0.190783511439804`, cap `0.50`, status
  `PASS`
- post-rebalance route decision:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_POST_REBALANCE_ROUTE_DECISION_20260514_0437.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_post_rebalance_route_decision_20260514_0437/route_decision_summary.json`
- pressure-relaxation design:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_PRESSURE_RELAXATION_DESIGN_20260514_0450.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_pressure_relaxation_design_20260514_0450/pressure_relaxation_design_summary.json`
- pressure-relaxation grid route review:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_PRESSURE_RELAXATION_GRID_H200_ROUTE_20260514.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_pressure_relaxation_grid_route_20260514_0500/route_review_summary.json`
- grid wrapper:
  `scripts/natural_evidence_v2/slurm/r4_candidate_v3_pressure_relaxation_grid_h200.sbatch`
- disabled allowlist entry:
  `v2_r4_candidate_v3_pressure_relaxation_grid_h200`
- grid job `857764` review:
  `results/natural_evidence_v2/status/r4_candidate_v3_pressure_relaxation_grid_857764_review/review_summary.json`
- review status:
  `PASS_AT_LEAST_ONE_ARM_TEACHER_FORCED_GATE_NO_GENERATION`
- passing arms: `A_ceiling_lambda_1_0`, `B_ceiling_lambda_0_5`
- selected generation-route arm: `B_ceiling_lambda_0_5`
- artifact-only generation-route review:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_PRESSURE_RELAXATION_B_DEV_GENERATION_ROUTE_20260514.md`
  and
  `results/natural_evidence_v2/status/r4_candidate_v3_pressure_relaxation_b_dev_generation_route_20260514_0522/route_review_summary.json`
- local route validation:
  `results/natural_evidence_v2/status/r4_candidate_v3_pressure_relaxation_b_dev_generation_route_20260514_0522/local_plan_validation_summary.json`
- local wrapper syntax: passed
- local all-shard plan-only validation: passed for shards `0..3`
- local zero-enabled allowlist safety: passed
- disabled allowlist entry:
  `v2_r4_candidate_v3_pressure_relaxation_b_dev_diagnostic_h200`

## Current Allowed Action

Artifact-only scorer/controller integration review and patch planning for the
teacher-forced pressure-controller route. Do not resubmit `859491` unchanged.
Do not use `859491` transcripts for post-hoc surface mining or threshold
relaxation. No Slurm, generation, model scoring, training, Llama, same-family
null, sanitizer benchmark, FAR aggregation, payload-diversity work, or
paper-facing positive claim is unlocked by the current state.

## Conditionally Authorized After Later Gates Pass

- generation / Qwen E2E rerun
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim

The user has already granted standing permission to proceed automatically once
the relevant prerequisite route gates pass. These actions are not permanently
forbidden, but they require a later reviewed route, allowlist safety, and
Hermes/Codex state sync before execution.

## Chimera Policy

Future Chimera tokenizer/model/CPU/GPU work must run through reviewed Slurm
routes. GPU work must default to H200 on `pomplun` with account
`cs_yinxin.wan`, QoS `pomplun`, `--gres=gpu:h200:1`, and max time
`30-00:00:00`, unless a later recorded route decision overrides this.
