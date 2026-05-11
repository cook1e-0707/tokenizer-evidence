# natural_evidence_v2 Current State

Last synchronized: 2026-05-11T19:31Z

## Canonical Phase

`V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT`

This compact state file is the first file Codex/Hermes should read for routine
ticks. Use the long historical files only when this file is ambiguous:

- `docs/natural_evidence_v1/AUTOMATION_STATE.md`
- `docs/natural_evidence_v1/next_step_codex_plan.md`
- `results/natural_evidence_v1/status/gate_status.json`
- `results/natural_evidence_v2/status/gate_status.json`

## Current Route

Route R3: Qwen v2 controlled-natural micro-slot paper-readiness route.

The user has recorded standing authorization for Codex and Hermes to continue
the already approved R3 route without repeatedly asking for explicit approval.
This does not waive gates, allowlist requirements, notification requirements,
or the one-Slurm-job-per-reviewed-submission limit.

The user has also recorded conditional authorization for later-stage training,
Llama, FAR/null expansion, sanitizer, and paper-claim work after their
prerequisite gates pass. These work classes are not permanently forbidden.
They are gate-controlled: each class must first have a recorded route decision,
passing gate evidence, reviewed wrapper or artifact plan, allowlist where
applicable, and TG/email notification before any state-changing action.

## Latest Hermes Sync

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

Artifact-only R3.2 prompt split repair is recorded:

`docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_CONTRACT_REPAIR_20260511_1747.md`

The repaired allocation is now implemented in the precommit builder, wrapper
split calls, and config. Local plan-only precommit passed:

`results/natural_evidence_v2/status/r3_2_prompt_split_repair_precommit_20260511_1801`

`852426` replay compatibility has now been re-reviewed under the repaired R3.2
prompt split contract:

`results/natural_evidence_v2/status/r3_2_852426_replay_compatibility_rereview_20260511_1817.json`

R3.2 allowlist safety has now been rechecked again under the repaired prompt
split contract for the 2026-05-11 19:16Z Hermes tick, and a later single-job
submission route is recorded:

`docs/natural_evidence_v2/R3_2_REPAIRED_PROMPT_SPLIT_ALLOWLIST_RECHECK_AND_ROUTE_20260511_1916.md`

`results/natural_evidence_v2/status/r3_2_allowlist_recheck_repaired_prompt_split_20260511_1916.json`

`results/natural_evidence_v2/status/r3_2_repaired_prompt_split_single_job_route_20260511_1916.json`

Next action: stop this tick without Slurm submission. On a later notified
submission tick only, enable exactly `v2_r3_2_qwen_locked_scale_eval`, submit
exactly one reviewed R3.2 Chimera Slurm job, immediately disable the allowlist
entry after `sbatch` returns, record the submission, and stop.

The 2026-05-11 19:31Z Hermes worker failed before taking a project action
because the Codex CLI was not found on PATH. This did not change experiment
state and did not submit Slurm.

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

No active Chimera Slurm job is expected for R3.2 after job `853070` failed
immediately before generation.
