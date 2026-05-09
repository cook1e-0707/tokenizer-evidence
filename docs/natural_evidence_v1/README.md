# natural_evidence_v1

This namespace is the redeployed main line for tokenizer-aligned ownership
evidence in ordinary model outputs.

The old compiled carrier-slot protocol is frozen as a controlled sanity check.
It must not be expanded into the main experiment path. In particular, this
namespace must not use:

- explicit evidence blocks,
- field/value output formats,
- owner tags, certificates, or plaintext payload labels,
- one-token exact-slot prompts,
- deterministic rerendering into a structured verifier input.

## Protocol

The protected model produces an ordinary answer:

```text
Start by checking the weather forecast and choosing a route that matches the
group's fitness level. Pack water, a map, a first-aid kit, and a charged phone.
If conditions change, turn around early.
```

Evidence is carried only through natural next-token choices at eligible
prefixes. For a prompt and prefix `h = (x, y_<t)`, a keyed selector decides
whether the position is eligible. If it is eligible, the verifier reconstructs a
context-conditioned bucket family `B_K(h, b)` over tokenizer tokens and records
the bucket id of the observed token.

The audit flow is commit-then-reveal, as specified in
`docs/natural_evidence_v1/protocol_commitment.md`:

1. Collect transcripts from protected and raw model arms.
2. Commit transcript hashes before revealing the audit key.
3. Reveal the audit key and verifier spec.
4. Check the revealed key/payload/policy against the pre-audit commitment.
5. Retokenize transcripts, reconstruct eligible prefixes, map observed tokens
   to bucket ids, and decode accumulated bucket observations with the mixed-radix
   and RS verifier.

The formal context-conditioned bucket object and transcript-level error/erasure
model are recorded in `docs/natural_evidence_v1/formal_protocol.md`.

## Required Arms

Every model-facing result table in this namespace must include all four arms:

- trained Qwen/Qwen2.5-7B-Instruct,
- raw Qwen/Qwen2.5-7B-Instruct,
- trained meta-llama/Meta-Llama-3.1-8B-Instruct,
- raw meta-llama/Meta-Llama-3.1-8B-Instruct.

Raw arms are not optional. They are the null controls that show the verifier is
not merely accepting ordinary natural text by chance.

Task-only LoRA arms are also required before paper-facing claims. They use the
same prompts, responses, and adapter recipe but disable bucket-mass loss, so
they control for LoRA and data-style drift.

## First Execution Order

Do not start training first. The first executable target is opportunity-bank
and verifier validation:

1. Build tokenizer-specific natural bucket opportunity banks from reference
   top-k candidate records.
2. Validate bucket coverage, token filters, mass thresholds, and manifest
   determinism.
3. Validate transcript-level decoding on static observation fixtures.
4. Once the 4-way bank passes basic gates, run a paper-facing Qwen end-to-end
   pilot instead of continuing to optimize banks indefinitely.
5. Replicate on Llama only after Qwen demonstrates end-to-end recovery under
   null controls.

The configured 24,000 raw bank-entry value is now only a static opportunity
scaling placeholder. It is not an end-to-end training gate, not a success
criterion, and not a comparison to Scalable Fingerprinting's 24,576 implanted
fingerprint identities. A bank entry is a context-conditioned measurable
opportunity, not a fingerprint. It becomes ownership evidence only after a
payload, audit key, ECC schedule, bucket-mass training objective, transcript
commitment, and transcript-level decoder are instantiated and evaluated.

The current training gate is compatibility-adjusted natural evidence capacity:
how many usable bucket observations survive compatibility filtering,
reconstructability, null checks, and the declared query budget. Raw static
opportunity count may be reported as a scaling axis, but it must not block a
controlled Qwen viability pilot when compatibility-adjusted gates are met.
The metric and current Qwen thresholds are recorded in
`docs/natural_evidence_v1/compatibility_adjusted_capacity.md`.

## Current Post-846699 Override

The current phase is
`POST_846699_REPAIRED_TARGET_MASS_SCORE_REVIEW_COMPLETE`.

Job 846699 completed the Qwen natural variable-radix five-arm eval with
`protected_accept_count=0`, `null_accept_count=0`, and all 120 decode rows
`insufficient_symbols`. Expert review accepted this as a completed negative
diagnostic, not as provider/model failure and not as payload-codec arithmetic
failure. The confirmed blockers are:

- frame observability: prompt-level eval observations do not complete any
  variable-radix frame under the global repeated-payload frame policy;
- symbol survival: compatible variable-radix digits survive at below 1% in the
  protected arm.

This overrides the generic pilot flow until resolved. Do not start new Qwen
training, Qwen E2E reruns, Llama, same-family nulls, sanitizer benchmarks, or
paper-facing positive claims. Provenance normalization for 846699 is complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_recovery/observation_erasure_summary_846699.json`
has `status=PASS_EXPLAINED_RECOVERY_DIR_NAME`, observation rows=`372216`,
compatible variable-radix digit rows=`1885`, erasure rows=`370331`, decode
rows=`120`, and `provenance_mismatches=[]`. Frame completion replay is also
complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_frame_completion_replay/qwen_846699_frame_completion_replay_summary.json`
has
`status=COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE`,
observed complete frames=`0`, max observed slots per frame=`1`, and scheduled
complete frames under no-erasure=`5370` across decode budgets. This refines the
blocker: the current schedule is formally frame-completable under no-erasure,
but actual symbol survival/anchor drift prevents any observed complete frame.
Oracle schedule simulation is complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_oracle_schedule_simulation/qwen_846699_oracle_schedule_simulation_summary.json`
has `status=COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME`,
`any_subset_observed_complete_frames_total=0`,
`max_any_subset_observed_slots_per_frame=2`, and max greedy iid complete-frame
probability about `5.79e-54`. Schedule-only repair cannot recover the completed
846699 transcript. On-policy survival by slot/source is complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_on_policy_survival/qwen_846699_on_policy_survival_summary.json`
has `status=COMPLETE_ON_POLICY_SURVIVAL_DIAGNOSTIC`,
`compatible_hit_rows=1885/372216`, `target_hit_rows=299/143160`,
`bucket_miss_rows=370331`, `token_index_out_of_response_rows=0`, and no oracle
provenance mismatch. Protected-vs-task-only lift by slice is complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_protected_vs_task_only_lift/qwen_846699_protected_vs_task_only_lift_summary.json`
has `status=COMPLETE_PROTECTED_VS_TASK_ONLY_LIFT_DIAGNOSTIC`. Aggregate
protected target-hit rate is higher than task-only (`0.002392` vs `0.001851`),
but protected compatible-hit rate is lower (`0.004750` vs `0.006164`), and
target survival remains far below 1%. Teacher-forced bucket-mass probe is
complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_teacher_forced_bucket_mass_probe/qwen_846699_teacher_forced_bucket_mass_probe_summary.json`
has `status=COMPLETE_TEACHER_FORCED_BUCKET_MASS_PROBE`,
`position_row_count=143160`, base mean target candidate mass `0.406997`,
protected mean target candidate mass `0.410354`, task-only mean target candidate
mass `0.405440`, protected minus base `+0.003357`, and protected minus task-only
`+0.004914`. Protected LoRA therefore shows only a small teacher-forced
target-mass lift; this does not resolve the free-generation survival or frame
completion failure. Decoder oracle substitution is also complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_decoder_oracle_substitution/qwen_846699_decoder_oracle_substitution_summary.json`
has
`status=COMPLETE_DECODER_ORACLE_SUBSTITUTION_EVALUATOR_CAN_DECODE_TARGET_DIGITS`.
Under committed target-digit substitution, protected rows accept `16/16`,
wrong-payload rows accept `0/16`, and eligible-position mismatches are `0`.
This means the current evaluator/frame schedule can decode if target digits are
observed. The remaining blocker is therefore not decoder arithmetic or frame
contract feasibility under no erasure; it is symbol survival, strict token-index
anchoring/free-generation drift, and weak learned target-bucket mass. The next
required work is a protocol repair decision and anchor/survival repair plan; no
new training or E2E rerun is allowed from this oracle result. The repair
decision is now recorded in
`docs/natural_evidence_v1/post_846699_protocol_repair_decision.md`. It rejects
the current strict token-index contract as the main protocol and selects
prefix-conditioned observed-text eligible selection plus anchor/survival repair
as the next direction. Phase R1 prefix-conditioned selector replay is now
complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_prefix_conditioned_selector_replay/prefix_conditioned_selector_replay_summary.json`
has `status=COMPLETE_PREFIX_CONDITIONED_SELECTOR_REPLAY_ARTIFACT_ONLY`.
Slurm job `847879` completed 0:0 on DGXA100/chimera12 in 00:00:37. The replay
read only existing generated transcripts, variable-radix train positions, and
expanded bucketized candidate artifacts; it did not train, generate, rerun E2E,
decode payload recovery, or claim FAR.

At budget 512, `exact_full` replay scheduled `35840` events, rediscovered
`11582` prefixes (`0.3232`), produced `11122` compatible hits (`0.3103`), and
`4681` target hits (`0.1306`). `suffix_8` replay scheduled the same denominator,
rediscovered `12652` prefixes (`0.3530`), produced `12109` compatible hits
(`0.3379`), and `5080` target hits (`0.1417`). These are coordinate-level
diagnostics only. They are not payload recovery and not FAR. The result is still
negative for launch: raw actual-prefix replay has much higher target-hit rates
around `0.386` at budget 512, while protected arms range from roughly `0.000`
to `0.054` depending on payload/seed/policy and task-only arms are often higher
than protected. This means prefix-conditioned replay can reduce strict
token-index erasure, but the current protected training does not create an
ownership-specific signal over raw/task-only null behavior.

At that point, the next allowed action was artifact-only R1 interpretation and
selector-contract repair planning: quantify protected-minus-raw/task-only
coordinate lift, multiple-testing/null risk, and whether a prefix-conditioned
selector should be paired with branch-aware/regenerated-suffix training targets
or sparse coordinate-level coding. No training, E2E rerun, Llama, same-family
null, sanitizer, or paper-facing positive claim was allowed from the R1 result.

That R1 interpretation is now complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_r1_selector_contract_analysis/r1_selector_contract_repair_summary.json`
has `status=COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS`. Across 64
protected-vs-null comparison slices, protected has positive lift over raw in
`0/64` rows and positive lift over task-only in `0/64` rows. At budget 512,
mean protected target-hit rates are `0.020089` to `0.030134` across policies,
while raw is about `0.384905` to `0.386440` and task-only is about `0.113979`
to `0.130999`. Direct replay-verifier use, training, and E2E rerun remain
blocked. The decision is recorded in
`docs/natural_evidence_v1/r1_selector_contract_repair_analysis.md`.

The next allowed action is artifact-only selector-contract precommit design and
branch-aware/regenerated-suffix training-target preflight.

That preflight is now complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_selector_contract_preflight/selector_contract_training_target_preflight_summary.json`
has
`status=COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY`.
The selector draft is
`DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT`; direct replay verifier,
training, generation, and E2E rerun remain disabled. The draft contract and gate
plan are recorded in
`docs/natural_evidence_v1/selector_contract_training_target_preflight.md`.

The next allowed action is to prepare artifact-only branch-aware compatibility
and regenerated/local-suffix repair diagnostics under the draft selector
contract. If Chimera CPU/GPU is needed, use Slurm.

The balanced branch-aware compatibility proxy diagnostic is now complete.
Slurm job `848414` (`nat-ev-qwen-brscore`) completed 0:0 on DGXA100/chimera12
in 00:00:55 and wrote:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_compatibility_scored_balanced/`.
Status is
`COMPLETE_BRANCH_AWARE_COMPATIBILITY_MODEL_SCORED_PROXY_NOT_GENERATED`.
The diagnostic scored 209 balanced rows and reports branch-aware proxy pass
153/209 (`0.7321`), response proxy pass 155/209 (`0.7416`), and suffix proxy
pass 169/209 (`0.8086`). By condition, branch-aware proxy pass is protected
57/76 (`0.7500`), raw 52/74 (`0.7027`), and task-only 44/59 (`0.7458`).
This is useful evidence that many local target substitutions are plausible
under a reference-model NLL proxy, but it does not establish protected-specific
ownership signal because protected is essentially tied with task-only. It is
not branch continuation generation, suffix regeneration, payload recovery, FAR,
training, or E2E. The next allowed action is artifact-only branch-aware score
interpretation and repaired training-target preflight.

That score interpretation and repaired target-mass probe preflight is now
complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_score_interpretation/`.
Status is
`COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING`.
The analysis found 75 primary repaired target-mass probe candidates, 78
secondary/ablation candidates, and 56 rejected rows. Primary candidates are
protected=39, task-only=20, raw=16; drift reasons are
`compatible_non_target=58` and `observed_token_not_candidate_set=17`; token
classes are word=74 and function_word=1. This is useful for the next
artifact-only repaired target-mass probe, but it still does not unlock training:
the decision is
`PRIMARY_CANDIDATES_EXIST_BUT_NO_TRAINING_GATE_PROTECTED_CONTROL_SEPARATION_WEAK`.
The artifact-only repaired teacher-forced target-mass probe design is now
complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/`.
It joins all 75 primary candidates to full bucket-token inputs and writes a
257-row scoring plan: base=75, protected=91, task-only=91. This was design
only, not model scoring, training, generation, E2E, payload recovery, FAR, or a
paper-facing claim. The repaired teacher-forced target-mass probe was then
scored as Slurm job `848547` and reviewed in
`results/natural_evidence_v1/status/hermes_reports/20260508_0811_repaired_target_mass_score_review.md`.
All `257` planned rows were scored, but `threshold_pass=false`: protected-base
target-mass lift was `-0.007645810655699581`, protected-task-only lift was
`-0.04776975171334799`, and protected-task-only rank-1 lift was
`-0.03296703296703296` against a required `+0.05` lift. This rejects repaired
dataset or training preflight from job `848547`. Training and E2E remain
forbidden.

That artifact-only preparation is now complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation/branch_aware_compatibility_summary.json`
has
`status=COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED`.
It prepared `68` branch-aware scoring rows and `68` regenerated/local-suffix
repair input examples from `240` R1 replay examples, with `68` train metadata
matches. Drift reasons in the prepared set are `compatible_non_target=60` and
`observed_token_not_candidate_set=8`. These are diagnostic inputs only:
`model_scoring_started=false`, `generation_started=false`,
`training_started=false`, `e2e_eval_started=false`, and
`paper_claim_allowed=false`. A limitation of the current prepared set is that
the selected examples are raw-only (`model_condition_counts.raw=68`) because of
the available R1 example export and deduplication priority; protected-specific
branch-aware analysis will need a richer protected/task-only example source or
an expanded replay-example export. The next allowed action is to run
branch-aware compatibility scoring through Slurm if model scoring is needed, or
to construct an artifact-only local-suffix repair dry-run from these inputs.

That artifact-only local-suffix repair dry-run is now complete:
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run/local_suffix_repair_dry_run_summary.json`
has `status=COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED`. It processed
`68` repair examples, found `36` approximate text-substitution-ready rows, and
found `32` rows that need tokenizer-aligned or branch-aware regeneration because
the observed token text could not be located in the original response text. It
did not load a model, regenerate suffixes, score branch-aware compatibility,
train, run E2E, decode payload recovery, or estimate FAR. Because the dry-run
input remains raw-only, protected/task-only branch-aware comparison is still
blocked. The next allowed action is a Slurm-scored branch-aware compatibility
diagnostic or a richer protected/task-only example export before scoring.

A richer protected/task-only example export completed as Slurm job `848405`
(`nat-ev-qwen-babr`) with exit code `0:0` and elapsed time `00:00:46`. It is
artifact-only and did not score branch-aware compatibility, regenerate suffixes,
train, generate, rerun E2E, decode payload recovery, or estimate FAR. The synced
balanced export selected `768` examples with condition counts
`protected_trained=288`, `task_only_lora=288`, and `raw=192`, all with generated
response text. Using those richer examples, balanced branch-aware/local-suffix
inputs were prepared under
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_branch_aware_suffix_repair_preparation_balanced/`:
`planned_branch_aware_rows=209`, with condition counts
`protected_trained=76`, `task_only_lora=59`, and `raw=74`. The balanced
local-suffix dry-run under
`results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_local_suffix_repair_dry_run_balanced/`
has `209/209` text-substitution-ready rows. This is still not compatibility
scoring, naturalness scoring, training-data approval, payload recovery, or FAR.
The next allowed action is a Slurm-scored branch-aware compatibility diagnostic
from the balanced scoring plan.

The detailed plan is in
`docs/natural_evidence_v1/next_step_codex_plan.md`, with the technical diagnosis
in `docs/natural_evidence_v1/eligible_positions_sparse_diagnosis.md`.

## Opportunity-Bank V2 Gates

The precomputed bank is calibration/cache data. The verifier-facing policy is:

```text
observed transcript prefix -> reference top-k candidates -> deterministic
filtering and keyed bucket construction -> observed token bucket id
```

This means verification must not depend on exact lookup of a training-time
prefix id. For generated transcripts, the same bucket-construction policy must
be applied on observed prefixes after transcript commitment.

Training must not start until the following are reported:

- raw opportunity entries per tokenizer as a diagnostic, not a training gate,
- compatibility-aware min1 and multi-member opportunity counts,
- held-out prompt coverage and eligible density,
- bucket mass balance and entropy,
- counterfactual suffix compatibility,
- on-policy reconstructability on generated transcripts,
- raw/wrong-key/wrong-payload null behavior,
- channel capacity and bucket-count ablations for 4 and 8 buckets.

The Qwen E2E viability pilot gate is intentionally lower than the raw static
bank size: 4-way min1-compatible entries around 1,500-2,500, at least 200
multi-member compatible entries, nontrivial held-out density, and raw/wrong-key
pre-null behavior that is not high risk. This is a pilot gate, not a final paper
claim. Main-paper scale should target 5,000-10,000 compatibility-aware entries
per tokenizer and report an effective capacity curve.

The current audit scripts report static opportunity quality and mark model-run
dependent gates as `NEEDS_RESULTS` rather than treating raw entry count as a
fingerprint claim.

The complete end-to-end work order is in
`docs/natural_evidence_v1/end_to_end_audit_plan.md`. The first paper-facing
pilot is Qwen protected/raw/task-only/wrong-key/wrong-payload with two payloads,
two seeds, at least 2048 owner probes, at least 2048 organic null prompts, and
query budgets 8, 16, 32, 64, and 128.

## Automation Reliability

Hourly automation that needs Chimera must follow
`docs/natural_evidence_v1/chimera_ssh_reliability.md`: the first
`ssh chimera` DNS failure is a known intermittent resolver cold-start issue, not
a final access gate failure. Warm DNS and retry non-interactively before marking
remote Slurm or artifact status as unverified.

Before any future `natural_evidence_v1` Chimera GPU run, also follow the GPU
resource preflight in `docs/chimera_runbook.md`. Check the requested H200/A100
partitions for available full, non-MIG devices, then size shard counts or Slurm
array concurrency to use the remaining eligible full GPUs where the work is
independent and artifact-safe. Do not leave a long serial GPU job running alone
when reference scoring, candidate scoring, null arms, seeds, payloads, or shards
can be split across idle full GPUs.

This resource policy does not loosen experiment gates. It only changes how
approved Chimera work is packed onto the cluster: use full H200/A100 capacity
aggressively when it is available, but do not start protected training,
counterfactual scoring, Llama rebuilds, or old structured-path expansions unless
the current phase explicitly allows that action.

Hourly automation must also enforce the Chimera Slurm mail-notification policy.
Every `natural_evidence_v1` sbatch script submitted by an hourly action must
include:

```bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
```

The local static validator checks this requirement through
`require_chimera_mail_notifications: true` in
`configs/natural_evidence_v1/run_allowlist.yaml`. A short Chimera CPU test job
(`844015`, `nat-ev-mail-test`) completed with exit code `0:0`, and mailbox-side
delivery was confirmed, so future hourly-submitted Slurm jobs should use the
same notification path.

## Current Entrypoints

Static validation:

```bash
python3 scripts/natural_evidence_v1/validate_static.py \
  --config configs/natural_evidence_v1/pilot.yaml
```

Reference top-k candidate scoring, to run on a GPU node:

```bash
python3 scripts/natural_evidence_v1/generate_reference_outputs.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen \
  --output-jsonl results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --prompt-count 4096 \
  --require-cuda

python3 scripts/natural_evidence_v1/score_reference_candidates.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen \
  --input-jsonl results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --require-cuda
```

Opportunity-bank construction from scored candidates:

```bash
python3 scripts/natural_evidence_v1/build_bucket_bank.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen
```

Opportunity-bank quality audit:

```bash
python3 scripts/natural_evidence_v1/audit_opportunity_bank.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --entries results/natural_evidence_v1/bucket_banks/qwen_bucket_bank_entries.jsonl \
  --reference-outputs results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --candidate-jsonl results/natural_evidence_v1/reference_candidates/qwen_topk_candidates.jsonl
```

Counterfactual suffix compatibility scoring, to run on a GPU node:

```bash
python3 scripts/natural_evidence_v1/score_counterfactual_compatibility.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen \
  --reference-outputs results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --candidate-jsonl results/natural_evidence_v1/reference_candidates/qwen_topk_candidates.jsonl \
  --output-jsonl results/natural_evidence_v1/bucket_banks/qwen_counterfactual_compatibility.jsonl \
  --require-cuda
```

Natural training dataset compilation:

```bash
python3 scripts/natural_evidence_v1/compile_train_dataset.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --reference-outputs results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --bucket-bank-entries results/natural_evidence_v1/bucket_banks/qwen_bucket_bank_entries.jsonl \
  --payload-id P0421 \
  --output-jsonl results/natural_evidence_v1/datasets/qwen_train_P0421.jsonl \
  --contract-json results/natural_evidence_v1/datasets/qwen_train_P0421_contract.json
```

Transcript observation decoding after commit-then-reveal:

```bash
python3 scripts/natural_evidence_v1/verify_observations.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --observations results/natural_evidence_v1/decoded_observations/qwen_observations.jsonl
```

Full Phase A Slurm job, one tokenizer at a time:

```bash
TOKENIZER_KEY=qwen \
SCRATCH_ROOT=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v1 \
sbatch scripts/natural_evidence_v1/slurm_phase_a_bucket_bank.sbatch
```
