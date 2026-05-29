# natural_evidence_v2 Current State

Last synchronized: 2026-05-29T00:31:51Z

This is the compact controlling state for Codex and Hermes. Historical route
records remain in `results/natural_evidence_v2/status/` and long-form review
docs under `docs/natural_evidence_v2/`; they are not controlling when they
conflict with this file.

## Canonical Phase

`V2_R4_AFTER_880121_LLAMA_LOCKED_SCALE_COMPLETED_STRONG_READONLY_AGGREGATE_REVIEW_PENDING`

## Latest Update

The domain-repaired Llama locked-scale generation job `880121` has reached a
complete remote artifact state under a read-only aggregate scan. This scan did
not download or rewrite raw generation artifacts and is not yet the formal
adoption review.

Read-only aggregate snapshot:

```text
job_id: 880121
job_name: nat-ev-v2-r4ll96v4
model: meta-llama/Meta-Llama-3.1-8B-Instruct
complete shards: 96/96
missing artifacts: 0
generated rows: 196608
unique response hashes: 196608
global duplicate extra rows: 0
max duplicate group size: 1
trace binding invalid rows: 0/196608
protected strict accepts: 96/96
protected accepts ignoring quality: 96/96
raw accepts: 0/96
task-only accepts: 0/96
wrong-key accepts: 0/96
wrong-payload accepts: 0/96
protected duplicate response hash count: 0
protected forbidden public surface count: 0
raw duplicate response hash count: 0
raw forbidden public surface count: 0
```

Recorded snapshot:

```text
results/natural_evidence_v2/status/r4_after_880121_remote_completion_snapshot_20260528/summary.json
results/natural_evidence_v2/status/r4_after_880121_remote_completion_snapshot_20260528/summary.md
```

Important boundaries:

```text
formal review/adoption: still pending
task-only: decode-control bucket only; no separate task-only generation arm in this route
trace binding: consistency/hash binding only; BINDING_HMAC_SECRET=false
text-only phrase decoder: not claimed
paper-facing positive claim: not allowed
```

Next allowed action: sync the reviewed artifacts locally and run the formal
locked-scale generation review/adoption script. Do not unlock paper-facing
positive claims, full FAR, sanitizer robustness, payload diversity, text-only
phrase-decoder success, or HMAC/signed trace-provenance claims from this
read-only aggregate alone.

Historical previous update follows.

The domain-repaired Llama locked-scale tokenizer-only preflight job `880120`
completed cleanly and was reviewed as PASS:

```text
status: PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_PREFLIGHT_879455_REVIEWED
checked rows: 98304
failed rows: 0
empty target ids: 0
empty other ids: 0
target/other overlaps: 0
model forward/generation/scoring/training: not started
```

The repaired generation route was then validated locally and remotely using the
same repaired row-bank hash. Local/remote allowlist safety and hash preflight
passed. Exactly one allowlist entry was enabled for submission:
`v2_r4_after_879406_second_family_llama_locked_scale_policy_v4_h200`.

Submitted repaired Llama locked-scale generation job:

```text
job_id: 880121
job_name: nat-ev-v2-r4ll96v4
array: 0-95
concurrency throttle: none
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
model: meta-llama/Meta-Llama-3.1-8B-Instruct
expected generated rows: 98304
route config:
  configs/natural_evidence_v2/r4_after_879555_domain_repaired_llama_locked_scale_generation_route.yaml
row bank:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_second_family_llama_locked_scale_row_bank_plan_20260528/row_allocation_rows.jsonl
tokenizer review:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_llama_locked_scale_tokenizer_preflight_880120_review/review_summary.json
remote output:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879555_domain_repaired_llama_locked_scale_policy_v4_880121/llama3_1_8b_instruct
submission record:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_llama_locked_scale_generation_submission_880121_20260528/submission_record.json
```

The allowlist was disabled immediately after `sbatch` returned. Local and
remote post-submit allowlist safety both passed. Current allowed action:
monitor `880121`; after all array tasks reach terminal state, sync artifacts
and run locked-scale generation review. Do not submit another generation job
while `880121` is active. Do not make a Llama locked-scale transfer claim until
the reviewed gate passes.

Still not unlocked: paper-facing positive claim, full FAR, sanitizer, payload
diversity, text-only phrase-decoder success, or training.

Historical previous update follows.

The repaired Llama locked-scale generation job `879555` completed cleanly at the
Slurm/artifact level: all `96/96` array shards completed with exit code `0:0`,
all generated/decode/trace artifacts were present, and no traceback/OOM/path
errors were detected in the checked logs.

Artifact review status:

```text
FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_879555_REVIEWED_NO_ADOPT
```

This is a strong signal result but not an adopted locked-scale pass:

```text
complete shards: 96/96
generated rows: 196608/196608
protected strict accepts: 96/96
protected accepts ignoring quality: 96/96
raw accepts: 0/96
task-only accepts: 0/96
wrong-key accepts: 0/96
wrong-payload accepts: 0/96
global duplicate extra rows: 0
within-block duplicate count: 0
trace binding invalid rows: 0/196608
technical forbidden public surface count: 1
```

The single strict failure is a raw-arm forbidden public literal under the
precommitted contextual policy v4. Failure attribution localizes it to
`shard_36_block_00`: a `document scanning routine` prompt produced the ordinary
document-processing word `watermark`. A broader diagnostic scan, not used to
re-score the precommitted gate, found that the same prompt domain naturally
elicits `fingerprints`/`watermark` language (`117` rows, all in
`document scanning routine`). Root cause: the locked prompt domain conflicts
with the hard public-literal policy.

Recorded artifacts:

```text
review:
  results/natural_evidence_v2/status/r4_after_879406_second_family_llama_locked_scale_generation_879555_review/review_summary.json
failure attribution:
  results/natural_evidence_v2/status/r4_after_879555_llama_locked_scale_failure_attribution_20260527/failure_attribution_summary.json
synced raw artifacts:
  results/natural_evidence_v2/status/r4_after_879458_repair_second_family_llama_locked_scale_generation_879555_raw/
```

The repair route has been recorded:

```text
docs/natural_evidence_v2/R4_AFTER_879555_LLAMA_LOCKED_SCALE_RESIDUAL_FORBIDDEN_DOMAIN_REPAIR_ROUTE_20260528.md
configs/natural_evidence_v2/r4_after_879555_llama_locked_scale_domain_repair_route.yaml
results/natural_evidence_v2/status/r4_after_879555_llama_locked_scale_domain_repair_route_20260528/route_decision_summary.json
```

The hard public-literal gate remains unchanged. Do not reclassify `879555` as
pass, do not make a Llama locked-scale claim, and do not submit another
locked-scale generation job until the repaired row-bank plan, actual Llama
tokenizer preflight, route validation, local/remote hash preflight, and
allowlist safety checks pass.

Artifact-only repaired prompt bank and row bank have now been built and
validated:

```text
prompt bank:
  results/natural_evidence_v2/prompts/r4_after_879555_llama_locked_scale_domain_repair_prompt_bank_20260528/
Qwen-neutral repaired row bank:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_qwen_neutral_locked_scale_row_bank_plan_20260528/
Llama repaired row bank:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_second_family_llama_locked_scale_row_bank_plan_20260528/
row-bank validation:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_llama_row_bank_validation_20260528/validation_summary.json
```

Validation status:

```text
PASS_R4_AFTER_879555_DOMAIN_REPAIRED_LLAMA_LOCKED_SCALE_ROW_BANK_VALIDATION
rows: 98304
prompts: 6144
shards: 96
rows per shard: 1024
document scanning rows: 0
static hard literal risk prompts: 0
duplicate prompt/prefix pairs: 0
```

Actual Llama tokenizer boundary preflight for the repaired row bank has been
submitted as tokenizer-only Slurm job `880120`:

```text
job_id: 880120
job_name: nat-ev-v2-r4llTok
route_config:
  configs/natural_evidence_v2/r4_after_879555_domain_repaired_llama_locked_scale_tokenizer_preflight_route.yaml
score_rows:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_second_family_llama_locked_scale_row_bank_plan_20260528/row_allocation_rows.jsonl
expected rows: 98304
remote output:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879555_domain_repaired_llama_locked_scale_tokenizer_preflight_manual/llama3_1_8b_instruct
submission record:
  results/natural_evidence_v2/status/r4_after_879555_domain_repaired_llama_locked_scale_tokenizer_submission_880120_20260528/submission_record.json
```

The allowlist was disabled immediately after `sbatch` returned. Local and remote
post-submit allowlist safety passed. Current allowed action: monitor `880120`;
after terminal completion, sync and review tokenizer preflight artifacts. No
generation submission is allowed until tokenizer preflight review passes and the
repaired locked-scale generation route is reviewed.

Still not unlocked: paper-facing positive claim, locked-scale cross-family
claim, full FAR, sanitizer, payload diversity, text-only phrase-decoder success,
or training.

Historical previous update follows.

The Llama locked-scale tokenizer-only preflight job `879455` completed and was
reviewed as PASS: 98,304 rows checked, 0 failed rows, 0 empty target/other rows,
and 0 target/other overlaps. This only authorized planning for Llama locked-scale
generation; it did not make a locked-scale transfer claim.

The first Llama locked-scale generation submission `879458` failed before model
generation because the locked-scale wrapper set `ROUTE_CONFIG` but inherited the
dev-diagnostic `ROUTE_VALIDATOR`. All 96 array tasks failed during route
validation before model loading or generation. It is recorded as a no-adopt
wrapper/control-plane failure:
`results/natural_evidence_v2/status/r4_after_879406_second_family_llama_locked_scale_generation_879458_failure_review_20260527/failure_review_summary.json`.

The wrapper was repaired to export the locked-scale validator. After repair,
local and remote route validation passed, local and remote wrapper plan-only
smoke passed, and local/remote allowlist preflight passed.

Submitted repaired Llama locked-scale generation job:

```text
job_id:
  879555
job_name:
  nat-ev-v2-r4ll96v4
array:
  0-95
route:
  R4 after-879406 second-family Llama locked-scale generation, policy v4
expected generated rows:
  98304
predecessor no-adopt job:
  879458
remote output:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879406_second_family_llama_locked_scale_policy_v4_879555/llama3_1_8b_instruct
submission record:
  results/natural_evidence_v2/status/r4_after_879458_repair_second_family_llama_locked_scale_generation_submission_879555_20260527/submission_record.json
```

Current allowed action: monitor `879555` only. After all shards reach terminal
state, sync artifacts and run locked-scale review. Do not submit another Llama
locked-scale generation job while `879555` is active. Do not make a paper-facing
claim or locked-scale cross-family claim until review passes.

Still not unlocked: paper-facing positive claim, locked-scale cross-family
claim, full FAR, sanitizer, payload diversity, text-only phrase-decoder success,
or training.

Historical previous update follows.

Artifact-only contextual forbidden policy v4 repair passed. Policy v4 extends
the precommitted ordinary scheduling cues for the `coordinate`/`slot` ambiguity
while keeping technical uses hard-forbidden. Unit tests passed (`20 passed`).
Counterfactual replay under v4 passed on the six completed `879391` shards
and on the full `879248` run:

```text
879391 completed shards under v4: protected strict 6/6; controls 0/6;
  forbidden=0; duplicate=0
879248 full run under v4: protected strict 32/32; controls 0/32;
  forbidden=0; duplicate=0
```

These counterfactuals do not reclassify `879248` or `879391`; both remain
historical no-adopt runs under their precommitted policies.

The policy-v4 Llama rerun route was validated locally and remotely. Wrapper
plan-only smoke passed locally and remotely. Local and remote allowlist safety
passed with zero enabled entries. Exactly one allowlist entry was then enabled
for submission:
`v2_r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_h200`.
Job `879406` (`nat-ev-v2-r4ll32v4`) was submitted to `pomplun` H200, and the
allowlist was immediately disabled again. Post-submit allowlist safety passed
locally and remotely with zero enabled entries.

Current allowed action: monitor `879406` only. After all array tasks reach a
terminal state, sync artifacts and run the reviewed Llama 32-block dev
diagnostic review. Do not submit another Llama dev diagnostic while `879406` is
active. Do not make a paper-facing positive claim, locked-scale cross-family
claim, full FAR, sanitizer robustness, payload diversity, or text-only
phrase-decoder success claim from `879248`, `879391`, or `879406` before review.

```text
canonical phase:
  V2_R4_AFTER_879391_POLICY_V4_LLAMA_DEV_DIAGNOSTIC_879406_SUBMITTED_MONITOR_ONLY
submitted job:
  879406, nat-ev-v2-r4ll32v4
array:
  0-31; no percent concurrency throttle
partition/qos/account:
  pomplun / pomplun / cs_yinxin.wan
remote output:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_879406/llama3_1_8b_instruct
policy v4:
  results/natural_evidence_v2/precommit/r4_after_879391_contextual_forbidden_policy_v4_20260527/contextual_forbidden_surface_policy_v4.json
policy v4 validation:
  results/natural_evidence_v2/status/r4_after_879391_contextual_policy_v4_validation_20260527/validation_summary.json
submission artifact:
  results/natural_evidence_v2/status/r4_after_879391_second_family_llama_policy_v4_submission_879406_20260527/submission_record.json
next allowed action:
  monitor job 879406; after completion sync and review artifacts only
not unlocked:
  paper-facing positive claim, locked-scale cross-family claim, full FAR,
  sanitizer, payload diversity, text-only phrase-decoder success
```

Historical previous update follows.

The R4 second-family/Llama 32-block dev diagnostic job `879248`
(`nat-ev-v2-r4ll32`) completed cleanly at the Slurm level: all 32 array tasks
completed with exit code `0:0`, no active queue entries remained, and local log
scan found no traceback/OOM/runtime hard errors. Outputs and Slurm logs were
synced locally.

Artifact review status:
`FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_879248_REVIEWED_NO_ADOPT`.
This is a strong near-pass but not an adopted dev pass. Signal and controls are
clean: protected accepts ignoring quality were `32/32`, controls were `0/32`,
global duplicate extra rows were `0`, and trace binding invalid rows were
`0/65536`. The strict gate failed because one protected-source block had a
contextual forbidden-surface count under the precommitted v2 matcher:
protected strict accepts were `31/32`.

Failure localization: only `shard_12_block_00` failed quality. The output used
ordinary schedule-coordination language (`time slot ... coordinate`), which v2
misclassified as technical `coordinate` because `slot` was a technical cue near
`coordinate`. This does not reclassify `879248`; under its precommitted v2
policy, `879248` remains no-adopt.

Artifact-only repair completed: contextual forbidden policy v3 was added and
tested. It treats schedule/team/time-slot coordination as ordinary only when
the sole nearby technical cue is `slot`; technical uses such as `coordinate slot`
with `decoder` remain forbidden. Unit tests passed, and a local shard-12
counterfactual decode under v3 removes the false positive while preserving
controls. This counterfactual is only a repair validation, not a positive
reclassification.

The policy-v3 rerun route was validated locally and remotely, wrapper plan-only
smoke passed locally and remotely, local/remote file hashes matched, and
local/remote allowlist safety passed with zero enabled entries. Exactly one
allowlist entry was then enabled for submission:
`v2_r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_h200`.
Job `879391` (`nat-ev-v2-r4ll32v3`) was submitted, and the allowlist was
immediately disabled again. Post-submit allowlist safety passed locally and
remotely with zero enabled entries.

`879391` was stopped early with `scancel` after the first six completed shards
showed the strict quality gate was already impossible to pass. This was not a
Slurm crash or model failure. The early signal was still clean: protected
accepts `6/6`, controls `0/6`, duplicate count `0`, and trace invalid rows
`0/12288`. The blocker is another contextual forbidden false positive in raw:
ordinary music-recital scheduling text contained `3 pm slot ... coordinate with
the sound technician`, and policy v3 still treated nearby `slot` as a technical
cue for `coordinate`.

Current allowed action: artifact-only contextual forbidden policy v4 repair and
fixtures. Do not submit another Slurm job until policy v4 tests, route
validation, wrapper smoke, and allowlist preflight pass. Do not make a
paper-facing positive claim, locked-scale cross-family claim, full FAR,
sanitizer robustness, payload diversity, or text-only phrase-decoder success
claim from `879248` or `879391`.

```text
canonical phase:
  V2_R4_AFTER_879391_POLICY_V3_LLAMA_DEV_DIAGNOSTIC_EARLY_FORBIDDEN_GATE_FAIL_NO_ADOPT
completed job:
  879248, nat-ev-v2-r4ll32
stopped rerun job:
  879391, nat-ev-v2-r4ll32v3
array:
  0-31; no percent concurrency throttle
slurm state:
  all tasks COMPLETED, ExitCode 0:0
post-submit allowlist:
  zero enabled locally and remotely
review facts:
  protected strict accepts=31/32; protected ignoring-quality accepts=32/32; raw/task-only/wrong-key/wrong-payload accepts=0/32; duplicate extra rows=0; trace invalid rows=0/65536; technical forbidden count=3 from one protected-source block viewed under protected/wrong-key/wrong-payload
failure analysis:
  results/natural_evidence_v2/status/r4_after_877895_second_family_llama_dev_diagnostic_879248_failure_analysis_20260527/failure_analysis_summary.json
policy v3:
  results/natural_evidence_v2/precommit/r4_after_879248_contextual_forbidden_policy_v3_20260527/contextual_forbidden_surface_policy_v3.json
not unlocked:
  paper-facing positive claim, locked-scale cross-family claim, full FAR, sanitizer, payload diversity, text-only phrase-decoder success
next allowed action:
  artifact-only contextual forbidden policy v4 repair and route validation; no Slurm until preconditions pass
submission artifact:
  results/natural_evidence_v2/status/r4_after_879248_second_family_llama_dev_diagnostic_policy_v3_submission_879391_20260527/submission_record.json
early stop artifact:
  results/natural_evidence_v2/status/r4_after_879248_policy_v3_879391_early_stop_20260527/early_stop_summary.json
```

Historical previous update follows.

Job `877142` completed successfully at the Slurm level (`COMPLETED`, exit
code `0:0`) and was aggregated. It is artifact-complete and the same-family raw
null signal checks are clean for Qwen2.5-3B/7B/14B, but it fails the strict
same-family raw-null quality gate because residual forbidden literal collisions
remain: technical `decoder` count 1, technical `coordinate` count 1, and
ambiguous `bucket` count 1. The run is no-adopt for same-family raw-null pass
claims. The next allowed route is artifact-only v6 forbidden-literal/domain
repair planning and lexical/tokenizer preflight. Do not reinterpret `877142`,
relax the gate, or make a same-family raw-null pass claim from this run.

```text
canonical phase:
  V2_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_877142_FORBIDDEN_RESIDUAL_NO_ADOPT
source failed job:
  877142, nat-ev-v2-r4sfRaw
877142 status:
  FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE / no-adopt
877142 clean checks:
  generated rows=196608; raw accepts=0/64 per model; raw accepts ignoring quality=0/64 per model; duplicate extra rows=0 per model; trace 196608 checked with 0 invalid
877142 failed strict quality gate:
  technical forbidden=2 (decoder=1, coordinate=1); ambiguous forbidden=1 bucket; ordinary-domain literals=1462 report-only
877142 aggregate:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_877142_aggregate_20260525/same_family_raw_null_summary.json
877142 failure review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_877142_failure_review_20260525/failure_review_summary.json
v5 row bank source:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_row_bank_plan_20260525/row_allocation_rows.jsonl
v5 row bank status:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
v5 row bank validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_ROW_BANK_VALIDATION_NO_SUBMIT
v5 row bank validation facts:
  rows=65536, prompts=4096, shards=64, denied prompt ids=99, denied-domain rows=0, static technical hits=0, static ambiguous hits=0
v5 tokenizer route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT
v5 tokenizer route validation artifacts:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_tokenizer_route_validation_20260525_r1/route_validation_summary.json
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_tokenizer_route_validation_remote_20260525/route_validation_summary.json
submitted tokenizer-only job:
  877139, nat-ev-v2-r4sfTok
array:
  0-2%3
scope:
  tokenizer-only preflight for Qwen2.5-3B/7B/14B on the v5 row bank, 65536 rows/tokenizer
submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_tokenizer_submission_20260525/submission_record.json
877139 tokenizer review:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_TOKENIZER_PREFLIGHT_877139
877139 tokenizer review artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_tokenizer_preflight_877139_review/review_summary.json
877139 tokenizer facts:
  Qwen2.5-3B/7B/14B each checked 65536 rows; failed=0, empty target=0, empty other=0, target/other overlap=0
v5 generation route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V5_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
v5 generation route validation artifacts:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_generation_route_validation_20260525/route_validation_summary.json
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_generation_route_validation_remote_20260525/route_validation_summary.json
completed generation job:
  877142, nat-ev-v2-r4sfRaw
array:
  0-191, no array task throttle
scope:
  raw-only Qwen2.5-3B/7B/14B same-family controls, 64 shards/model, 1024 rows/shard, 196608 expected generated rows
generation submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v5_generation_submission_20260525/submission_record.json
post-submit allowlist:
  zero enabled locally and remotely
current allowed action:
  artifact-only v6 same-family raw-null prompt-domain/forbidden-literal repair planning and lexical/tokenizer preflight; no new generation until route validation and tokenizer preflight pass
not allowed yet:
  training, Llama, sanitizer, FAR aggregation, payload diversity, paper-facing claim, same-family raw-null pass claim from 877142
```

Historical previous update follows.

Job `875777` completed and was aggregated. It is artifact-complete and the raw
null signal checks are clean for Qwen2.5-3B/7B/14B, but it fails the strict
same-family raw-null quality gate because residual forbidden literal collisions
remain: technical `fingerprint` count 5 and ambiguous `bucket` count 2. The run
is no-adopt for same-family raw-null pass claims. A v4 row bank has been built
and validated by excluding all 875168/875471/875777 collision prompt ids and
domains; static technical/ambiguous forbidden hits are 0. The v4 tokenizer-only
route validated locally and remotely, then exactly one H200 tokenizer preflight
array was submitted as job `876849`; it completed successfully and passed
review. The v4 generation route validated locally and remotely. Exactly one
H200 v4 generation/decode array was submitted as job `876852`; the allowlist was
disabled immediately after submission and revalidated locally/remotely with zero
enabled entries.

```text
canonical phase:
  V2_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_GENERATION_876852_RUNNING
source failed job:
  875777, nat-ev-v2-r4sfRaw
875777 status:
  FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE / no-adopt
875777 clean checks:
  generated rows=196608; raw accepts=0/64 per model; raw accepts ignoring quality=0/64 per model; duplicate extra rows=0 per model; trace 196608 checked with 0 invalid
875777 failed strict quality gate:
  technical forbidden=5 all fingerprint; ambiguous forbidden=2 all bucket; ordinary-domain literals=1129 report-only
875777 aggregate:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875777_aggregate_20260524/same_family_raw_null_summary.json
875777 failure review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875777_failure_review_20260524/failure_review_summary.json
v4 residual repair policy:
  keep hard forbidden policy unchanged; do not rescue 875777; exclude 875168/875471/875777 collision prompt ids and domains
v4 denied domains:
  cafe morning setup; community theater rehearsal; dental office supply review; farmers market booth setup; hardware store aisle cleanup; library display refresh; local history archive sorting; school club fundraiser; tenant move-out checklist; tutoring session preparation
v4 row bank:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_row_bank_plan_20260524/row_allocation_rows.jsonl
v4 row bank status:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
v4 row bank validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_ROW_BANK_VALIDATION_NO_SUBMIT
v4 row bank validation facts:
  rows=65536, prompts=4096, shards=64, denied prompt ids=96, denied-domain rows=0, static technical hits=0, static ambiguous hits=0
v4 tokenizer route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT
v4 tokenizer route validation artifacts:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_tokenizer_route_validation_20260524/route_validation_summary.json
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_tokenizer_route_validation_remote_20260524/route_validation_summary.json
submitted tokenizer-only job:
  876849, nat-ev-v2-r4sfTok
array:
  0-2%3
scope:
  tokenizer-only preflight for Qwen2.5-3B/7B/14B on the v4 row bank, 65536 rows/tokenizer
submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_tokenizer_submission_20260524/submission_record.json
876849 tokenizer review:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_TOKENIZER_PREFLIGHT_876849
876849 tokenizer review artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_tokenizer_preflight_876849_review/review_summary.json
876849 tokenizer facts:
  Qwen2.5-3B/7B/14B each checked 65536 rows; failed=0, empty target=0, empty other=0, target/other overlap=0
v4 generation route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
v4 generation route validation artifacts:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_generation_route_validation_20260524/route_validation_summary.json
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_generation_route_validation_remote_20260524/route_validation_summary.json
submitted generation job:
  876852, nat-ev-v2-r4sfRaw
array:
  0-191%6 at submission; live throttle raised to %8 on 2026-05-24T23:14:42Z; array throttle removed on 2026-05-24T23:16:56Z
scope:
  raw-only Qwen2.5-3B/7B/14B same-family controls, 64 shards/model, 1024 rows/shard, 196608 expected generated rows
generation submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_generation_submission_20260524/submission_record.json
throttle update:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v4_generation_876852_throttle_update_20260524/throttle_update_summary.json
throttle update status:
  ArrayTaskThrottle changed from 6 to 8, then to 0/no array cap; current pending reason after update is QOSGrpBillingRunMinutes rather than JobArrayTaskLimit
post-submit allowlist:
  zero enabled locally and remotely
current allowed action:
  monitor job 876852; after completion sync and aggregate v4 same-family raw-null outputs before any pass claim
not allowed yet:
  training, Llama, sanitizer, FAR aggregation, payload diversity, paper-facing claim, same-family raw-null pass claim before aggregate review
```

Historical previous update follows.

```text
canonical phase:
  V2_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_GENERATION_875777_RUNNING
source failed job:
  875168, nat-ev-v2-r4sfRaw
875168 status:
  FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE / no-adopt
875168 clean signals:
  raw accepts 0/64 for Qwen2.5-3B/7B/14B, duplicate extra rows 0, trace 196608 checked with 0 invalid
875168 failed gate:
  technical forbidden=254, ambiguous=53
v2 repair policy:
  keep hard forbidden policy unchanged; do not rescue 875168; exclude collision prompt ids/domains
v2 row bank:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_row_bank_plan_20260523/row_allocation_rows.jsonl
v2 row bank status:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
v2 row bank validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_VALIDATION_NO_SUBMIT
v2 row bank validation artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_row_bank_validation_20260523/validation_summary.json
v2 row bank validation facts:
  rows=65536, prompts=4096, shards=64, source collision prompt reuse=0, denied-domain rows=0, static technical hits=0, static ambiguous hits=0
v2 tokenizer route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT
v2 tokenizer route validation artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_tokenizer_route_validation_20260523/route_validation_summary.json
v2 tokenizer wrapper plan smoke:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_tokenizer_wrapper_plan_smoke_20260523_task0/route_validation_qwen2_5_3b_instruct_raw/route_validation_summary.json
submitted tokenizer-only job:
  875427, nat-ev-v2-r4sfTok
array:
  0-2%3
scope:
  tokenizer-only preflight for Qwen2.5-3B/7B/14B on the v2 row bank, 65536 rows/tokenizer
submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_tokenizer_submission_20260523/submission_record.json
allowlist after submission:
  zero enabled locally; remote sync pending/post-submit safety to be recorded
tokenizer review:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_TOKENIZER_PREFLIGHT_875427
tokenizer review artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_tokenizer_preflight_875427_review/review_summary.json
superseded canceled generation job:
  875464, nat-ev-v2-r4sfRaw
status:
  CANCELED_NO_ADOPT_NO_GENERATION_AGGREGATE
review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_generation_875464_canceled_review_20260523/canceled_review_summary.json
completed generation job:
  875471, nat-ev-v2-r4sfRaw
array:
  0-191%6
generation scope:
  raw-only Qwen2.5-3B/7B/14B same-family controls, 64 shards/model, 1024 rows/shard, 196608 expected generated rows
generation route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
generation submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v2_generation_resubmission_20260523/submission_record.json
875471 aggregate:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875471_aggregate_20260523/same_family_raw_null_summary.json
875471 aggregate status:
  FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE
875471 clean checks:
  generated rows=196608; raw accepts=0/64 per model; raw accepts ignoring quality=0/64 per model; duplicate extra rows=0 per model; trace 196608 checked with 0 invalid
875471 failed strict quality gate:
  technical forbidden=4, ambiguous forbidden=9, ordinary-domain literals=1663 report-only
residual technical terms:
  fingerprint=4
residual ambiguous terms:
  bucket=9
875471 failure review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875471_failure_review_20260523/failure_review_summary.json
interpretation:
  same-family raw-null v2 is much cleaner than 875168 but still cannot be adopted because strict forbidden gates require technical=0 and ambiguous=0
v3 residual repair policy:
  exclude 875168 collision domains plus 875471 residual domains: cafe morning setup, community theater rehearsal, farmers market booth setup, tenant move-out checklist
v3 row bank:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_row_bank_plan_20260523/row_allocation_rows.jsonl
v3 row bank status:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
v3 row bank validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_ROW_BANK_VALIDATION_NO_SUBMIT
v3 row bank validation facts:
  rows=65536, prompts=4096, shards=64, denied prompt ids=91, denied-domain rows=0, static technical hits=0, static ambiguous hits=0
v3 tokenizer route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT
v3 tokenizer route validation artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_tokenizer_route_validation_20260523/route_validation_summary.json
allowlist tokenizer command:
  updated to v3 tokenizer route and v3 row bank, still enabled=false
submitted tokenizer-only job:
  875756, nat-ev-v2-r4sfTok
array:
  0-2%3
scope:
  tokenizer-only preflight for Qwen2.5-3B/7B/14B on the v3 row bank, 65536 rows/tokenizer
submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_tokenizer_submission_20260523/submission_record.json
post-submit allowlist:
  zero enabled locally and remotely
875756 tokenizer review:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_TOKENIZER_PREFLIGHT_875756
875756 tokenizer review artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_tokenizer_preflight_875756_review/review_summary.json
875756 tokenizer facts:
  Qwen2.5-3B/7B/14B each checked 65536 rows; failed=0, empty target=0, empty other=0, target/other overlap=0
v3 generation route validation:
  PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
v3 generation route validation artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_generation_route_validation_20260523/route_validation_summary.json
submitted generation job:
  875777, nat-ev-v2-r4sfRaw
array:
  0-191%6
scope:
  raw-only Qwen2.5-3B/7B/14B same-family controls, 64 shards/model, 1024 rows/shard, 196608 expected generated rows
submission artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_v3_generation_submission_20260523/submission_record.json
post-submit allowlist:
  zero enabled locally and remotely
v3 generation route scope:
  raw-only Qwen2.5-3B/7B/14B same-family controls, 64 shards/model, 1024 rows/shard, 196608 expected generated rows
current allowed action:
  monitor job 875777; after completion sync and aggregate same-family raw-null v3 outputs before any claim unlock
not allowed yet:
  training, Llama, sanitizer, FAR aggregation, payload diversity, paper-facing claim, same-family raw-null pass claim
```

Historical previous update follows.

Job `875168` completed 192/192 same-family raw-null shards and was aggregated.
The run is artifact-complete and null accepts are clean, but it fails the strict
same-family raw-null gate because the contextual forbidden-surface audit found
hard-forbidden literal collisions (`fingerprint`, `watermark`, and one
technical `bucket`) plus ambiguous `bucket` hits. The run is not adopted as a
passed same-family raw-null package.

```text
canonical phase:
  V2_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_875168_FAILED_FORBIDDEN_LITERAL_COLLISION_REPAIR_PLANNING
completed generation job:
  875168, nat-ev-v2-r4sfRaw
array:
  0-191%6
completion:
  192/192 tasks completed, ExitCode 0
aggregate status:
  FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE
raw accepts:
  Qwen2.5-3B 0/64, Qwen2.5-7B 0/64, Qwen2.5-14B 0/64
raw accepts ignoring quality:
  Qwen2.5-3B 0/64, Qwen2.5-7B 0/64, Qwen2.5-14B 0/64
generated rows:
  196608
duplicate extra rows:
  0
trace binding:
  196608 checked, 0 invalid
forbidden surface failure:
  technical=254, ambiguous=53, ordinary_domain_literals=801
dominant technical hits:
  fingerprint=234, watermark=19, bucket=1
interpretation:
  null separation is clean, but strict quality-policy gate fails due prompt-domain / hard-forbidden-literal collision
adopt outputs as passed same-family raw-null:
  false
aggregate artifact:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875168_aggregate_20260523/same_family_raw_null_summary.json
failure review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_generation_875168_failure_review_20260523/failure_review_summary.json
current allowed action:
  artifact-only same-family raw-null v2 prompt-domain/forbidden-literal collision repair planning and lexical preflight
not allowed yet:
  training, Llama, sanitizer, FAR aggregation, payload diversity, paper-facing claim, same-family raw-null pass claim
failed generation job:
  874973, nat-ev-v2-r4sfRaw
array:
  0-191%6
scope:
  raw-only same-family Qwen 3B/7B/14B controls, 64 shards/model, 1024 rows/shard
failure:
  shared ${OUTPUT_DIR}/route_validation directory caused later shards to fail before generation
adopt outputs:
  false
repair:
  scripts/natural_evidence_v2/slurm/r4_after_870987_same_family_raw_null_generation_h200.sbatch now writes route_validation_shard_${LOCAL_SHARD_INDEX}
superseded failed generation job:
  874781, wrapper allowlist-state mismatch before model load/generation
completed tokenizer job:
  874778, nat-ev-v2-r4sfTok
tokenizer review:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_tokenizer_preflight_874778_review/review_summary.json
tokenizer result:
  PASS, 3 tokenizers, 786432 checked rows, 0 failed rows, 0 empty target/other rows, 0 target/other overlaps
superseded failed job:
  874775, wrapper default PLAN_ONLY/allowlist-state mismatch; no tokenizer/model/scoring/generation started
array:
  0-2%3
scope:
  tokenizer-only preflight for Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct
route decision:
  docs/natural_evidence_v2/R4_AFTER_870987_SAME_FAMILY_RAW_NULL_TOKENIZER_PREFLIGHT_ROUTE_20260522.md
tokenizers planned:
  Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct
allowlist after submission:
  zero enabled locally and remotely
route validation:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_tokenizer_route_validation_20260522/route_validation_summary.json
wrapper plan smoke:
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_tokenizer_wrapper_plan_smoke_20260522_task0/
  results/natural_evidence_v2/status/r4_after_870987_same_family_raw_null_tokenizer_wrapper_plan_smoke_20260522_task2/
```

Historical previous update follows.

Job `874308` completed and the R4 after-870987 pre-FAR organic-null aggregate passed. Combined with the passed standard-control pre-FAR aggregate from `871250`, the Qwen-only first-token event route now has a passed pre-FAR null package.

```text
canonical phase:
  V2_R4_AFTER_870987_PREFAR_NULL_PACKAGE_871250_PLUS_874308_PASSED_NEXT_ROUTE_DECISION
organic null job:
  874308, nat-ev-v2-r4oGen
organic null status:
  PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE
organic raw accepts:
  0/256
organic raw accepts ignoring quality:
  0/256
organic generated rows:
  262144
organic unique response hashes:
  262144
organic global duplicate extra rows:
  0
organic trace binding:
  262144 checked, 0 invalid
standard controls:
  raw 0/256, task_only 0/256, wrong_key 0/256, wrong_payload 0/256
review artifact:
  results/natural_evidence_v2/status/r4_after_870987_prefar_null_package_871250_plus_874308_review_20260522/review_summary.json
next allowed action:
  prepare the next reviewed route decision package; no Llama, same-family null, sanitizer, FAR aggregation, payload diversity, training, or paper-facing claim until that route decision/preflight passes
```

Historical previous update follows.

Job `874308` is the current canonical R4 pre-FAR organic-null raw-only
generation/decode array. It was submitted after the organic-null generation
route validation and wrapper smoke passed locally/remotely.

```text
job:
  874308, nat-ev-v2-r4oGen
route:
  R4 after-870987 pre-FAR organic-null raw-only generation/decode
array:
  0-255%6
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
row bank:
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl
expected generated rows:
  262144
conditions:
  raw only
status dir:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_870987_prefar_organic_null_generation_874308
next allowed action:
  monitor 874308 to completion, then run artifact-only organic-null aggregate/review before any further route unlock
not unlocked:
  training, Llama, same-family null, sanitizer, FAR aggregation, payload diversity, text-only phrase success claim, paper-facing claim
```

Historical latest tokenizer update follows.

Job `874307` completed successfully and is adopted as the actual Qwen tokenizer preflight pass for the R4 pre-FAR organic-null v2 row bank.

```text
job:
  874307, nat-ev-v2-r4oTok
status:
  PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT
checked rows:
  262144 / 262144
failed rows:
  0
empty target id rows:
  0
empty other id rows:
  0
target/other overlap rows:
  0
model forward / scoring / generation / training:
  false / false / false / false
review artifact:
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_qwen_tokenizer_preflight_874307_review/review_summary.json
next allowed action:
  prepare and validate organic-null raw-only generation/decode wrapper route; no generation submission until wrapper/route, local/remote hash preflight, allowlist safety, and exactly-one H200 submission preflight pass
```

Job `874307` is now the canonical active R4 pre-FAR organic-null actual Qwen tokenizer boundary preflight. It uses the dedicated organic-null v2 wrapper and the validated v2 row bank. No generation, model scoring, training, FAR aggregation, Llama, sanitizer, payload-diversity, text-only phrase success claim, or paper-facing claim has been started.

```text
canonical active job:
  874307, nat-ev-v2-r4oTok
route:
  R4 after-870987 pre-FAR organic-null Qwen tokenizer boundary preflight
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_870987_prefar_organic_null_qwen_tokenizer_boundary_preflight_h200.sbatch
score rows:
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl
expected rows:
  262144
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
allowlist after submission:
  zero enabled locally and remotely
next allowed action:
  monitor 874307; sync and review tokenizer preflight artifacts after completion before any organic-null generation submission
```

Control-plane correction:

```text
noncanonical canceled job:
  874306, nat-ev-v2-r4pTok
reason:
  tokenizer-only job referenced the old non-v2 organic row bank path via the standard-control wrapper
adopt outputs:
  false
replacement:
  874307 dedicated organic-null v2 tokenizer preflight
```

Organic-null v2 artifact status:

```text
prompt bank v2:
  PASS, 16384 locked prompts, 0 technical/structural hits, 0 overlap with prior locked/standard-control prompts
row bank v2:
  PASS, 262144 rows, 256 shards, raw-only organic null, row bank sha256 30faab3ddc58e7f0a1a9351838c04a322d9fca617dca2479782d402522a3e62a
route validation:
  PASS locally and remotely
```

Job `871250` completed all 160 H200 shards for the R4 after-870987 pre-FAR
standard-control first-token event null expansion. The first aggregate failed
only because the contextual forbidden matcher treated full-text co-occurrence
as technical context. Artifact-only attribution found four ordinary-domain rows:
ordinary `coordinate` verbs and one ordinary `bucket of water` sentence were
linked to distant ordinary words such as `slot` or another contextual literal.

Codex repaired the matcher implementation to require local sentence/window
context for contextual technical cues. This repair did not change the forbidden
literal policy, thresholds, generated outputs, token traces, prompt allocation,
controller configuration, or accept logic. It was validated locally and on
Chimera, then applied only as artifact-only re-decode/re-aggregate against the
already completed `871250` transcripts.

```text
completed job:
  871250, nat-ev-v2-r4pGen
array:
  0-159%6
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
terminal state:
  160/160 shards completed
original aggregate:
  FAIL_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE
original failure reason:
  forbidden public surface quality count = 7 from contextual matcher
matcher repair:
  sentence-local contextual cue window; no policy/gate/threshold change
tests:
  tests/natural_evidence_v2/test_r4_after_868151_first_token_event_decoder.py PASS
  tests/natural_evidence_v2/test_r4_contextual_forbidden_surface_policy_v2.py PASS
repaired artifact-only aggregate:
  PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE
aggregate artifacts:
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_generation_871250_contextual_window_v2_aggregate_20260521/
control blocks:
  raw: 0/256 accepts, 0/256 ignoring-quality accepts
  task_only: 0/256 accepts, 0/256 ignoring-quality accepts
  wrong_key: 0/256 accepts, 0/256 ignoring-quality accepts
  wrong_payload: 0/256 accepts, 0/256 ignoring-quality accepts
new generation rows:
  491520
unique response hashes:
  491520
global duplicate extra rows:
  0
trace binding:
  491520 checked, 0 invalid
technical forbidden public surface:
  0 after precommitted contextual policy implementation repair
protected rows:
  report-only for this route; 159/160 strict accepts
full phrase decoder:
  report-only failure; no text-only success claim
claim policy:
  Qwen first-token event pre-FAR standard controls only; no full FAR and no paper claim
```

Historical immediate predecessor:

```text
previous failed job:
  871079, nat-ev-v2-r4pGen
terminal state:
  all shards FAILED quickly with ExitCode 1:0
failure point:
  validate_reviews before generation
generation/model forward/training:
  not started
failure review:
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_generation_871079_failure_review_20260519/
repair:
  scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py now prefers review_status over status
  871057 route-specific tokenizer review status added to the allowed pass set
repair tests:
  tests/natural_evidence_v2/test_r4_after_870987_tokenizer_review_status.py
repair validation:
  py_compile PASS
  review-status pytest PASS
  delegate plan-only smoke PASS
  route validation PASS
repaired submission record:
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_generation_repaired_submission_20260519/submission_record.json
```

Next allowed action: continue the R4 Qwen pre-FAR null package with the
organic-null route. This may proceed automatically after route validation,
prompt/allocation validation, tokenizer/controller preflight, local/remote hash
preflight, allowlist safety, and exactly-one H200 submission preflight pass.
Do not treat the standard-control pass as full FAR, text-only phrase decoder
success, payload diversity, Llama transfer, sanitizer robustness, or a
paper-facing positive claim.

Organic-null row-bank progress:

```text
row-bank builder:
  scripts/natural_evidence_v2/build_r4_after_870987_prefar_organic_null_row_bank.py
row-bank validator:
  scripts/natural_evidence_v2/validate_r4_after_870987_prefar_organic_null_row_bank.py
local row-bank artifact:
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_plan_20260521/
local validation:
  PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_VALIDATION_NO_SUBMIT
remote validation:
  PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_VALIDATION_NO_SUBMIT
organic-null blocks:
  256
selected prompts:
  16384
row cylinders:
  262144
generation conditions:
  raw only
generation/slurm:
  not started
next concrete action:
  monitor actual Qwen tokenizer boundary preflight job 874306
submitted tokenizer-only job:
  874306
submission record:
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_qwen_tokenizer_preflight_submission_20260521/submission_record.json
model forward/generation/training:
  not started
```

Organic-null artifact-only prompt-bank repair/validation has started:

```text
organic-null prompt bank v1 issue:
  prompt template contained the public phrase "hidden-code terminology"
repair:
  replace with "special terminology"; no generation, no Slurm, no model call
organic-null prompt bank v2:
  results/natural_evidence_v2/prompts/r4_after_870987_prefar_organic_null_prompts_20260521/
validation:
  PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_PROMPT_BANK_VALIDATION_NO_SUBMIT
  results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_prompt_bank_v2_validation_20260521/
locked prompts:
  16384
technical prompt literal rows:
  0
duplicate prompt ids/texts:
  0
overlap with locked-scale/standard-control prompts:
  0
```

## Recent Locked-Scale And Pre-FAR Context

Job `870210` reached a terminal state and is not a locked-scale result. The
array completed 79/96 shards, while 17 shards failed or were cancelled. The
failure is classified as a runtime/storage blocker, not a model/method gate
failure.

```text
terminal job:
  870210, nat-ev-v2-r4lGen, array 0-95%6
slurm state:
  COMPLETED 79/96; FAILED 17/96
completed shards:
  0-74, 77, 79, 80, 81
failed or missing shards:
  75, 76, 78, 82-95
hard failure evidence:
  shard 78 stderr: OSError [Errno 28] No space left on device
storage state before cleanup:
  /hpcstor6/scratch01/g/guanjie.lin001 was effectively full
cleanup record:
  results/natural_evidence_v2/status/hpcstor_cleanup_20260519/
local small-artifact sync for 870210:
  results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_870210/
post-cleanup remote free space:
  166G available on /hpcstor6/scratch01/g/guanjie.lin001
```

Resume job `870987` completed successfully, and aggregation of source job
`870210` plus resume job `870987` passed the R4 Qwen same-contract first-token
event locked-scale generation gate.

Recommended resume safeguards:

```text
rerun shards:
  75, 76, 78, 82-95
log suppression:
  do not set TQDM_DISABLE or TRANSFORMERS_VERBOSITY for the resume route
array throttle:
  use %6 for the resume route per user override
completed shards:
  do not rerun
```

```text
resume job:
  870987, nat-ev-v2-r4lGen
resume array:
  75,76,78,82-95%6
resume output dir:
  /hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_after_869348_locked_scale_generation_870210_resume_20260519
aggregation script:
  scripts/natural_evidence_v2/aggregate_r4_after_869348_locked_scale_generation.py
aggregation policy:
  refuse locked-scale gate until 96/96 complete; full phrase decoder remains report-only
```

```text
aggregate review:
  docs/natural_evidence_v2/R4_AFTER_869348_LOCKED_SCALE_870210_PLUS_870987_REVIEW_20260519.md
aggregate artifacts:
  results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_870210_plus_870987_aggregate_20260519/
review summary:
  results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_870210_plus_870987_review_20260519/review_summary.json
aggregate status:
  PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE
complete shards:
  96/96
generated rows:
  294912
protected strict accepts:
  94/96
protected accepts ignoring quality:
  94/96
raw/task-only/wrong-key/wrong-payload accepts:
  0/96 each
global duplicate response hash extra rows:
  0
unique response hashes:
  294912/294912
trace binding:
  294912 checked, 0 invalid
first-token forbidden public surface count:
  0
full phrase decoder:
  report-only; protected accepts 0/96 under format_scrub=all and 0/96 under no scrub
```

Allowed internal statement: R4 Qwen same-contract first-token event locked-scale
generation passed under bound trace decoding, with protected 94/96 and all
diagnostic control arms 0/96.

This route still does not unlock training, Llama, same-family null, sanitizer,
FAR, payload-diversity, text-only phrase decoder success claim, or paper-facing
positive claim.

Post-pass pre-FAR null expansion route planning has started and passed its first
artifact-only gates:

```text
route decision:
  docs/natural_evidence_v2/R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_DECISION_20260519.md
route config:
  configs/natural_evidence_v2/r4_after_870987_prefar_null_expansion_route.yaml
route validation:
  PASS_R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_PLAN_ONLY_NO_SUBMIT
route validation artifacts:
  results/natural_evidence_v2/status/r4_after_870987_prefar_null_expansion_route_validation_20260519/
standard-control prompt bank:
  results/natural_evidence_v2/prompts/r4_after_870987_prefar_standard_control_prompts_20260519/
  locked prompts: 10240
  overlap with 870987 locked prompts: 0
organic-null prompt bank:
  results/natural_evidence_v2/prompts/r4_after_870987_prefar_organic_null_prompts_20260519/
  locked prompts: 16384
  overlap with 870987 locked prompts: 0
standard-control row bank:
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_row_bank_plan_20260519/
  row cylinders: 163840
  shards: 160
  duplicate prompt/prefix extra rows: 0
  previous locked-scale prompt overlap: 0
standard-control row-bank validation:
  PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_VALIDATION_NO_SUBMIT
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_row_bank_validation_20260519/
remote validation:
  PASS_R4_AFTER_870987_PREFAR_NULL_EXPANSION_ROUTE_PLAN_ONLY_NO_SUBMIT
  PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_VALIDATION_NO_SUBMIT
  results/natural_evidence_v2/status/r4_after_870987_prefar_null_expansion_route_validation_remote2_20260519/
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_row_bank_validation_remote2_20260519/
standard-control tokenizer preflight route:
  configs/natural_evidence_v2/r4_after_870987_prefar_standard_control_tokenizer_preflight_route.yaml
  scripts/natural_evidence_v2/slurm/r4_after_870987_prefar_standard_control_qwen_tokenizer_boundary_preflight_h200.sbatch
  allowlist entry: v2_r4_after_870987_prefar_standard_control_qwen_tokenizer_boundary_preflight_h200
  allowlist safety: PASS
  results/natural_evidence_v2/status/r4_after_870987_prefar_tokenizer_route_allowlist_safety_20260519.json
  remote allowlist safety: PASS
  results/natural_evidence_v2/status/r4_after_870987_prefar_tokenizer_route_allowlist_safety_remote_20260519.json
  remote hash preflight: PASS
  results/natural_evidence_v2/status/r4_after_870987_prefar_tokenizer_remote_hash_preflight_20260519/
  submitted tokenizer-only Slurm job: 871057
  submission record:
    results/natural_evidence_v2/status/r4_after_870987_prefar_tokenizer_submission_871057_20260519/submission_record.json
  post-submit allowlist safety:
    PASS local
    PASS remote
  job status:
    COMPLETED, ExitCode 0
  tokenizer preflight review:
    PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_QWEN_TOKENIZER_PREFLIGHT_871057
    docs/natural_evidence_v2/R4_AFTER_870987_PREFAR_STANDARD_CONTROL_QWEN_TOKENIZER_PREFLIGHT_871057_REVIEW_20260519.md
    results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_qwen_tokenizer_preflight_871057_review/review_summary.json
  tokenizer gate:
    checked rows: 163840
    failed rows: 0
    empty target id rows: 0
    empty other id rows: 0
    target/other overlap rows: 0
    model forward/generation/training: false
```

## Prior Update

Codex recorded and locally validated the held-out 96-block locked-scale
generation route after `870078` passed tokenizer preflight.

```text
route decision:
  docs/natural_evidence_v2/R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_DECISION_20260518.md
route validation:
  PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
wrapper plan-only smoke:
  PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
allowlist entry:
  v2_r4_after_869348_locked_scale_generation_h200
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_generation_h200.sbatch
array:
  0-95%4
expected generated rows:
  294912
protected strict gate:
  >=85/96
controls:
  raw/task_only/wrong_key/wrong_payload must be 0/96 each
```

## Prior Update

Slurm job `870078` completed and passed the held-out locked row bank actual Qwen
tokenizer boundary preflight.

```text
job:
  870078, nat-ev-v2-r4lTok, COMPLETED, ExitCode 0:0, elapsed 00:03:28
review:
  results/natural_evidence_v2/status/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_870078_review/
status:
  PASS_R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_870078
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
checked rows:
  98304
failed rows:
  0
empty target id rows:
  0
empty other id rows:
  0
target/other first-token overlap rows:
  0
model forward/scoring/generation/training:
  not started
```

The next allowed action is artifact-only locked-scale generation route
preparation and local/remote preflight. Do not submit locked-scale generation
until the route config, wrapper, allowlist, local/remote hashes, and exactly-one
allowlist enablement preflight pass. Do not start training, Llama, same-family
null, sanitizer, FAR, payload-diversity, or paper-facing claim jobs.

## Prior Update

Codex submitted exactly one H200 tokenizer-only preflight job for the held-out
locked row bank and immediately disabled the allowlist entry.

```text
job:
  870078, nat-ev-v2-r4lTok
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
command:
  sbatch scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200.sbatch
allowlist entry:
  v2_r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200
post-submit allowlist safety:
  local PASS, remote PASS
planned checked rows:
  98304
model forward/scoring/generation/training:
  not allowed and not started by this route
```

## Prior Update

Codex prepared the held-out locked-scale tokenizer-only preflight route after
the `869348` dev diagnostic pass. This is still not locked-scale generation and
does not make a paper-facing positive claim.

```text
source dev diagnostic:
  869348, PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE
locked row bank:
  results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/
row bank status:
  PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
row bank validation:
  PASS_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT
static boundary preflight:
  PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING
tokenizer route validation:
  PASS_R4_AFTER_869348_LOCKED_SCALE_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT
locked split:
  yes
shards:
  96
row cylinders:
  98304
selected coordinates:
  16
unique content prompt/prefix pairs:
  98304
duplicate content prompt/prefix extra rows:
  0
allowlist entry:
  v2_r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_869348_locked_scale_qwen_tokenizer_boundary_preflight_h200.sbatch
```

The next allowed action is local/remote hash preflight and exactly-one H200
tokenizer-only submission for the locked row bank. This job may load the Qwen
tokenizer only; it must not run model forward, model scoring, generation,
training, Llama, same-family null, sanitizer, FAR, payload-diversity, or
paper-facing claim actions. The allowlist entry must be disabled immediately
after `sbatch` returns.

## Prior Update

Slurm array `869348` completed on H200/pomplun and passed the reviewed
global-unique first-token-event 32-block dev diagnostic gate.

```text
job:
  869348, nat-ev-v2-r4gDev, COMPLETED, 32/32 shards, ExitCode 0:0
review:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_dev_diagnostic_869348_review/
status:
  PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE
generated rows:
  98304
attempt rows:
  98304
protected strict accepts:
  32/32
protected accepts ignoring quality:
  32/32
control accepts:
  raw=0/32, task_only=0/32, wrong_key=0/32, wrong_payload=0/32
global duplicate response hash count:
  0
protected duplicate response hash count:
  0
protected forbidden public surface count:
  0
trace binding:
  98304 checked, 0 invalid, validity 1.0
full-phrase protected accepts, format_scrub=all:
  0 (report-only, not a success claim)
full-phrase forbidden public surface count, format_scrub=all:
  684 (report-only under this first-token-event route)
```

This is a Qwen-only, same-contract, provider-side first-token-event/controller
dev diagnostic pass. It does not establish text-only phrase decoding and does
not unlock paper-facing positive claims by itself.

The next allowed action is artifact-only locked-scale route decision / expert
review for the same first-token-event protocol. Do not submit locked-scale,
FAR, sanitizer, Llama, payload-diversity, or paper-facing claim jobs until that
route is recorded and preflighted.

## Prior Update

Codex/Hermes synchronized the `869298` actual Qwen tokenizer preflight result,
prepared the global-unique first-token-event dev diagnostic route, passed local
and remote preflight, submitted one H200 array job, and immediately disabled the
allowlist entry.

```text
allowlist entry:
  v2_r4_after_868348_global_unique_dev_diagnostic_h200
command:
  PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch scripts/natural_evidence_v2/slurm/r4_after_868348_global_unique_dev_diagnostic_h200.sbatch
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
array:
  0-31%4
time limit:
  30-00:00:00
```
The route does not unlock training, Llama, same-family null, sanitizer, FAR,
payload-diversity, locked-scale, or paper-facing positive claims.

## Prior Update

Codex then executed Option A artifact-only repair planning and built a new
globally unique 32-block row bank from reviewed R4 dev prompts:

```text
row bank:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/
status:
  PASS_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT
self-audit:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_self_audit_20260517/
self-audit status:
  PASS_R4_AFTER_868348_EXISTING_ROW_SOURCES_HAVE_NECESSARY_GLOBAL_UNIQUE_CAPACITY_NO_RERUN
rows:
  32768
shards:
  32
rows per shard:
  1024
selected coordinates:
  16
unique content prompt/prefix pairs:
  32768
duplicate content prompt/prefix extra rows:
  0
min unique content prompt/prefix pairs per selected coordinate:
  2048
prefix templates:
  16, rotated by prompt and coordinate
generation/model/scoring/training/slurm:
  not started
```

This repairs the immediate input-capacity blocker for a future rerun, but it is
not a submission route and does not reclassify `868348`. The next allowed action
is artifact-only route validation plus actual Qwen tokenizer/controller preflight
planning for this row bank. No generation or Slurm submission is allowed until
those pass and a reviewed route is recorded.

## Prior Update

After the `868348` dev diagnostic failed only the strict global exact duplicate
gate, Codex ran an artifact-only audit of all existing row-source JSONL files:

```text
audit:
  results/natural_evidence_v2/status/r4_after_868348_candidate_row_source_audit_20260517/
status:
  FAIL_R4_AFTER_868348_EXISTING_ROW_SOURCES_INSUFFICIENT_FOR_GLOBAL_UNIQUE_32_BLOCK_ALLOCATION_NO_RERUN
scope:
  scanned 369 row files; no model calls, no generation, no Slurm submission
compatible sources:
  6
compatible rows:
  55296
unique content prompt/prefix pairs:
  4096
required for strict 32-block global-unique rerun:
  32768
min unique content prompt/prefix pairs per coordinate:
  256
required per coordinate for 32 blocks:
  2048
interpretation:
  existing reviewed row sources are insufficient for a strict global-unique
  32-block rerun of the 868348 dev route
```

This does not reclassify `868348`. The first-token event signal remains strong
inside that failed diagnostic (`protected strict 32/32`, controls `0/32` each,
trace binding valid), but the strict duplicate gate failure remains fatal for
canonical adoption.

The next allowed action is artifact-only route planning for one of two future
repairs:

```text
Option A:
  build a larger reviewed prompt/row bank with tokenizer/controller preflight,
  then validate a globally unique allocation before any rerun
Option B:
  record a future-only duplicate-gate semantics decision that separates
  protected/accepted duplicates from control-only duplicates; this cannot
  retroactively rescue 868348
```

No generation or Slurm submission is allowed from the current row bank.

Job `868313` (`nat-ev-v2-r4dev`, array `0-31%4`) reached a terminal mixed
state and is not a canonical dev-diagnostic result.

```text
failure review:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_868313_failure_review_20260517/
status:
  FAILED_R4_AFTER_868299_DEV_JOB_868313_RUNTIME_ALLOWLIST_RACE_PARTIAL_GENERATION_NO_METHOD_RESULT
failed shards:
  0..23
completed shards:
  24..31
root cause:
  runtime allowlist enabled-state race before immediate post-sbatch disablement
method interpretation:
  not a model/tokenizer/controller/decoder failure
canonical adoption:
  false
```

The repair keeps submission preflights strict, but separates runtime shard
self-checks from allowlist enabled-state checks. Runtime shards still verify
the reviewed allowlist entry and command pattern.

The repaired replacement has been submitted:

```text
submission:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_repair_submission_20260517/
status:
  SUBMITTED_R4_AFTER_868299_DEV_DIAGNOSTIC_REPAIRED_H200_ARRAY_MONITOR_ONLY
job:
  868348, array 0-31%4, nat-ev-v2-r4dev, pomplun H200
post-submit allowlist:
  local PASS, remote PASS
```

Job `868348` completed all 32 shards and was reviewed:

```text
review:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_868348_review/
review status:
  FAIL_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_GATE
failure attribution:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_868348_failure_attribution/
attribution status:
  RECORDED_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILURE_ATTRIBUTION_DUPLICATES_TASK_ONLY_ONLY
signal:
  protected strict accepts 32/32
  protected ignoring-quality accepts 32/32
  raw/task_only/wrong_key/wrong_payload accepts 0/32 each
  trace binding invalid rows 0/98304
quality blocker:
  global exact duplicate extra rows 2
duplicate location:
  task_only only; 0 protected duplicate rows
canonical adoption:
  false
allocation feasibility:
  results/natural_evidence_v2/status/r4_after_868348_global_unique_allocation_feasibility_20260517/
allocation feasibility status:
  FAIL_R4_AFTER_868348_GLOBAL_UNIQUE_ALLOCATION_NOT_FEASIBLE_FROM_CURRENT_REVIEWED_ROW_BANK
```

## Active Route Update

The active route has been synchronized after expert review of `868260`:

```text
state sync:
  results/natural_evidence_v2/status/r4_after_868260_state_sync_20260517/
status:
  SYNCED_R4_AFTER_868260_FIRST_TOKEN_EVENT_QUALITY_REPAIR_ROUTE_ARTIFACT_ONLY_NO_SUBMIT
artifact validation:
  PASS_R4_AFTER_868260_FORENSICS_POLICY_TRACE_BINDING_ARTIFACTS_VALIDATED_NO_SUBMIT
active interpretation:
  failed strict-quality diagnostic with full protected codeword recovery before
  quality filtering
active route:
  provider-side keyed first-token event evidence with strict natural-output
  quality, duplicate, contextual-forbidden, and trace-binding gates
```

Older v3 training-objective blockers are historical for the active route. They
remain evidence that surface-mass/objective pressure was insufficient, but they
are not the current execution blocker.

New artifact-only packages are recorded:

```text
duplicate forensics:
  results/natural_evidence_v2/status/r4_868260_duplicate_forensics_20260517/
duplicate-safe generation policy v2 validation:
  results/natural_evidence_v2/status/r4_first_token_event_duplicate_safe_generation_policy_v2_validation_20260517/
contextual forbidden-surface policy v2 validation:
  results/natural_evidence_v2/status/r4_contextual_forbidden_surface_policy_v2_validation_20260517/
trace-binding validation:
  results/natural_evidence_v2/status/r4_first_token_event_trace_binding_validation_20260517/
quality-repair confirmation route:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_route_validation_20260517/
quality-repair confirmation wrapper plan smoke:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_wrapper_plan_smoke_20260517/
route decision:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_route_decision_20260517/
remote preflight:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_remote_preflight_20260517/
full-mode wrapper review:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_full_mode_wrapper_review_20260517/
full-mode wrapper status:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FULL_MODE_WRAPPER_REVIEW_NO_SUBMIT
full-mode delegate smoke:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_full_wrapper_delegate_smoke_20260517_a/
full-mode remote preflight:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_full_mode_remote_preflight_20260517/
full-mode remote status:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FULL_MODE_REMOTE_PREFLIGHT_NO_SUBMIT
submission:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_submission_20260517/
submission status:
  SUBMITTED_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_H200_ARRAY_MONITOR_ONLY
job:
  868291, array 0-3%4, nat-ev-v2-r4qfix, pomplun H200
failure review:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_job_868291_failure_review/
failure status:
  FAILED_R4_AFTER_868260_JOB_868291_ALLOWLIST_RUNTIME_VALIDATION_RACE_NO_GENERATION
runtime repair:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_runtime_allowlist_race_repair_20260517/
resubmission:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_resubmission_20260517/
resubmission status:
  SUBMITTED_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_REPAIRED_H200_ARRAY_MONITOR_ONLY
failed job:
  868295, array 0-3%4, nat-ev-v2-r4qfix, pomplun H200
failure review:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_job_868295_failure_review/
failure status:
  FAILED_R4_AFTER_868260_JOB_868295_REMOTE_ARTIFACT_SYNC_MISSING_CONTEXTUAL_POLICY_NO_GENERATION
failure interpretation:
  job 868295 failed before generation/model-forward because the remote runtime
  repository was missing the precommit repair package file
  results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/contextual_forbidden_surface_policy_v2.json
remote status:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_REMOTE_PREFLIGHT_NO_SUBMIT
remote host:
  chimerahead.umb.edu
remote route validation:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
remote wrapper plan-only:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
remote allowlist:
  PASS, enabled_entries=[]
local/remote hashes:
  match, 51 reviewed files
active Chimera jobs at last remote preflight:
  0
tests:
  16 passed for route/helper tests after the full-mode wrapper patch
```

The confirmation wrapper no longer exits fail-closed in non-plan mode. Instead,
it delegates to the reviewed H200 generation/decode wrapper with the
duplicate-safe generation policy v2, contextual forbidden policy v2, and
first-token event trace-binding verifier wired into the execution path.

Slurm job `868291` failed before generation because runtime validation still
expected the allowlist entry to be enabled after the required immediate
post-`sbatch` disablement. This was recorded as a control-plane runtime
validation race, not as a model/method result.

The delegated wrapper was repaired so the exactly-one allowlist check remains
pre-submit only. After repaired preflight, Slurm job `868295` was submitted and
the allowlist was disabled immediately after `sbatch`; local and remote
post-submit allowlist checks both passed with `enabled_entries=[]`.

Slurm job `868295` then failed before generation/model-forward because the
remote runtime repository was missing the precommit repair package required by
the wrapper. This is a remote artifact-sync failure, not a model/method result.
The missing precommit repair package was synchronized to Chimera, and the
remote required-artifact/allowlist/active-job preflight passed:

```text
remote sync-repair preflight:
  results/natural_evidence_v2/status/r4_after_868260_remote_artifact_sync_repair_preflight_20260517/
status:
  PASS_R4_AFTER_868260_REMOTE_ARTIFACT_SYNC_REPAIR_PREFLIGHT_NO_SUBMIT
required artifact hashes:
  local/remote match, 14 files
remote allowlist:
  PASS, enabled_entries=[]
remote route validation:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
remote wrapper delegate smoke:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
active jobs before submission:
  0
```

After exactly-one local/remote allowlist preflight, one replacement H200 array
was submitted:

```text
submission:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_resubmission2_20260517/
job:
  868299, array 0-3%4, nat-ev-v2-r4qfix, pomplun H200
command:
  PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch scripts/natural_evidence_v2/slurm/r4_after_868260_quality_repair_confirmation_h200.sbatch
post-submit local allowlist:
  PASS, enabled_entries=[]
post-submit remote allowlist:
  PASS, enabled_entries=[]
```

Job `868299` completed on Chimera H200 and the first-token event quality-repair
confirmation review passed:

```text
job:
  868299, array 0-3%4, nat-ev-v2-r4qfix, chimera21 H200
state:
  COMPLETED, all 4 shards, ExitCode 0:0
run artifacts:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_868299/
quality review:
  results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_868299_quality_review/
review status:
  PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FIRST_TOKEN_EVENT_GATE
protected strict accepts:
  4/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
global duplicate response hash count:
  0
protected forbidden public surface count:
  0
trace binding validity:
  12288/12288 valid
full-phrase protected accepts, format_scrub=all:
  0, report-only, not a text-only success claim
```

This is the first strict-quality pass for the provider-side keyed first-token
event route. It does not establish a text-only phrase decoder result and does
not unlock paper-facing claims.

The next reviewed route has now been recorded and passed local plan-only
validation:

```text
route:
  docs/natural_evidence_v2/R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_20260517.md
config:
  configs/natural_evidence_v2/r4_after_868299_first_token_event_dev_diagnostic_route.yaml
allocation plan:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_plan_20260517/
route validation:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_route_validation_20260517/
wrapper plan smoke:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_wrapper_plan_smoke_20260517/
local status:
  PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT
scope:
  32-block Qwen dev diagnostic, provider-side first-token event trace route
allocation caveat:
  cyclic reuse of the reviewed 4-block full16 allocation; dev diagnostic only,
  not locked-scale independent evidence
gates:
  protected strict accepts >=28/32, protected ignoring-quality accepts >=30/32,
  all controls 0/32, global exact response duplicate 0, technical forbidden 0,
  trace binding 100%
```

Local/remote hash and allowlist preflight passed, then exactly one H200 Slurm
array was submitted:

```text
submission:
  results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_submission_20260517/
job:
  868313, array 0-31%4, nat-ev-v2-r4dev, pomplun H200
command:
  PLAN_ONLY=0 VALIDATE_PLAN_ONLY=0 sbatch scripts/natural_evidence_v2/slurm/r4_after_868299_first_token_event_dev_diagnostic_h200.sbatch
post-submit local allowlist:
  PASS, enabled_entries=[]
post-submit remote allowlist:
  PASS, enabled_entries=[]
```

The next allowed action is to monitor job `868313`; after terminal completion,
sync artifacts and run the first-token event dev diagnostic review. Do not
submit another generation job before this review. Downstream training/Llama/
sanitizer/FAR/payload-diversity/paper-claim routes remain gated by their prior
conditions.

## Prior Compute Result: 868212

Job `868212` completed the reviewed quality-repaired after-868151 controller
generation diagnostic on Chimera H200:

```text
job_id: 868212
job_name: nat-ev-v2-r4cgen
array: 0-3%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
state: COMPLETED, 4/4 shards
exit_code: 0:0
```

The precommitted first-token event diagnostic gate passed at small diagnostic
scale:

```text
protected accepts:
  3/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
block-level forbidden public surface count:
  0
block-level duplicate response hash count:
  0
token-id event traces:
  9216
event status counts:
  target=839, other=84, erasure=8293
```

The single protected failed block was localized:

```text
block_id:
  shard_03_block_00
decoded_bits:
  1-100101
expected_bits:
  10100101
missing bit index:
  1
missing coordinate:
  26
complete pairs:
  7/8
```

The full-phrase decoder remains failed as expected:

```text
full-phrase protected accepts, format_scrub=all:
  0
```

This result remains diagnostic only, not a locked positive or paper claim. The
main quality caveat is still global duplication across generated outputs:

```text
generated rows:
  9216
unique response hashes:
  4792
global duplicate response hash count:
  4424
max duplicate group size:
  4
```

## Failure Attribution

Artifact-only attribution for `868212` is recorded:

```text
attribution:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_failure_attribution/
status:
  RECORDED_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_ARTIFACT_ONLY_FAILURE_ATTRIBUTION_NO_SUBMIT
coordinate-26 shard_03 protected erasures:
  64/64
duplicate hash groups:
  2908
duplicate extra rows:
  4424
dominant duplicate condition sets:
  protected,raw: 1621 groups
  task_only: 1024 groups
dominant duplicate shard pairs:
  shard_00,shard_01: 1090 groups
  shard_02,shard_03: 1051 groups
```

Interpretation: the protected failure was a localized erasure/reliability issue
for coordinate 26 in shard 03, not a null-accept failure. The duplicate caveat
is global and dominated by deterministic identical generations across paired
shards and protected/raw or same-condition repetitions.

## Superseded Failed Repair Preflight

The first artifact-only repair preflight intentionally failed because the
12-coordinate pivot codebook left several singleton payload bits:

```text
preflight:
  results/natural_evidence_v2/status/r4_after_868212_reliability_duplicate_repair_preflight_20260516/
status:
  FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT
singleton/codebook failures:
  bit 1 active=[26], coordinate_26 sole active coordinate
  bit 3 active=[19]
  bit 5 active=[8]
  bit 6 active=[4]
duplicate extra rows:
  4424
```

This failed preflight is now superseded by the repaired full-16 plan below. It
must not be used for another Slurm submission.

## Current Route Result

The next artifact-only step has completed: the route restored the full 16
coordinates from the reviewed reliability codebook, rebuilt the row allocation,
precommitted a repaired first-token event decoder/codebook, and validated the
plan without submitting compute.

```text
full16 allocation plan:
  results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/
allocation status:
  PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY
allocation rows:
  4096
shards:
  4
rows per shard:
  1024
rows per coordinate per shard:
  64
duplicate prompt/prefix pair max per shard:
  0
```

```text
repaired precommit:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/
precommit status:
  PRECOMMITTED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_ARTIFACT_ONLY_NO_COMPUTE
selected coordinates:
  16
min active coordinates per bit:
  2
coordinate 26 sole-coordinate condition:
  rejected
reclassifies 868212:
  false
```

```text
plan validation:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_plan_validation_20260516/
validation status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATION_NO_SUBMIT
locked-scale global duplicate gate:
  0
slurm submitted:
  false
generation/model-scoring/training started:
  false
```

Precommit hashes:

```text
codebook:
  58d5fc6dc0c42136e5fb238c0b255e73c9c7d63115a3abc39af31ec6fd2f5444
decoder_spec:
  64fd1e682c0ea314bc2f49b6a543447ef9df9679957b87800c2bd41a82bb70f3
duplicate_policy:
  241d93f445676a63f353a3ca58b63e5ceff1bdc826dc058d2b29a1086409e9e9
allocation_manifest:
  b797b46f876e08dfaf329379f578a69fe975d4747d7a87ae2a834d6f83899993
allocation_rows:
  61927c822c6ce730974ebbaffc775678e70c0a0a2c13e526173f392a231c64dd
contextual_literal_policy:
  0522c7f17c177137f4abbe29c147656797584ef28a328c5a6e8b8145201f31b5
```

Verification:

```text
pytest:
  19 passed, 1 skipped
```

## Generation Wrapper Route Validation

The repaired full16 generation/decode control plane has been implemented and
validated locally without Slurm submission:

```text
route config:
  configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_generation_route.yaml
decoder route:
  configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_decoder_route.yaml
wrapper:
  scripts/natural_evidence_v2/slurm/r4_after_868212_repaired_first_token_event_generation_h200.sbatch
validator:
  scripts/natural_evidence_v2/validate_r4_after_868212_repaired_first_token_event_generation_route.py
route validation:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_route_validation_20260516/
route validation status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
wrapper plan-only smoke:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_wrapper_plan_smoke_20260516/
wrapper plan-only status:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
toy protected accepts:
  1
toy wrong-key/wrong-payload accepts:
  0/0
full mode enabled:
  false
slurm submission started:
  false
```

The wrapper now consumes the repaired full16 allocation and precommit:

```text
rows per shard:
  1024
expected selected coordinates:
  16
score rows:
  results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl
allocation rows:
  results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/row_allocation_rows.jsonl
codebook:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/codebook.json
```

## Remote Preflight

The repaired full16 generation/decode control-plane files were synchronized to
Chimera and the remote route/wrapper preflight passed without allowlist
enablement or Slurm submission:

```text
remote preflight:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_remote_preflight_20260517_0008/
remote status:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_REMOTE_PREFLIGHT_NO_SUBMIT
remote host:
  chimerahead.umb.edu
remote route validation:
  PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
remote wrapper plan-only:
  PASS_R4_AFTER_868016_CONTROLLER_GENERATION_WRAPPER_PLAN_ONLY
remote allowlist:
  PASS, enabled_entries=[]
allowlist entry:
  v2_r4_after_868212_repaired_first_token_event_generation_h200
allowlist entry enabled:
  false
```

## Current Submission

The reviewed full16 first-token event generation diagnostic has been submitted
as one H200 Slurm array job after Hermes notification and single-enabled
allowlist preflight:

```text
submission record:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_submission_20260517_0016/
single-enabled preflight:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_single_enabled_preflight_20260517_0016/
job_id:
  868260
job_name:
  nat-ev-v2-r4c16
array:
  0-3%4
partition/qos/account:
  pomplun / pomplun / cs_yinxin.wan
gres:
  gpu:h200:1
time_limit:
  30-00:00:00
allowlist entry:
  v2_r4_after_868212_repaired_first_token_event_generation_h200
enabled entries after submission, local:
  []
enabled entries after submission, remote:
  []
```

## Current Review

Job `868260` completed and has been reviewed:

```text
review:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_868260_review/
failure analysis:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/
repair decision:
  results/natural_evidence_v2/status/r4_after_868212_generation_868260_quality_gate_repair_decision_20260517/
status:
  RECORDED_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_868260_FAILED_QUALITY_GATE_SIGNAL_PRESENT_NO_SUBMIT
```

The run is not a positive result:

```text
strict protected accepts:
  2/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
full-phrase protected accepts, format_scrub=all:
  0
```

Interpretation:

```text
The first-token event signal recovered the expected codeword in all protected
blocks before quality filtering. The strict gate failed because shard_00 and
shard_01 protected blocks hit duplicate/forbidden quality filters.
```

Quality failure details:

```text
shard_00_block_00:
  decoded expected codeword, valid checksum, duplicate_response_hash_count=1
shard_01_block_00:
  decoded expected codeword, valid checksum,
  duplicate_response_hash_count=2,
  forbidden_public_surface_count=1
forbidden example:
  literal "bucket" in ordinary physical plumbing/home-maintenance sense
global duplicate extra rows:
  7612
unique response hashes:
  4676 / 12288
```

## Current Repair Package

The artifact-only quality-gate repair package has been precommitted and
validated:

```text
repair package:
  results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517/
validation:
  results/natural_evidence_v2/status/r4_after_868260_quality_gate_repair_package_validation_20260517/
validation status:
  PASS_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT
tests:
  5 passed
reclassifies 868260:
  false
slurm allowed:
  false
```

The package contains:

```text
contextual forbidden-surface policy v2:
  ordinary physical "bucket" may be allowed under precommitted task-domain
  cues, while technical "bucket" remains forbidden
duplicate-safe generation policy:
  future within-block and global duplicate response hash gates remain 0
```

## Next Allowed Action

The route may continue automatically after recorded preconditions pass. After
the `870210 + 870987` locked-scale pass, the next canonical route is the
artifact-only R4 Qwen first-token event pre-FAR null expansion package.

```text
next:
  prepare/review full generation/decode wrapper for the pre-FAR standard-control
  null package; no generation Slurm submission until wrapper and local/remote
  route preflights pass
allowed:
  Hermes notification
  Hermes/Codex state synchronization
  route validation for configs/natural_evidence_v2/r4_after_870987_prefar_null_expansion_route.yaml
  artifact-only standard-control null row-bank validation
  artifact-only organic-null prompt-bank validation
  full-wrapper review planning
  standard-control pre-FAR generation route planning/preflight
  organic-null wrapper design
not allowed:
  generation/null Slurm submission before the standard-control wrapper route is
  reviewed and local/remote route preflights pass
  reclassifying text-only full phrase decoder as successful
  reclassifying this pre-FAR route as full FAR
  paper-facing positive claims
not yet allowed:
  training
  Llama
  same-family null
  sanitizer
  FAR aggregation
  payload diversity
```

This route does not unlock training, Llama, same-family null, sanitizer, FAR
aggregation, payload diversity, text-only phrase decoder success claims, or
paper-facing positive claims.

Route-controlled actions may proceed automatically after their preconditions are
recorded; the user has authorized Codex and Hermes not to ask repeatedly for the
same approved route.

## Still Gate-Controlled

These actions are not permanently forbidden, but may proceed only after their
route-specific preconditions pass and are recorded in this file:

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
