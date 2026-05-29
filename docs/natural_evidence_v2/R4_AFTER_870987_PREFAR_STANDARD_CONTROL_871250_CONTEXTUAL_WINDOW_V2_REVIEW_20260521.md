# R4 After-870987 Pre-FAR Standard-Control 871250 Contextual Window V2 Review

## Decision

`871250` is adopted as a Qwen first-token event pre-FAR standard-control null
package after artifact-only contextual matcher implementation repair.

This is not full FAR, not text-only phrase decoder success, not payload
diversity, not Llama transfer, and not a paper-facing positive claim.

## Job And Scope

```text
job_id: 871250
job_name: nat-ev-v2-r4pGen
array: 0-159%6
partition/qos/account/gres: pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
route: R4 after-870987 pre-FAR standard-control first-token event null expansion
controls: raw, task_only, wrong_key, wrong_payload
protected rows: report-only for this route
```

## Original Aggregate

The first aggregate completed all shards but failed the strict gate:

```text
status: FAIL_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE
complete_shards: 160/160
control accepts: 0/256 for raw/task_only/wrong_key/wrong_payload
trace binding invalid rows: 0
global duplicate extra rows: 0
failure item: forbidden public surface count = 7
```

Artifact attribution found four generated rows responsible for the seven block
counts. They were ordinary-domain uses: `bucket of water`, `Coordinate with
family`, `coordinate volunteer schedules`, and similar coordination verbs. The
old implementation treated technical cues by full-response co-occurrence, so a
nearby-but-ordinary `bucket`/`coordinate`, or a distant ordinary `slot`, could
make a contextual literal fatal.

## Repair

The repair changes implementation semantics only:

```text
old matcher: contextual literal + technical cue anywhere in the full response
new matcher: contextual literal + technical cue in the same sentence-local token window
policy changed: no
threshold changed: no
generated outputs changed: no
token traces changed: no
controller config changed: no
accept logic changed: no
Slurm submitted: no
```

Validated tests:

```text
tests/natural_evidence_v2/test_r4_after_868151_first_token_event_decoder.py
tests/natural_evidence_v2/test_r4_contextual_forbidden_surface_policy_v2.py
```

## Repaired Aggregate

Artifact-only re-decode/re-aggregate output:

```text
results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_generation_871250_contextual_window_v2_aggregate_20260521/
status: PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE
complete_shards: 160/160
generated rows: 491520
unique response hashes: 491520
global duplicate extra rows: 0
trace binding: 491520 checked, 0 invalid
technical forbidden public surface count: 0
```

Combined standard-control counts:

```text
raw: 0/256 accepts, 0/256 ignoring-quality accepts
task_only: 0/256 accepts, 0/256 ignoring-quality accepts
wrong_key: 0/256 accepts, 0/256 ignoring-quality accepts
wrong_payload: 0/256 accepts, 0/256 ignoring-quality accepts
```

Report-only protected first-token event result:

```text
protected: 159/160 strict accepts, 159/160 ignoring-quality accepts
```

Full phrase decoder remains report-only and failing; it does not unlock a
text-only phrase claim.

## Next Route

Continue R4 Qwen pre-FAR null package with organic-null planning/execution after
the required route validation, prompt/allocation validation, tokenizer/controller
preflight, local/remote hash preflight, allowlist safety, and exactly-one H200
submission preflight pass.
