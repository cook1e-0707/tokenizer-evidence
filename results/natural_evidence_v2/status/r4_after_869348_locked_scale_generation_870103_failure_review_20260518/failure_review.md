# R4 After-869348 Locked-Scale Generation 870103 Failure Review

Recorded: 2026-05-18T14:10:49Z

## Classification

`870103` is a wrapper/runtime contract failure, not a model or method result.

```text
status:
  FAILED_R4_AFTER_869348_LOCKED_SCALE_GENERATION_870103_RUNTIME_TOKENIZER_REVIEW_STATUS_GUARD_NO_METHOD_RESULT
job:
  870103, nat-ev-v2-r4lGen, array 0-95%4
partition/qos/account/gres:
  pomplun / pomplun / cs_yinxin.wan / gpu:h200:1
observed terminal state:
  all array tasks FAILED with ExitCode 1:0 in 0-2 seconds
generation result:
  none
paper claim:
  not allowed
```

## Root Cause

The route validator and route config correctly required the reviewed held-out
tokenizer preflight status:

```text
PASS_R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_870078
```

The runtime generation script still had an older explicit allowlist in
`scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py`.
It accepted earlier tokenizer preflight review statuses but did not include the
new reviewed `870078` status. All inspected shards failed before generation at
the same guard:

```text
ValueError: tokenizer review is not an allowed reviewed pass:
PASS_R4_AFTER_869348_LOCKED_SCALE_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_870078
```

The wrapper printed `generation_started=false` before the guard raised. No
locked-scale outputs or decode rows were produced by this job.

## Interpretation

Do not classify `870103` as a protected-signal failure, null failure, duplicate
failure, forbidden-surface failure, or locked-scale generation result. The run
did not reach model generation.

## Repair

The immediate repair is to add the reviewed `870078` tokenizer status to the
generator's explicit tokenizer-review allowlist and test both acceptance of the
reviewed status and rejection of an unreviewed status.

Local checks after patch:

```text
.venv/bin/pytest -q \
  tests/natural_evidence_v2/test_r4_after_868016_controller_generation.py \
  tests/natural_evidence_v2/test_r4_controller_generation_binding_helpers.py

result: PASS, 3 passed

python3 scripts/natural_evidence_v2/validate_r4_after_869348_locked_scale_generation_route.py \
  --skip-allowlist-state-check \
  --output-dir results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_route_validation_after_870103_repair_20260518

result: PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_ROUTE_PLAN_ONLY_NO_SUBMIT
```

## Next Allowed Action

Sync the reviewed status-guard patch and tests to Chimera, rerun remote route
validation and wrapper plan-only smoke, verify local/remote zero-enabled
allowlist safety, then resubmit exactly one H200 locked-scale generation array
under the same reviewed route. Disable the allowlist entry immediately after
`sbatch` returns.

Still not unlocked:

```text
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity
paper-facing positive claim
text-only phrase decoder success claim
```
