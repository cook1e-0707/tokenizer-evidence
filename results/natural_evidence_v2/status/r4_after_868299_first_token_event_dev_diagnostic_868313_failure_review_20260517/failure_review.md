# R4 after-868299 dev diagnostic job 868313 failure review

## Verdict

`868313` is a control-plane/runtime validation failure, not a model, tokenizer,
controller, or decoder failure.

The Slurm array ended `Failed, Mixed` because shards `0..23` started while the
reviewed allowlist entry was still enabled during the immediate post-`sbatch`
disablement window. Those shards failed before generation. Shards `24..31`
started after the entry had been disabled, passed runtime validation, and
completed generation/decode.

## Evidence

Observed Slurm state:

```text
job_id: 868313
job_name: nat-ev-v2-r4dev
failed_shards: 0..23
completed_shards: 24..31
failed_elapsed: 00:00:00 to 00:00:02
completed_elapsed: about 01:19:43 to 01:21:09
```

Representative failed shard stdout contains:

```text
errors:
  - allowlist enabled entries must be empty during plan validation:
    ['v2_r4_after_868299_first_token_event_dev_diagnostic_h200']
  - allowlist entry enabled state must be False
status:
  FAIL_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT
generation_started: false
```

Representative completed shards (`24`, `31`) reached model execution, wrote
`r4_generated_outputs.jsonl`, passed trace binding, and completed both
`decode_all` and `decode_none`. Their partial outputs are not a canonical
32-block result because the array did not complete all shards.

## Root Cause

The route validator used the same allowlist enabled-state rule for both:

1. pre-submission control-plane checks, where the allowlist should be either
   zero-enabled or exactly-one-enabled depending on phase; and
2. runtime shard self-checks, where tasks can begin before or after the required
   immediate post-submission disablement.

This creates a race: early tasks see exactly-one-enabled and fail, while later
tasks see zero-enabled and pass.

## Canonical Status

```text
status:
  FAILED_R4_AFTER_868299_DEV_JOB_868313_RUNTIME_ALLOWLIST_RACE_PARTIAL_GENERATION_NO_METHOD_RESULT
adopt_as_dev_result: false
aggregate_outputs: false
paper_claim_allowed: false
```

## Repair

The repair is to keep submission preflights strict, but allow runtime shard
validation to skip only the allowlist enabled-state check. Runtime validation
must still verify that the allowlist entry exists and that its command pattern
matches the reviewed wrapper.

Next allowed action:

```text
patch runtime route validation
rerun local tests and wrapper smoke
sync repaired control-plane files to Chimera
run local/remote preflights
submit one repaired 32-shard H200 dev diagnostic replacement if preflights pass
```
