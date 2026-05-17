# R4 After-868299 First-Token Event Dev Diagnostic Plan

Date: 2026-05-17

## Status

`PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ALLOCATION_PLAN_NO_SUBMIT`

This is a 32-block Qwen dev diagnostic route plan for the provider-side keyed
first-token event channel. It is not a locked-scale result and does not unlock
paper-facing claims, Llama, FAR, sanitizer, training, or payload diversity.

## Source Confirmation

The route is only allowed because job `868299` passed the strict 4-block
quality-repair confirmation:

- protected strict accepts: `4/4`
- protected accepts ignoring quality: `4/4`
- raw/task-only/wrong-key/wrong-payload accepts: `{'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}`
- global duplicate response hashes: `0`
- trace-binding invalid rows: `0`

## Allocation Policy

The reviewed full16 row bank contains four fully unique 1024-row shards. A
blind 32-block dev diagnostic therefore cannot claim 32 independent prompt
allocations from the current bank. This route precommits cyclic reuse of the
reviewed four-shard allocation across `8` cycles.
Each shard still has zero within-shard prompt/prefix duplicates, and each shard
uses a distinct public shard id and public sampling seed.

This reuse is acceptable only for a dev diagnostic. It must not be described as
locked-scale independent evidence. The global exact response-hash duplicate
gate remains zero.

## Next Allowed Action

Run local/remote route validation and wrapper plan-only smoke. If those pass,
the existing user authorization permits one reviewed H200 Slurm submission with
the allowlist enabled for exactly one entry and disabled immediately after
`sbatch`.
