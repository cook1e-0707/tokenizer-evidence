# Hermes sync: R4 after-868299 dev job 868348 reviewed

phase:
`V2_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILED_GLOBAL_DUPLICATE_GATE_SIGNAL_PASSING_NO_RERUN`

summary:

```text
Job 868348 completed all 32 H200 shards and was reviewed.

Signal/null/trace:
- protected strict accepts: 32/32
- protected ignoring-quality accepts: 32/32
- raw/task_only/wrong_key/wrong_payload accepts: 0/32 each
- trace binding invalid rows: 0 / 98304
- protected forbidden public surface count: 0
- protected duplicate response hash count: 0

Strict-quality blocker:
- global exact duplicate extra rows: 2
- duplicate rows are task_only only
- protected duplicate rows: 0

Duplicate groups:
- task_only, prompt r4_cover_dev_4409a3670c843c3b1383, prefix useful_habit, shards 10 and 31
- task_only, prompt r4_cover_dev_54dcac7d434267b59ff1, prefix practical_option, shards 9 and 24
```

recorded:

```text
results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_868348_review/
results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_868348_failure_attribution/
docs/natural_evidence_v2/CURRENT_STATE.md
```

interpretation:

```text
868348 is not a pass and must not be reclassified as positive. It is a strong
signal-passing diagnostic that fails the precommitted global exact duplicate
quality gate. The blocker is now duplicate-gate/allocation policy, not
first-token event signal recovery.
```

next_allowed_action:

```text
Prepare expert route decision package: either build a globally unique
prompt/prefix allocation before rerun, or precommit narrower duplicate-gate
semantics for future runs. No Slurm rerun before a reviewed route.
```
