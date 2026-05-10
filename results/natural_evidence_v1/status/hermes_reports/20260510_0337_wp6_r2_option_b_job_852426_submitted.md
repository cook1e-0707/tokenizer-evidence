# WP6-R2 Option B Job 852426 Submitted

timestamp_utc:
2026-05-10T03:37:58Z

## Decision

Hermes notification had already succeeded through Telegram and email:

```text
results/natural_evidence_v1/status/hermes_reports/20260510_0335_scheduled_tick_notification.json
```

Codex synced the reviewed WP6-R2 Option B wrapper and locked inputs to Chimera
and submitted exactly one allowlisted Slurm job: `852426`
(`nat-ev-v2-wp6r2b`).

The allowlist entry `v2_wp6_r2_option_b_scale_eval` was disabled immediately
after submission with condition
`submitted_once_as_job_852426_pending_wp6_r2_option_b_scale_result_review`.

## Slurm Check

```text
852426|DGXA100|nat-ev-v2-wp6r2b|guanjie.lin001|PENDING|0:00|1|(Priority)
852426|nat-ev-v2-wp6r2b|DGXA100|PENDING|0:0|00:00:00|None assigned
```

Remote output directory:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/wp6_r2_option_b_scale_eval_852426
```

## Scope

- Qwen WP6-R2 Option B robust-block diagnostic only.
- 512 locked `wp3_r1_eval` prompts in eight 64-query blocks, file rows
  `768..1279`.
- Protected/raw/task-only generation plus wrong-key and wrong-payload decode
  controls.
- No new training, no FAR aggregation, no Llama, no same-family null, no
  sanitizer, and no paper-facing positive claim.

## Next Allowed Action

Monitor job `852426`; after completion, sync
`wp6_r2_option_b_scale_eval_852426` artifacts and review the precommitted R2
Option B gates. Do not submit another WP6 job before that review.
