# Hermes natural_evidence_v1 Codex worker failed

phase:
V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT

codex_report:
`results/natural_evidence_v1/status/hermes_reports/20260511_1931_scheduled_tick_codex_report.md`

summary:
```text
codex CLI not found on PATH.

```

next_allowed_action:
Artifact-only next step: recheck R3.2 allowlist safety under the repaired prompt split contract, then record a new single-job submission route before any Slurm submission. Do not submit another R3.2 Slurm job in this tick.

gate_controlled_actions_not_yet_unlocked:
Gate-controlled and not yet unlocked: training; Llama; same-family null; sanitizer benchmark; FAR aggregation; paper-facing positive claims. Hard-blocked unless explicitly allowed by the current next_allowed_action: generation and Qwen E2E reruns.
