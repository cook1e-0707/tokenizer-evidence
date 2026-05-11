# Hermes R3.2 prompt allocation decision

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

state_changing_action:
Recorded the R3.2 prompt allocation decision before wrapper implementation.

decision:
Use the reviewed 2560-row WP3-R1 strict-density prompt source with a
deterministic five-window circular reuse rule across the 12 payload/seed cells.
This is explicitly not cell-disjoint prompt allocation because a fully disjoint
R3.2 package would require 6144 prompt rows.

artifacts_written:
```text
docs/natural_evidence_v2/R3_2_PROMPT_ALLOCATION_DECISION_20260511.md
results/natural_evidence_v2/status/r3_2_prompt_allocation_decision_20260511_0244.json
```

selected_prompt_manifest_sha256:
```text
4d49ae100b272f184a8b2563e5b64f768e6db01425a2384f1457a4eb10eedb67
```

forbidden_actions_confirmed:
No Slurm job, training, generation, Qwen E2E rerun, Llama, same-family null,
sanitizer benchmark, FAR aggregation, or paper-facing positive claim was
started.

next_allowed_action:
Implement or review an R3.2-specific Qwen locked-scale wrapper and disabled
allowlist entry, then run local plan-only validation only. Do not submit Slurm.

timestamp_utc:
2026-05-11T02:44:00Z
