# R4 after-867621 adapter-gain route validated locally

phase: V2_R4_AFTER_867621_RELIABILITY_ADAPTER_GAIN_SWEEP_ROUTE_VALIDATED_NO_SUBMIT

summary:
```text
867849 completed cleanly but failed teacher-forced surface-mass gate.
Tokenizer boundary valid; task-only lift vs base is negative; protected pressure is too weak.
Codex recorded a protected-adapter gain-sweep pivot route on the same 4096 reliability rows.
Local route validator: PASS.
Wrapper plan-only smoke: DRY_RUN_VALIDATED_INPUTS.
Local allowlist safety: PASS with zero enabled entries.
No Slurm job submitted yet.
```

next_allowed_action:
Remote Chimera hash/preflight for `v2_r4_after_867621_reliability_adapter_gain_sweep_h200`; if remote preflight passes, submit exactly one H200/pomplun teacher-forced scoring-only gain sweep, then immediately disable the allowlist entry.

not_unlocked:
generation, training, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, payload diversity claim, paper-facing positive claim.
