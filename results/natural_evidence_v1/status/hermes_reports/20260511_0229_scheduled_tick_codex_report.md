Recorded a blocker instead of implementing the R3.2 wrapper, because the prompt allocation policy is not fixed.

Added:
- [20260511_0231_r3_2_wrapper_prompt_allocation_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0231_r3_2_wrapper_prompt_allocation_blocker.md)
- [20260511_0231_r3_2_wrapper_prompt_allocation_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0231_r3_2_wrapper_prompt_allocation_blocker.json)

Updated:
- [AUTOMATION_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v1/AUTOMATION_STATE.md)
- both v1/v2 `gate_status.json` files with `BLOCK_R3_2_WRAPPER_PROMPT_ALLOCATION_POLICY_MISSING_NO_SLURM`

Reason: R3.2 implies `4 payloads * 3 seeds * 8 blocks * 64 prompts = 6144` prompt responses per arm, but the current package does not specify disjoint windows vs deterministic reuse, and the apparent prompt source has `2560` rows. I did not invent that policy.

Validation passed: JSON parsing for the new blocker and both gate files; confirmed the reserved wrapper path does not exist. No Slurm, training, generation, Qwen E2E rerun, Llama, sanitizer, FAR aggregation, or paper-facing claim was started.