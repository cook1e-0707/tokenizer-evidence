# Hermes natural_evidence_v1 Codex worker completed

phase:
V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT

codex_report:
`results/natural_evidence_v1/status/hermes_reports/20260511_1746_scheduled_tick_codex_report.md`

summary:
```text
Recorded the artifact-only R3.2 prompt split contract repair.

Changed:
- [R3_2_PROMPT_SPLIT_CONTRACT_REPAIR_20260511_1747.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_PROMPT_SPLIT_CONTRACT_REPAIR_20260511_1747.md)
- [r3_2_prompt_split_contract_repair_20260511_1747.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_prompt_split_contract_repair_20260511_1747.json)
- [CURRENT_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/CURRENT_STATE.md)

The repair records `wp3_r1_eval` as the required split, uses eval rows `512..2559`, supersedes the prior 5-window policy with a 4 eval-window circular policy, and keeps resubmission blocked until implementation, preflight, replay review/supersession, allowlist safety, and a new single-job route are recorded.

Validation run:
- `python3 -m json.tool ...`
- non-empty artifact checks

No Slurm, generation, Qwen E2E rerun, training, Llama, null, sanitizer, FAR, or claim work was started.
```

next_allowed_action:
Repair the R3.2 prompt allocation and wrapper split contract artifact-only. Do not submit another R3.2 Slurm job until the repaired allocation is recorded, plan-only preflight is rerun, 852426 replay compatibility is re-reviewed or superseded, allowlist safety is rechecked, and a new single-job submission route is recorded.

gate_controlled_actions_not_yet_unlocked:
Gate-controlled and not yet unlocked: training; Llama; same-family null; sanitizer benchmark; FAR aggregation; paper-facing positive claims. Hard-blocked unless explicitly allowed by the current next_allowed_action: generation and Qwen E2E reruns.
