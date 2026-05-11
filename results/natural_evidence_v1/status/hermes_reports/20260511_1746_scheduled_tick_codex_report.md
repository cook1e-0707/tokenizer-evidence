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