Recorded the 11:17 R3.2 submission blocker instead of submitting Slurm.

New files:
- [20260511_1117_r3_2_submission_preflight_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1117_r3_2_submission_preflight_blocker.md)
- [20260511_1117_r3_2_submission_preflight_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_1117_r3_2_submission_preflight_blocker.json)

Reason: the 11:17 TG/email notification succeeded, but submission is still unsafe because local `configs/natural_evidence_v2/run_allowlist.yaml` has `llama_v2_wp6_e2e_eval` enabled while `llama_allowed=false`, and no post-11:02 preflight repair record is present in the compact state/gate files.

Validation passed: JSON parses, blocker decision is present, and `git diff --check` passed for the new files. No allowlist entry was enabled and no Slurm job was submitted.