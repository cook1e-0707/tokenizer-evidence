Blocked the 11:47 submission instead of enabling/submitting.

Reason: the notification path passed, but `configs/natural_evidence_v2/run_allowlist.yaml` still has forbidden `llama_v2_wp6_e2e_eval` enabled while `llama_allowed=false`, and no post-11:32 repair record exists. Submitting R3.2 would violate the guardrails.

Recorded:
- `results/natural_evidence_v1/status/hermes_reports/20260511_1147_r3_2_submission_preflight_blocker.md`
- `results/natural_evidence_v1/status/hermes_reports/20260511_1147_r3_2_submission_preflight_blocker.json`
- Updated compact status pointers to `FAIL_20260511_1147`

Validation: `jq empty` passed for the new blocker JSON and both gate status JSON files. No Slurm job was submitted, and no allowlist entry was enabled.