Recorded a blocker for this Hermes tick instead of enabling/submitting R3.2.

Reason: the current R3.2 wrapper is still reviewed as plan-only, and this tick explicitly forbids generation/Qwen E2E rerun. Submitting it now would either fail by design or violate the tick constraints.

Updated:
- `results/natural_evidence_v1/status/hermes_reports/20260511_0430_r3_2_full_eval_submission_blocker.md`
- `results/natural_evidence_v1/status/hermes_reports/20260511_0430_r3_2_full_eval_submission_blocker.json`
- `results/natural_evidence_v1/status/gate_status.json`
- `results/natural_evidence_v2/status/gate_status.json`

Validated JSON with `python3 -m json.tool` on the new blocker JSON and both gate status files. No allowlist entry was enabled, no Slurm job was submitted, and no generation/training/Llama/null/sanitizer/FAR/paper-claim action was started.