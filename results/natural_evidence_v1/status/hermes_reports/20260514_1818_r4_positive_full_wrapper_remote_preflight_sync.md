# Hermes/Codex Sync: R4 Positive Full Wrapper Remote Preflight

phase:
`V2_R4_POSITIVE_FULL_GENERATION_DECODE_WRAPPER_REMOTE_PREFLIGHT_PASS_SUBMISSION_ROUTE_REVIEW_NEXT`

summary:
```text
Implemented and locally reviewed the full generation/decode wrapper for the
R4 positive event-bank dev diagnostic. The wrapper no longer fails with the
old implementation-pending marker outside plan-only mode. Full mode now runs
Qwen protected/raw/task-only generation, then keyed phrase-event decode under
format_scrub=all and format_scrub=none with wrong-key/wrong-payload controls.

Local validation:
- bash syntax: PASS
- focused pytest: 13 passed
- local plan-only wrapper smoke: PASS
- static keyed-decoder fixture: protected 1/1, wrong-key 0/1, wrong-payload 0/1
- local allowlist safety: PASS, enabled entries []

Remote validation:
- remote plan-only wrapper smoke: PASS
- remote allowlist safety: PASS, enabled entries []
- local/remote hash preflight: PASS
- Chimera active-job preflight: PASS, no active jobs

No Slurm job was submitted. No generation/training/Llama/null/sanitizer/FAR or
paper-claim action was started.
```

next_allowed_action:
Record the R4 positive single-submission route, send Hermes TG/email
pre-submit notification, then if final preflight remains PASS enable exactly
`v2_r4_positive_dev_diagnostic_h200`, submit exactly one H200/pomplun Slurm
array job, and immediately disable the allowlist after `sbatch` returns.

gate_controlled_actions:
Training, Llama, same-family null, sanitizer, FAR aggregation, payload
diversity, and paper-facing claims remain gated. They are conditionally
authorized only after their route-specific prerequisite gates pass.

