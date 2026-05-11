# R3.2 full wrapper upgrade blocker

timestamp_utc:
2026-05-11T04:14:00Z

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

controlling_report:
results/natural_evidence_v1/status/hermes_reports/20260511_0414_scheduled_tick.md

decision:
BLOCK_R3_2_ALLOWLIST_AND_SUBMISSION_PENDING_FULL_WRAPPER_SPEC

reason:
The controlling tick authorizes the R3.2 route, but the current reviewed
R3.2 wrapper is still explicitly plan-only and exits before any non-plan run.
Submitting it would not perform the locked-scale generation/eval.

The next route step is also not yet safe to implement automatically because
the locked R3.2 package names payload cells `P00/P01/P02/P03`, while the
available production WP4 payload contract used by the current Qwen generation
and decoder path fixes one payload-plus-checksum value, `a55e`. Earlier WP4
oracle artifacts contain `P00/P01` contracts only, and no reviewed
production R3.2 mapping for `P02/P03` payload bytes, checksums, adapters, and
wrong-payload controls is recorded in the inspected state.

actions_taken:
- Read the required v1/v2 automation, protocol, claim, and gate status files.
- Reviewed the R3.2 package review, prompt allocation decision, current wrapper
  review, disabled allowlist entry, and current Slurm wrapper.
- Recorded this blocker report only.

forbidden_actions_confirmed_not_started:
- training
- generation
- Qwen E2E rerun
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- paper-facing positive claim
- Slurm submission
- allowlist enablement

next_allowed_action:
Record or implement a reviewed full R3.2 wrapper specification that resolves
the payload-cell contract mapping before enabling the allowlist or submitting
Slurm. The wrapper must remain Qwen-only, Slurm-only for Chimera work, refuse
overwrites, and preserve the locked prompt allocation and gate targets.
