# R4 autonomous route permission

Timestamp UTC: 2026-05-13T01:35:00Z

## User Permission

The user authorized Codex and Hermes to keep executing the next work items when
the route is already clear. Codex/Hermes should not stop for repeated per-step
approval inside the same reviewed route.

## Guardrails That Still Apply

This permission does not waive:

- route decision records for new phases;
- gate evidence;
- allowlist safety checks;
- exactly one enabled allowlist entry for a reviewed submission;
- Hermes TG/email notification before state-changing compute;
- Slurm-only execution for Chimera CPU/GPU work;
- no-overwrite artifact checks;
- post-submission allowlist disablement;
- paper-claim restrictions.

Training, Llama, same-family null, sanitizer, FAR aggregation, payload diversity
claims, and paper-facing positive claims remain gate-controlled work classes.
They are allowed only after their prerequisite gates and route decisions pass.
