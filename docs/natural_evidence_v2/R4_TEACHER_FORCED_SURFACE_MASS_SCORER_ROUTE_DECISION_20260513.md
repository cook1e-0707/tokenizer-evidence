# R4 teacher-forced surface-mass scorer route decision

Timestamp UTC: 2026-05-13T01:28:25Z

## Decision

The user explicitly authorizes R4 teacher-forced surface-mass scorer wrapper
preparation.

This decision supersedes the 2026-05-13T01:26Z no-Slurm hold only for the
artifact/wrapper preparation scope below. It does not authorize allowlist
enablement or Slurm submission.

## Scope

Allowed:

- prepare a Slurm-only Qwen scorer wrapper;
- add a disabled allowlist entry for that wrapper;
- run local plan-only smoke validation with `VALIDATE_PLAN_ONLY=1`;
- record route, preflight, and Hermes/Codex sync artifacts.

The wrapper must score the frozen R4 binary-repair teacher-forced surface rows:

- score rows:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/r4_surface_teacher_forced_probe_rows.jsonl`;
- candidate surface bank:
  `results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/candidate_binary_surface_bank.json`;
- contract: `a55e`;
- model/tokenizer: `Qwen/Qwen2.5-7B-Instruct`;
- conditions: `base`, `protected`, `task_only`.

## Not Authorized

This decision does not authorize:

- Slurm submission;
- allowlist enablement;
- free generation;
- locked-scale rerun;
- training;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload diversity claims;
- paper-facing positive claims.

## Next Gate

After wrapper preparation and plan-only smoke pass, the next route decision may
authorize exactly one Slurm scoring submission. That future submission must keep
all other allowlist entries disabled and must notify Hermes by TG/email before
any state-changing action.
