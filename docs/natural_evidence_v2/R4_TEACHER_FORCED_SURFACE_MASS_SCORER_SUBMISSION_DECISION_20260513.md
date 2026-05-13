# R4 teacher-forced surface-mass scorer submission decision

Timestamp UTC: 2026-05-13T01:35:00Z

## Decision

The R4 teacher-forced surface-mass scorer wrapper and plan-only smoke are
reviewed for a single Slurm scoring submission.

This decision authorizes exactly one allowlisted H200 Slurm submission for
Qwen teacher-forced surface-mass scoring.

## Authorized Submission

Allowlist entry:

`v2_r4_teacher_forced_surface_mass_score_h200`

Command pattern:

`sbatch scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch`

Scope:

- model/tokenizer: `Qwen/Qwen2.5-7B-Instruct`;
- conditions: `base`, `protected`, `task_only`;
- score rows:
  `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/r4_surface_teacher_forced_probe_rows.jsonl`;
- row count: `8192`;
- contract: `a55e`;
- wrapper:
  `scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch`;
- partition / account / QoS: `pomplun` / `cs_yinxin.wan` / `pomplun`;
- GPU: `h200`;
- one Slurm job only.

## Required Pre-Submission Checks

- local allowlist preflight must pass with exactly one enabled entry:
  `v2_r4_teacher_forced_surface_mass_score_h200`;
- remote Chimera files must match local hashes for wrapper, scorer script,
  allowlist, score rows, and state artifacts;
- Hermes TG/email notification must succeed before `sbatch`;
- all unrelated allowlist entries must remain disabled.

## Post-Submission Requirement

Immediately after `sbatch` returns a job id, the allowlist entry must be
disabled again locally and remotely. A submission record must be written.

## Not Authorized

This submission does not authorize:

- free generation;
- locked-scale rerun;
- training;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim.
