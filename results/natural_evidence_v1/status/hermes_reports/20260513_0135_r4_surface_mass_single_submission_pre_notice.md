# Hermes / Codex pre-submission notice: R4 surface-mass scoring

timestamp_utc: 2026-05-13T01:35:00Z

phase:
`V2_R4_TEACHER_FORCED_SURFACE_MASS_SCORER_SINGLE_SUBMISSION_READY`

permission:

- User authorized Codex/Hermes to continue clear-route next steps without
  repeated per-step approval.
- Submission decision:
  `docs/natural_evidence_v2/R4_TEACHER_FORCED_SURFACE_MASS_SCORER_SUBMISSION_DECISION_20260513.md`

authorized action:

Submit exactly one allowlisted Chimera H200 Slurm job for Qwen teacher-forced
surface-mass scoring.

allowlist:

- enabled entry: `v2_r4_teacher_forced_surface_mass_score_h200`
- all other entries disabled
- local single-enabled preflight: `PASS`
- remote single-enabled preflight: `PASS`
- remote hash preflight: `PASS`

scope:

- model/tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- conditions: `base`, `protected`, `task_only`
- score rows: `8192`
- contract: `a55e`
- no free generation
- no training
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no payload-diversity claim
- no paper-facing positive claim

next action:

Run:

`ssh chimera 'cd ~/tokenizer-evidence && sbatch scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch'`

Immediately after `sbatch` returns a job id, disable the allowlist entry locally
and remotely and record the submission.
