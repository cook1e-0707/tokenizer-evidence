# Hermes / Codex sync: R4 surface-mass scoring submitted

timestamp_utc: 2026-05-13T01:35:27Z

phase:
`V2_R4_TEACHER_FORCED_SURFACE_MASS_SCORING_JOB_853815_RUNNING`

submitted_job:

- job id: `853815`
- job name: `nat-ev-v2-r4tfm`
- partition/QoS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`
- GPU: `h200`
- initial state: `RUNNING`
- initial node: `chimera21`

scope:

- Qwen teacher-forced surface-mass scoring only
- conditions: `base`, `protected`, `task_only`
- score rows: `8192`
- contract: `a55e`

post-submit safety:

- allowlist entry disabled locally and remotely after `sbatch`
- local post-submit allowlist safety: `PASS`
- remote post-submit allowlist safety: `PASS`

actions_not_started:

- free generation
- training
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim

next_allowed_action:

Monitor Slurm job `853815`. After completion, sync and review the
teacher-forced surface-mass summary. Do not submit another scoring job or run
generation/training/Llama/FAR/sanitizer/paper claims until this result is
reviewed.
