# Hermes pre-notice: R4 prefix-native surface-mass Slurm submission

phase:
V2_R4_PREFIX_NATIVE_SURFACE_MASS_SINGLE_JOB_SUBMISSION

summary:
Codex is about to submit exactly one Chimera H200 Slurm job for R4 prefix-native teacher-forced tokenizer/model surface-mass scoring.

Scope:
- Qwen tokenizer/model forward scoring only
- conditions: base, protected, task_only
- score rows: `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/r4_prefix_native_surface_probe_rows.jsonl`
- wrapper: `scripts/natural_evidence_v2/slurm/r4_prefix_native_surface_mass_score_h200.sbatch`
- allowlist entry: `v2_r4_prefix_native_surface_mass_score_h200`
- partition/QoS/account: `pomplun` / `pomplun` / `cs_yinxin.wan`
- GPU: H200

Preflight:
- local zero-enabled allowlist safety: PASS
- remote zero-enabled allowlist safety: PASS
- local/remote hash preflight: PASS

Not allowed by this route:
- generation
- training
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- payload diversity claim
- paper-facing positive claim

Submission rule:
Enable exactly one allowlist entry, submit one Slurm job, then immediately disable the entry locally and remotely.
