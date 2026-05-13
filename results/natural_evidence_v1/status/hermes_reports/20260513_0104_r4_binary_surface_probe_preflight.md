# Hermes / Codex sync: R4 binary surface probe preflight

phase: `V2_R4_COVER_NATURAL_ECC_ARTIFACT_ONLY_REPAIR_PREFLIGHT_NO_SLURM`

Summary:

- Reviewed the completed R4 dev diagnostic `853691`: clean H200 completion, but positive channel failed (`0/32` protected accepts with no scrub and `0/32` with `format_scrub=all`; null arms remained `0/32`).
- Confirmed the original R4 surface bank cannot support a valid teacher-forced target-vs-other mass probe because every coordinate has only one polarity side.
- Built an artifact-only binary surface-bank repair candidate with `32` coordinates, `256` phrase-level entries, and `4` bit-0 plus `4` bit-1 entries per coordinate.
- Built teacher-forced surface probe rows against the repair candidate: `256` dev prompts, `8192` score rows, `32` coordinates, contract `a55e`.
- Ran scorer dry-run validation on the row plan. No model scoring, generation, training, Slurm submission, Llama, FAR, sanitizer, same-family null, payload diversity claim, or paper claim was started.

Key artifacts:

- `results/natural_evidence_v2/status/r4_binary_surface_bank_repair_plan_20260513/binary_surface_bank_repair_summary.json`
- `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_preflight_binary_repair_20260513/r4_surface_teacher_forced_probe_plan_summary.json`
- `results/natural_evidence_v2/status/r4_surface_teacher_forced_probe_dry_run_binary_repair_20260513/r4_teacher_forced_surface_mass_summary.json`
- `docs/natural_evidence_v2/CURRENT_STATE.md`

next_allowed_action:

Review the artifact-only R4 binary surface repair and teacher-forced surface probe plan. If accepted, prepare a Slurm-only Qwen teacher-forced surface-mass scorer wrapper for base/protected/task-only. Do not run free generation or locked-scale until the surface teacher-forced gate is actually scored and reviewed.

gate_controlled_actions_not_yet_unlocked:

Training, Llama, same-family null, sanitizer benchmark, FAR aggregation, payload-diversity claim, and paper-facing positive claim remain gate-controlled and not unlocked by this artifact-only preflight.
