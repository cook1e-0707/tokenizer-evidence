# Hermes sync: R4 prefix-native repair candidate built

phase:
V2_R4_PREFIX_NATIVE_SURFACE_REPAIR_CANDIDATE_PROXY_VALIDATED_NO_COMPUTE

summary:
Codex executed the next allowed artifact-only step after the `853815` teacher-forced surface-mass failure diagnosis.

New artifacts:
- `scripts/natural_evidence_v2/build_r4_prefix_native_surface_repair_candidate.py`
- `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/candidate_prefix_native_surface_bank.json`
- `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/r4_prefix_native_surface_probe_rows.jsonl`
- `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/static_validation_summary.json`
- `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/static_validation_report.md`
- `results/natural_evidence_v2/status/r4_prefix_native_surface_score_wrapper_plan_smoke_20260513/r4_teacher_forced_surface_mass_summary.json`

Static proxy validation:
- status: PASS_PROXY_STATIC_VALIDATION_TOKENIZER_PENDING
- coordinates: 32
- entries: 256
- probe rows: 8,192
- prompts: 256
- missing binary-side coordinates: 0
- normalized first-word proxy overlap coordinates: 0
- forbidden surface hits: 0
- measured span-start failures: 0

Design change:
The candidate uses prefix-native continuations whose measured span starts immediately after the local lead-in prefix. Binary sides reuse the R3/WP5 learned action families in cover-natural phrases: set/plan versus create/prepare.

Important limitation:
Qwen tokenizer validation was not run locally because local `transformers` is unavailable. The local check is proxy-only. Any tokenizer/model scoring must be a separate reviewed Slurm-only route.

next_allowed_action:
Review this repaired candidate and prepare a separate Slurm-only tokenizer/model scoring route if accepted. Do not enable allowlist or submit Slurm from this artifact alone. Do not run generation, training, Llama, FAR, sanitizer, or paper-claim actions from this state.
