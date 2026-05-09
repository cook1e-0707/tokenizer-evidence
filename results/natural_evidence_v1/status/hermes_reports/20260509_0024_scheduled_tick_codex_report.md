Prepared the artifact-only WP3 context-mass prefix-boundary repair. No Slurm job, model scoring, training, generation, E2E, FAR, or positive claim was started.

Changed the scorer to handle `prefix + surface` retokenization with a shared longest-token-prefix policy and reject unsafe mixed-prefix rows: [score_wp3_context_mass.py](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/score_wp3_context_mass.py:225). The Slurm wrapper now points at the repaired plan and runs tokenizer-only validation before model load: [wp3_context_mass_score.sbatch](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch:26). The GPU allowlist is disabled pending review: [run_allowlist.yaml](/Users/guanjie/Documents/tokenizer_alignment/configs/natural_evidence_v2/run_allowlist.yaml:20).

New repaired plan artifact:
`results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/`
with `230` rows, `115` lowercase and `115` sentence-case.

Status/report updated:
[20260509_0024_wp3_context_mass_prefix_boundary_repair_prepared.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260509_0024_wp3_context_mass_prefix_boundary_repair_prepared.md:1)

Validation passed:
`py_compile`, `bash -n`, `pytest -q tests/test_natural_evidence_v2_context_mass.py`, and scorer `--validate-plan-only`. Configured-Qwen tokenizer-only validation was implemented but not run locally because `transformers` is not installed in this local environment.