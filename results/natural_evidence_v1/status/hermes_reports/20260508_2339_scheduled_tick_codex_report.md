Prepared the WP3 context-specific mass scorer, without submitting Slurm or starting model scoring.

Added [score_wp3_context_mass.py](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/score_wp3_context_mass.py:21) and [wp3_context_mass_score.sbatch](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch:1). The scorer consumes `qwen_v2_wp3_context_mass_score_plan.jsonl`, scores `bucket_surfaces` at `prefix_before_candidate`, derives token IDs from contextual `prefix + surface`, keeps casing variants separate, and writes context-score, mass, audit, and summary artifacts.

Updated the v2 allowlist at [run_allowlist.yaml](/Users/guanjie/Documents/tokenizer_alignment/configs/natural_evidence_v2/run_allowlist.yaml:20), plus state/docs in [gate_status.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/gate_status.json:87), [AUTOMATION_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v1/AUTOMATION_STATE.md:2442), and the Hermes report.

Validation passed:
- `python3 -m py_compile scripts/natural_evidence_v2/score_wp3_context_mass.py`
- `bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch`
- `python3 scripts/natural_evidence_v2/score_wp3_context_mass.py --validate-plan-only`

Plan validation: `230` rows, `115` lowercase, `115` sentence-case, `2` empty-prefix rows handled with BOS/EOS fallback metadata.

No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR aggregation, positive claim, Slurm submission, or model scoring was started. Next allowed action is exactly one allowlisted Chimera Slurm submission of `scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch`, then artifact sync/review.