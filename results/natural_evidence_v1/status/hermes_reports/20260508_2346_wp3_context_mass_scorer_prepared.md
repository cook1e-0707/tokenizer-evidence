# Hermes Codex action report

phase:
V2_WP3_CONTEXT_SPECIFIC_MASS_SCORER_PREPARED_NEEDS_SLURM_SUBMISSION

action:
Prepared the plan-consuming WP3 context-specific base-Qwen mass scorer and
Slurm wrapper. No Slurm job was submitted.

artifacts:
```text
scripts/natural_evidence_v2/score_wp3_context_mass.py
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
configs/natural_evidence_v2/run_allowlist.yaml
```

plan_review:
```text
score_plan_rows=230
lowercase_rows=115
sentence_case_rows=115
empty_prefix_rows=2
```

validation:
```text
python3 -m py_compile scripts/natural_evidence_v2/score_wp3_context_mass.py
bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
python3 scripts/natural_evidence_v2/score_wp3_context_mass.py --validate-plan-only
```

result:
The scorer reads `qwen_v2_wp3_context_mass_score_plan.jsonl`, scores
`bucket_surfaces` at `prefix_before_candidate`, keeps casing variants separate,
and writes context-score, mass, and audit JSON artifacts. It derives candidate
token IDs from contextual `prefix + surface` tokenization and records BOS/EOS
fallback metadata for start-of-response rows.

forbidden_actions_confirmed:
No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer,
FAR aggregation, positive claim, Slurm submission, or model scoring was started.

next_allowed_action:
Submit exactly one allowlisted Chimera Slurm job using
`scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch`, then sync
and review the mass/audit artifacts.
