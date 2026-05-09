# Hermes Codex action report

phase:
V2_WP3_CONTEXT_MASS_JOB_850372_FAILED_NEEDS_PREFIX_BOUNDARY_REPAIR

action:
Prepared an artifact-only WP3 context-mass plan/scorer repair for the tokenizer
prefix-boundary retokenization observed in Slurm job `850372`. No Slurm job was
submitted.

artifacts:
```text
scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
scripts/natural_evidence_v2/score_wp3_context_mass.py
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
configs/natural_evidence_v2/run_allowlist.yaml
tests/test_natural_evidence_v2_context_mass.py
results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/qwen_v2_wp3_context_mass_score_plan.jsonl
results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/qwen_v2_wp3_context_mass_score_plan_summary.json
```

repair:
- The scorer now resolves each row's bucket surfaces against `prefix + surface`
  tokenization and, when the tokenizer merges across the boundary, scores the
  shared longest token prefix.
- The repair refuses mixed rows: all bucket surfaces must share one adjusted
  scoring prefix and each candidate continuation must be exactly one next token.
- The Slurm wrapper now runs tokenizer-only boundary validation before model
  load or scoring.
- The v2 GPU allowlist entry is disabled pending review and explicit
  allowlisting.

plan_review:
```text
score_plan_rows=230
lowercase_rows=115
sentence_case_rows=115
empty_prefix_rows=2
prefix_boundary_tokenization_policy=score_longest_common_token_prefix_when_candidate_retokenizes_boundary
```

validation:
```text
python3 -m py_compile scripts/natural_evidence_v2/score_wp3_context_mass.py scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
pytest -q tests/test_natural_evidence_v2_context_mass.py
python3 scripts/natural_evidence_v2/score_wp3_context_mass.py --validate-plan-only
```

result:
The repaired plan/scorer is prepared and locally validated without model
scoring. Configured-Qwen tokenizer-only validation is implemented but not run
locally because `transformers` is unavailable in this local environment.

forbidden_actions_confirmed:
No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer,
FAR aggregation, positive claim, Slurm submission, or model scoring was started.

next_allowed_action:
Review the repaired WP3 context-mass plan/scorer and validation record. Do not
submit another Slurm scoring job until the repair is reviewed and explicitly
allowlisted.
