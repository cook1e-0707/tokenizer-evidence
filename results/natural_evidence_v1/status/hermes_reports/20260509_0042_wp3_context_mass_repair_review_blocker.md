# Hermes Codex action report

phase:
V2_WP3_CONTEXT_MASS_PREFIX_BOUNDARY_REPAIR_PREPARED_NEEDS_REVIEW_ALLOWLIST

action:
Reviewed the repaired WP3 context-mass plan/scorer and validation record. The
repair is not allowlisted.

reviewed_artifacts:
```text
scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
scripts/natural_evidence_v2/score_wp3_context_mass.py
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
configs/natural_evidence_v2/run_allowlist.yaml
tests/test_natural_evidence_v2_context_mass.py
results/natural_evidence_v2/status/wp3_context_mass_plan_prefix_boundary_repair_20260509_0024/qwen_v2_wp3_context_mass_score_plan_summary.json
results/natural_evidence_v1/status/hermes_reports/20260509_0024_wp3_context_mass_prefix_boundary_repair_prepared.md
```

validation_recheck:
```text
python3 -m py_compile scripts/natural_evidence_v2/score_wp3_context_mass.py scripts/natural_evidence_v2/build_wp3_context_mass_plan.py
PASS

bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
PASS

python3 scripts/natural_evidence_v2/score_wp3_context_mass.py --validate-plan-only
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
score_plan_rows=230

pytest -q tests/test_natural_evidence_v2_context_mass.py
FAIL
TypeError: validate_tokenizer_boundaries() missing 1 required keyword-only argument: 'skip_invalid'
```

blockers:
- The recorded local validation claim is stale for the current working tree:
  `tests/test_natural_evidence_v2_context_mass.py` no longer matches the
  current scorer API.
- The GPU allowlist entry must stay disabled until this mismatch is repaired
  and reviewed. The current review does not authorize Slurm scoring.
- Configured-Qwen tokenizer-only validation is still not locally recorded.
- Local references to later NVIDIA/step-local repair paths are out of scope for
  the controlling `20260509_0039` review-only tick and were not monitored or
  extended by this action.

decision:
`NOT_ALLOWLISTED_BLOCKED_LOCAL_VALIDATION_MISMATCH`

No Slurm scoring job is allowed from this review. Keep the GPU allowlist
disabled until the scorer/test API mismatch is repaired, no-model validation
passes in the current working tree, and a later review explicitly allowlists
the scoring wrapper.

forbidden_actions_confirmed:
No Slurm job, model scoring, training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer, FAR aggregation, or positive claim was started.

next_allowed_action:
Repair the WP3 context-mass scorer/test validation mismatch and rerun the local
no-model validation only. Do not submit Slurm scoring until a later review
explicitly allowlists the repaired wrapper.
