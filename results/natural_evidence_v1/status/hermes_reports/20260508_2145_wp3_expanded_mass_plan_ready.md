# Hermes/Codex progress report

## Event

Reviewed the restricted Step-label repair plan and prepared a Slurm-only
tokenizer/context-mass score plan for expanded Step-label action-verb banks.

## Virtual Environment Note

User reminded Codex to use virtual environments for Python locally and on
Chimera. This run used:

```text
local=.venv/bin/python
chimera_default=/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Artifacts

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_expanded_mass_plan.py
results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_REPAIR_AND_EXPANDED_MASS_PLAN_REVIEW.md
```

## Score Plan

```text
score_plan_rows=128
expanded_bank_candidate_count=8
prefix_contexts=Step 1: ... Step 16:
casing_variant=sentence_case
compatible_wrapper=scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

Validation:

```text
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch passed
.venv/bin/python -m pytest -q tests/test_natural_evidence_v2_restricted_density.py passed
```

## Allowlist

The existing `v2_wp3_context_mass_score` allowlist entry remains disabled:

```text
enabled=false
enable_condition=pending_review_and_explicit_submission_approval_for_restricted_step_label_expanded_mass_plan_20260508_2148
```

## Next Allowed Action

Explicitly review/approve one Slurm context-mass scoring submission. If
approved, Codex can sync the score plan to Chimera, temporarily enable exactly
one allowlist entry, and submit one `wp3_context_mass_score.sbatch` job with
`SCORE_PLAN` pointing to the expanded plan.

Still forbidden: WP4, training, Qwen E2E, Llama, same-family null, sanitizer,
FAR aggregation, and positive paper claims.
