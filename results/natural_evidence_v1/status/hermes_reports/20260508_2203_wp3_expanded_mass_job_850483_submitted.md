# Hermes/Codex progress report

## Event

Submitted the approved restricted Step-label expanded action-verb
context-mass scoring job.

## Submission

```text
job_id=850483
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
node=chimera13
initial_state=RUNNING
```

The job uses the intended score plan:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl
```

The Chimera wrapper uses the configured virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Guardrails

The `v2_wp3_context_mass_score` allowlist entry was enabled only for this
submission and disabled immediately afterward:

```text
enabled=false
enable_condition=submitted_once_as_job_850483_pending_restricted_step_label_expanded_mass_result_review
```

No training, model-output generation, WP4, Qwen E2E, Llama, same-family null,
sanitizer, FAR aggregation, or paper-facing positive claim was started.

## Initial Wrapper Output

Plan validation passed:

```text
score_plan_rows=128
status=PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
```

Tokenizer validation skipped `16` rows because `Organize` is not one Qwen next
token, leaving `112` valid rows for scoring under the intended
`--skip-invalid-tokenization` policy.

## Next Allowed Action

Monitor job `850483`. After completion, sync and review its mass artifacts
before any further action.
