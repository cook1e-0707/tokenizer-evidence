# R3.2 Qwen locked-scale wrapper review

Status:

```text
R3_2_QWEN_LOCKED_SCALE_WRAPPER_REVIEWED_PLAN_ONLY_NO_SLURM
```

Implemented and locally validated the R3.2 Qwen locked-scale plan-only wrapper:

```text
scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
tests/test_natural_evidence_v2_wp6_coordinate_majority.py
```

The wrapper reconstructs the selected prompt manifest and refuses a mismatch
against:

```text
4d49ae100b272f184a8b2563e5b64f768e6db01425a2384f1457a4eb10eedb67
```

Local plan-only validation wrote:

```text
results/natural_evidence_v2/status/r3_2_wrapper_plan_validation_20260511_0318/precommit/r3_2_selected_prompt_manifest.json
results/natural_evidence_v2/status/r3_2_wrapper_plan_validation_20260511_0318/precommit/r3_2_qwen_locked_scale_contract.json
```

Validation:

```text
py_compile = PASS
bash -n = PASS
plan-only wrapper validation = PASS
pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py = PASS_8_TESTS
```

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

Next allowed action:

```text
Stop until a later explicit, notified R3.2 submission tick authorizes exactly
one reviewed Slurm job. Llama, same-family nulls, sanitizer, FAR aggregation,
and paper-facing positive claims remain disabled.
```
