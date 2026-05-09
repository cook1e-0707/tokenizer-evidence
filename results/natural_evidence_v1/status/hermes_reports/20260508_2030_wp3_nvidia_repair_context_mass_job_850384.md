# Hermes/Codex progress report

## Status

Prepared and submitted the WP3 NVIDIA-assisted repair context-mass scoring job.

## What changed

- Added `scripts/natural_evidence_v2/nvidia_assisted_context_repair_design.py`.
- Used NVIDIA `qwen/qwen3.5-397b-a17b` and `z-ai/glm-5.1` only as design assistants.
- Wrote proposal artifacts to:
  `results/natural_evidence_v2/status/nvidia_assisted_context_repair_20260508_2021/`.
- Added `scripts/natural_evidence_v2/build_wp3_nvidia_repair_context_mass_plan.py`.
- Built an 8-row artifact-only repair score plan at:
  `results/natural_evidence_v2/status/wp3_nvidia_repair_context_mass_plan_20260508_2028/`.
- Dropped two risky GLM suggestions before scoring.
- Validated the plan locally and on Chimera with `--validate-plan-only`.
- Synced required v2 scripts/configs/artifacts to Chimera.
- Submitted exactly one allowlisted Slurm job:
  `850384` (`nat-ev-v2-wp3ctxm`) on `DGXA100`.

## Job

```text
job_id=850384
state=PENDING(Resources)
score_plan=results/natural_evidence_v2/status/wp3_nvidia_repair_context_mass_plan_20260508_2028/qwen_v2_wp3_nvidia_repair_context_mass_score_plan.jsonl
```

## Guardrails

No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, or positive paper claim was started. The NVIDIA model outputs are
not gates; final validation remains base-Qwen scoring through Chimera Slurm.

## Next action

Monitor job `850384`. After it completes, sync and review the context-mass
artifacts before any further scoring submission.
