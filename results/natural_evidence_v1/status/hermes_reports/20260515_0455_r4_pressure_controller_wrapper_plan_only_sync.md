# Hermes Sync: R4 Pressure-Controller Wrapper Plan-Only

Timestamp UTC: `2026-05-15T04:55:00Z`

Status:
`PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY`

Codex completed the artifact-only H200 wrapper plan-only review:

```text
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRAPPER_PLAN_ONLY_20260515_0455.md
results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_wrapper_plan_only_20260515_0455/wrapper_plan_only_summary.json
```

Key points:

```text
wrapper: scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
H200 policy: pomplun / cs_yinxin.wan / gpu:h200:1 / 30-00:00:00
array grid: 72 controller cells
allowlist entry added disabled: v2_r4_positive_selectivity_pressure_controller_score_h200
plan-only wrapper run: PASS
allowlist safety: PASS with zero enabled entries
Hermes TG/email notification: SENT_ALL_REQUIRED_CHANNELS
```

Full scoring mode remains intentionally fail-closed:

```text
R4_PRESSURE_CONTROLLER_FULL_SCORING_REQUIRES_WRONG_CONTROL_WRAPPER_REVIEW
```

Current phase:

```text
V2_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRAPPER_PLAN_ONLY_PASS_NO_SUBMIT
```

Current blocker:

```text
BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_NEXT
```

Next allowed action:

```text
Artifact-only wrong-key / wrong-payload controller mapping design and wrapper review.
```

No Slurm/model scoring/generation/training/Llama/null/sanitizer/FAR/payload-diversity/paper-claim action is unlocked by this sync.
