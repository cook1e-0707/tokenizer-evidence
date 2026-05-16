# Hermes Sync: R4 After 867621 Reliability Tokenizer Route Validated

phase:
`V2_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATED_NO_SUBMIT`

blocker:
`BLOCK_R4_AFTER_867621_RELIABILITY_ACTUAL_QWEN_TOKENIZER_PREFLIGHT_PENDING`

summary:

```text
Codex completed the artifact-only route package after job 867621 failed the
protected generation positive gate.

New route:
- Purpose: actual Qwen tokenizer-boundary preflight for coordinate-unique
  reliability surface rows.
- Rows: 4096.
- Selected prompts: 256.
- Selected coordinates: 16.
- Static boundary preflight: PASS, 4096 checked rows, 0 failed rows.
- Route validation: PASS_R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_VALIDATION_NO_SUBMIT.
- Allowlist safety: PASS with zero enabled entries.

No Slurm job was submitted. No model forward, teacher-forced scoring,
generation, training, Llama, same-family null, sanitizer, FAR, payload-diversity
claim, or paper-facing claim action was started.
```

artifacts:

```text
docs/natural_evidence_v2/R4_AFTER_867621_RELIABILITY_TOKENIZER_PREFLIGHT_ROUTE_20260516.md
configs/natural_evidence_v2/r4_after_867621_reliability_tokenizer_preflight_route.yaml
scripts/natural_evidence_v2/build_r4_after_867621_reliability_surface_mass_rows.py
scripts/natural_evidence_v2/validate_r4_after_867621_reliability_tokenizer_route.py
scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch
results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/
results/natural_evidence_v2/status/r4_after_867621_reliability_static_boundary_preflight_20260516/
results/natural_evidence_v2/status/r4_after_867621_reliability_tokenizer_route_validation_20260516/
results/natural_evidence_v2/status/r4_after_867621_reliability_tokenizer_route_allowlist_safety_20260516.json
```

next_allowed_action:

```text
Fresh local/remote hash preflight and Hermes notification for exactly one
H200/pomplun tokenizer-only Slurm submission:

entry: v2_r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```

not_unlocked:

```text
teacher-forced surface-mass scoring
generation
training
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```
