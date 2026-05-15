# Hermes/Codex Sync: R4 Selectivity Prompt-Policy Static Validation

Timestamp UTC: 2026-05-15T02:42:59Z

## Phase

`V2_R4_POSITIVE_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_PASS_NO_COMPUTE`

## Result

Codex executed the next allowed artifact-only step and built the selectivity
prompt-policy package. No Slurm job was submitted, no generation/model
scoring/training was started, and no claim gate was unlocked.

Static validation:

- policy id: `r4_positive_selectivity_prompt_policy_v1`
- prompt count: `2048`
- duplicate prompt ids: `0`
- forbidden prompt violations: `0`
- max policy family fraction: `0.1669921875`
- expected fixture events: `48`
- fixture events per family: `8`
- prompt bank sha256:
  `c22134d3f3d8510a07ca6104a1278abb64f94247c40758a14b6e97fbbdb856d6`

Artifacts:

- `configs/natural_evidence_v2/r4_positive_selectivity_prompt_policy.yaml`
- `scripts/natural_evidence_v2/build_r4_positive_selectivity_prompt_policy.py`
- `tests/natural_evidence_v2/test_r4_positive_selectivity_prompt_policy.py`
- `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/`
- `docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_20260515_0242.md`

## Next Allowed Action

Artifact-only generation/decode route planning and wrapper review for a small
H200 dev diagnostic using the selectivity package and prompt-policy package.

No Slurm submission is unlocked until wrapper review, local/remote plan-only
validation, allowlist safety, Hermes TG/email notification, H200/pomplun policy,
and exactly-one submission gates pass. Training, Llama, same-family null,
sanitizer, FAR, payload diversity, and paper-facing positive claims remain
locked until later route gates pass.
