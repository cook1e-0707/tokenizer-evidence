# R4 Positive Selectivity Prompt-Policy Static Validation

## Decision

Status: `PASS_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_NO_COMPUTE`.

The artifact-only prompt-policy elicitation package has been built and
statically validated for the `r4_positive_selectivity_repair_v1` event bank.
No generation, model scoring, training, Slurm submission, or claim action was
started.

## Package

- policy id: `r4_positive_selectivity_prompt_policy_v1`
- prompt count: `2048`
- source prompt bank:
  `results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl`
- source prompt bank sha256:
  `5fb7a5309c4afc02330bb7b2890d5ffaec954a3b221fab2c4da9ca7398740bac`
- selectivity package:
  `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158`
- event-window bank sha256:
  `0b3624281bce0637667e629b7d940eaae579c00efadce8be017fd03784a646f6`
- prompt bank sha256:
  `c22134d3f3d8510a07ca6104a1278abb64f94247c40758a14b6e97fbbdb856d6`

## Static Validation

- duplicate prompt ids: `0`
- forbidden prompt violations: `0`
- max policy family fraction: `0.1669921875`
- policy family counts:
  - `constraint_reasoning`: `342`
  - `handoff_trace`: `342`
  - `risk_review`: `341`
  - `context_alignment`: `341`
  - `communication_choice`: `341`
  - `quality_review`: `341`
- expected fixture events per family: `8`
- total expected fixture events: `48`
- validation errors: `[]`

The policy uses natural task-language instructions to elicit constraint,
handoff, risk, context, communication, and quality-review language. It does not
introduce Step labels, fixed slot counts, visible coordinates, or public
technical literals.

## Artifacts

- `configs/natural_evidence_v2/r4_positive_selectivity_prompt_policy.yaml`
- `scripts/natural_evidence_v2/build_r4_positive_selectivity_prompt_policy.py`
- `tests/natural_evidence_v2/test_r4_positive_selectivity_prompt_policy.py`
- `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/dev_prompts.jsonl`
- `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/prompt_policy_manifest.json`
- `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/expected_elicitation_fixtures.jsonl`

## Verification

- `uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_prompt_policy.py`
- `uv run python scripts/natural_evidence_v2/build_r4_positive_selectivity_prompt_policy.py`

## Next Allowed Action

Artifact-only generation/decode route planning for a small H200 dev diagnostic
using:

- selectivity package:
  `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158`
- prompt policy package:
  `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242`

No Slurm submission is unlocked by this validation alone. The route must still
pass wrapper review, local/remote plan-only validation, allowlist safety,
Hermes TG/email notification, H200/pomplun policy, and exactly-one submission
rules before any compute.
