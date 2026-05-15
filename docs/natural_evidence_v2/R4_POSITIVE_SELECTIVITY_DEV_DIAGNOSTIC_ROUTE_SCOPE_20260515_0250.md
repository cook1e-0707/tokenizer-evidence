# R4 Positive Selectivity Dev Diagnostic Route Scope Review

Date: 2026-05-15T02:50:38Z

## Decision

Status:

```text
PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT
```

This is an artifact-only route-scope review. It does not submit Slurm, enable an
allowlist entry, start generation, start training, or create a paper-facing
claim.

## Inputs

- Selectivity package:
  `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158`
- Prompt-policy package:
  `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242`
- Route config:
  `configs/natural_evidence_v2/r4_positive_selectivity_dev_diagnostic_route.yaml`

The reviewed config keeps this as a Qwen-only, same-contract `a55e` dev
diagnostic route using the R4 positive selectivity package and the new
selectivity prompt-policy package.

## Route Scope

- Model family: Qwen only.
- Contract: `r4_positive_selectivity_repair_v1`.
- Payload: same-contract `a55e`; no payload diversity is tested.
- Future compute policy: H200 on `pomplun`, account `cs_yinxin.wan`, max time
  `30-00:00:00`.
- Future dev scale: 32 blocks, 64 prompts per block, 4 shards, 512 prompts per
  shard.
- Conditions: `protected`, `raw`, `task_only`, `wrong_key`, `wrong_payload`.
- Primary decode: `format_scrub=all`.

Future dev gate, if a later wrapper/submission route is reviewed and unlocked:

```text
protected accepts >= 26/32
raw accepts = 0/32
task_only accepts = 0/32
wrong_key accepts = 0/32
wrong_payload accepts = 0/32
min specificity margin >= 4.0
min keyed score >= 8.0
forbidden technical public surface count = 0
duplicate generated-output hash count = 0
duplicate decode-row hash count = 0
```

## Added Local Review Surface

- `scripts/natural_evidence_v2/decode_r4_positive_support_window_correlation.py`
  adds an explicit support-window keyed-correlation decode path for generated
  outputs.
- `scripts/natural_evidence_v2/validate_r4_positive_selectivity_dev_diagnostic_route.py`
  validates the route scope and keeps compute disabled at this stage.
- Focused tests cover the route scope and a toy support-window decode fixture.

## Validation

Commands:

```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_support_window_decode.py tests/natural_evidence_v2/test_r4_positive_selectivity_dev_diagnostic_route.py
uv run python -m py_compile scripts/natural_evidence_v2/decode_r4_positive_support_window_correlation.py scripts/natural_evidence_v2/validate_r4_positive_selectivity_dev_diagnostic_route.py
uv run python scripts/natural_evidence_v2/validate_r4_positive_selectivity_dev_diagnostic_route.py --output-dir results/natural_evidence_v2/status/r4_positive_selectivity_dev_diagnostic_route_scope_20260515_0250
```

Results:

```text
pytest: 6 passed
py_compile: pass
route scope validator: PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT
```

## Current Control

New canonical phase:

```text
V2_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_PASS_NO_SUBMIT
```

New blocker:

```text
BLOCK_R4_POSITIVE_SELECTIVITY_H200_WRAPPER_PLAN_ONLY_NEXT
```

Next allowed action:

```text
Artifact-only H200 generation/decode wrapper implementation and plan-only
validation for this selectivity dev diagnostic route. No Slurm submission is
unlocked until wrapper review, local/remote plan-only validation, allowlist
safety, Hermes TG/email notification, H200/pomplun policy, active-job preflight,
exactly-one allowlist enablement, and immediate post-submit allowlist
disablement all pass.
```

