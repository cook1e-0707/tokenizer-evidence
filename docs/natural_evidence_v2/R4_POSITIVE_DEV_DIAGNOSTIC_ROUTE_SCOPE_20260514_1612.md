# R4 Positive Dev Diagnostic Route Scope

Timestamp: `2026-05-14T16:12:00Z`

## Scope

This is an artifact-only route-scope review for a future R4 dev diagnostic
using the precommitted positive event bank. It does not submit Slurm, enable
allowlist entries, start generation, or unlock downstream claims.

Route id:

```text
r4_positive_event_bank_dev_diagnostic_v1
```

Source precommit:

```text
results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605/
```

Precommit hash:

```text
9ea75e28abf1842e78017fd9100a03fad75ff0b3ad316e5018f94644baf39b30
```

## Future Diagnostic Shape

If later implementation and preflight gates pass, the future diagnostic scope is:

```text
model family: Qwen only
contract: same-contract a55e only
payload diversity: false
split: dev
blocks: 32
prompts per block: 64
conditions: protected, raw, task_only, wrong_key, wrong_payload
primary scrub mode: all
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gpu: H200
time limit: 30-00:00:00
```

## Future Gates

```text
protected accepts >= 26/32
raw/task_only/wrong_key/wrong_payload accepts = 0/32 each
forbidden technical public surface count = 0
shallow structural AUC <= 0.60
duplicate generated-output hashes = 0
duplicate decode-row hashes = 0
min specificity margin >= 3.0
min weighted margin >= 3.0
```

## Required Before Any Submission

The route scope is not a submission approval. Before any Slurm job can be
submitted, the following must be recorded and pass:

```text
event extractor implementation review
generation wrapper plan-only validation
keyed decoder replay or fixture validation
local/remote hash preflight
zero-enabled allowlist preflight
active-job preflight
Hermes Telegram/email notification
exactly-one allowlist enablement
immediate allowlist disablement after sbatch
post-submit zero-enabled allowlist safety
```

## Validation

Config:

```text
configs/natural_evidence_v2/r4_positive_dev_diagnostic_route.yaml
```

Validator:

```text
scripts/natural_evidence_v2/validate_r4_positive_dev_diagnostic_route.py
```

Summary:

```text
results/natural_evidence_v2/status/r4_positive_dev_diagnostic_route_scope_20260514_1612/route_scope_validation_summary.json
```

Status:

```text
PASS_R4_POSITIVE_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT
```

Focused tests:

```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_event_bank_precommit.py tests/natural_evidence_v2/test_r4_positive_dev_diagnostic_route.py -q
10 passed
```

## Next Allowed Action

Implement and review the artifact-only event extractor and generation/decode
wrapper plan-only path for this route. Do not submit Slurm or enable allowlist
until those implementation/preflight gates pass.
