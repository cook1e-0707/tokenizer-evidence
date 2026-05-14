# R4 Positive Event-Bank Precommit Package

Timestamp: `2026-05-14T16:05:00Z`

## Decision

The R4 redesigned positive evidence contract now has an artifact-only
event-bank precommit package. This is a precompute/control artifact, not a
model result. It does not unlock Slurm, generation, training, tokenizer/model
scoring, Llama, null/FAR, sanitizer, payload diversity, or paper-facing claims.

## Inputs

- Contract config:
  `configs/natural_evidence_v2/r4_positive_evidence_contract_redesign.yaml`
- Event-bank precommit config:
  `configs/natural_evidence_v2/r4_positive_event_bank_precommit.yaml`
- Builder:
  `scripts/natural_evidence_v2/build_r4_positive_event_bank_precommit.py`
- Toy keyed-correlation decoder:
  `scripts/natural_evidence_v2/r4_keyed_correlation_decoder.py`

## Output Package

Package directory:

```text
results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605/
```

Primary artifacts:

```text
surface_bank.json
coordinate_mapping.jsonl
codebook.json
decoder_spec.json
dev_gate.json
precommit_manifest.json
package_summary.json
package_report.md
```

## Validation Result

Status:

```text
PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE
```

Key checks:

| Field | Value |
| --- | ---: |
| surface count | 96 |
| surface families | 8 |
| max family fraction | 0.125 |
| distinct coordinates | 29 |
| positive polarity events | 35 |
| key material exposed | false |
| Slurm/model/generation started | false |

Precommit hash:

```text
9ea75e28abf1842e78017fd9100a03fad75ff0b3ad316e5018f94644baf39b30
```

Artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| contract config | `ec01b3f145bd3a1444b495b897c0484c7f25e7887e843477e63179a1cd3fb3b1` |
| surface bank | `98559cc0821f96071b551c7bb4104cb475ddae4cc68080d8745feeae121b7d27` |
| coordinate mapping | `f69c8f718b220a148297c13742a04bcadb14748dd0fd6631752cfc87afd85ea5` |
| codebook | `640fba3811a245b8ccc6997df441d310eb7b459e38ef54fbff895cbe9b208f40` |
| decoder spec | `49b3b5f30dc2a250e0d0681bd51779616142451041873d3342e286960b62cff1` |
| dev gate | `cb6d8f910b66c3c90f442b03ade9bbd5e031dd2095ca536b8aeb7efd094b812a` |
| manifest | `6799238e3c40becabf8a50a7c31ebf31a68b5d0f8016b4f49d3194777bb13b94` |

## Local Verification

Commands run:

```text
uv run pytest tests/natural_evidence_v2/test_r4_keyed_correlation_decoder.py tests/natural_evidence_v2/test_r4_positive_evidence_contract.py tests/natural_evidence_v2/test_r4_positive_event_bank_precommit.py -q
uv run python -m py_compile scripts/natural_evidence_v2/build_r4_positive_event_bank_precommit.py scripts/natural_evidence_v2/r4_keyed_correlation_decoder.py scripts/natural_evidence_v2/validate_r4_positive_evidence_contract.py
uv run python scripts/natural_evidence_v2/build_r4_positive_event_bank_precommit.py
uv run python scripts/natural_evidence_v2/check_allowlist_safety.py --output results/natural_evidence_v2/status/r4_positive_event_bank_precommit_allowlist_safety_20260514.json
```

Results:

```text
focused pytest: 15 passed
py_compile: pass
builder status: PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE
allowlist safety: PASS
```

## Current Claim Boundary

Allowed internal statement:

```text
The redesigned R4 keyed-correlation evidence contract now has a frozen
artifact-only event-bank/codebook/decoder/dev-gate precommit package.
```

Not allowed:

```text
positive generation result
payload recovery
Qwen E2E success
Llama or cross-family success
same-family null rejection
FAR
sanitizer robustness
payload diversity
paper-facing positive claim
```

## Next Allowed Action

Prepare an artifact-only dev diagnostic route review for the precommitted
event bank, including extractor/wrapper design, local/remote hash preflight,
zero-enabled allowlist safety, and Hermes notification requirements. Do not
submit Slurm or enable an allowlist entry until that route is recorded and all
preconditions pass.
