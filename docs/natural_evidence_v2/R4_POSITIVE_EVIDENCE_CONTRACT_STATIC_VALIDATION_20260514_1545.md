# R4 Positive Evidence Contract Static Validation

Timestamp UTC: 2026-05-14T15:45:00Z

## Scope

This artifact records static validation for the redesigned R4 positive evidence
contract. It did not run Slurm, generation, Qwen E2E rerun, training,
tokenizer/model scoring, Llama, same-family null, sanitizer, FAR aggregation,
payload-diversity work, or paper-facing claim work.

## Contract

- Config:
  `configs/natural_evidence_v2/r4_positive_evidence_contract_redesign.yaml`
- Contract id: `r4_keyed_correlation_evidence_v1`
- Validator:
  `scripts/natural_evidence_v2/validate_r4_positive_evidence_contract.py`
- Toy decoder:
  `scripts/natural_evidence_v2/r4_keyed_correlation_decoder.py`

## Validation Result

- static contract validation:
  `PASS_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE`;
- focused pytest for contract + toy decoder: `10 passed`;
- `py_compile`: passed;
- local allowlist safety: `PASS`;
- active Chimera jobs: none observed in the previous blocker check.

## What Is Now Resolved

The redesigned contract now has an artifact-only specification for:

- key/payload specificity before accept scoring;
- support not being equivalent to acceptance;
- wrong-key and wrong-payload recomputation;
- structural features excluded from votes;
- primary `format_scrub=all` decoding;
- pre-registered dev pass/fail table.

## What Is Still Not Ready

No compute route is unlocked. The contract is still missing a concrete
artifact-only precommit package:

- frozen event/surface bank;
- coordinate/polarity mapping manifest;
- checksum/codeword manifest;
- decoder spec hash;
- dev-only preflight manifest;
- local/remote hash preflight plan.

The next project-advancing action is artifact-only event-bank/precommit package
construction or static review. After that package passes, a future compute
route may be reviewed under the standing conditional execution authorization.
