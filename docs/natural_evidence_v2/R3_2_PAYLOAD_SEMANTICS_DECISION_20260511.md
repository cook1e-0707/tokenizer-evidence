# R3.2 Payload Semantics Decision: 2026-05-11

Superseded: this earlier cell-label decision is superseded by
`docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION.md`. The canonical
R3.2 route must not use `P00/P01/P02/P03` labels at all; it is now a
same-contract `a55e` replicate-block stability route.

## Decision

For R3.2, `P00/P01/P02/P03` are package cell labels, not distinct payload
contracts.

Each R3.2 cell intentionally reuses the fixed WP5-R2 reviewed prompt-local
contract:

```text
contract_path = results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
payload_data_byte_hex = a5
checksum_byte_hex = 5e
payload_plus_checksum_hex = a55e
audit_key_id = KWP4_QWEN_PILOT_001
checksum_domain = natural_evidence_v2_wp4_prompt_local_checksum_v1
bucket_policy_id = qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1
```

`P00/P01/P02/P03` remain labels for the locked-scale grid:

```text
payload labels = P00, P01, P02, P03
seeds = 17, 23, 29
cells = 12
cell order = payload-major then seed
blocks per cell = 8
block size = 64
```

The labels do not authorize trying four payload byte/checksum contracts. They
only partition the R3.2 12-cell package for scale accounting over the same
learned WP5-R2 `a55e` contract.

## Rationale

WP5-R2 trained and passed the teacher-forced gate against the single fixed
WP4 prompt-local `a55e` contract. The reviewed WP6 generation/decode path and
repeated-coordinate decoder spec also bind the recovery predicate to
`payload_plus_checksum_hex = a55e`.

Treating `P00/P01/P02/P03` as distinct payloads would require separate reviewed
contract paths, expected bytes/checksums, checksum predicates, and adapter
compatibility evidence. Those artifacts do not exist in the current approved
R3.2 route.

## Consequences

- R3.2 wrapper review must validate that all 12 cells point to the same fixed
  `a55e` contract while preserving the `P00/P01/P02/P03` labels for reporting.
- Wrong-payload remains a null control evaluated against the reviewed wrong
  expected payload predicate; it is not one of `P00/P01/P02/P03`.
- No Slurm job is authorized by this decision.
- No generation, Qwen E2E rerun, training, Llama, same-family null, sanitizer,
  FAR aggregation, or paper-facing positive claim was started.

## Status

```text
R3_2_PAYLOAD_SEMANTICS_RESOLVED_A55E_CELL_LABELS_NO_SLURM
```
