# R3.2 Payload Semantics Decision

## Decision

R3.2 is a same-contract locked-scale stability package, not a
payload-diversity package.

The only adopted positive contract for this route is the reviewed WP5-R2/WP6-R2
`a55e` contract:

```text
contract_path = results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
contract_id = a55e
payload_data_byte_hex = a5
checksum_byte_hex = 5e
payload_plus_checksum_hex = a55e
```

Any table unit in R3.2 must be interpreted as an independent replicate block
under the same payload contract, not as a distinct payload.

## Consequences

R3.2 must not use `P00/P01/P02/P03` as payload labels. Those labels are
removed from the canonical R3.2 route. The canonical unit names are:

```text
contract_id = a55e
replicate_group = shard_00..shard_11
block_id = C_A55E_shard_XX_block_YY
generation_seed = deterministic seed cycle [17, 23, 29]
```

Distinct payload evaluation is deferred to R3.4. R3.4 requires independent
payload contracts, checksums, precommitments, and teacher-forced
protected-vs-task-only compatibility gates. It may not reuse the R3.2
same-contract locked-scale result as payload-diversity evidence.

## Forbidden Interpretations

These interpretations are explicitly forbidden:

```text
P00/P01/P02/P03 are four payloads, but all reuse a55e.
Use the current adapter to evaluate four distinct payloads without separate
contract/checksum/precommit/teacher-forced compatibility evidence.
```

## Current Gate

R3.2-S may proceed with artifact-only wrapper/precommit repair. Full Slurm
submission remains disabled until:

1. the same-contract config validates;
2. the wrapper hard-validates that no fake payload labels are present;
3. the plan-only precommit artifacts are regenerated under the same-contract
   schema;
4. the full wrapper is reviewed and replayed against job `852426`.

## Claim Control

R3.2 may only support this internal statement after its locked-scale run passes:

```text
Qwen v2 same-contract controlled-natural micro-slot protocol passes a
locked-scale stability package under the precommitted repeated-coordinate
majority decoder.
```

R3.2 still does not support full FAR, payload diversity, Llama success,
cross-family generality, sanitizer robustness, paper-facing positive claims, or
superiority over Scalable/Perinucleus.

