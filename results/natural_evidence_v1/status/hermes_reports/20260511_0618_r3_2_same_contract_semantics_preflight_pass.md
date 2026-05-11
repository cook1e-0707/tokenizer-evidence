# R3.2 same-contract semantics and preflight pass

Timestamp UTC: `2026-05-11T06:18Z`

## Expert Route Update Applied

R3.2 is now a same-contract locked-scale stability package for the reviewed
`a55e` contract. It is not a payload-diversity package.

Canonical R3.2 no longer uses `P00/P01/P02/P03` as payload labels or cell
labels. The canonical units are:

```text
contract_id = a55e
replicate_group = shard_00..shard_11
block_id = C_A55E_shard_XX_block_YY
```

Distinct payload evaluation is deferred to R3.4.

## Artifacts Written

```text
docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION.md
docs/natural_evidence_v2/R3_2_LOCKED_SCALE_PROTOCOL.md
configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml
results/natural_evidence_v2/status/r3_2_wrapper_preflight_summary.json
```

The older dated payload-semantics and prompt-allocation docs were marked as
superseded where needed.

## Validation

```text
py_compile build_r3_2_locked_scale_precommit.py = PASS
bash -n r3_2_qwen_locked_scale_eval.sbatch = PASS
plan-only wrapper preflight = PASS
fake payload_ids hard-fail = PASS
```

Plan-only preflight outputs:

```text
results/natural_evidence_v2/status/r3_2_same_contract_preflight_20260511_0615/precommit/r3_2_selected_prompt_manifest.json
results/natural_evidence_v2/status/r3_2_same_contract_preflight_20260511_0615/precommit/r3_2_qwen_locked_scale_contract.json
```

Selected prompt manifest SHA-256:

```text
71f6ce51fb1e4cfd8ef07fe74e284cf14d16a19651de95aa4b8e717eb1e78820
```

## Current Blocker

Full wrapper/aggregation path is still missing. R3.2 must replay job `852426`
exactly before any new Slurm submission.

## Next Allowed Action

Implement/review a full R3.2 same-contract `a55e` wrapper aggregation path and
replay job `852426` artifacts exactly. Do not enable allowlist or submit Slurm
until replay and wrapper review pass.

## Still Forbidden

```text
wp6_r3_2_locked_scale_allowed=false
training_allowed=false
llama_allowed=false
same_family_null_allowed=false
sanitizer_allowed=false
far_aggregation_allowed=false
paper_claim_allowed=false
```

No Slurm job was submitted.

