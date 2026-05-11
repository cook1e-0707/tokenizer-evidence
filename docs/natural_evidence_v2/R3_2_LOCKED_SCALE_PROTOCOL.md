# R3.2 Same-Contract Locked-Scale Protocol

## Scope

R3.2 tests Qwen v2 controlled-natural micro-slot stability for one reviewed
contract:

```text
contract_id = a55e
payload_diversity_tested = false
model_family = Qwen only
```

R3.2 is not a distinct-payload route. Distinct payloads are deferred to R3.4.

## Schedule

```text
replicate_groups = 12
blocks_per_group = 8
block_size = 64
total_blocks_per_arm = 96
generation_seed_cycle = [17, 23, 29]
prompt_window_policy = deterministic_5_window_circular_reuse_by_replicate_group_index
```

Canonical identifiers:

```text
replicate_group_id = shard_00..shard_11
block_id = C_A55E_shard_XX_block_YY
```

## Arms

```text
protected
raw
task_only
wrong_key
wrong_payload
```

## Decoder

The decoder is the precommitted repeated-coordinate majority decoder:

```text
support_threshold = 16
majority_margin_threshold = 3
query_budgets = [16, 32, 64]
primary_budget = 64
checksum_required = true
threshold_changes_allowed_after_transcript = false
```

## Pass Gate

```text
protected accepts @64 >= 80/96
raw accepts @64 = 0/96
task_only accepts @64 = 0/96
wrong_key accepts @64 = 0/96
wrong_payload accepts @64 = 0/96
min accepted-block support >= 16
min accepted-block majority margin >= 3
forbidden public surface count = 0
all 12 replicate groups complete = true
```

Any null accept stops the route. Protected accepts below `80/96` stop expansion
and require analysis by shard, step index, prompt family, target bucket hit,
support, and margin. Thresholds may not be lowered after transcript generation.

## Required Preflight

Before any R3.2 full Slurm submission:

1. same-contract payload semantics decision must be recorded;
2. config must contain `contract_id=a55e` and no `payload_ids`;
3. wrapper must hard-fail fake distinct-payload labels;
4. plan-only precommit must pass with the same-contract schema;
5. full wrapper must replay job `852426` exactly before running new Slurm;
6. allowlist must be enabled for exactly one reviewed job and disabled after
   submission;
7. TG/email notification must succeed before submission.

## Claim Boundaries

R3.2 passing is not full FAR, not payload diversity, not Llama, not
cross-family, not sanitizer robustness, and not a paper-facing positive claim.

