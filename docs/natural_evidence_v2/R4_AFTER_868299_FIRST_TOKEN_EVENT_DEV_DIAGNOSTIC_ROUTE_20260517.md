# R4 after-868299 first-token event dev diagnostic route

Date: 2026-05-17

## Decision

The next route is a 32-block Qwen dev diagnostic for the provider-side keyed
first-token event channel.

This route is allowed because job `868299` passed the strict 4-block
quality-repair confirmation:

- protected strict accepts: 4/4
- protected accepts ignoring quality: 4/4
- raw/task-only/wrong-key/wrong-payload accepts: 0/4 each
- global duplicate response hashes: 0
- protected forbidden public surface count: 0
- trace binding: 12288/12288 valid

This route does not establish a text-only phrase decoder result and does not
unlock paper-facing claims.

## Scope

```text
blocks: 32
shards: 32
row cylinders per block: 1024
conditions: protected, raw, task_only
decode controls: protected, raw, task_only, wrong_key, wrong_payload
contract: a55e
model family: Qwen only
compute: H200 pomplun / cs_yinxin.wan
```

## Allocation Caveat

The reviewed full16 row bank currently supports four fully unique 1024-row
shards. This dev diagnostic therefore uses a precommitted cyclic reuse policy
over the reviewed four-shard allocation:

```text
dev_shard_i uses base_shard_(i mod 4)
```

Each dev shard still has zero within-shard prompt/prefix duplicates and uses a
distinct public shard id and public sampling seed. The global exact
response-hash duplicate gate remains zero.

This allocation is acceptable only for a dev diagnostic. It must not be called
locked-scale independent evidence.

## Gates

```text
protected strict accepts >= 28/32
protected accepts ignoring quality >= 30/32
raw/task-only/wrong-key/wrong-payload accepts = 0/32 each
within-block duplicate response hash = 0
global duplicate response hash = 0
technical forbidden public surface = 0
trace binding validity = 100%
full phrase decoder = report only, not a success claim
post-generation template leakage review required
```

## Allowed Next Action

After route validation, wrapper plan-only smoke, local/remote hash preflight,
zero-enabled allowlist safety, and Hermes synchronization pass, one H200 Slurm
array may be submitted using exactly one allowlist entry. The allowlist entry
must be disabled immediately after `sbatch` returns.

## Not Unlocked

This route does not unlock training, Llama, same-family null, sanitizer, FAR,
payload diversity, locked-scale claims, or paper-facing positive claims.
