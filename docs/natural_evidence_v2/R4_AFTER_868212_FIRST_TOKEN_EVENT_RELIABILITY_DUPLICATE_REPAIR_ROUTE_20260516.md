# R4 After-868212 First-Token Event Reliability/Duplicate Repair Route

Date: 2026-05-16

## Decision

Do not scale or rerun the 868212 route as-is. Record 868212 as a diagnostic
small-scale result:

```text
protected first-token accepts: 3/4
raw/task-only/wrong-key/wrong-payload accepts: 0/4 each
block-level forbidden public surface count: 0
block-level duplicate response hash count: 0
global duplicate response hash count: 4424
failed protected block: shard_03_block_00
failed bit index: 1
missing coordinate: 26
coordinate-26 protected erasures in shard_03: 64/64
```

This is a real first-token event-channel signal, but it is not a locked
positive and not a paper-facing claim.

## Route

Current phase:

```text
V2_R4_AFTER_868212_FIRST_TOKEN_EVENT_RELIABILITY_DUPLICATE_REPAIR_ROUTE_RECORDED_ARTIFACT_ONLY_NEXT
```

The next route is artifact-only reliability and duplicate repair. No Slurm
generation/scoring/training job is authorized by this route.

## Required Repairs

### 1. No Singleton Code Bits

The 868212 failed block exposed a codebook fragility:

```text
bit_index: 1
coordinates: [26]
support in failed protected shard: 0
```

The next codebook/precommit must not contain any bit represented by a single
active coordinate. Each committed bit must have at least two precommitted active
coordinates, or the route must explicitly mark the experiment as a
single-coordinate ablation.

Hard preflight checks:

```text
all bits have active_coordinate_count >= 2
coordinate 26 cannot be the sole coordinate for any bit
source coordinate reliability table is recorded
868212 is not reclassified under the repaired codebook
wrong-key/wrong-payload controls remain configured
```

### 2. Global Duplicate Policy Repair

The per-block duplicate gate was clean, but global generated-output duplicates
remain high:

```text
generated rows: 9216
unique response hashes: 4792
duplicate hash groups: 2908
duplicate extra rows: 4424
max duplicate group size: 4
dominant condition set: protected,raw
dominant shard pairs: shard_00/shard_01 and shard_02/shard_03
```

The next wrapper/preflight must separate and report:

```text
within-arm duplicate response hashes
within-shard duplicate response hashes
cross-shard duplicate response hashes
cross-arm duplicate response hashes
per-block decoder duplicate hashes
```

Before any locked-scale claim, the duplicate policy must specify which duplicate
classes are hard failures and which are diagnostic caveats. A route that keeps
cross-arm prompt sharing must not silently call global duplicates a pass.

## Implementation Tasks

Codex/Hermes may proceed automatically with these artifact-only tasks:

```text
1. Build a reliability repair preflight that rejects singleton-bit codebooks.
2. Build a duplicate taxonomy/preflight for future generation wrappers.
3. Add tests covering singleton-bit rejection and duplicate taxonomy.
4. Validate locally and, if needed, remotely in plan-only mode.
5. Update CURRENT_STATE.md and gate_status.json.
```

## Not Allowed By This Route

```text
new Slurm generation
new model scoring
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
reclassifying 868212 as a locked positive
```

## Next Compute Unlock Condition

Only after the artifact-only repair/preflight passes may Codex/Hermes record a
new single-submission diagnostic route. That future route must explicitly state:

```text
codebook has no singleton bits
duplicate policy is precommitted
quality gates are unchanged or explicitly justified
868212 remains diagnostic-only
```
