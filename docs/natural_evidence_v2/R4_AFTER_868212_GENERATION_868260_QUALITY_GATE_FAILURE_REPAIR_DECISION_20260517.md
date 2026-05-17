# R4 After-868212 Generation 868260 Quality-Gate Failure Repair Decision

Date: 2026-05-17

## Decision

Job `868260` is reviewed as a failed strict diagnostic, not a positive result.
It must not be reclassified by changing thresholds, duplicate policy, forbidden
surface policy, decoder settings, payload, or codebook after seeing the
transcripts.

The run did show first-token event signal:

```text
protected accepts ignoring quality:
  4/4
strict protected accepts:
  2/4
raw/task-only/wrong-key/wrong-payload strict accepts:
  0/4 each
```

The strict gate failed because the quality filters rejected two protected
blocks:

```text
shard_00_block_00:
  decoded expected codeword, valid checksum, duplicate_response_hash_count=1

shard_01_block_00:
  decoded expected codeword, valid checksum,
  duplicate_response_hash_count=2,
  forbidden_public_surface_count=1
```

The forbidden hit was literal `bucket` in an ordinary physical
home-maintenance/plumbing sense. The current policy still hard-forbids `bucket`,
so the block correctly fails under the precommitted policy. This is a policy
repair candidate for future runs only; it does not rescue `868260`.

Global duplicate caveat remains severe:

```text
generated rows:
  12288
unique response hashes:
  4676
global duplicate extra rows:
  7612
max duplicate group size:
  8
```

## Next Route

Current phase:

```text
V2_R4_AFTER_868212_GENERATION_868260_FAILED_QUALITY_GATE_REPAIR_DECISION_RECORDED_NO_SUBMIT
```

Next allowed action:

```text
artifact-only quality-gate repair package:
  1. Build and validate contextual forbidden-surface policy v2.
  2. Build and validate duplicate-safe generation/allocation policy.
  3. Update route validator/wrapper only in plan-only mode.
  4. Record a new reviewed rerun route before any Slurm submission.
```

## Required Repair Constraints

The contextual forbidden-surface policy v2 must:

```text
- keep technical bucket/fingerprint/watermark/payload/secret-key/decoder/hidden-signal surfaces forbidden;
- allow ordinary physical "bucket" only under precommitted task-domain cues;
- report ordinary-domain literal uses separately;
- keep technical public literal count max at 0;
- fail closed for ambiguous technical contexts.
```

The duplicate-safe generation/allocation policy must:

```text
- reject within-block duplicate response hashes;
- track global duplicate response hash count;
- reduce deterministic prompt/prefix collisions before compute;
- preserve raw/task-only/wrong-key/wrong-payload controls;
- not use 868260 transcripts to reclassify 868260;
- not lower the strict accept gate.
```

## Not Unlocked

This decision does not unlock:

```text
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
another Slurm generation rerun
```
