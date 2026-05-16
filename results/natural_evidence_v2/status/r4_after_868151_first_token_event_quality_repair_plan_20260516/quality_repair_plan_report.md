# R4 After-868151 First-Token Event Quality Repair Plan

Date: 2026-05-16

## Status

`PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY`

This is an artifact-only planning record. No generation, scoring, training,
Slurm submission, Llama, FAR, sanitizer, payload-diversity route, or paper claim
is unlocked by this artifact.

## Literal Policy

The previous quality audit showed `coordinate` literal hits. The prompt bank
contains 44 / 256
coordination-domain prompts, so simply deleting that domain would leave too few
rows to preserve the existing 4-shard x 768-row diagnostic scope. The planned
repair is a contextual matcher:

- hard-forbid technical literals such as `bucket`, `fingerprint`, `watermark`,
  `payload`, `secret key`, `decoder`, and `hidden signal`;
- treat `coordinate` as technical only when hidden-channel technical cues are
  present;
- still require zero technical public literal hits in any future generation
  route.

## Duplicate-Output Mitigation

The repaired allocation manifest assigns all 3072 rows to
4 shards while enforcing one unique `(prompt_index,
prefix_family_id)` pair per shard. Each shard has 768
rows and 64 rows per selected
coordinate.

This does not prove future generation will have zero duplicate response hashes,
but it removes the deterministic duplicate source caused by evaluating multiple
coordinates with the same prompt/prefix inside a shard.

## Next Allowed Action

Patch the generation wrapper to consume the allocation manifest and patch the
decoder/quality gate to use the reviewed contextual literal policy. Then run
local and remote plan-only validation again. Do not submit Slurm from this
artifact alone.
