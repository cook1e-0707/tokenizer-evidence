# R4 Same-Family Raw-Null V2 Forbidden-Literal Repair Route

Date: 2026-05-23

Status: `ARTIFACT_ONLY_REPAIR_PLANNING_NO_SUBMIT`

## Decision

Job `875168` is not adopted as a passed same-family raw-null package. It
completed all 192 H200 shards and showed clean null separation, but failed the
strict contextual forbidden-surface gate:

```text
raw accepts: 0/64 for Qwen2.5-3B, 0/64 for Qwen2.5-7B, 0/64 for Qwen2.5-14B
raw accepts ignoring quality: 0/64 for each model
generated rows: 196608
duplicate extra rows: 0
trace binding: 196608 checked, 0 invalid
technical forbidden hits: 254
ambiguous forbidden hits: 53
ordinary-domain literals: 801
dominant technical hits: fingerprint=234, watermark=19, bucket=1
```

The failure is a prompt-domain / hard-forbidden-literal collision. Archive,
document, photo, library, and hardware-store cleanup prompts can naturally
elicit words that the current policy hard-forbids. The hard policy remains
unchanged for this route; `875168` must not be retroactively rescued.

## V2 Repair Scope

Build a new same-family raw-null v2 row bank from the existing organic-null row
bank while excluding:

- prompt ids that produced forbidden collisions in `875168`;
- prompt domains implicated by the collision review;
- static prompt/candidate rows with technical or ambiguous forbidden literals.

The v2 row bank preserves:

```text
conditions: raw only
models planned later: Qwen2.5-3B / 7B / 14B Instruct raw
shards per model: 64
prompts per shard: 64
rows per shard: 1024
selected rows per model: 65536
array policy if later submitted: 0-191%6 on H200 pomplun
contract: a55e
payload diversity tested: false
```

## Current Allowed Actions

Allowed now:

```text
artifact-only v2 row-bank construction
artifact-only v2 row-bank validation
lexical/static forbidden preflight
route documentation and Hermes/Codex state synchronization
```

Not allowed by this document alone:

```text
Slurm submission
generation
training
Llama
sanitizer
FAR aggregation
payload diversity
paper-facing claims
same-family raw-null pass claim
```

## Next Gate

Before any same-family raw-null rerun, the v2 row bank must pass validation:

```text
rows = 65536
shards = 64
rows per shard = 1024
selected prompts = 4096
each prompt has 16 rows
no reused 875168 collision prompt ids
no denied prompt-domain rows
static technical forbidden hits = 0
static ambiguous forbidden hits = 0
raw-only generation conditions
generation/scoring/training/slurm flags are false
```

After this pass, the next route may prepare tokenizer preflight and reviewed
single H200 generation submission artifacts. That later route must still use
zero-enabled allowlist safety and exactly-one-entry submission control.
