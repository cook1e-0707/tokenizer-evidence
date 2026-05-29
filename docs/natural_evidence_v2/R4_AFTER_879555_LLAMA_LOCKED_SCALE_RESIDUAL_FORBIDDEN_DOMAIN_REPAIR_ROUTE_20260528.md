# R4 After-879555 Llama Locked-Scale Residual Forbidden Domain Repair Route

Status: `ROUTE_RECORDED_ARTIFACT_ONLY_NO_SUBMIT`

## Decision

Job `879555` is not adopted as a Llama locked-scale pass. It completed cleanly
and recovered the protected first-token event codeword at locked scale, but the
strict quality gate failed:

```text
protected strict accepts: 96/96
protected accepts ignoring quality: 96/96
raw accepts: 0/96
task-only accepts: 0/96
wrong-key accepts: 0/96
wrong-payload accepts: 0/96
global duplicate extra rows: 0
trace binding invalid rows: 0/196608
technical forbidden public surface count: 1
```

The failing precommitted hit is a raw-arm `watermark` mention in the
`document scanning routine` domain. A broader diagnostic scan found that this
domain naturally elicits ordinary document-processing terms such as
`fingerprints` and `watermark`; those broader counts are diagnostic only and do
not re-score `879555`.

## Repair Choice

Use a prompt/domain repair, not a gate relaxation.

The next Llama locked-scale candidate must remove or replace
`document scanning routine` prompts from the locked-scale allocation before any
new Slurm generation submission. The hard public-literal gate remains unchanged:

```text
technical_forbidden_public_surface_count == 0
global_duplicate_response_hash_extra_rows == 0
trace_binding_validity == 100%
protected strict accepts >= 80/96
protected accepts ignoring quality >= 85/96
raw/task-only/wrong-key/wrong-payload accepts == 0/96
```

## Required Artifact-Only Work

1. Build a repaired Llama locked-scale row-bank plan that excludes
   `document scanning routine` and any replacement prompts with high literal
   risk.
2. Preserve the same locked-scale shape:

```text
shards: 96
rows per shard: 1024
generated rows expected after protected/raw generation: 196608
same contract: a55e
payload diversity tested: false
model: meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Validate the repaired row bank:

```text
selected prompt count == 6144
duplicate prompt/prefix pair == 0
document scanning prompts == 0
static hard literal risk prompts == 0
rows per shard == 1024
shards == 96
```

4. Rerun actual Llama tokenizer boundary preflight for the repaired row bank.
5. Only after tokenizer preflight passes and route hashes are reviewed may a
   single H200 locked-scale generation job be submitted.

## Claim Control

This route does not reclassify `879555`, does not make a Llama locked-scale
claim, and does not unlock paper-facing claims, FAR, sanitizer, payload
diversity, or text-only phrase-decoder success.
