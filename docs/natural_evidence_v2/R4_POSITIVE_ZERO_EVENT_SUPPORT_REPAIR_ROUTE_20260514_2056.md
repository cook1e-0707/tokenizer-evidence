# R4 Positive Zero-Event Support Repair Route

Timestamp: `2026-05-14T20:56:17Z`

## Decision

Route status:

```text
V2_R4_POSITIVE_ZERO_EVENT_SUPPORT_REPAIR_ROUTE_RECORDED_ARTIFACT_ONLY_NO_SLURM
```

This route records the required repair/pivot decision after the reviewed
`859277` positive event-bank diagnostic failed with zero phrase-event support.
It does not authorize Slurm, generation, training, Llama, FAR, sanitizer,
payload diversity, or any paper-facing positive claim.

## Triggering Evidence

The reviewed `859277` artifacts establish:

```text
Slurm status: clean H200/pomplun completion
generated rows: 6144
primary format_scrub=all protected accepts: 0/32
no-scrub protected accepts: 0/32
raw/task_only/wrong_key/wrong_payload accepts: 0/32
extracted frozen phrase events: 0 in every block
distinct coordinates: 0 in every block
forbidden surface hits: coordinate=439, bucket=28
```

The run is a method failure, not a cluster or wrapper failure. The decoder did
not fail because of thresholds; it had no frozen phrase events to score.

## Reuse Policy

`859277` may be used only for:

```text
failure taxonomy
prompt-policy diagnosis
surface-bank coverage diagnosis
forbidden matcher semantics diagnosis
wrapper/provenance audit
```

`859277` must not be used for:

```text
post-hoc phrase mining into the next locked bank
threshold tuning
key/payload remapping
decoder relaxation to reclassify 859277 as passing
claiming a positive natural-evidence result
unchanged route resubmission
```

## Repair Hypothesis

The current frozen event bank was too exact for free generation. Protected
outputs often used natural task-action language such as `keep`, `use`,
`prepare`, `create`, and `plan`, but none matched the frozen multi-word phrase
events exactly.

The next repair must prove a support contract before compute:

```text
prompt policy -> naturally elicits carrier opportunities
surface policy -> matches carrier opportunities after format scrub
extractor policy -> records support without visible structural labels
decoder policy -> uses precommitted keyed correlation, not post-hoc transcript repair
```

## Immediate Artifact-Only Tasks

1. Build a support-gap audit for `859277` outputs.
   - Measure exact phrase, lemma phrase, action-opener, and natural event-window
     coverage.
   - Report protected/raw/task-only separately.
   - Treat all generated text as diagnostic only, not as a source for new locked
     surfaces.

2. Review forbidden-surface matcher semantics.
   - Separate technical reserved terms from ordinary domain language.
   - Keep the future locked gate conservative for technical literals.
   - Do not use matcher repair to rescue `859277`.

3. Design a new support contract.
   - Avoid fixed Step labels, coordinate-visible structure, and repeated global
     templates.
   - Require phrase/event surfaces to come from independent rules or dev-only
     precommit sources, not from locked generated transcripts.
   - Require primary decode under `format_scrub=all`.

4. Static-validate the new repair package.
   - No exposed key material.
   - No forbidden technical literals.
   - No post-hoc surfaces from `859277`.
   - No duplicate prompt windows or duplicate selected blocks.
   - Surface family concentration bounded.
   - Extractor has toy positive and wrong-key/wrong-payload negative fixtures.

## Future Compute Eligibility

A future H200 route may be reviewed only after artifact-only support repair
passes static validation and records:

```text
new contract id
surface-bank source policy
prompt-bank source policy
extractor version
decoder version
format-scrub modes
null arms
query/block budget
allowlist entry
Hermes notification path
local/remote hash preflight
```

Any future compute must still use:

```text
partition/QOS: pomplun
account: cs_yinxin.wan
GPU: h200
time limit: max available policy limit
exactly one reviewed allowlist entry
disable allowlist immediately after sbatch
```

## Current Allowed Action

```text
artifact-only support-gap audit and repair-package planning only
```

## Current Not Unlocked

These actions are conditionally allowed only after their prerequisite gates pass;
they are not unlocked by this route record:

```text
Slurm submission
free generation
model scoring
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity
paper-facing positive claim
```

