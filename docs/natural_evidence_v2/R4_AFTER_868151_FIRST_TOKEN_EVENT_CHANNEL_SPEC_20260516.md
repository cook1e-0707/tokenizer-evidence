# R4 After-868151 First-Token Event-Channel Spec

Date: 2026-05-16

## Scope

This document formalizes the artifact-only next step after failed generation
job `868151`. It does not reclassify `868151` as a pass.

The new route studies a precommitted first-token / lemma event channel, because
the failed full-phrase decoder observed zero locked phrase events while a
posthoc oracle found protected first-token event recovery.

## Source Facts

```text
source job:
  868151
source review:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_review/controller_generation_review_summary.json
source failure analysis:
  results/natural_evidence_v2/status/r4_after_868016_controller_generation_868151_failure_analysis/failure_analysis_summary.json
posthoc oracle:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_oracle_20260516_v2/first_token_oracle_summary.json
```

The posthoc oracle is route-planning evidence only:

```text
protected accepts ignoring forbidden: 4/4
raw/task_only/wrong_key/wrong_payload accepts ignoring forbidden: 0/4 each
not precommitted: true
ignores forbidden-surface gate: true
paper positive claim allowed: false
```

## Event Definition

For a generated row-cylinder with model-visible assistant prefix `h` and
response text `y`, define the continuation:

```text
c = y after exact prefix h
```

If `y` does not start with `h`, the row has no event.

Normalize the first lexical event from `c`:

```text
1. strip leading whitespace and punctuation;
2. take the first ASCII alphabetic word;
3. lowercase;
4. optionally map through a precommitted lemma alias table.
```

The initial route uses identity lemma aliases only. Any non-identity lemma table
must be precommitted before generation and may not be derived from locked
outputs.

For each row, the target event set and opposite event set are derived only from
the reviewed tokenizer-boundary row contract:

```text
target_event_set = first lexical events of target_tokenizer_scored_surface_texts
other_event_set  = first lexical events of other_tokenizer_scored_surface_texts
```

Hard validation:

```text
target_event_set non-empty
other_event_set non-empty
target_event_set ∩ other_event_set = empty
event extraction does not use generated-output phrase mining
event extraction does not use Step labels, line indices, or slot labels
```

## Decoder

For each block and arm, collect votes by selected coordinate:

```text
if first_event in target_event_set: vote = committed target bit for coordinate
if first_event in other_event_set:  vote = opposite bit for coordinate
else: erasure
```

Pairs and checksum use the already reviewed coordinate-pivot codebook:

```text
codebook:
  results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516/codebook.json
decoder_spec:
  results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516/decoder_spec.json
```

Accept only if:

```text
all required bit pairs are complete;
pair majority has no tie;
checksum is valid;
decoded payload matches the committed payload;
wrong-key and wrong-payload controls reject;
forbidden public technical surface count = 0;
duplicate response hash checks pass.
```

The primary reported generation gate must include `format_scrub=all`. If the
event extractor changes the scrub semantics, the scrub transform must be
precommitted before generation.

## Not Allowed

This spec does not allow:

```text
Slurm submission
generation rerun
teacher-forced scoring rerun
training
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
```

These actions can proceed later only after their route-specific preconditions
are recorded in `CURRENT_STATE.md`.
