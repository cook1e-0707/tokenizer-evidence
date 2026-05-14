# R4 Positive Event-Bank Dev Diagnostic 859277 Failure Analysis

Timestamp: `2026-05-14T20:18:00Z`

## Scope

This is an artifact-only analysis of job `859277`. It does not relabel the run
as positive, does not change thresholds, and does not authorize a resubmission
of the same route. The reviewed result remains:

```text
FAIL_R4_POSITIVE_DEV_DIAGNOSTIC_GATE_NO_RESUBMIT_UNTIL_FAILURE_ANALYSIS
```

## Established Facts

The H200 Slurm array completed cleanly:

```text
859277_0 COMPLETED 0:0 00:43:32 chimera21
859277_1 COMPLETED 0:0 00:42:58 chimera21
859277_2 COMPLETED 0:0 00:45:03 chimera21
859277_3 COMPLETED 0:0 00:39:12 chimera21
```

The wrapper produced the expected artifact shape:

```text
generated_rows = 6144
protected rows = 2048
raw rows = 2048
task_only rows = 2048
decode_rows per scrub mode = 160
duplicate prompt-condition rows = 0
duplicate generated response hashes = 0
all shards complete = true
```

The primary `format_scrub=all` decode failed the positive gate:

```text
protected accepts = 0/32
raw accepts = 0/32
task_only accepts = 0/32
wrong_key accepts = 0/32
wrong_payload accepts = 0/32
protected observed events max = 0
protected distinct coordinates max = 0
```

No-scrub decode had the same support pattern:

```text
protected accepts = 0/32
protected observed events max = 0
protected distinct coordinates max = 0
```

## Primary Failure Mode

The route failed because the frozen phrase-event bank had zero exact support in
the generated text. This is not a decoder arithmetic failure: the decoder had no
events to score.

The frozen bank uses specific phrase events such as:

```text
ask a focused question
confirm the main constraint
check the current status
review the latest details
```

The generated protected outputs instead contain many natural but nonmatching
action openings:

```text
keep 1706
use 1208
prepare 1072
create 1041
when 907
make 790
have 755
encourage 747
plan 688
```

The exact surface hit count was `0` for protected, raw, and task-only outputs.
That means the event-bank surfaces were not merely weak; they were absent under
the frozen extractor contract.

## Control Interpretation

The null arms were clean:

```text
raw accepts = 0/32
task_only accepts = 0/32
wrong_key accepts = 0/32
wrong_payload accepts = 0/32
```

This does not establish a positive result because protected support was also
zero. The result only shows that the current detector did not false-accept when
no phrase events were found.

## Forbidden-Surface Matcher Finding

The primary scrubbed decode counted `467` forbidden public surface hits:

```text
coordinate = 439
bucket = 28
```

The examples indicate many hits are ordinary language, especially:

```text
volunteer coordination
physical bucket / cleaning bucket contexts
```

This is a matcher-semantics issue. The current matcher is too broad for ordinary
domain words, but it is secondary for `859277`: even if these false positives
were removed, the positive gate would still fail because phrase-event support is
zero.

## Evidence Against Unchanged Resubmission

An unchanged rerun would exercise the same mismatch:

```text
prompt policy -> useful natural answers
protected model -> natural action-verb openings
frozen phrase bank -> exact multi-word phrases
extractor -> exact phrase matches only
decoder -> keyed correlation over extracted events
```

The generated text shows the model is producing task-relevant language, but not
the precommitted event phrases. More samples of the same route would likely
increase the count of nonmatching action openings, not create support for the
locked phrase bank.

## Current Blocker

```text
BLOCK_R4_POSITIVE_859277_ZERO_EVENT_SUPPORT_FAILURE_ANALYSIS_RECORDED
```

The unresolved technical problem is a free-generation support mismatch:

```text
teacher-forced / static phrase-bank design did not transfer into exact emitted
phrase events under free generation.
```

This is distinct from earlier tokenizer-boundary failures and distinct from a
Slurm/control-plane failure.

## Claim Status

Allowed internal statement:

```text
R4 positive event-bank wrapper completed on H200, null arms were clean, but the
positive channel failed because generated outputs contained zero frozen phrase
events.
```

Still not allowed:

```text
positive natural evidence claim
paper-facing positive result
payload diversity claim
Llama or cross-family claim
FAR claim
sanitizer robustness claim
unchanged 859277-route resubmission
```

