# R4 After-870987 Pre-FAR Standard-Control 871250 Contextual-Window Review

Date: 2026-05-21

## Status

`871250` completed all 160 H200 `%6` array shards. The original aggregate failed only because the first-token decoder's contextual forbidden-surface implementation treated technical cues as global document-level co-occurrences.

The repaired artifact-only contextual-window decoder review passes:

```text
aggregate:
  PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_GENERATION_GATE
aggregate artifact:
  results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_generation_871250_contextual_window_v2_aggregate_20260521/
generation/model/training:
  not rerun
slurm:
  not resubmitted
```

## Original Failure Attribution

The strict aggregate initially failed with 7 forbidden-public-surface counts. Artifact-only attribution found 4 generated rows:

```text
shard_82 raw:
  ordinary "bucket of water" and ordinary "Coordinate with family"
shard_107 protected:
  ordinary "coordinate volunteer schedules" plus distant ordinary scheduling "slots"
shard_108 raw:
  ordinary "coordinate volunteer schedules"
shard_147 raw:
  ordinary "Coordinate with local schools"
```

No row contained hard technical literals such as `fingerprint`, `watermark`, `payload`, `secret key`, `decoder`, or `hidden signal`.

## Repair

The matcher repair keeps the precommitted policy vocabulary unchanged. It changes implementation semantics from whole-response cue co-occurrence to sentence-local/token-window cue matching:

```text
hard literals:
  still fatal anywhere
contextual literals:
  technical only when the literal and cue occur in the same local sentence/window
ordinary uses:
  not counted as technical forbidden surface
```

Unit coverage added:

```text
tests/natural_evidence_v2/test_r4_after_868151_first_token_event_decoder.py
tests/natural_evidence_v2/test_r4_contextual_forbidden_surface_policy_v2.py
```

Validation:

```text
local py_compile: PASS
local pytest selected tests: 15 passed
remote py_compile: PASS
remote pytest selected tests: 15 passed
full 871250 artifact-only policy-hit rescan: 0 technical hits
```

## Passing Aggregate

The contextual-window re-decode/re-aggregate uses the same 871250 generated transcripts and trace-binding artifacts. It does not change codeword thresholds, retries, prompts, seeds, controller config, or generated text.

```text
new shards decoded: 160/160
generated rows: 491,520
unique response hashes: 491,520
global duplicate extra rows: 0
trace binding invalid rows: 0 / 491,520
```

Combined standard-control null package:

```text
raw:           0/256 strict accepts, 0/256 ignoring-quality accepts
task_only:     0/256 strict accepts, 0/256 ignoring-quality accepts
wrong_key:     0/256 strict accepts, 0/256 ignoring-quality accepts
wrong_payload: 0/256 strict accepts, 0/256 ignoring-quality accepts
```

Report-only protected arm in the added 160 blocks:

```text
protected strict accepts: 159/160
protected ignoring-quality accepts: 159/160
```

## Claim Control

This passes the Qwen same-contract first-token event pre-FAR standard-control null expansion. It still does not unlock:

```text
full FAR claim
paper-facing positive claim
text-only phrase decoder success
payload diversity
Llama
same-family null
sanitizer robustness
training
```

Next route should continue the same pre-FAR package with organic-null wrapper planning/preflight.
