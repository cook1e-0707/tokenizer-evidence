# R4 Positive Event-Bank Dev Diagnostic 859277 Review

Timestamp: `2026-05-14T20:05:00Z`

## Verdict

Status: `FAIL_R4_POSITIVE_DEV_DIAGNOSTIC_GATE_NO_RESUBMIT_UNTIL_FAILURE_ANALYSIS`

The Slurm job completed cleanly, and the control arms remain clean, but the
positive channel did not appear. Protected accepts are `0/32` under the primary
`format_scrub=all` decode and also `0/32` under no-scrub decode. The extractor
found zero phrase events in every block, so the keyed decoder had no support to
score.

This is a clean experimental failure, not a Slurm failure.

## Slurm Completion

| task | state | exit | elapsed | node |
| --- | --- | --- | --- | --- |
| `859277_0` | `COMPLETED` | `0:0` | `00:43:32` | `chimera21` |
| `859277_1` | `COMPLETED` | `0:0` | `00:42:58` | `chimera21` |
| `859277_2` | `COMPLETED` | `0:0` | `00:45:03` | `chimera21` |
| `859277_3` | `COMPLETED` | `0:0` | `00:39:12` | `chimera21` |

No traceback, OOM, CUDA error, or wrapper exception was found in the synced
Slurm logs.

## Artifact Completeness

- generated rows: `6144`
- generated rows by condition: protected `2048`, raw `2048`, task-only `2048`
- decode rows per scrub mode: `160`
- duplicate prompt-condition rows: `0`
- duplicate generated response hashes: `0`
- shards complete: `4/4`

## Primary Decode Gate

Primary mode: `format_scrub=all`

| arm | accepts | blocks | observed events max | distinct coords max | forbidden hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| protected | `0` | `32` | `0` | `0` | `103` |
| raw | `0` | `32` | `0` | `0` | `73` |
| task_only | `0` | `32` | `0` | `0` | `85` |
| wrong_key | `0` | `32` | `0` | `0` | `103` |
| wrong_payload | `0` | `32` | `0` | `0` | `103` |

Gate result:

- protected accept gate `>=26/32`: `FAIL` (`0/32`)
- null accept gate: `PASS` (`0/32` for all controls)
- forbidden public surface gate: `FAIL` (`467` total hits)
- phrase-event support gate: `FAIL` (`0` events in all blocks)

No-scrub decode produced the same accept/support pattern.

## Failure Evidence

The generated protected outputs often begin with task-natural action verbs such
as `Create`, `Prepare`, `Keep`, and `Use`, but the frozen phrase-event bank was
much more specific. Exact phrase hits against the frozen bank were `0` for all
conditions.

Top protected output openers:

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

Frozen bank first words are broader/different, but require exact phrases such
as `ask a focused question` or `confirm the main constraint`; no generated
responses matched those phrases exactly.

Forbidden surface hits are dominated by ordinary task words:

```text
coordinate: 439
bucket: 28
```

Examples show `coordinate` usually appears in the normal domain phrase
`volunteer coordination`, while `bucket` appears as an ordinary physical object
in maintenance/cleaning contexts. This is a matcher-semantics problem for the
forbidden-surface gate, but it does not rescue the positive failure because
phrase-event support is still zero.

## Interpretation

The job confirms the wrapper and decoder can run end-to-end on H200, but the
current R4 positive phrase-event bank is not aligned with free-generation
outputs from the selected protected adapter. The protected adapter still
induces visible natural action-verb style, but not the exact frozen phrase
events required by the keyed-correlation decoder.

The null result is clean, but it is uninformative as a positive evidence result
because there is no protected support.

## Artifacts

- aggregate summary:
  `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/aggregate_summary.json`
- combined decode rows:
  `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/combined_decode_rows.csv`
- surface/openers analysis:
  `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/surface_openers_and_exact_hit_analysis.json`
- forbidden examples:
  `results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/forbidden_surface_examples.csv`

