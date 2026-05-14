# R4 Positive Phrase Event Extractor Static Review

Timestamp: `2026-05-14T16:18:00Z`

## Scope

Implemented the artifact-only phrase event extractor for the R4 precommitted
positive event bank.

Files:

```text
scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py
tests/natural_evidence_v2/test_r4_positive_phrase_event_extractor.py
```

## Behavior

The extractor:

- loads the frozen `surface_bank.json`;
- refuses structural-marker surfaces;
- supports `scrub_mode=all` and `scrub_mode=none`;
- strips bullets, numbering, and simple public action labels before matching
  when `scrub_mode=all`;
- normalizes lowercase, punctuation, and whitespace;
- uses word-boundary phrase matching;
- emits only `normalized_phrase_event` rows with `surface_id`,
  `surface_family`, `canonical_phrase`, `weight`, span offsets, and scrub mode.

It does not run a model, score logits, generate outputs, submit Slurm, or
enable any allowlist entry.

## Validation

Commands:

```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_phrase_event_extractor.py tests/natural_evidence_v2/test_r4_keyed_correlation_decoder.py -q
uv run python -m py_compile scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py
uv run python scripts/natural_evidence_v2/extract_r4_positive_phrase_events.py --text '1. Ask a focused question. Next action: use a calm tone and compare with a known case.' --scrub-mode all
```

Results:

```text
focused pytest: 10 passed
py_compile: pass
smoke extraction: emitted 3 normalized_phrase_event rows
```

## Current Boundary

This extractor is necessary but not sufficient for a Slurm route. The next
artifact-only task is generation/decode wrapper plan-only implementation and
review for the R4 positive dev diagnostic route.
