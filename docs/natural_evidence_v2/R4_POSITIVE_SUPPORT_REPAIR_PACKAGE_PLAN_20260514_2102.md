# R4 Positive Support-Repair Package Plan

Timestamp: `2026-05-14T21:02:34Z`

## Decision

Current phase:

```text
V2_R4_POSITIVE_ZERO_EVENT_SUPPORT_GAP_AUDIT_RECORDED_REPAIR_PACKAGE_PLANNING
```

The support-gap audit confirms that `859277` failed because the frozen exact
phrase-event bank had zero support in free generation. This plan defines the
artifact-only repair package that must be built before any further compute can
be reviewed.

This is not a Slurm route and does not authorize generation, model scoring,
training, Llama, FAR, sanitizer, payload diversity, or paper-facing claims.

## Audit Findings

The audit output is:

```text
results/natural_evidence_v2/status/r4_positive_zero_event_support_gap_audit_20260514_2102/
```

Key facts:

```text
generated rows = 6144
surface bank rows = 96
protected exact frozen phrase hits = 0
raw exact frozen phrase hits = 0
task_only exact frozen phrase hits = 0
protected loose stem hits = 1
raw loose stem hits = 0
task_only loose stem hits = 0
protected rows with bank-first-word opener = 2032/2048
raw rows with bank-first-word opener = 2042/2048
task_only rows with bank-first-word opener = 2046/2048
```

Interpretation:

```text
The generated outputs contain ordinary action language, but the locked
multi-word phrase events are too exact for free generation. Bank first-word
overlap is high across protected/raw/task-only, so first-word support alone is
not ownership evidence and cannot be used as a positive claim.
```

## Repair Scope

The next repair package must solve support without post-hoc mining:

```text
Allowed:
  independent rule-derived event windows
  dev-only support grammar design
  static fixtures
  toy extractor positives and wrong-key/wrong-payload negatives
  forbidden matcher semantics repair

Forbidden:
  copying 859277 generated phrases into the locked bank
  tuning thresholds on 859277
  relabeling 859277 as positive
  resubmitting 859277 route unchanged
  using exact opener frequency as ownership signal
```

## Required Repair Package Components

1. `r4_positive_support_repair_v2` contract draft.
   - New contract id.
   - Same-contract `a55e` only unless a later payload route is reviewed.
   - `format_scrub=all` remains primary.

2. Independent surface source policy.
   - Surfaces must come from static lexical rules, dev-only prompt policy, or a
     frozen external lexical construction rule.
   - `859277` transcripts may be used only to reject unsafe assumptions, not to
     add surfaces.

3. Extractor v2 plan.
   - Move from exact multi-word phrase-only support to precommitted natural
     event windows.
   - Avoid fixed labels, Step numbering, repeated public templates, and
     coordinate-visible structure.
   - Record support, coordinate coverage, surface family concentration, and
     null separation under `format_scrub=all`.

4. Forbidden matcher semantics policy.
   - Hard-forbid technical public literals.
   - Treat ordinary domain uses separately from technical protocol terms.
   - Do not use matcher repair to rescue old runs.

5. Static validation package.
   - No exposed key material.
   - No forbidden technical literals.
   - No surfaces copied from `859277`.
   - Toy positive fixture accepts.
   - Wrong-key and wrong-payload fixtures reject.
   - Surface-family concentration bounded.
   - Duplicate prompt windows and duplicate selected blocks rejected.

## Future Compute Gate

A future H200 route can be reviewed only after:

```text
support-repair contract exists
surface source policy passes audit
extractor v2 static fixtures pass
forbidden matcher policy passes static review
local tests pass
CURRENT_STATE and gate_status are synchronized
Hermes TG/email notification path passes
allowlist entry is disabled by default
local/remote hash preflight is clean
```

Until then:

```text
slurm_allowed = false
generation_allowed = false
model_scoring_allowed = false
training_allowed = false
paper_claim_allowed = false
```

