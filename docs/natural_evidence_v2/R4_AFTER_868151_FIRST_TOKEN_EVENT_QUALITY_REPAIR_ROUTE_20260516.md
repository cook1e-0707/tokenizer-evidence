# R4 After-868151 First-Token Event Quality Repair Route

Date: 2026-05-16

## Decision

The first-token event decoder replay shows recoverable event-level signal only
when quality gates are ignored. The next route is therefore quality repair, not
another Slurm generation run.

This route is artifact-only until wrapper/preflight records prove that future
generation will store token-id event traces and reduce public-literal /
duplicate-output quality failures.

## Evidence

The implemented event decoder replay:

```text
decoder replay:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_decoder_replay_20260516/
protected accepts with quality gates:
  0/4
protected accepts ignoring quality:
  4/4
null accepts ignoring quality:
  0/4 each
```

The quality audit:

```text
quality audit:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_audit_20260516/
coordinate literal hits:
  14
likely ordinary domain-sense coordinate hits:
  10
within-condition duplicate response hash count:
  2803
protected duplicate response hash count:
  755
```

The `coordinate` literal is entangled with ordinary volunteer-coordination
prompt semantics, not just hidden-channel terminology. The deterministic greedy
row-cylinder generation also creates many duplicate response hashes across
repeated prompt/prefix patterns.

## Required Repairs

Before any new generation Slurm route:

```text
1. Event-trace wrapper:
   - store first_generated_token_id
   - store first_generated_token_text
   - store target_first_token_ids and other_first_token_ids
   - store event_side and event_trace

2. Public literal policy:
   - either exclude coordination-domain prompts from this diagnostic route
   - or precommit a contextual technical-literal matcher that distinguishes
     ordinary domain use from hidden-channel terminology
   - the final route gate must still require zero technical public literals

3. Duplicate-output policy:
   - add precommitted row/window allocation or decoding diversity controls
   - record duplicate generated-output hash checks
   - refuse duplicate decode-row hashes

4. Wrapper review:
   - local plan-only pass
   - remote plan-only pass
   - zero-enabled allowlist safety before enablement
   - exactly one allowlist entry if a future Slurm route is submitted
```

## Not A Positive Claim

This route does not claim natural-output success. It only records the conditions
that must be met before another first-token event generation diagnostic can be
submitted.
