# Hermes/Codex State Reconciliation 2026-05-11

## Decision

The canonical project state remains:

```text
V2_WP6_R2_OPTION_B_SCALE_GATE_852426_REVIEWED_PASS_HOLD_FOR_NEXT_ROUTE
```

WP6-R2 Option B job `852426` is the latest formally reviewed route result.
It passed the precommitted Qwen robust-block scale gate, and the current safe
action remains to stop until a new route is explicitly recorded.

No new formal route is adopted by this reconciliation.

## Canonical Reviewed Result

Reviewed artifact:

```text
docs/natural_evidence_v2/WP6_R2_OPTION_B_SCALE_EVAL_852426_REVIEW.md
results/natural_evidence_v2/status/wp6_r2_option_b_scale_eval_852426/
```

Key reviewed facts:

```text
job_id = 852426
partition = DGXA100
node = chimera12
state = COMPLETED
exit_code = 0:0
protected accepts @64 = 7/8
raw accepts @64 = 0/8
task-only accepts @64 = 0/8
wrong-key accepts @64 = 0/8
wrong-payload accepts @64 = 0/8
min accepted-block support = 26
min accepted-block majority margin = 5
forbidden public surface count = 0
```

This is a Qwen-only controlled-natural micro-slot scale diagnostic. It is not a
full FAR result, not a Llama result, not a same-family null, not a sanitizer
benchmark, and not a paper-facing positive claim.

## Conflict Observed

Read-only Chimera inspection on 2026-05-11 found Slurm history after `852426`
that is not represented in the canonical Hermes/Codex state:

```text
852810 llama-bank DGXA100 COMPLETED 0:0
852811 llama-v2-* DGXA100 COMPLETED 0:0
852844 llama-v2-* DGXA100 COMPLETED 0:0
852853 llama-v2-* DGXA100 COMPLETED 0:0
852881 llama-v2-* DGXA100 COMPLETED 0:0
```

Remote out-of-band artifact root:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/
```

The final inspected Llama WP5 teacher-forced summary under job `852881`
reported a local teacher-forced gate pass, but these Llama jobs were not
recorded as an allowed route in the canonical state and were not synced or
reviewed by the formal Hermes/Codex progression.

Local worktree inspection also found untracked or noncanonical artifacts/scripts
for FAR aggregation, Llama migration, and sanitizer work. These are not adopted
as formal project state by this reconciliation.

## Resolution

The out-of-band Llama, FAR, and sanitizer artifacts are quarantined as
noncanonical until a separate explicit route is recorded and reviewed.

Rules after this reconciliation:

- Do not treat jobs `852810`, `852811`, `852844`, `852853`, or `852881` as
  formal project progress.
- Do not use the out-of-band Llama artifacts to claim cross-family generality.
- Do not use the untracked FAR summary to claim full FAR or paper-facing FAR.
- Do not run or submit Llama, same-family null, sanitizer, FAR aggregation, or
  additional WP6 jobs from the current hold state.
- Any future adoption of these artifacts requires a provenance review,
  explicit route decision, allowlist review, and a fresh Hermes/Codex state
  update.

## Allowlist Sync

The CPU allowlist entry `build_llama_v2_bucket_bank` was disabled because the
current canonical state does not permit a Llama route. Llama WP5, Llama WP6,
and sanitizer GPU entries remain disabled.

## Current Next Allowed Action

```text
Stop until a new route is explicitly recorded. A future route may be a
Qwen-result interpretation package or a separately approved Llama/FAR/sanitizer
route, but no such route is active after this reconciliation.
```

## Forbidden Claims

- natural-output success beyond the reviewed Qwen WP6-R2 diagnostic scope
- full FAR
- cross-family generality
- sanitizer robustness
- stealth guarantee
- superiority over Scalable/Perinucleus
- paper-facing positive claim
