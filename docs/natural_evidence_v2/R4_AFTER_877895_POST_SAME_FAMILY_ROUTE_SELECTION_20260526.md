# R4 After-877895 Post Same-Family Route Selection - 2026-05-26

Status: `ROUTE_DECISION_R4_AFTER_877895_POST_SAME_FAMILY_ROUTE_SELECTION_NO_SUBMIT`

## Reviewed Inputs

- R4 locked-scale Qwen first-token event positive package has passed under the
  provider-side trace-bound decoder route.
- R4 pre-FAR standard controls and organic raw null package have passed.
- Full v8 same-family raw-null package `877895` has passed across
  Qwen2.5-3B/7B/14B raw-only controls:
  - 64/64 shards complete per model;
  - 65,536 generated rows per model;
  - 196,608 generated rows total;
  - raw accepts `0/64` per model;
  - raw accepts ignoring quality `0/64` per model;
  - duplicate extra rows `0` per model;
  - technical forbidden public surface count `0`;
  - ambiguous forbidden surface count `0`;
  - trace binding invalid rows `0`.

## Decision

The next active route should be canonical R4 second-family/Llama migration
planning, starting artifact-only. This is not an immediate Slurm submission.

The reason is that the current evidence package is now strong within Qwen:
positive recovery, standard nulls, organic nulls, and same-family raw nulls are
all reviewed. The largest remaining scientific gap is whether the provider-side
first-token event channel can transfer to a second tokenizer/model family. That
gap must be addressed before any cross-family claim.

## Allowed Now

```text
artifact-only R4 second-family/Llama migration planning
inventory existing Llama artifacts and mark out-of-band artifacts noncanonical
design tokenizer-native first-token event row bank for the second family
prepare tokenizer-only preflight route validation, disabled by default
prepare H200 wrapper/hash-preflight plan, disabled by default
update state and gate_status for 877895 pass and this route decision
```

## Not Allowed Yet

```text
Slurm submission
allowlist enablement
Llama model scoring
Llama generation
training
sanitizer benchmark
FAR aggregation
payload-diversity claim
paper-facing positive claim
text-only phrase-decoder success claim
```

## Required Preconditions Before Any Second-Family Slurm

1. A route-specific R4 Llama/second-family migration config exists and validates
   locally.
2. Existing Llama artifacts are inventoried and any out-of-band artifacts are
   explicitly noncanonical.
3. The second-family row bank is tokenizer-native and does not reuse Qwen token
   boundaries silently.
4. Actual second-family tokenizer boundary preflight is planned as tokenizer-only:
   no model forward, no scoring, no generation, no training.
5. Local and remote hashes agree for config, wrapper, row bank, tokenizer review
   inputs, and allowlist.
6. Allowlist safety passes with zero enabled entries before route review and
   exactly one enabled entry only during a reviewed single-job submission
   preflight.
7. H200 policy remains in force unless superseded by a later recorded route
   decision.

## Deferred Routes

- Sanitizer benchmarking remains deferred until a second-family route decision
  or an explicit boundary-only sanitizer route is recorded.
- FAR aggregation remains deferred until the exact FAR scope is defined. The
  current package is still pre-FAR/null evidence, not a full FAR claim.
- Payload diversity remains deferred; current successful route is same-contract
  `a55e`.
- Paper-facing claims remain deferred until the claim surface is rewritten around
  provider-side first-token event evidence and the required boundary experiments
  pass.

## Next Allowed Action

Prepare the artifact-only R4 second-family/Llama migration planning package and
state synchronization. Do not submit Slurm in this route-selection step.
