# R4 After-877895 Second-Family Llama Small Diagnostic Plan - 2026-05-26

Status: `ROUTE_PLAN_R4_AFTER_877895_SECOND_FAMILY_LLAMA_SMALL_DIAGNOSTIC_ARTIFACT_ONLY_NO_SUBMIT`

## Purpose

Plan the first Llama/second-family generation diagnostic for the current R4
provider-side first-token event route. This is not a paper-facing claim and not
the old Step-label Llama route.

## Preconditions Already Met

- Full Qwen same-family raw-null v8 package `877895` passed.
- Llama migration inventory completed; old WP5/WP6 Llama wrappers are
  noncanonical for R4.
- Llama candidate row bank was built artifact-only with 65,536 tokenizer-neutral
  rows.
- Llama tokenizer-only preflight job `879100` passed review:
  - checked rows: 65,536;
  - failed rows: 0;
  - empty target first-token id rows: 0;
  - empty other first-token id rows: 0;
  - target/other overlap rows: 0.

## Blocking Code/Route Work Before Generation

1. Add `PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_TOKENIZER_PREFLIGHT_879100_REVIEWED`
   to the generation wrapper's allowed tokenizer-review statuses.
2. Define a provider-side Llama generation route that does not depend on the old
   Step-label task-only adapter path.
3. Decide the initial Llama control set. Recommended minimum:
   `protected_controller`, `raw_no_controller`, `wrong_key_replay`, and
   `wrong_payload_replay`; task-only should remain absent unless a reviewed
   Llama task-only baseline is created.
4. Run plan-only wrapper validation over a small subset before any H200
   generation submission.
5. Preserve trace binding, duplicate gates, contextual forbidden policy, and
   no paper-claim controls.

## Recommended Initial Scale

```text
blocks: 4
rows per block: same R4 first-token event row policy
model: meta-llama/Meta-Llama-3.1-8B-Instruct
tokenizer review: 879100
controller: same reviewed controller config used by Qwen R4 route
claim scope: second-family proof-of-life diagnostic only
```

## Diagnostic Gate

```text
protected strict accepts: target 4/4
protected accepts ignoring quality: target 4/4
raw accepts: 0/4
wrong-key replay accepts: 0/4
wrong-payload replay accepts: 0/4
technical forbidden public surface: 0
ambiguous forbidden surface: 0
duplicate response hash extra rows: 0
trace binding validity: 100%
```

## Not Allowed by This Plan

```text
Slurm submission
allowlist enablement
Llama generation
Llama model scoring
training
sanitizer benchmark
FAR aggregation
payload diversity
paper-facing positive claim
cross-family success claim
```

## Next Allowed Action

Implement artifact-only wrapper/code review for this small diagnostic route.
Do not submit Slurm until route validation, local/remote hash preflight, and
allowlist safety pass.
