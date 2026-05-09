# Actual-prefix candidate scoring plan

This plan replaces the failed static reference-prefix verifier path for
diagnostic natural evidence. It is a plan, not a result and not a paper claim.

## Motivation

The Qwen diagnostic E2E partial run failed because generated responses diverged
from static reference prefixes. A follow-up static-bucket salvage diagnostic did
not recover payload and only raised observed-symbol rate from about 0.163 to
about 0.195. Therefore, reusing old static bucket token sets is not enough.

## Construction order

1. Freeze generated natural transcripts.
2. Enumerate actual generated prefixes with the precommitted keyed selector.
3. Score the reference model top-k next-token distribution at those actual
   prefixes.
4. Apply token surface filters before bucketization.
5. Run counterfactual compatibility on actual-prefix suffix windows.
6. Bucketize after compatibility filtering, not before.
7. Audit density, bucket mass, entropy, reconstructability, and decode capacity.
8. Only after the audit passes, decide whether another Qwen diagnostic eval is
   justified.

## Current selector for the plan artifact

The CPU plan generator enumerates every possible next-token prefix in each
generated response after `min_response_prefix_tokens=1`, ranks offsets using the
audit key, then applies:

- `min_spacing_tokens=12`
- `max_evidence_positions_per_response=4`
- `audit_key_id=K001`

The output is a scoring-input JSONL. It is not a candidate bank, not payload
recovery, and not a training artifact.

## GPU boundary

The current step does not need GPU. GPU becomes necessary only for the next
phase if we score Qwen top-k next-token candidates at the actual generated
prefixes. That future job must remain allowlisted and diagnostic-only.

## Stop conditions

Do not run Llama, 8-way main, full matrix, or more LoRA training before the
actual-prefix scoring plan and selector repair are reviewed. Do not claim
natural-output success until protected recovery and raw/task-only/wrong-key/
wrong-payload rejection are demonstrated under the repaired verifier.
