# Formal Natural-Output Protocol

This note records the formal object for `natural_evidence_v1`. It replaces the
old `y_org || y_car` carrier-block framing for the main line.

## Context-Conditioned Buckets

For a prompt `x` and generated prefix `h_t = (x, y_<t)`, the bucket policy is a
deterministic prefix-only function:

```text
B_K(h_t, b) subset V
```

The function may depend on the current prefix, audit key, tokenizer, fixed
reference model, and committed policy config. It must not depend on future
tokens.

The bucket event is:

```text
E_{K,b}(h_t) = union_{v in B_K(h_t,b)} [v]
```

Conditional on prefix `h_t`, this is first-token measurable because it is a
token set in the current tokenizer vocabulary. Its model probability is exactly:

```text
P_theta(E_{K,b}(h_t) | h_t) =
  sum_{v in B_K(h_t,b)} P_theta(v | h_t)
```

This is the reason bucket-mass supervision remains a next-token objective in
natural outputs.

## What This Does Not Prove

The measurability statement does not prove:

- enough eligible positions occur during free generation,
- candidate tokens are semantically interchangeable,
- training can recover a payload,
- utility or naturalness is preserved,
- non-keyed or keyed sanitizers cannot remove evidence,
- false accepts are low under raw or same-family nulls.

Those are empirical gates.

## Transcript-Level Decoding

The verifier operates on committed transcripts, not on a contiguous evidence
block. For each observed output, it scans prefixes left-to-right, applies the
committed eligible-position selector, reconstructs buckets for the observed
prefix, and records a bucket observation if the generated token belongs to one
bucket.

Missing or unbucketed positions are erasures. Mismatched buckets are symbol
errors. Payload recovery is performed over the accumulated transcript-level
observation stream with the committed mixed-radix/ECC decoder.

## Required Controls

The paper-facing protocol must compare the prefix-only construction against:

- static reference-prefix lookup,
- non-prefix or future-dependent events,
- multi-token phrase targets or semantic predicates when practical,
- task-only and raw nulls.

These controls are needed to show that first-token measurability is not merely
notation but the tractable training and verification boundary.
