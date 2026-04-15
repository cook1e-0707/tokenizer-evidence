# Manual Decisions For `real_pilot_catalog` / `gpt2` / `v1`

## Scope

This document records the manual decisions taken after reviewing:

- the failed strict freeze audit for `gpt2`
- the remediation table
- the remediation markdown review

This revision is for a minimal strict-pass pilot only.

The goal is not to maximize coverage.
The goal is to produce a small catalog that is honest about what survives under `gpt2`, does not silently distort semantics, and can pass a strict freeze gate.

## Decision policy for this revision

- `TONE` -> `drop_field`
- `REGION` -> `drop_field`
- `SECTION` -> keep surviving members only
- `TOPIC` -> keep surviving members only

Rules applied in this revision:

- If a field has empty buckets after filtering, drop the field for the pilot.
- If a field remains non-empty across all buckets and the surviving members still form a usable field without obvious semantic collapse, keep surviving members only.
- Do not do automatic regroup in this revision.
- Do not mine new candidate members in this revision.
- Do not silently rewrite source semantics just to preserve bucket size symmetry.

Explicit constraint for this revision:

- no automatic regroup
- no new candidate mining

## Field decisions

### `TONE`

Decision: `drop_field`

Reason:

- `TONE` was blocked in the failed freeze review.
- Bucket `0` became empty after filtering.
- A pilot catalog cannot keep a field that already fails the strict freeze structure check.
- Replacing members or regrouping would require a new manual semantic pass, which is out of scope for this revision.

### `REGION`

Decision: `drop_field`

Reason:

- `REGION` was blocked in the failed freeze review.
- Buckets `1`, `2`, and `3` became empty after filtering.
- The field is too degraded for a minimal strict-pass pilot.
- Any rescue path would require manual replacement or a redesigned field, not a small conservative edit.

### `SECTION`

Decision: keep surviving members only

Surviving members by bucket:

- Bucket `0`: `news`
- Bucket `1`: `report`
- Bucket `2`: `guide`
- Bucket `3`: `update`, `review`

Reason:

- All buckets remain non-empty after filtering.
- The surviving labels still behave like plausible section/category carriers.
- The field is reduced, but not obviously semantically collapsed.
- Uneven bucket cardinality is acceptable for this pilot revision; strict correctness matters more than symmetry.

Non-decision for this revision:

- do not regroup `review` or `update`
- do not search for substitute members for buckets `0`-`2`

### `TOPIC`

Decision: keep surviving members only

Surviving members by bucket:

- Bucket `0`: `market`
- Bucket `1`: `travel`
- Bucket `2`: `health`
- Bucket `3`: `science`, `climate`

Reason:

- All buckets remain non-empty after filtering.
- The surviving labels still form a usable topic field for a pilot.
- The field is reduced, but not obviously broken for a strict-pass integration test.
- As with `SECTION`, uneven bucket sizes are acceptable in this revision.

Non-decision for this revision:

- do not regroup `science` / `climate`
- do not mine new one-token topical members yet

## Pilot objective

The immediate pilot objective is:

- produce a minimal, truthful, strict-pass catalog for `gpt2`
- keep only fields that remain structurally valid after filtering
- avoid any automatic semantic surgery
- keep provenance and decisions explicit

This means the pilot is allowed to be smaller than the original source catalog.
It is not allowed to pretend that dropped or degraded fields are still valid.

## Deferred work

The following are explicitly deferred to a later revision:

- automatic regroup
- new candidate mining
- replacement-member search
- broader semantic balancing across buckets
- expanding the pilot back toward the original field set
