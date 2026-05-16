# R4 After-864832 Reliability Dev Generation Plan Failure

## Decision

Do not submit the reliability-codebook dev generation route yet.

The route scope validator passed, but the wrapper plan-only toy decode failed.
This exposed a protocol blocker in the two-sided surface bank: normalized phrase
surfaces are reused across many selected coordinates, so phrase-only decoding
cannot identify the reliability-codebook coordinate.

## Evidence

Route validation:

```text
status: PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
allowlist entry: v2_r4_after_864832_reliability_dev_generation_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_reliability_dev_generation_h200.sbatch
slurm submitted: false
generation started: false
```

Wrapper plan-only failed:

```text
toy protected reliability decode did not accept
protected accepts: 0/1
wrong-key accepts: 0/1
wrong-payload accepts: 0/1
matched_surface_count: 256
selected_surface_count: 120
tie pairs: bit 0 and bit 1
```

Surface uniqueness audit:

```text
status: FAIL_R4_RELIABILITY_SURFACE_UNIQUENESS_SELECTED_COORDINATES_AMBIGUOUS
surface entries: 256
selected coordinates: 16
unique normalized phrases: 21
phrases ambiguous for selected coordinates: 21
phrases with opposite polarity for selected coordinates: 0
```

Interpretation:

```text
The current surface bank does not create coordinate-identifiable evidence.
It creates phrase-family evidence that repeats across coordinates. The
reliability codebook is pair-coordinate based, so the decoder cannot safely map
observed phrases back to selected coordinate pairs without extra structure.
```

## Control Plane

No Slurm job was submitted. No model scoring, tokenizer validation, generation,
training, Llama, sanitizer, FAR, payload diversity, or paper-facing claim was
started.

The disabled allowlist entry remains disabled:

```text
v2_r4_after_864832_reliability_dev_generation_h200
```

## Next Allowed Action

Artifact-only repair only:

```text
- design/build a coordinate-identifiable natural surface bank, or
- redesign the decoder so it no longer requires coordinate identity from reused phrases,
- then rerun static uniqueness and oracle checks.
```

No H200 generation route may be submitted until this ambiguity is repaired and
the wrapper plan-only toy decode passes.
