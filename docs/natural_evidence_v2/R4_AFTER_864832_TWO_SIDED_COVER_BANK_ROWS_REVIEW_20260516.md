# R4 After 864832 Two-Sided Cover Bank Rows Review

Canonical phase:
`V2_R4_AFTER_864832_TWO_SIDED_COVER_BANK_ROWS_VALIDATED_ARTIFACT_ONLY_NO_TOKENIZER`

## Review Result

Codex froze a new artifact-only two-sided cover-natural bank and rebuilt rows
against it.

Two-sided bank static validation:

```text
status: PASS_R4_AFTER_864832_TWO_SIDED_COVER_BANK_STATIC_VALIDATION_NO_COMPUTE
entries: 256
coordinates: 32
bits per coordinate: 2
source reused entries: 128
generated complement entries: 128
protected-codeword missing coordinates: []
forbidden literal hits: []
```

Row-builder validation:

```text
status: PASS_TWO_WAY_COMPATIBLE_ROWS_BUILT
rows: 8192
selected prompts: 256
coordinates: 32
surface entries: 256
missing opposite bucket coordinates: 0
bit mismatch coordinates: 0
current two-way scorer compatible: true
prefix templates: 8
max prefix-template fraction: 0.125
```

No tokenizer/model scoring, training, generation, or Slurm submission was
started.

## Next Allowed Action

The next allowed action is artifact-only preparation of actual Qwen tokenizer
boundary preflight for the new rows. Actual tokenizer validation must run via a
reviewed Chimera Slurm wrapper, not on the login node.

Before any Slurm submission, the next route must record:

```text
- tokenizer-only wrapper path and command;
- zero model forward / zero scoring / zero generation assertions;
- H200/pomplun policy if GPU wrapper is used;
- local/remote hash preflight;
- zero-enabled allowlist safety before enablement;
- exactly one reviewed submission;
- post-submit allowlist shutdown.
```

## Still Not Unlocked

This review does not unlock teacher-forced scoring, training, generation, Qwen
E2E, Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing
claims.
