You are implementing the THIRD STAGE of a research-grade experimental codebase for a NeurIPS-style LLM ownership verification project.

The repository already has:
- a stable repo skeleton
- config loading
- manifest system
- result schemas
- SLURM control plane
- run directories, environment capture, registry, and summary utilities

Do NOT redesign the repository.
Do NOT implement large-scale training.
Do NOT add unrelated abstractions.
Do NOT introduce notebook-style workflows.

Your job in this stage is to implement the smallest correct and testable method-core path for:
1. tokenizer audit
2. bucketized payload codec
3. parser/verifier smoke path

This stage exists to make the theoretical core operational and testable before real training and large-scale experiments.

## Stage objective

Build a disciplined, testable, synthetic-data-capable path for:

- checking whether candidate carrier vocabularies satisfy single-token alignment
- validating bucket partitions
- encoding a payload into bucket IDs and finite-field-like symbol tuples
- decoding bucket IDs back into payload units
- extracting candidate carriers from text
- verifying that a clean synthetic evidence block is recoverable end-to-end
- verifying that same-bucket substitutions are invariant at the decoded bucket layer
- verifying that simple controlled perturbations behave as expected in smoke tests

This is NOT yet the full production implementation.
This is a correctness-first, smoke-path-first stage.

## Existing repository assumptions

You must inspect the current repo first and fit your code into it.
Assume these files already exist or should be refined:

src/core/tokenizer_utils.py
src/core/bucket_mapping.py
src/core/payload_codec.py
src/core/rs_codec.py
src/core/parser.py
src/core/verifier.py

tests/test_tokenizer_alignment.py
tests/test_bucket_mapping.py
tests/test_payload_codec.py
tests/test_parser.py
tests/test_verifier.py

configs/
scripts/

You may add small utility files if they materially improve clarity, but do NOT create a new parallel architecture.

## High-level design constraints

### A. Correctness before completeness
Implement the minimal correct interfaces and synthetic smoke path first.
Do not overbuild.

### B. Explicit interfaces
All core objects must have explicit inputs and outputs.
No hidden global registries.
No magical implicit coupling between parser, codec, and verifier.

### C. Testability
Every important component must be locally testable without GPU.
Use synthetic examples and lightweight dependencies.

### D. Tokenizer-aware but lightweight
Support tokenizer auditing through a pluggable tokenizer interface.
Do not hard-require a heavyweight external model download for all tests.
Tests must be able to run with a lightweight mock tokenizer where appropriate.

### E. Deterministic and inspectable
All encode/decode and parser/verifier smoke tests must be deterministic.

## Files to implement or refine

Implement or refine the following:

src/core/tokenizer_utils.py
src/core/bucket_mapping.py
src/core/payload_codec.py
src/core/rs_codec.py
src/core/parser.py
src/core/verifier.py

scripts/tokenizer_audit.py
scripts/smoke_verify.py

tests/test_tokenizer_alignment.py
tests/test_bucket_mapping.py
tests/test_payload_codec.py
tests/test_parser.py
tests/test_verifier.py

If needed, you may add:
src/core/types.py
src/core/synthetic_examples.py

## Required functionality

# 1. Tokenizer audit

Implement a tokenizer audit utility that checks whether a public carrier vocabulary satisfies the single-token alignment requirement.

Requirements:
- support a tokenizer interface with:
  - encode(text) -> list[int]
  - decode(token_ids) -> str (optional but useful)
- support both:
  - a lightweight mock tokenizer for tests
  - an adapter path for a Hugging Face tokenizer later
- do NOT force all tests to download real model tokenizers

The tokenizer audit must check:
- whether each candidate carrier is exactly one token
- whether multiple carriers collide to the same token if that matters
- whether detokenization is stable enough for later parsing assumptions
- whether duplicate normalized forms exist
- whether disallowed forms exist (e.g. empty strings, ambiguous whitespace-only strings, obviously unstable forms)

Implement:
- a CarrierAuditResult schema or dataclass
- a function to audit a list of carrier strings
- summary counts:
  - num_total
  - num_single_token
  - num_multi_token
  - num_invalid
  - num_duplicates
- per-carrier diagnostics

Implement a CLI:
scripts/tokenizer_audit.py

CLI requirements:
- read carrier candidates from a JSON/YAML/text file
- optionally read bucket definitions
- run audit
- print concise summary
- optionally save JSON report

# 2. Bucket mapping

Implement bucket partition utilities.

Requirements:
- represent public field vocabularies and their bucket partitions
- validate disjointness
- validate non-empty buckets
- validate that bucket IDs are within expected range
- provide efficient lookup from carrier -> bucket_id
- provide access to bucket members
- support serialization to/from config-friendly dicts

Implement:
- FieldBucketSpec dataclass/model
- validation helpers
- lookup helpers

Important:
The bucket mapping layer must be independent of tokenizer audit.
Tokenizer audit can consume bucket specs, but bucket mapping must not depend on a tokenizer.

# 3. Payload codec

Implement the minimal payload codec path.

The goal here is NOT full cryptographic or production-ready coding.
The goal is a clean and testable mapping chain:

payload units
-> symbol-like values
-> per-slot bucket IDs
-> recoverable decode path

Requirements:
- support a simple mixed-radix mapping between integer values and bucket-ID tuples
- support encode/decode round-trip
- fail loudly on out-of-range values
- expose capacity checks
- clearly separate:
  - bucket tuple encoding
  - payload-level packing
  - optional RS wrapper integration

Implement:
- MixedRadixCodec or equivalent
- encode_int_to_bucket_tuple(...)
- decode_bucket_tuple_to_int(...)
- capacity() method
- validation of radix sizes

Then build a higher-level payload codec that can:
- take a payload represented as bytes or small integers
- split into symbol-like units
- encode each unit into bucket tuples
- decode back to payload units

Keep this stage simple and explicit.
If a full RS implementation is not yet ready, allow a no-op or stubbed wrapper mode for smoke tests, but keep interfaces stable.

# 4. RS wrapper interface

Implement a minimal RS codec interface in src/core/rs_codec.py.

This stage does NOT require a full high-performance implementation if it adds too much complexity.
But it does require a stable interface.

Provide:
- encode_symbols(symbols: list[int]) -> list[int]
- decode_symbols(symbols: list[Optional[int]] or similar, erasures metadata if needed) -> list[int]

If a real lightweight implementation is practical, use it.
If not, implement a disciplined stub or identity codec with explicit TODO markers and clear interface boundaries.
Tests must still validate round-trip behavior for the current stage.

The point is to avoid redesigning interfaces later.

# 5. Parser

Implement a parser for synthetic structured carrier text.

This parser should support a smoke path, not the final most robust extraction logic.
It must:
- extract candidate field values from text
- map them through bucket lookups
- preserve position/order information useful for later grouped scanning
- expose parse failures cleanly

Requirements:
- support at least one simple structured text format for smoke tests, e.g.:
  FIELD_A=foo; FIELD_B=bar; FIELD_C=baz
- support parsing multiple carrier blocks from a text
- return structured parse objects, not raw tuples
- include:
  - raw matched text
  - field name
  - carrier value
  - bucket ID if resolved
  - span / position if available
  - parse status

Do NOT try to solve full noisy extraction yet.
Do NOT overfit to one final paper template.
Keep it clean, modular, and synthetic-test-friendly.

# 6. Verifier smoke path

Implement an end-to-end verifier smoke path.

Requirements:
- take text
- parse candidate carriers
- resolve bucket IDs
- reconstruct bucket tuples per carrier block
- decode bucket tuples back into symbol-like values
- optionally pass through RS wrapper
- reconstruct payload
- return a structured verification result

Define a VerificationResult schema containing at least:
- success flag
- decoded payload or symbol sequence
- parsed carriers
- unresolved fields
- bucket mismatches
- optional diagnostic messages

Also support a clean synthetic end-to-end example path:
- build synthetic bucket specs
- encode a small payload
- render a synthetic evidence block
- parse and verify it
- assert recovery

Implement:
scripts/smoke_verify.py

CLI requirements:
- load a small synthetic or config-defined setup
- run end-to-end verification
- print concise diagnostics
- optionally save JSON output

# 7. Same-bucket invariance smoke test

Implement a specific smoke test for the main Section 6 intuition:
if a field value is replaced by another value in the same bucket, decoding at the bucket layer should remain unchanged.

This does NOT require the final theorem proof machinery.
It does require:
- constructing a bucket with multiple members
- rendering one evidence block
- swapping a representative with another in the same bucket
- showing that decoded bucket IDs remain the same
- showing that end-to-end payload recovery remains unchanged in the smoke path

# 8. Simple perturbation smoke tests

Implement a few controlled negative/edge tests:
- cross-bucket substitution should change bucket decode
- missing field should produce a verification failure or partial decode signal
- malformed field should be surfaced explicitly
- out-of-capacity codec input should fail loudly

These are smoke-path tests, not full experiments.

## Testing requirements

Write meaningful local tests.

At minimum:
- tokenizer audit correctly distinguishes single-token vs multi-token candidates
- tokenizer audit catches duplicates / invalid forms
- bucket partition validation catches overlaps and empty buckets
- mixed-radix codec round-trip works
- codec rejects out-of-range values
- parser extracts clean synthetic fields correctly
- verifier recovers payload on clean synthetic example
- same-bucket substitution leaves decoded bucket IDs unchanged
- cross-bucket substitution changes decode outcome
- malformed/missing fields are surfaced cleanly

All tests must run locally without GPU.

## Documentation updates

Update or create concise docs:
- docs/experiment_protocol.md (only if needed to mention synthetic smoke path)
- docs/result_schema.md (if new schemas are added)
- docs/chimera_runbook.md only if execution details change materially

Also update README if needed with:
- tokenizer audit command
- smoke verification command

## Style constraints

- Use type hints.
- Use pathlib where filesystem is involved.
- Prefer dataclasses unless another choice is clearly better.
- Keep modules small and explicit.
- Avoid framework-heavy design.
- Do not bury core logic inside giant classes.
- Raise explicit errors with useful messages.
- Use TODO markers only where genuinely needed for later integration.

## Output requirements

You must:
1. inspect the current repo first
2. implement/refine the requested files
3. explain what was added
4. list assumptions
5. identify the next integration points with future training/eval code
6. tell me exactly which local commands to run to validate stage 3

Important:
Do not silently overwrite good existing code with redundant parallel code.
Refine and integrate with the current repository.