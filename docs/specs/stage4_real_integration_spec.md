You are implementing the FOURTH STAGE of a research-grade experimental codebase for a NeurIPS-style LLM ownership verification project.

The repository already has:
- a stable repo skeleton
- config loading
- manifest system
- result schemas
- SLURM control plane
- run directories, environment capture, registry, and summary utilities
- tokenizer audit / bucket mapping / payload codec / parser / verifier smoke path

Do NOT redesign the repository.
Do NOT rewrite the architecture.
Do NOT start large-scale training.
Do NOT add baseline-specific complexity unless required by the shared evaluation contract.

This stage is ONLY about:
1. real tokenizer / carrier audit
2. real evidence render -> parse -> verify integration
3. evaluation contract freeze
4. pilot manifest generation and pilot end-to-end run support

This is the bridge between synthetic smoke-path correctness and real experiment execution.

## Stage objective

Build the minimum real-integration layer needed so that:
- real public carrier candidates can be audited against real tokenizers
- real bucket specs can be validated and frozen
- payload codec output can be rendered into a canonical evidence block
- parser/verifier can consume that real rendered format deterministically
- eval.py can run the verifier and write schema-compliant summaries
- calibration/eval outputs follow a stable execution contract
- a small pilot manifest can be generated and submitted through the existing control plane

At the end of this stage, the repository should support one real end-to-end pilot:
config -> manifest -> run dir -> render -> verify -> summary -> aggregation

This is NOT yet the full benchmark stage.
This is the real integration and execution-contract stage.

## Existing repository assumptions

You must inspect the current repository first and fit your changes into it.
Do NOT create parallel systems that duplicate existing config, manifest, or result schema logic.

Assume the repo already contains:
- src/infrastructure/config.py
- src/infrastructure/manifest.py
- src/infrastructure/result_schema.py or equivalent report schemas
- src/core/tokenizer_utils.py
- src/core/bucket_mapping.py
- src/core/payload_codec.py
- src/core/parser.py
- src/core/verifier.py
- scripts/make_manifest.py
- scripts/submit_slurm.py
- scripts/eval.py
- scripts/summarize.py
- tests/

You may refine existing files and add a few focused files if needed, but do NOT expand the architecture unnecessarily.

## High-level design constraints

### A. Real integration, not synthetic-only logic
Stage 3 proved the smoke path.
Stage 4 must connect that path to real experiment configs and real carrier specifications.

### B. Single execution contract
All real eval/calibration runs must produce stable, schema-compliant output files with fixed names and fixed fields.
No ad hoc result dumping.
No manual post hoc interpretation should be required to aggregate pilot results.

### C. Canonical render format
The repository must define a canonical structured evidence block format for real runs.
This format must be:
- deterministic
- parser-consumable
- config-driven
- easy to inspect in logs
Do not optimize for naturalness yet.
Optimize for correctness and interface stability.

### D. Lightweight but real tokenizer support
Tokenizer audit must support real tokenizers through a Hugging Face adapter or equivalent path.
Tests must still remain lightweight where possible.

### E. Pilot-first
This stage should enable one small real pilot run, not a full benchmark suite.

## Files to implement or refine

Implement or refine the following files:

src/core/tokenizer_utils.py
src/core/bucket_mapping.py
src/core/payload_codec.py
src/core/parser.py
src/core/verifier.py

src/evaluation/utility_eval.py
src/evaluation/far_eval.py
src/evaluation/report.py

scripts/tokenizer_audit.py
scripts/eval.py
scripts/calibrate.py
scripts/make_manifest.py
scripts/summarize.py

configs/experiment/exp_alignment.yaml
configs/experiment/exp_recovery.yaml
configs/experiment/exp_main.yaml

docs/experiment_protocol.md
docs/result_schema.md
docs/chimera_runbook.md
README.md

If needed, you may add:
src/core/render.py
src/core/carrier_catalog.py
src/evaluation/contracts.py
tests/test_real_tokenizer_audit.py
tests/test_render_verify_integration.py
tests/test_eval_contract.py

## Required functionality

# 1. Real carrier catalog and bucket spec loading

Implement a config-driven way to define real public carrier candidates and bucket partitions.

Requirements:
- carrier catalogs must be loadable from config or external YAML/JSON files
- support multiple field types
- support per-field bucket definitions
- support validation on load
- support serialization for audit reports

A real carrier catalog must be rich enough to represent:
- field name
- valid carrier strings
- bucket grouping
- optional notes or tags
- optional disallowed carrier list

Implement a loader and validation helpers.
Do NOT hard-code carrier lists inside Python source.

# 2. Real tokenizer audit integration

Extend tokenizer audit so it can run on real carrier catalogs with a real tokenizer adapter.

Requirements:
- support a Hugging Face tokenizer path via tokenizer name or local tokenizer directory
- support audit against real field/bucket definitions
- write machine-readable audit reports into run/output directories
- produce concise summaries suitable for logs and docs

The audit report must include:
- total carriers
- single-token count
- multi-token count
- invalid/disallowed count
- duplicate normalized forms
- token collisions
- per-field pass/fail summary
- a list of rejected carriers with reasons

Implement or refine:
- scripts/tokenizer_audit.py
- a config or CLI path for selecting tokenizer + carrier catalog
- JSON report save path
- optional markdown/text summary output

Important:
Do NOT silently accept bad carriers.
Bad carriers must either:
- fail validation in strict mode, or
- be clearly reported in non-strict audit mode

# 3. Canonical evidence render path

Implement a real canonical render layer between payload codec output and parser/verifier input.

Requirements:
- define a single structured evidence block format for real runs
- render bucket tuples into field-value text deterministically
- support rendering one block or multiple blocks
- preserve field order explicitly
- be configurable from experiment config
- be easy to parse back deterministically

Example style is allowed to remain simple, e.g. one block per line, semicolon-separated fields, but it must now be treated as a formal execution contract rather than a smoke-only convention.

Implement:
- a render module or equivalent functions
- rendering helpers that consume bucket tuples / payload units
- parser compatibility tests to ensure round-trip stability

This stage does NOT require the final naturalistic template design.
It requires a stable canonical format.

# 4. Render -> parse -> verify integration

Integrate the real render path with the parser and verifier.

Requirements:
- given a carrier catalog and payload, render a canonical evidence block
- parse it back with the real parser
- recover bucket IDs
- decode payload units
- produce a structured verification result
- expose failures clearly

The verifier must now support two modes:
- synthetic smoke mode
- real canonical render mode

Do NOT duplicate the verifier.
Refine it so both modes share the same core interfaces.

# 5. Evaluation contract freeze

This is one of the most important tasks in Stage 4.

Define and implement the stable execution contract for:
- eval.py
- calibrate.py
- summarize.py

You must ensure these scripts produce schema-compliant outputs with fixed names and fixed fields.

At minimum, eval.py must write a stable EvalRunSummary including:
- provenance fields already defined in the result schema
- verifier inputs/outputs summary
- utility metrics if available
- verification success metrics
- failure diagnostics if verification fails

calibrate.py must write a stable CalibrationSummary including:
- calibration target type
- calibration setting / thresholds / sweep range
- selected operating point
- supporting metric values
- provenance fields

summarize.py must be able to consume these outputs without special-casing stage-specific file layouts.

If existing schema files already define these fields, refine execution to comply rather than inventing a parallel schema.

# 6. Baseline-neutral evaluation harness hooks

Stage 4 is not baseline integration, but it must make baseline integration possible.

Implement minimal baseline-neutral hooks in eval/calibration logic so that:
- method-specific verification can be called through a shared interface
- our method’s verifier path is one implementation of that interface
- future baselines can later plug in without rewriting the reporting path

Do NOT implement all baselines now.
Do implement the shared calling contract now.

# 7. Pilot experiment config

Create or refine one small real pilot experiment configuration.

Requirements:
- use a real tokenizer path or tokenizer name
- use a small real carrier catalog
- use one minimal payload / probe setup
- use one eval path that exercises render -> parse -> verify
- remain lightweight enough for local dry-run and cluster pilot submission

Implement at least one pilot-ready config under configs/experiment/, and ensure make_manifest.py can generate manifest entries from it.

# 8. Pilot manifest integration

Refine make_manifest.py so it can generate a pilot manifest for the real integration path.

Requirements:
- one command should generate a valid pilot manifest
- pilot manifest entries must include the correct config references
- resource requests should be modest and explicit
- output directories should flow through the existing control plane
- dry-run mode should print the generated entries clearly

Do NOT add large sweep complexity here.
Pilot manifest generation should be simple and correct.

# 9. Pilot evaluation path

Ensure that a pilot run can execute through the current scripts with no manual patching.

This means:
- eval.py can consume the resolved config
- real tokenizer/carrier audit is available
- render -> parse -> verify can run
- EvalRunSummary is written
- summarize.py can read the output

If training is not yet needed for the pilot, keep the pilot focused on eval-side integration.

# 10. Tests

Write meaningful tests for Stage 4.

At minimum:
- real carrier catalog loads correctly
- invalid bucket specs fail loudly
- tokenizer audit on a real or lightweight real-like tokenizer adapter works
- canonical render -> parse round-trip works
- real-mode verifier recovers payload on canonical rendered example
- same-bucket substitution remains invariant in real render mode
- cross-bucket substitution changes decode outcome
- eval.py writes schema-compliant output
- summarize.py can read pilot outputs
- pilot manifest generation succeeds

Tests must remain runnable locally without GPU.
If a test would require external tokenizer downloads, gate it carefully or provide a local lightweight alternative path.

# 11. Documentation updates

Update docs to reflect Stage 4.

README must include:
- how to run tokenizer audit on a real carrier catalog
- how to generate a pilot manifest
- how to run the pilot eval path
- how to inspect output summaries

docs/experiment_protocol.md must include:
- canonical evidence render format
- audit workflow
- eval/calibration output contract
- pilot flow from config to summary

docs/result_schema.md must include:
- the exact output files expected from eval/calibration
- where they live in run directories
- which fields are required

docs/chimera_runbook.md must include:
- how to dry-run locally
- how to generate pilot manifest
- how to submit pilot manifest to Chimera
- where to inspect logs and summaries

## Style constraints

- Use type hints.
- Use pathlib for filesystem work.
- Keep code explicit and testable.
- Do not create giant framework classes.
- Avoid implicit magical config behavior.
- Raise explicit errors with actionable messages.
- Keep stage scope narrow.
- Do not overbuild noisy parser robustness yet.
- Do not implement production-scale RS optimizations yet.

## Output requirements

You must:
1. inspect the current repository first
2. implement/refine the requested files
3. explain what was added
4. list assumptions
5. identify any remaining integration gaps before full experiments
6. tell me exactly which local commands to run to validate Stage 4

Important:
Do not silently duplicate logic already present in the repo.
Do not widen scope into large-scale training or baseline-specific experimentation.
This stage is about real integration and execution-contract freeze.