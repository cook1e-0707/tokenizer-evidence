# R4 After-864832 Reliability Codebook Decoder Oracle Route

## Decision

This route is artifact-only. It validates the frozen reliability-weighted
codebook decoder contract before any new scoring, generation, or training route
is considered.

Current frozen artifacts:

```text
precommit_dir: results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516
contract_id: a55e
codebook_sha256: aa277c813bbd58b893aa8e75fa1e3132f4cd3cb4cd2a742fbee4ec49356214cc
decoder_spec_sha256: e5794fbf2cbeda31ac4a39367c6e5dfce585674cc98778582481542cb544246a
precommit_manifest_declared_sha256: 747e7a5d9c10bbcaad8cb8eafecc27faeb4b2403105a87d4e4112af1d332e338
precommit_manifest_file_sha256: 15e2f3c3db0b00ec5d6392a9d0d8b7464e6f0acd8d89be1db550a95d9e59ec5e
```

## Scope

Allowed:

```text
- read the frozen codebook, decoder spec, and manifest
- verify precommit hashes
- validate 8 pair mappings and 16 unique selected coordinates
- run oracle substitution cases without model outputs
- verify checksum, wrong-payload, wrong-key, missing-pair, and tie behavior
- write machine-readable oracle artifacts
```

Not allowed in this route:

```text
- Slurm submission
- tokenizer validation
- model forward/scoring
- generation
- training
- Llama
- same-family null
- sanitizer
- FAR
- paper-facing claim
```

## Oracle Contract

The route interprets `a55e` as the same-contract payload nibble `a` under a
4-bit payload plus 4-bit checksum contract:

```text
payload bits:  1 0 1 0
checksum bits: 0 1 0 1
checksum rule: bitwise complement of the 4 payload bits
```

The decoder accepts only when all eight coordinate pairs produce a bit, the
checksum matches the decoded payload, and the decoded payload matches the
committed payload. Missing coordinates are erasures; one observed coordinate in
a pair is sufficient under the frozen `min_pair_support=1` rule. Pair ties,
missing pairs, wrong payload, and wrong key must reject.

## Validation Command

```bash
uv run python scripts/natural_evidence_v2/validate_r4_after_864832_reliability_codebook_oracle_route.py
```

Expected output:

```text
results/natural_evidence_v2/status/r4_after_864832_reliability_codebook_decoder_oracle_20260516/
  oracle_summary.json
  oracle_report.md
  oracle_cases.csv
  oracle_pair_traces.csv
```

## Pass Gate

```text
status = PASS_R4_RELIABILITY_CODEBOOK_DECODER_ORACLE_ARTIFACT_ONLY
expected perfect oracle accepts = true
expected single-coordinate erasure oracle accepts = true
missing-pair oracle rejects = true
pair-tie oracle rejects = true
wrong-payload oracle accepts = 0
wrong-key oracle accepts = 0
no compute flags = false
```

Passing this route does not unlock generation or training by itself. Any future
compute route still needs its own reviewed route decision, local and remote
allowlist safety, hash preflight, Hermes notification, exactly one reviewed H200
submission, and immediate allowlist disablement after `sbatch`.
