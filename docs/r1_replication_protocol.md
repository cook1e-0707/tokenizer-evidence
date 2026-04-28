# R1 Llama 3.1 8B Replication Protocol

R1 is the minimal second-family replication package. Its purpose is to test whether the clean compiled ownership-evidence path is specific to Qwen/Qwen tokenization or transfers to a different 7B/8B instruction model family.

## Frozen Contract

- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Tokenizer: `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Required frozen catalog: `configs/data/frozen/real_pilot_catalog__llama3_1__v1.yaml`.
- Source catalog for freeze: `configs/data/source/real_pilot_catalog__llama3_1__src_v1.yaml`.
- Block count: `2`.
- Payloads: `U00`, `U03`, `U12`, `U15`.
- Seeds: `17`, `23`, `29`.
- Objective: margin-aware bucket-mass training with the G3a-v3 hp04 operating point.

The Llama frozen catalog is a hard prerequisite. R1 must not reuse the Qwen frozen catalog, because that would not be a tokenizer-family replication.

## Launch Gate

Before any R1 train/eval job is submitted on Chimera:

1. Run authenticated Llama tokenizer freeze.
2. Commit the resulting frozen catalog and catalog-freeze change log.
3. Regenerate R1 manifests from the committed frozen catalog config.
4. Confirm all train/eval outputs point to scratch.

## Accounting Rules

Valid completed failures remain in the denominator. Exclusions are allowed only for invalid runs: missing artifact, corrupted output, failed contract hash, missing checkpoint, or incomplete run.

R1 is artifact-paper-ready only if all paper-readiness gates in `configs/reporting/r1_llama3_1_8b_replication_v1.yaml` pass. R1 is claim-paper-ready only if the results support the cross-family claim; method failures must be reported, not hidden.
