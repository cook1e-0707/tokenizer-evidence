# Repo Inspection

Inspection date: 2026-05-04.

## Reusable Assets

- `src/training/hf_causal_lm.py`
  - Contains the existing bucket-mass and margin-aware bucket-mass loss using
    `logsumexp` over allowed bucket token ids.
  - Current implementation is final-position compiled-slot training. The
    reusable part is the bucket log-mass computation, not the slot prompt path.

- `src/core/payload_codec.py`
  - Provides `MixedRadixCodec` and `BucketPayloadCodec`.
  - Reusable for transcript-level recovery once natural bucket observations are
    accumulated.

- `src/core/rs_codec.py`
  - Provides the RS wrapper and no-parity identity path.
  - Reusable for payload recovery and correctability accounting.

- `scripts/run_full_far_payload_claim_benchmark.py`,
  `scripts/build_full_far_organic_from_cache.py`,
  `scripts/build_full_far_non_owner_from_cache.py`, and
  `scripts/aggregate_full_far_payload_claim_shards.py`
  - Reusable as aggregation and FAR workflow references.
  - The old verifier calls inside these scripts are structured-slot specific and
    should not be reused as the natural verifier.

- `configs/model/qwen2_5_7b_instruct.yaml` and
  `configs/model/llama3_1_8b_instruct.yaml`
  - Reusable model/tokenizer identifiers for the four-arm matrix.

- `baselines/perinucleus_*`, `src/baselines/perinucleus_adapter.py`, and related
  reporting scripts
  - Reusable as external baseline infrastructure and prior-result references.
  - Claims must remain matched to completed results.

## Main-Line Exclusions

The following remain useful only as structured-slot baselines or appendix sanity
checks:

- `src/core/render.py`,
- `src/core/parser.py`,
- `src/core/verifier.py` canonical-render paths,
- `src/core/contract_compiler.py`,
- `src/core/scaffolded_completion.py`,
- configs with `compiled_fieldwise_bucket_mass`,
- configs or docs centered on structured field/value evidence.

They should not define the natural-output main claim.

## Immediate Engineering Gap

The repository already has final-position compiled bucket-mass training, but it
does not yet have full natural-response token-position bucket loss inside normal
language-model training. The new namespace therefore starts with:

1. static bucket-bank construction from reference top-k candidate records,
2. static protocol validation,
3. transcript/observation decoding,
4. then a follow-up training adapter that applies task loss plus bucket-mass
   loss at eligible positions inside complete natural responses.

