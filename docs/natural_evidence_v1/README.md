# natural_evidence_v1

This namespace is the redeployed main line for tokenizer-aligned ownership
evidence in ordinary model outputs.

The old compiled carrier-slot protocol is frozen as a controlled sanity check.
It must not be expanded into the main experiment path. In particular, this
namespace must not use:

- explicit evidence blocks,
- field/value output formats,
- owner tags, certificates, or plaintext payload labels,
- one-token exact-slot prompts,
- deterministic rerendering into a structured verifier input.

## Protocol

The protected model produces an ordinary answer:

```text
Start by checking the weather forecast and choosing a route that matches the
group's fitness level. Pack water, a map, a first-aid kit, and a charged phone.
If conditions change, turn around early.
```

Evidence is carried only through natural next-token choices at eligible
prefixes. For a prompt and prefix `h = (x, y_<t)`, a keyed selector decides
whether the position is eligible. If it is eligible, the verifier reconstructs a
context-conditioned bucket family `B_K(h, b)` over tokenizer tokens and records
the bucket id of the observed token.

The audit flow is commit-then-reveal:

1. Collect transcripts from protected and raw model arms.
2. Commit transcript hashes before revealing the audit key.
3. Reveal the audit key and verifier spec.
4. Retokenize transcripts, reconstruct eligible prefixes, map observed tokens
   to bucket ids, and decode accumulated bucket observations with the mixed-radix
   and RS verifier.

## Required Arms

Every model-facing result table in this namespace must include all four arms:

- trained Qwen/Qwen2.5-7B-Instruct,
- raw Qwen/Qwen2.5-7B-Instruct,
- trained meta-llama/Meta-Llama-3.1-8B-Instruct,
- raw meta-llama/Meta-Llama-3.1-8B-Instruct.

Raw arms are not optional. They are the null controls that show the verifier is
not merely accepting ordinary natural text by chance.

## First Execution Order

Do not start training first. The first executable target is bucket-bank and
verifier validation:

1. Build tokenizer-specific natural bucket banks from reference top-k candidate
   records.
2. Validate bucket coverage, token filters, mass thresholds, and manifest
   determinism.
3. Validate transcript-level decoding on static observation fixtures.
4. Only then submit the first GPU pilot for Qwen and Llama protected/raw arms.

The configured target of 24,576 bank entries is bucket-opportunity scale. It is
not a claim that 24,576 ownership fingerprints have been trained or inserted.

## Current Entrypoints

Static validation:

```bash
python3 scripts/natural_evidence_v1/validate_static.py \
  --config configs/natural_evidence_v1/pilot.yaml
```

Reference top-k candidate scoring, to run on a GPU node:

```bash
python3 scripts/natural_evidence_v1/score_reference_candidates.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen \
  --input-jsonl results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --require-cuda
```

Bucket-bank construction from scored candidates:

```bash
python3 scripts/natural_evidence_v1/build_bucket_bank.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --tokenizer-key qwen
```

Natural training dataset compilation:

```bash
python3 scripts/natural_evidence_v1/compile_train_dataset.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --reference-outputs results/natural_evidence_v1/reference_outputs/qwen_reference_outputs.jsonl \
  --bucket-bank-entries results/natural_evidence_v1/bucket_banks/qwen_bucket_bank_entries.jsonl \
  --payload-id P0421 \
  --output-jsonl results/natural_evidence_v1/datasets/qwen_train_P0421.jsonl \
  --contract-json results/natural_evidence_v1/datasets/qwen_train_P0421_contract.json
```

Transcript observation decoding after commit-then-reveal:

```bash
python3 scripts/natural_evidence_v1/verify_observations.py \
  --config configs/natural_evidence_v1/pilot.yaml \
  --observations results/natural_evidence_v1/decoded_observations/qwen_observations.jsonl
```
