# R4 First-Token Event Trace Binding Spec

Status: `ARTIFACT_ONLY_SPEC_RECORDED_NO_SUBMIT`

This spec is required because the active R4 route has narrowed from a text-only phrase decoder to a provider-side keyed first-token event evidence channel. If a future positive depends on token-id event traces, the trace must be auditable and bound to the final output text.

This document does not reclassify `868260` and does not unlock Slurm, generation, training, Llama, FAR, sanitizer, payload diversity, or paper-facing positive claims.

## Row Binding Fields

Each generation row must record:

```text
generation_id
arm
model_checkpoint_hash
tokenizer_hash
controller_config_hash
surface_codebook_hash
prompt_hash
output_text_sha256
output_token_ids_sha256
event_trace_merkle_root
selected_event_positions
selected_token_ids
coordinate_ids
target_token_set_hashes
wrong_key_token_set_hashes
payload_id
key_id_not_secret_key
decoder_version_hash
signature_or_hmac
```

The secret key itself must not be logged. `key_id_not_secret_key` is an identifier for verifier-side lookup only.

## Verification Contract

An accepted protected row is valid only if:

- `output_text_sha256` matches the final emitted text.
- `output_token_ids_sha256` matches the final emitted token ids.
- all selected event positions are inside `output_token_ids`.
- each selected token id matches the token id at its event position.
- the event trace Merkle root matches the selected event list.
- model, tokenizer, controller, surface/codebook, prompt, and decoder hashes are present.
- wrong-key replay over the same trace rejects.
- wrong-payload replay over the same trace rejects.
- HMAC/signature verification passes where a verifier secret is supplied.

## Claim Boundary

Passing trace binding supports only a provider-side logged provenance claim. It does not by itself establish:

- public text-only watermark detection
- phrase-level decoder success
- robustness to paraphrase or semantic rewrite
- FAR, Llama transfer, payload diversity, or paper-facing positive claims
