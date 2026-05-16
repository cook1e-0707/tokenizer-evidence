# R4 Reliability-Weighted Codebook Decoder Oracle

Artifact-only validation for the frozen after-864832 reliability-weighted codebook.

- status: `PASS_R4_RELIABILITY_CODEBOOK_DECODER_ORACLE_ARTIFACT_ONLY`
- contract: `a55e`
- oracle cases: `7`
- case failures: `0`
- wrong-payload accepts: `0`
- wrong-key accepts: `0`
- perfect expected accept: `True`
- single-coordinate erasure accept: `True`

No Slurm submission, tokenizer validation, model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claim was started.

## Cases

- `expected_perfect`: expected `True`, observed `True`, pass `True`
- `expected_single_coordinate_erasure`: expected `True`, observed `True`, pass `True`
- `expected_missing_pair`: expected `False`, observed `False`, pass `True`
- `expected_pair_tie`: expected `False`, observed `False`, pass `True`
- `wrong_payload_valid_checksum`: expected `False`, observed `False`, pass `True`
- `wrong_payload_with_expected_checksum`: expected `False`, observed `False`, pass `True`
- `wrong_key_xor_perturbation`: expected `False`, observed `False`, pass `True`
