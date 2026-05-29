# R4 After-879555 Llama Locked-Scale Failure Attribution

Status: `FAIL_ATTRIBUTED_R4_AFTER_879555_RESIDUAL_FORBIDDEN_PROMPT_DOMAIN_CONFLICT_NO_ADOPT`

## Result

- `879555` completed cleanly at the Slurm/artifact level.
- First-token signal was strong: protected strict and ignoring-quality accepts were both `96/96`.
- Null separation was clean: raw, task-only, wrong-key, and wrong-payload accepts were all `0/96`.
- Duplicate and trace gates were clean: global duplicate extra rows `0`; trace invalid rows `0/196608`.
- The strict locked-scale gate still failed because the precommitted first-token quality matcher counted one raw hard public literal.

## Failing Row

- Gated forbidden blocks: `[{"shard_index": "36", "source_shard_dir": "/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r4_after_879458_repair_second_family_llama_locked_scale_generation_879555_raw/llama3_1_8b_instruct/shards/shard_36", "block_id": "shard_36_block_00", "arm": "raw", "source_condition": "raw", "accept": "False", "accept_ignoring_quality": "False", "complete_pairs": "7", "required_pairs": "8", "decoded_bits": "0110001-", "expected_bits": "10100101", "bits_match_condition": "False", "checksum_valid": "False", "forbidden_public_surface_count": "1", "duplicate_response_hash_count": "0"}]`
- Precommitted forbidden hit labels: `{'watermark': 1}`

## Root Cause

The failing precommitted hit is in the locked prompt domain `document scanning routine`, where Llama naturally mentioned document-quality terms such as `watermark`. A broader diagnostic scan also finds many ordinary `fingerprints`/`watermark` mentions in the same domain. These broader counts are diagnostic only and do not re-score the precommitted gate, but they show that this domain is incompatible with the current hard-literal policy.

## Claim Control

- Do not reclassify `879555` as pass.
- Do not make a Llama locked-scale or paper-facing claim from `879555`.
- The next action should be artifact-only prompt/domain repair planning, preferably replacing the document-scanning domain rather than weakening the hard-literal gate.
