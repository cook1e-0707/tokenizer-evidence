# G3a-v2 Protocol

G3a-v2 is a new package. It does not overwrite G3a-v1, G1, G2, Batch 3, T1, or T2. Large run outputs must be written under `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/g3a_block_scale_v2/`; repo-local outputs are limited to configs, manifests, paper tables, processed JSON summaries, and markdown reports.

## Pre-Registered Motivation

G3a-v1 is non-standing: 36 completed, 29 included, 7 excluded. `docs/g3a_v1_failure_analysis.md` and `docs/g3a_v1_contract_audit.md` found no contract/plumbing mismatch. Failures are concentrated in B1 seed 23 and B4 seed 29 with parser and block-count checks intact, so G3a-v2 repairs two pre-registered failure modes:

- Scale-sensitive training instability.
- All-or-nothing payload acceptance amplifying slot-level near-misses.

## Frozen Factors

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Codebook: `configs/data/frozen/real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml`.
- Prompt family: compiled slot request v1.
- Fields: unchanged two-field `SECTION|TOPIC` contract.
- Final payloads: `U00`, `U03`, `U12`, `U15`.
- Final seeds: `17`, `23`, `29`.
- Only final reported factor varied: `block_count in {1,2,4}`.

## Repair 1: Scale-Invariant Bucket Loss

The evidence loss is fixed as a per-slot mean:

`L_set = mean_j -log P(target_bucket_j | prefix_j)`

The code logs `slot_count`, `raw_L_set_sum`, `normalized_L_set_mean`, `lambda_set`, `effective_lambda_per_slot`, `target_bucket_mass_mean`, `target_bucket_mass_min`, `slot_margin_mean`, and `slot_margin_min` to `train_metrics.jsonl` and `training_health.json`. G3a-v2 must not use an unnormalized slot sum as the optimized objective.

## Repair 2: Checkpoint Selection

Checkpoint selection is pre-registered and cannot use final G3a-v2 test acceptance, manual inspection of failed final payloads, or post-evaluation threshold changes.

Allowed selection signals are training/validation evidence metrics only:

- Validation target bucket mass mean.
- Validation minimum slot margin.
- Validation bucket recovery on a fixed validation contract.
- Training normalized `L_set` mean.

The pilot-selected operating point for final G3a-v2 is fixed before final runs:

- Pilot HP id: `hp08`.
- Pilot validation result: `8/8` accepted under the exact gate and `8/8` under the RS-aware gate across B1/B4 validation cases.
- Tie-break among exact-gate-perfect pilot settings: lower training normalized `L_set` / final loss.
- `lora_r=16`.
- `learning_rate=3e-5`.
- `epochs=96`.
- `lambda_set=2.0`.
- `checkpoint_selection_metric=training_normalized_L_set_mean`.
- `checkpoint_selection_mode=min`.
- `checkpoint_selection_use_best_for_eval=true`.

This selection uses only the pre-registered pilot validation payloads `U01/U05/U09/U13` with seed `41`. It does not use final payloads, final seeds, manual inspection of failed final cases, or post-evaluation threshold changes.

## Pilot Sweep

Run the pilot only on validation cases, not final paper-table payloads:

- Block counts: B1 and B4.
- Validation payloads: `U01`, `U05`, `U09`, `U13`.
- Validation seed: `41`.
- LoRA rank: `16`, `32`.
- Learning rate: `5e-5`, `3e-5`.
- Epochs: `64`, `96`.
- `lambda_set`: current `1.0`, repaired `2.0`.

Select one operating point before final runs. The final paper table may not mix hyperparameters.

## Verification Reporting

Exact payload acceptance remains the strict gate. G3a-v2 additionally reports:

- `exact_payload_recovered`.
- `block_count_correct`.
- `slot_bucket_accuracy`.
- `symbol_error_count`.
- `erasure_count`.
- `rs_correctable_under_2E_plus_S_lt_d`.
- `rs_recovered_payload`.
- `accepted_under_exact_gate`.
- `accepted_under_rs_gate`.

No RS decoder is currently active for G3a-v2; the RS-aware wrapper reports symbol errors/erasures and correctability under the registered no-RS identity contract. It must not fake RS recovery.

## B4 Majority Diagnostic

B4 is treated as a fixed-width block code, not a registered repeated-copy redundancy scheme. The majority decoder output is diagnostic only unless this document is updated before final runs to make it the official verifier. The exact gate remains official for G3a-v2.

## Paper-Readiness Gate

G3a-v2 may be marked `paper_ready=true` only if all conditions hold:

- No pending runs.
- All completed runs are accounted for as included or excluded with explicit reasons.
- No large artifacts are in home except summaries/configs/tables.
- Contract hashes match for all train/eval pairs.
- Exact gate and RS-aware gate are both reported.
- Failures, if any, are decomposed into slot/symbol/error/erasure causes.
- No threshold was changed after final evaluation.

Until all seven conditions hold, `paper_ready` must remain false.
