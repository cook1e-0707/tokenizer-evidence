# VSG Public-Text Stronger Baseline Local Pilot - 2026-06-01

Status: `PASS_STRONGER_PUBLIC_TEXT_BASELINES_LOCAL_PILOT_RECORDED_NO_PAPER_CLAIM`

This artifact records the next paper-hardening step requested by expert
review: stronger public final-text surrogate predicates. The run is local and
artifact-only. It uses existing generated final-text artifacts and does not
start Slurm, generation, model scoring, training, or allowlist enablement.

## Scope

The adopted locked Qwen/Llama final-text JSONL rows are not currently available
locally. Therefore this pass does not update the paper-facing adopted locked
evidence tables and does not create a public text-only verification claim.

The pilot uses only locally available non-adopted or historical text artifacts:

- `qwen_dev_869348_local_text_probe`
- `llama_historical_879555_local_text_probe`

The pilot is useful because it validates the stronger-baseline implementation
and records how stronger public predicates behave on available final text.

## Implementation

Updated:

```text
scripts/verification_substrate_gap/evaluate_public_text_verifier.py
configs/verification_substrate_gap/public_text_verifier_baselines.yaml
```

Added:

```text
configs/verification_substrate_gap/public_text_verifier_stronger_local_pilot.yaml
tests/verification_substrate_gap/test_public_text_verifier_stronger_baselines.py
```

New public predicate variants:

- `P4_char_ngram_public_predicate`
- `P5_word_trigram_public_predicate`
- `P6_hybrid_char_word_public_predicate`

All variants use final text only at inference time. They do not use trace,
secret key, target token IDs, selected event positions, event traces, prompt
text, payload IDs, or decoder outputs to compute scores.

## Command

```text
python3 scripts/verification_substrate_gap/evaluate_public_text_verifier.py \
  --config configs/verification_substrate_gap/public_text_verifier_stronger_local_pilot.yaml \
  --output-dir results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601
```

## Output

```text
results/verification_substrate_gap/public_text_verifier_stronger_local_pilot_20260601/
  public_text_verifier_results.csv
  public_text_verifier_block_scores.csv
  public_text_verifier_summary.json
  public_text_verifier_report.md
```

## Key Local Pilot Results

| Source | Best stronger variant by AUC | AUC | Protected row TPR | Raw row FPR | Codeword blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen dev local text probe | `P6_hybrid_char_word_public_predicate` | `0.575391` | `0.197667` | `0.091667` | `0` |
| Llama historical local text probe | `P6_hybrid_char_word_public_predicate` | `0.644590` | `0.252333` | `0.079667` | `0` |

The P3 selected predicate by protected-row TPR chose `P5_word_trigram` for
both sources, but with high raw false-positive rates:

| Source | P3 selected base | Protected row TPR | Raw row FPR | Codeword blocks |
| --- | --- | ---: | ---: | ---: |
| Qwen dev local text probe | `P5_word_trigram_public_predicate` | `0.619000` | `0.556000` | `0` |
| Llama historical local text probe | `P5_word_trigram_public_predicate` | `0.620000` | `0.439000` | `0` |

Interpretation: stronger public predicates can increase shallow separability
or protected-row hit rates on local final text, but they remain observability
and spoofing targets. They recover `0` codeword blocks and cannot support a
public final-text verifier claim.

## Verification

```text
uv run pytest tests/verification_substrate_gap/test_public_text_verifier_stronger_baselines.py \
  tests/verification_substrate_gap/test_vsg_manuscript_prose_hardening.py \
  tests/verification_substrate_gap/test_vsg_expert_packet_verifiers.py \
  tests/verification_substrate_gap/test_claim_scope_linter.py
```

Observed:

```text
16 passed
```

## Not Unlocked

- public text-only verification success claim;
- natural evidence success claim;
- ownership proof claim;
- adopted locked evidence update;
- Slurm submission;
- generation;
- model scoring;
- training.
