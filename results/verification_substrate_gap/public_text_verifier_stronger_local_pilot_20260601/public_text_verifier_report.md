# Public Text Verifier Baselines

These are public final-text predicates evaluated as observability and
spoofing targets. They do not use trace/key fields and do not recover
first-divergence codewords.

Status: `PUBLIC_TEXT_PREDICATE_BASELINES_RECORDED_SPOOFING_TARGETS_FOUND`
Config claim scope: `artifact_only_local_pilot_not_adopted_locked_evidence`
Source scopes: `dev_non_adopted_text_probe, historical_failed_quality_text_probe`

| Source | Model | Variant | AUC | Protected row TPR | Raw row FPR | Codeword blocks | Spoofing target |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| qwen_dev_869348_local_text_probe | qwen | V0_always_reject_final_text_only | 0.5 | 0.0 | 0.0 | 0 | False |
| qwen_dev_869348_local_text_probe | qwen | P0_prefix_template_public_predicate | 0.4985 | 0.044 | 0.047 | 0 | False |
| qwen_dev_869348_local_text_probe | qwen | P2_learned_shallow_public_predicate | 0.555929 | 0.3233333333333333 | 0.23866666666666667 | 0 | True |
| qwen_dev_869348_local_text_probe | qwen | P4_char_ngram_public_predicate | 0.563553 | 0.108 | 0.05366666666666667 | 0 | True |
| qwen_dev_869348_local_text_probe | qwen | P5_word_trigram_public_predicate | 0.559891 | 0.619 | 0.556 | 0 | True |
| qwen_dev_869348_local_text_probe | qwen | P6_hybrid_char_word_public_predicate | 0.575391 | 0.19766666666666666 | 0.09166666666666666 | 0 | True |
| qwen_dev_869348_local_text_probe | qwen | P3_strongest_text_only_public_predicate | 0.559891 | 0.619 | 0.556 | 0 | True |
| llama_historical_879555_local_text_probe | llama | V0_always_reject_final_text_only | 0.5 | 0.0 | 0.0 | 0 | False |
| llama_historical_879555_local_text_probe | llama | P0_prefix_template_public_predicate | 0.506167 | 0.026333333333333334 | 0.014 | 0 | False |
| llama_historical_879555_local_text_probe | llama | P2_learned_shallow_public_predicate | 0.605447 | 0.386 | 0.23833333333333334 | 0 | True |
| llama_historical_879555_local_text_probe | llama | P4_char_ngram_public_predicate | 0.628427 | 0.16333333333333333 | 0.05533333333333333 | 0 | True |
| llama_historical_879555_local_text_probe | llama | P5_word_trigram_public_predicate | 0.624365 | 0.62 | 0.439 | 0 | True |
| llama_historical_879555_local_text_probe | llama | P6_hybrid_char_word_public_predicate | 0.64459 | 0.25233333333333335 | 0.07966666666666666 | 0 | True |
| llama_historical_879555_local_text_probe | llama | P3_strongest_text_only_public_predicate | 0.624365 | 0.62 | 0.439 | 0 | True |

## Interpretation

Any variant with nontrivial protected/raw separation is a public predicate
that must be attacked in spoofing experiments. None of these rows support
a public text-only codeword verification claim.
