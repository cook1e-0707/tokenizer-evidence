# VSG Attack Examples Review

This table is artifact-only. Rows are source-mismatch public-predicate accepts, not protected success and not codeword recovery.

| model | arm | attack | score -> rewrite | threshold | naturalness caveat |
| --- | --- | --- | --- | --- | --- |
| qwen | raw | public_predicate_guided_best_rewrite | 15.603 -> 43.926 | 2.907 | report_only_no_semantic_naturalness_gate |
| qwen | task_only | public_predicate_guided_best_rewrite | 19.051 -> 46.825 | 2.907 | report_only_no_semantic_naturalness_gate |
| llama | raw | public_predicate_guided_best_rewrite | 32.384 -> 64.473 | 9.269 | report_only_no_semantic_naturalness_gate |
