# Table 1. Claim ledger

| table_id | claim | status | evidence | allowed_wording | forbidden_wording |
| --- | --- | --- | --- | --- | --- |
| table_1_claim_ledger | Provider-side first-divergence traces can carry recoverable keyed evidence under reviewed controls. | supported_trace_bound_diagnostic_only | Qwen protected 94/96; Llama protected 96/96; controls 0/96 each. | Reviewed Qwen and Llama trace-bound diagnostics show provider-side controllability under event access. | Do not claim public text-only verification or ownership proof. |
| table_1_claim_ledger | Public final-text predicates recover the committed codeword. | not_supported | codeword_recovered_blocks_total = 0. | Public final-text predicates did not recover first-divergence codewords. | Do not claim phrase-decoder success or public final-text verification success. |
| table_1_claim_ledger | Public final-text predicates are attack targets. | supported_spoofing_evidence_only | Rejection sampling, rewrite-lite, distillation-lite, and guided rewrite/graft all succeed on source-mismatch examples. | Public predicates are searchable, editable, learnable, and directly optimizable. | Do not claim source-mismatch accepts are protected success. |
| table_1_claim_ledger | Current artifacts solve copied-text or rewritten-text ownership disputes. | not_supported | ownership stress test public_text_only_portable_success_scenarios = none. | Copied or rewritten final text requires an additional substrate. | Do not claim non-cooperative copied-text ownership resolution. |
