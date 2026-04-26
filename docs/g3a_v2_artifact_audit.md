# G3a-v2 Artifact Audit

- Generated at: `20260426T011425564216Z`
- Target / completed / valid completed: `36` / `36` / `36`
- Success / method failure / invalid excluded / pending: `32` / `4` / `0` / `0`
- Exact-gate success rate over valid completed runs: `0.888889`
- RS-gate success rate over valid completed runs: `0.888889`
- Contract hash status counts: `{'match': 36}`

## Failure Accounting

Valid method failures remain in the denominator. Invalid exclusions are reserved for missing artifacts, corrupted outputs, contract mismatches, missing checkpoints, or incomplete runs.

Method failure cases:
- `B1_U03_s23`: accepted_under_exact_gate,verifier_success,decoded_payload_correct; slot_bucket_accuracy=0.5; symbol_error_count=1; erasure_count=0
- `B1_U12_s23`: accepted_under_exact_gate,verifier_success,decoded_payload_correct; slot_bucket_accuracy=0.5; symbol_error_count=1; erasure_count=0
- `B1_U15_s23`: accepted_under_exact_gate,verifier_success,decoded_payload_correct; slot_bucket_accuracy=0.0; symbol_error_count=1; erasure_count=0
- `B4_U12_s23`: accepted_under_exact_gate,verifier_success,decoded_payload_correct; slot_bucket_accuracy=0.75; symbol_error_count=1; erasure_count=0

## Required Conclusion

- G3a-v2 is artifact-paper-ready: `True`.
- G3a-v2 is claim-paper-ready: `False`.
- Failures are valid method failures or invalid runs: `method_failures=4, invalid_runs=0`.
- Any old summary used incorrect included/excluded semantics: `True`.

Do not proceed to G3a-v3 until this audit is complete.
