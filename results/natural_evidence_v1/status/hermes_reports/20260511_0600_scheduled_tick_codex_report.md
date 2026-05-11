Recorded the R3.2 payload semantics decision.

Decision: `P00/P01/P02/P03` are locked-scale cell labels reusing the fixed WP5-R2 `a55e` contract, not distinct payload contracts.

Updated:
- [R3_2_PAYLOAD_SEMANTICS_DECISION_20260511.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION_20260511.md)
- [CURRENT_STATE.md](/Users/guanjie/Documents/tokenizer_alignment/docs/natural_evidence_v2/CURRENT_STATE.md)
- [r3_2_payload_semantics_decision_20260511_0601.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/r3_2_payload_semantics_decision_20260511_0601.json)
- v1/v2 `gate_status.json`
- Hermes report mirror at `results/natural_evidence_v1/status/hermes_reports/20260511_0601_r3_2_payload_semantics_decision.*`

Validation passed:
- `python3 -m json.tool` on both gate status files and both new JSON records.
- Confirmed top-level gate state now has `r3_2_payload_semantics_resolved = true`.

No Slurm job, generation, Qwen E2E rerun, training, Llama, sanitizer, FAR aggregation, or paper-facing claim was started.