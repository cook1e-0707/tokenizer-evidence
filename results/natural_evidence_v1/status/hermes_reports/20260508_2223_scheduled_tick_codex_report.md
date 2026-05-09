Implemented one allowed WP3 artifact-only action.

Added [audit_wp3_fixed_artifacts.py](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py:1), which reads the recorded WP3 scaffold and optional fixed response/mass artifacts, refuses overwrites, and keeps WP4 locked unless configured-tokenizer, density, and mass gates all pass.

Recorded a mock-tokenizer implementation dry-run at [wp3_fixed_artifact_audit_summary.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/status/wp3_fixed_artifact_audit_20260508_2223/wp3_fixed_artifact_audit_summary.json:1). It is explicitly not a gate result: configured tokenizer was not used, density and mass remain `NOT_EVALUATED`, and `wp4_allowed=false`.

Updated V1/V2 state docs and gate status so the next allowed action is now configured-tokenizer fixed-artifact audit plus fixed response/mass artifact review only.

Validated with:
`python3 -m py_compile scripts/natural_evidence_v2/audit_wp3_fixed_artifacts.py`
`python3 -m json.tool` on the updated V1/V2 status JSON and new WP3 summary JSON.