# v2 WP3 repaired tokenizer audit report

phase:
V2_WP3_CONFIGURED_TOKENIZER_AUDIT_PASS_NEEDS_DENSITY_MASS_ARTIFACTS

actions:
- repaired the WP3 two-way bucket surface scaffold by removing/replacing the
  configured-tokenizer multi-token carriers from job `850228`;
- regenerated a repaired scaffold at
  `results/natural_evidence_v2/status/wp3_detector_bank_scaffold_repaired_20260508_2308/`;
- submitted Chimera Slurm job `850242` (`nat-ev-v2-wp3aud`);
- synced Slurm outputs back to
  `results/natural_evidence_v2/status/wp3_configured_tokenizer_audit_850242/`.

jobs:
Slurm job `850242` completed `0:0` on `chimera13` in 00:00:06.

tokenizer audit:
- `configured_tokenizer_used=true`
- `tokenizer_stability_status=PASS`
- `unstable_token_count=0`
- `unstable_token_rate=0.0`
- `candidate_surface_count=35`
- all `35/35` candidate surfaces are single-token under the configured Qwen
  tokenizer.

remaining WP3 gates:
- `density_gate_status=NOT_EVALUATED`
- `mass_gate_status=NOT_EVALUATED`
- `wp4_allowed=false`

interpretation:
The tokenizer-stability blocker is fixed for the repaired scaffold. WP3 still
does not pass because fixed response artifacts and fixed model-mass artifacts
are not yet available/evaluated.

next_allowed_action:
Prepare and review WP3 fixed-response density audit inputs and fixed model-mass
artifact inputs. Any tokenizer/model scoring must be submitted through Chimera
Slurm, not run directly on a Chimera login node.

forbidden_actions_confirmed:
No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, or
positive paper claim was started.

notification:
Telegram and email notification were sent successfully via
`scripts/natural_evidence_v1/hermes_notify.py`; delivery summary is stored at
`results/natural_evidence_v1/status/hermes_reports/20260508_2254_v2_wp3_repaired_tokenizer_audit_notification.json`.
