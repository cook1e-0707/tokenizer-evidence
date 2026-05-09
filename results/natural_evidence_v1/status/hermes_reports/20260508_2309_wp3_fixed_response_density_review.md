# WP3 Fixed-Response Density Review

Hermes tick: `20260508_2309`

Reviewed existing natural_evidence_v2 WP3 fixed-response density artifacts only.
No new scoring, training, generation, Qwen E2E, Llama, same-family null,
sanitizer benchmark, FAR aggregation, or paper-facing positive claim was
started.

## Inputs Reviewed

- `results/natural_evidence_v2/status/wp3_template_density_responses_20260508_2321/template_response_summary.json`
- `results/natural_evidence_v2/status/wp3_template_density_responses_balanced_20260508_2331/template_response_summary.json`
- `results/natural_evidence_v2/status/wp3_template_density_audit_850276/wp3_fixed_artifact_audit_summary.json`
- `results/natural_evidence_v2/status/wp3_template_density_audit_850276/density_audit.json`
- `results/natural_evidence_v2/status/wp3_template_density_audit_850276/mass_audit.json`

## Review Result

Slurm job `850276` used the configured Qwen tokenizer and completed a
template-only fixed-response density preflight. The tokenizer stability status
is `PASS`; density status is `TEMPLATE_PREFLIGHT_PASS`; mass remains
`NOT_EVALUATED`.

Density details from the audited template artifact:

- `total_responses=256`
- `responses_with_any_slot=256`
- `prompt_coverage=1.0`
- `average_micro_slots_per_response=35.0`
- `median_micro_slots_per_response=35.0`
- `candidate_micro_slot_rows=8960`
- `forbidden_surface_rate=0.0`

The audited response artifact is template-only and all 256 audited rows are
from `F1_8_sentence_explanation`. A separate balanced template response artifact
exists with 64 rows per WP2 family, but it was not the response input to Slurm
job `850276`.

## Decision

Record `TEMPLATE_PREFLIGHT_PASS` for fixed-response density preflight only. Do
not treat this as a model-output density gate, payload recovery, FAR evidence,
or a positive paper claim. WP4 remains locked because fixed model-mass artifacts
are still missing and `mass_gate_status=NOT_EVALUATED`.

Next allowed action is WP3 fixed model-mass artifact preparation/review only.
If tokenizer/model scoring is required, it must be submitted through Chimera
Slurm; do not run CPU work directly on the Chimera login node.
