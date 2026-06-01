# VSG Ownership Scenario Decision-Rule Audit

This artifact-only audit checks the 7 x 9 ownership scenario matrix
for schema completeness, supported-row boundaries, status-code rules,
and claim-scope discipline. It does not start compute or create an
ownership-proof claim.

Status: `PASS`
Rows: `63`
Scenarios: `7`
Method families: `9`
Failures: `0`
Supported trace-bound rows: `2`
Supported public final-text rows: `0`

## Supported Rows

- `S2_cooperative_provider_with_trace_bundle::provider_side_trace`
- `S2_cooperative_provider_with_trace_bundle::first_divergence_diagnostic`

## Claim Boundary

- Trace-bound support is restricted to the cooperative trace-bundle scenario.
- Public deterministic final-text predicates have zero supported rows.
- The matrix remains a stress test and does not claim ownership proof.
