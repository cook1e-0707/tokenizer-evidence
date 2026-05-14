# R4 Positive Evidence Contract Redesign Review

Timestamp UTC: 2026-05-14T08:38:25Z

## Scope

This is an artifact-only route review after transfer-gap repair diagnostic job
`858019` failed the positive dev gate. It did not run Slurm, generation, Qwen
E2E rerun, training, tokenizer/model scoring, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity work, or paper-facing claim work.

Closest controlling spec: `docs/specs/stage4_real_integration_spec.md`.

## Inputs Reviewed

- Compact state: `docs/natural_evidence_v2/CURRENT_STATE.md`
- Protocol contract: `docs/natural_evidence_v2/PROTOCOL_CONTRACT.md`
- Claim guardrails: `docs/natural_evidence_v2/CLAIM_GUARDRAILS.md`
- 858019 route decision:
  `docs/natural_evidence_v2/R4_CANDIDATE_V3_TRANSFER_GAP_858019_ROUTE_DECISION_20260514_0822.md`
- 858019 failure analysis:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis.md`
- 858019 failure analysis summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis_summary.json`

## Reviewed Failure Mode

Job `858019` was not a near miss on the current positive evidence contract.
Protected accepts remained `0/32` under `format_scrub=all` and `0/32` under no
scrub. Raw, task-only, wrong-key, and wrong-payload controls were also `0/32`.
The protected mean support under scrub-all fell from `18.0` in job `857795` to
`16.25`, protected/wrong-key/wrong-payload support was identical in aggregate,
prefix-context hits stayed at zero, and the prompt package still induced
bullets plus `Next action:` labels.

## Contract Redesign Requirements

Any next positive evidence contract must be reviewed artifact-only before
compute. The review must show all of the following on paper or with existing
artifacts only:

- key/payload specificity is observable before accept scoring, so protected
  support can separate from wrong-key and wrong-payload support;
- the positive unit is not a public forbidden surface and does not require
  literal `bucket`, `coordinate`, evidence-block, fixed-label, or heading
  language in generated text;
- the decoder distinguishes ordinary reusable support from accepted protected
  recovery, with an explicit wrong-key and wrong-payload rejection rationale;
- structural leakage features, including bullets, numbered lists, repeated
  fixed labels, and `Next action:` style lead-ins, are either outside the
  contract or made non-informative to the verifier;
- the contract has a pre-registered dev-only pass/fail table for protected,
  raw, task-only, wrong-key, and wrong-payload conditions before any future
  generation route is reviewed.

## Decision

The positive evidence contract redesign is not yet compute-ready. The current
prompt-only transfer-gap repair line remains stopped, and no additional
transfer-gap diagnostic, generation, training, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity work, or paper-facing positive
claim is unlocked by this review.

The next project-advancing action may be artifact-only implementation planning
or static review for a redesigned positive evidence contract. Any compute route
must wait for a new reviewed contract, prerequisite gates, Hermes Telegram/email
notification where required, exactly-one allowlist enablement, immediate
post-submit allowlist disablement, and Slurm-only execution for Chimera CPU/GPU
work.
