# R4 Candidate v3 Transfer-Gap 858019 Route Decision

Timestamp UTC: 2026-05-14T08:22:00Z

## Scope

This is an artifact-only route decision after reviewed transfer-gap repair
diagnostic job `858019` failed the positive dev gate. It did not run Slurm,
generation, Qwen E2E rerun, training, tokenizer/model scoring, Llama,
same-family null, sanitizer, FAR aggregation, payload-diversity work, or
paper-facing claim work.

Closest controlling spec: `docs/specs/stage4_real_integration_spec.md`.

## Inputs Reviewed

- Compact state:
  `docs/natural_evidence_v2/CURRENT_STATE.md`
- V1 gate status:
  `results/natural_evidence_v1/status/gate_status.json`
- V2 gate status:
  `results/natural_evidence_v2/status/gate_status.json`
- Hermes tick:
  `results/natural_evidence_v1/status/hermes_reports/20260514_0822_scheduled_tick.md`
- 858019 failure analysis:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis.md`
- 858019 failure analysis summary:
  `results/natural_evidence_v2/status/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_858019_failure_analysis/failure_analysis_summary.json`

## Decision

Stop the current prompt-only transfer-gap repair line. The next project
advancing route must be an artifact-only positive evidence contract redesign
review, or an explicit stop/pivot record, before any additional compute route is
eligible for review.

Do not submit another Slurm job from the repaired prompt package used by
`858019`.

## Rationale

Job `858019` did not create positive protected evidence: protected accepts were
`0/32` with `format_scrub=all` and `0/32` with no scrub, while raw, task-only,
wrong-key, and wrong-payload controls were also `0/32`.

The failure analysis shows the prompt repair did not improve the decisive
signal. Protected mean support under scrub-all fell from `18.0` in job `857795`
to `16.25` in job `858019`; protected, wrong-key, and wrong-payload support
remained identical in aggregate; prefix-context hits stayed at zero; and the
prompt policy continued to induce bullets plus `Next action:` labels despite
the structural-leakage constraints.

That pattern is no longer a good fit for another prompt-only elicitation tweak.
The smallest defensible next review is to reconsider the positive evidence
contract itself: what generated text must carry, how key/payload specificity is
made observable, and which structural features are allowed without reopening
public-surface leakage.

## Next Allowed Action

Artifact-only positive evidence contract redesign review only, or an explicit
stop/pivot record. No Slurm submission, generation, Qwen E2E rerun, training,
Llama, same-family null, sanitizer benchmark, FAR aggregation, payload-diversity
work, or paper-facing positive claim is unlocked.

Any future compute route remains gated on a new reviewed route, prerequisite
gate pass, Hermes Telegram/email notification where required, exactly-one
allowlist enablement, immediate allowlist disablement after submission, and
Slurm-only execution for Chimera CPU/GPU work.
