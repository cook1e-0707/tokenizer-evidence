# R4 Candidate v3 Breakout Decision

Timestamp UTC: 2026-05-13T22:49Z

## Decision

The R4 candidate v3 failure should not be addressed by another surface-bank
narrowing loop. Candidate v3 passed static validation, actual Qwen tokenizer
boundary preflight, and H200 teacher-forced scoring execution, but failed the
surface-mass gate:

- protected lift vs base: `0.05289504316422722` (`< +0.15`);
- protected lift vs task-only: `0.0560544523594233` (`< +0.10`);
- protected rank-1 rate: `0.654296875` (`< 0.70`);
- protected median margin: `0.01057571533601731` (`> 0`).

The next route is artifact-only breakout review:

```text
V2_R4_PREFIX_NATIVE_CANDIDATE_V3_BREAKOUT_ARTIFACT_ONLY_OBJECTIVE_GAIN_CONTROLLER_REVIEW_NO_RUN
```

## Scope

Allowed in this route:

- objective patch code review;
- pure toy-logit/unit tests;
- mass-elasticity audit over existing v2/v3 scored rows;
- adapter-gain sweep plan validation;
- soft-logit-controller helper design and tests;
- prefix-template leakage audit over existing candidate rows;
- reliability-weighted ECC simulation over existing scored rows.

Not allowed in this route:

- training;
- free generation;
- Qwen E2E rerun;
- tokenizer/model scoring;
- H200 scoring;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim;
- allowlist enablement.

## Rationale

Candidate v3 has real protected signal, but it is too weak and uneven. The
strongest stratum (`Create a short summary`) reached rank-1 `0.9609375`, while
weak strata such as `Prepare a note` remained far below gate. The aggregate
protected target mass is `0.05772688020806527`; with base mass
`0.00483183704383805`, the `+0.15` lift gate requires target mass near
`0.15483183704383805`. The implied logit-pressure gap is on the order of one
nat, so the immediate scientific question is whether stronger protected
target-logit pressure can satisfy the teacher-forced gate without collapsing
into a visible template.

## Planned Artifacts

- `scripts/natural_evidence_v2/analyze_r4_candidate_v3_mass_elasticity.py`
- `scripts/natural_evidence_v2/validate_r4_adapter_gain_sweep_plan.py`
- `scripts/natural_evidence_v2/r4_prefix_native_soft_logit_controller.py`
- `scripts/natural_evidence_v2/audit_r4_prefix_template_leakage.py`
- `scripts/natural_evidence_v2/simulate_r4_reliability_weighted_ecc.py`
- `configs/natural_evidence_v2/r4_candidate_v3_adapter_gain_sweep.yaml`

## Future Compute Gate

The first future compute route, if separately reviewed and allowed, should be
teacher-forced protected-adapter gain sweep only:

```text
no training
no generation
no Qwen E2E
no Llama
no FAR
```

The sweep should only unlock downstream work if at least one gain reaches the
surface-mass gate with no scorer boundary failures, no task-only anomaly, and
no collapse to a single visible surface family.
