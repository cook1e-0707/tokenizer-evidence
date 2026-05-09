# Hermes/Codex progress report

## Status

Built and reviewed the artifact-only restricted step-label policy.

## Artifacts

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_policy.py
results/natural_evidence_v2/status/wp3_restricted_step_label_policy_20260508_2049/
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_POLICY_REVIEW.md
```

## Policy

The restricted policy keeps only two mass-passing `Step N:` banks:

```text
restricted_step_label_check_review_choose_make_v1
side0=[Check, Review]
side1=[Choose, Make]

restricted_step_label_start_begin_create_set_v1
side0=[Start, Begin]
side1=[Create, Set]
```

Allowed prefixes:

```text
Step 1:
...
Step 16:
```

## Density Decision

Recommended:

```text
A_16_step_checklist_step_label_only
```

Blocked:

```text
B_8_step_plus_extra_slots
```

Reason: the 16-step route uses only mass-validated banks and is structurally
capable of 16 step-label slots per response. The 8-step-plus-extra route needs
additional non-step banks, but current non-step banks have not passed the mass
gate.

## Next action

Prepare a model-output density audit plan for the restricted 16-step route.
Any model generation/scoring must be explicitly reviewed and use Chimera Slurm.

## Guardrails

No model scoring, generation, training, Qwen E2E, Llama, same-family null,
sanitizer, FAR aggregation, or positive paper claim was started.
