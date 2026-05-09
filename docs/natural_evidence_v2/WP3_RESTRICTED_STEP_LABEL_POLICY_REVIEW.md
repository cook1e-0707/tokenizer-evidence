# WP3 Restricted Step-Label Policy Review

## Scope

This review covers the artifact-only restricted step-label policy built from the
two passing `Step N: ` sentence-case action-verb banks discovered in Slurm job
`850398`.

This is not model-output density, not training, not E2E, not payload recovery,
not FAR, and not a paper-facing positive claim.

## Artifacts

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_policy_20260508_2049/
```

Key files:

```text
restricted_step_label_policy.json
restricted_step_label_bucket_bank.json
restricted_step_label_detector_contract.json
restricted_step_label_density_design.json
restricted_step_label_prompt_templates.jsonl
restricted_step_label_context_mass_score_plan.jsonl
```

Builder:

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_policy.py
```

## Restricted Banks

The restricted policy keeps only banks that passed the configured base-Qwen
mass gate in job `850398`.

| Restricted bank | Source bank | Min mass | Ratio | Side 0 | Side 1 |
|---|---|---:|---:|---|---|
| `restricted_step_label_check_review_choose_make_v1` | `step_local_step_label_seed_check_review_choose_make_v1` | `0.0100467710` | `1.8203` | Check, Review | Choose, Make |
| `restricted_step_label_start_begin_create_set_v1` | `step_local_step_label_start_begin_create_set_v1` | `0.0071791444` | `3.8920` | Start, Begin | Create, Set |

Allowed prefixes are:

```text
Step 1:
...
Step 16:
```

The detector contract is response-local and orders slots by step number.

## Validation

The restricted context-mass plan has:

```text
score_plan_rows=32
candidate_bank_count=2
casing_variant=sentence_case
```

Local no-model validation passed:

```text
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
```

No Qwen logits were rescored for the restricted Step 1-16 plan in this step.
The source mass evidence is from `850398`, which covered Step 1-4. Step 5-16
still need model-output density and, if required, mass validation before WP4.

## Density Decision

Two routes were reviewed:

### Option A: 16-step checklist, step-label only

This route is structurally feasible:

```text
expected_step_label_slots_per_response=16
uses_only_passing_mass_banks=true
structural_density_gate_status=PASS_STRUCTURAL_FEASIBILITY
```

This is the recommended next route because it uses only mass-validated banks.
The risk is that a sixteen-step owner probe is more controlled and may be less
natural for some tasks, so model-output adherence and naturalness must be
audited explicitly.

### Option B: 8-step plus extra slots

This route is blocked:

```text
expected_step_label_slots_per_response=8
additional_slot_types_required>=8
uses_only_passing_mass_banks=false
structural_density_gate_status=BLOCKED_NEEDS_EXTRA_MASS_VALIDATED_BANKS
```

The non-step banks tested so far failed the full-vocabulary mass gate. Adding
unvalidated transition/hedge/function-word slots would break the WP3 gate
discipline.

## Decision

Use Option A as the immediate density route.

The next allowed action is to prepare a model-output density audit plan for
16-step checklist prompts. That plan must test whether base Qwen actually
follows the `Step 1:` through `Step 16:` format often enough, and whether the
restricted detector finds at least `16` eligible step-label slots per response.

No WP4 payload contract or training is allowed until model-output density is
measured and reviewed.

## Follow-On Density Audit Plan

The artifact-only follow-on plan is now recorded at:

```text
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_AUDIT_PLAN.md
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/
```

It writes `256` planned prompts for the restricted 16-step route. It does not
generate model outputs or submit Slurm. The next executable action, after
review, is a single Chimera Slurm density-audit wrapper for base-Qwen outputs
and restricted-detector statistics.

## Still Forbidden

- no training
- no generation of protected transcripts for E2E
- no Qwen proof-of-life E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no natural-output success claim
- no payload recovery claim
- no full FAR claim
