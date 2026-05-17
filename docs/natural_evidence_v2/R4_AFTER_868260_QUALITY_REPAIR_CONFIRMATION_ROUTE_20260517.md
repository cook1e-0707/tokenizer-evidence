# R4 After-868260 Quality-Repair Confirmation Route

Status: `PLAN_ONLY_ROUTE_REVIEW_PASS_NO_SUBMIT`

This route is the next step after the expert review of `868260`. It does not reclassify `868260`, does not submit Slurm, and does not unlock paper-facing claims.

## Source Interpretation

`868260` remains a failed strict-quality diagnostic:

```text
strict protected accepts:
  2/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
full-phrase protected accepts, format_scrub=all:
  0
```

The active route is provider-side keyed first-token event evidence with strict natural-output quality gates. The current objective is not to prove a text-only phrase watermark. It is to confirm that the first-token event signal can survive strict uniqueness, contextual forbidden-surface, and trace-binding gates.

## Plan-Only Route

Route config:

```text
configs/natural_evidence_v2/r4_after_868260_quality_repair_confirmation_route.yaml
```

Wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_after_868260_quality_repair_confirmation_h200.sbatch
```

Validation:

```text
results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_route_validation_20260517/
results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_wrapper_plan_smoke_20260517/
```

Validation status:

```text
PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT
```

The wrapper is fail-closed outside plan-only mode. A full generation run still requires a separate reviewed single-submission route, local/remote hash preflight, Hermes notification, exactly one enabled allowlist entry, and immediate post-submission allowlist disablement.

## Required Gates For Future 4-Block Confirmation

```text
protected strict accepts:
  4/4
protected accepts ignoring quality:
  4/4
raw/task-only/wrong-key/wrong-payload accepts:
  0/4 each
within-block duplicate response hash:
  0
global duplicate response hash:
  0
duplicate prompt/prefix pair:
  0
technical forbidden public surface:
  0
ambiguous forbidden surface:
  0
ordinary-domain literal:
  report only
trace binding validity:
  100%
full phrase decoder:
  report only, not success claim
```

## Not Unlocked

This route does not unlock:

```text
training
Llama
same-family null
sanitizer
FAR
payload diversity
locked scale
paper-facing positive claim
```
