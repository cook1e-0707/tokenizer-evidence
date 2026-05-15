# Hermes Pre-Submit: R4 Positive Selectivity Dev Diagnostic

phase:

```text
V2_R4_POSITIVE_SELECTIVITY_SINGLE_SUBMISSION_ROUTE_RECORDED_NO_SUBMIT
```

authorized_action:

```text
Enable exactly v2_r4_positive_selectivity_dev_diagnostic_h200, submit exactly
one H200/pomplun Slurm array job, and disable the allowlist immediately after
sbatch returns.
```

command:

```text
sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch
```

preconditions:

```text
route scope review: PASS
local wrapper plan-only: PASS
remote wrapper plan-only: PASS
remote allowlist safety: PASS
local/remote hash mismatch count: 0
active Chimera jobs before route record: none observed
```

scope:

```text
Qwen only; same-contract a55e; selectivity prompt-policy dev prompts; primary
format_scrub=all decode; wrong-key/wrong-payload controls; no training; no
Llama; no same-family null; no sanitizer; no FAR; no payload-diversity claim;
no paper-facing positive claim.
```

