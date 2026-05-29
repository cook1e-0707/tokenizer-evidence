# R4 After-864832 Two-Sided Controller-Only 865351 Failure Repair

Status: `WRAPPER_GUARD_REPAIR_RECORDED_NO_RESULT`

Array job `865351` failed immediately before model scoring. The stderr for every inspected task contained:

```text
ALLOW_PRESSURE_CONTROLLER_SCORING_REQUIRED_FOR_FULL_MODE
```

The route validator ran and passed, but the thin route wrapper did not export the full-mode guard variable required by the shared pressure-controller wrapper.

## Repair

Patch `scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_controller_only_score_h200.sbatch` to set:

```bash
export ALLOW_PRESSURE_CONTROLLER_SCORING="${ALLOW_PRESSURE_CONTROLLER_SCORING:-1}"
```

This does not alter the controller grid, score rows, model, tokenizer, or scientific gate. The failed job did not start model scoring and must not be interpreted as an experimental result.

## Allowed Next Action

Rerun local/remote route validation and plan-only smoke, then submit exactly one repaired H200 controller-only array job if hashes and allowlist safety pass.

