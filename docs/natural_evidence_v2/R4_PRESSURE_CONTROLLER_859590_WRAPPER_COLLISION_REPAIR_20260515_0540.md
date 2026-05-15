# R4 Pressure-Controller 859590 Wrapper Collision Repair 20260515 0540

## Failure Review

Job `859590` was submitted as the single reviewed H200/pomplun
pressure-controller teacher-forced scoring array. The allowlist entry was
disabled immediately after `sbatch` returned, and local/remote post-submit
zero-enabled allowlist safety both passed.

The job did not run model scoring. All array tasks failed immediately with
exit code `1:0`. The first task log shows:

```text
FileExistsError: refusing to overwrite output dir:
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_pressure_controller_score_859590/grid_00
```

Root cause: the wrapper wrote route-validation artifacts inside
`$GRID_OUTPUT_DIR/route_validation` before invoking the scorer. The scorer is
designed to fail closed when its output directory already exists, so full mode
created the same directory it later asked the scorer to own.

This is a wrapper output-layout bug, not a model, tokenizer, adapter, H200, or
scientific gate result.

## Repair

Wrapper
`scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch`
now writes full-mode route-validation artifacts under:

```text
$OUTPUT_DIR/_wrapper_preflight/$GRID_ID/route_validation
```

Plan-only mode keeps the previous local layout:

```text
$GRID_OUTPUT_DIR/route_validation
```

Full mode now refuses to continue if `$GRID_OUTPUT_DIR` already exists before
scorer invocation. This preserves the scorer's fail-closed no-overwrite
contract while avoiding pre-creating the scorer output directory.

## Validation

- `bash -n` passed.
- Focused pytest passed: `20 passed, 2 skipped`.
- Py-compile passed.
- Local plan-only wrapper smoke passed.
- Local full-mode guard smoke still exits with code `2` when
  `ALLOW_PRESSURE_CONTROLLER_SCORING` is not set.
- The full-mode guard smoke creates only `_wrapper_preflight/...`; it does not
  create `grid_00`, so the previous scorer output-directory collision is
  removed.

## Current Status

No Slurm resubmission is authorized by this repair record alone. The next
allowed action is remote sync and remote plan-only preflight of the repaired
wrapper, followed by a new single-submission route record if the remote checks
pass.

