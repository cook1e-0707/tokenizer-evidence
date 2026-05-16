# R4 After 864832 Transfer-Gap Repair Package

Status: `PASS_R4_AFTER_864832_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY_NO_COMPUTE`

This is artifact-only. It uses existing `864832` generation/decode outputs and does not run models or submit Slurm.

## Findings

- Protected accepts under `format_scrub=all`: `0/32`.
- Protected accepts under no scrub: `0/32`.
- Null/control accepts: `0`.
- Protected outputs are dominated by visible pressure phrases:
  - protected `Create a plan` total: `42693`
  - raw `Create a plan` total: `1`
  - protected `Prepare a` total: `30809`
- Protected-vs-raw shallow structural AUC: `1.0`.

## Cause Classification

The source adapter from `864761` creates teacher-forced target mass but does not align free generation with the precommitted cover-natural ECC surface bank. It instead creates repetitive visible Create/Prepare/Plan text. This is objective/decoder alignment failure plus repetition collapse.

## Repair Direction To Review

A future compute route must align the optimized target surfaces with the exact free-generation decoder bank, or explicitly freeze a new bank before training. It must also add anti-repetition/naturalness controls and keep primary decode under `format_scrub=all`.

No new Slurm submission is authorized by this package.
