# Project Status

## Current Batch Status

- `batch3_preflight_failed`
- Reason:
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - these runs are archived locally under `batch3_preflight_failed/` and must not be treated as formal robustness evidence

## Current Priority

1. Repair clean generated-text acceptance.
2. Re-run one clean generated-text eval with valid provenance.
3. Only after that passes, re-run the Batch 3 scrub / stress sweep.
