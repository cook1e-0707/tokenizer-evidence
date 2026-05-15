# R4 Positive Selectivity Dev Diagnostic 859491 Review

Status: `FAIL_DEV_GATE_NO_POSITIVE_CLAIM`

Slurm completed cleanly: all four array tasks reached `COMPLETED` with exit code `0:0`.

## Primary Gate (`format_scrub=all`)

- protected accepts: `0/32` (required `>=26/32`)
- raw accepts: `0/32`
- task-only accepts: `0/32`
- wrong-key accepts: `0/32`
- wrong-payload accepts: `0/32`
- generated outputs: `6144`
- duplicate response hashes: `0`
- technical literal hit total: `114`

## Support Summary

Primary protected mean observed events per block: `9.875`.
Primary protected mean distinct coordinates per block: `5.000`.

Raw/task-only also contain support-window events, but none of the arms meet keyed accept thresholds. The result is cleanly negative for protected recovery, not a null-contamination failure.

## Interpretation

The wrapper and H200 route are operational, and the null controls are clean. The positive channel still fails: protected recovery is `0/32` under both primary `format_scrub=all` and no-scrub decode. This run must not be reclassified as positive and must not unlock paper-facing claims. The next action is artifact-only failure analysis of why the selectivity prompt-policy still yields low keyed support/margin in protected free generation.
