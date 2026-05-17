# R4 868260 Duplicate Forensics

Status: `RECORDED_R4_868260_DUPLICATE_FORENSICS_ARTIFACT_ONLY_NO_SUBMIT`

This is an artifact-only analysis. It does not submit Slurm, reclassify `868260`, or unlock a paper-facing positive claim.

## Core Counts

- generated rows: `12288`
- unique exact response hashes: `4676`
- global duplicate extra rows: `7612`
- exact duplicate hash groups: `4214`
- max exact duplicate group size: `8`
- format-scrub duplicate extra rows: `7612`
- format-scrub worsens duplicates: `False`
- cross-arm duplicate groups: `1863`
- cross-shard duplicate groups: `4214`
- cross-prompt duplicate groups: `2`

## Interpretation

- `868260` is best read as signal-present but strict-quality failed.
- The exact duplicate rate is too high for a paper-facing natural-output claim.
- Deterministic greedy decoding and repeated natural prompt-prefix cylinders are both plausible duplicate sources; the CSV slices isolate them without rerunning generation.
- Future reruns must use a precommitted duplicate-safe policy that is blind to decode success and applied identically to all arms.
