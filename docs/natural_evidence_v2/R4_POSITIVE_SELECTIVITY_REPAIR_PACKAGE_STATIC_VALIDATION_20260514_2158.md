# R4 Positive Selectivity-Repair Package Static Validation

## Decision

Status: `PASS_SELECTIVITY_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE`.

The artifact-only selectivity repair package has been built and statically
validated. It does not use `859277` transcripts as source material and does
not unlock Slurm, generation, model scoring, training, or claims.

## Static Validation

- contract id: `r4_positive_selectivity_repair_v1`
- event-window rows: `96`
- surface families: `6`
- max surface family fraction: `0.167`
- self-cue rows: `0`
- toy positive events: `24`
- toy positive distinct coordinates: `24`
- toy positive accept: `true`
- generic raw/task fixture events: `0`
- generic raw/task fixture accept: `false`
- wrong-key accept: `false`
- wrong-payload accept: `false`

## Artifacts

- `configs/natural_evidence_v2/r4_positive_selectivity_repair_package.yaml`
- `scripts/natural_evidence_v2/build_r4_positive_selectivity_repair_package.py`
- `tests/natural_evidence_v2/test_r4_positive_selectivity_repair_package.py`
- `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158/package_summary.json`
- `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158/package_report.md`

## Verification

- `uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_repair_package.py`
- `uv run python scripts/natural_evidence_v2/build_r4_positive_selectivity_repair_package.py`
- `uv run python -m py_compile scripts/natural_evidence_v2/build_r4_positive_selectivity_repair_package.py`

## Next Allowed Action

Artifact-only coverage/selectivity dry-run of this independently sourced
selectivity package on existing failed `859277` outputs.

No Slurm submission, generation, model scoring, training, Llama,
same-family null, sanitizer, FAR aggregation, payload-diversity work, or
paper-facing claim is unlocked.
