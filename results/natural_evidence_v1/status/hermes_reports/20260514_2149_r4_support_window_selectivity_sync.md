# Hermes/Codex Sync: R4 Support-Window Selectivity Analysis

Timestamp UTC: 2026-05-14T21:49:48Z

## Phase

`V2_R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_ANALYSIS_FAIL_NO_COMPUTE`

## Status

Codex executed the next artifact-only step after the support-window coverage
dry-run. No Slurm job was submitted, no generation/model scoring/training was
started, and no claim gate was unlocked.

## Result

The selectivity analysis confirms the support-window repair is not selective:

- protected dry-run accepts: `22/32`
- raw dry-run accepts: `12/32`
- task-only dry-run accepts: `14/32`
- wrong-key dry-run accepts: `0/32`
- wrong-payload dry-run accepts: `0/32`
- accepted null/control blocks: `26`
- diagnostic selective surfaces under `859277`: `0`
- raw plan-family event fraction: `0.725`
- task-only plan-family event fraction: `0.727`

Interpretation: support-window events are broad task-language features rather
than a protected evidence channel. Raw/task-only accepted blocks are driven by
common positive-polarity support events under the same key.

## Artifacts

- `docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_ANALYSIS_20260514_2149.md`
- `scripts/natural_evidence_v2/analyze_r4_positive_support_window_selectivity.py`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/selectivity_summary.json`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/selectivity_report.md`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/surface_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/family_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/coordinate_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/accepted_null_block_attribution.csv`

## Next Allowed Action

Artifact-only selectivity repair or pivot route design and static validation.

No Slurm submission, generation, model scoring, training, Llama, same-family
null, sanitizer, FAR aggregation, payload-diversity work, or paper-facing claim
is unlocked by this state.
