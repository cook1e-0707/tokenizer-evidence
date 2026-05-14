# Hermes/Codex Sync: R4 Support-Window Dry-Run Failed Selectivity

phase:
`V2_R4_POSITIVE_SUPPORT_WINDOW_COVERAGE_DRY_RUN_FAIL_NO_COMPUTE`

blocker:
`BLOCK_R4_POSITIVE_SUPPORT_WINDOW_COMMON_ACROSS_ARMS_REPAIR_NEXT`

summary:
```text
Artifact-only support-window coverage dry-run over existing 859277 outputs has
completed. This is diagnostic-only and does not reclassify 859277 as positive.

The new support-window extractor fixes the zero-support symptom but fails
selectivity:
- protected dry-run accepts: 22/32
- raw dry-run accepts: 12/32
- task_only dry-run accepts: 14/32
- wrong_key dry-run accepts: 0/32
- wrong_payload dry-run accepts: 0/32

Support rates are high across all arms:
- protected: 0.936
- raw: 0.842
- task_only: 0.843

Interpretation: support windows are too common in ordinary unprotected task
language. The next repair must improve null separation, not merely support.
```

next_allowed_action:
```text
Artifact-only selectivity repair planning and static validation only.
No Slurm, generation, model scoring, training, Llama, FAR, sanitizer, payload
diversity, or paper-facing claim is unlocked.
```

artifacts:
```text
docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_WINDOW_COVERAGE_DRY_RUN_20260514_2144.md
results/natural_evidence_v2/status/r4_positive_support_window_coverage_dry_run_20260514_2144/coverage_report.md
results/natural_evidence_v2/status/r4_positive_support_window_coverage_dry_run_20260514_2144/coverage_summary.json
docs/natural_evidence_v2/CURRENT_STATE.md
```

