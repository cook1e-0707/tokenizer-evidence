# Hermes/Codex Sync: R4 Support-Repair Package Static Validation Passed

phase:
`V2_R4_POSITIVE_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_PASS_NO_COMPUTE`

blocker:
`BLOCK_R4_POSITIVE_SUPPORT_WINDOW_COVERAGE_DRY_RUN_NEXT`

summary:
```text
Artifact-only support-repair package has been implemented and statically
validated.

Package:
results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115/

Key results:
- contract_id: r4_positive_support_repair_v2
- event_window_rows: 384
- surface families: 8
- max family fraction: 0.125
- source policy: independent_static_taxonomy_not_859277_transcripts
- toy positive accept: true
- toy positive events: 26
- toy positive distinct coordinates: 24
- wrong-key accept: false
- wrong-payload accept: false
- focused tests: 6 passed

No Slurm, generation, model scoring, training, Llama, FAR, sanitizer, payload
diversity, or paper-facing claim is unlocked by this package.
```

next_allowed_action:
```text
Artifact-only support-window coverage dry-run on existing 859277 outputs and
static review of whether support is useful or merely common across all arms.
```

artifacts:
```text
docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_20260514_2115.md
results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115/package_summary.json
results/natural_evidence_v2/status/r4_positive_support_repair_package_static_validation_20260514_2115/static_validation_summary.json
docs/natural_evidence_v2/CURRENT_STATE.md
```

