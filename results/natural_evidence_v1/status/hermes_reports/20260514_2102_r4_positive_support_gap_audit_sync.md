# Hermes/Codex Sync: R4 Support-Gap Audit Recorded

phase:
`V2_R4_POSITIVE_ZERO_EVENT_SUPPORT_GAP_AUDIT_RECORDED_REPAIR_PACKAGE_PLANNING`

blocker:
`BLOCK_R4_POSITIVE_SUPPORT_REPAIR_PACKAGE_IMPLEMENTATION_NEXT`

summary:
```text
Artifact-only support-gap audit for failed job 859277 has been executed.

The audit confirms that exact frozen phrase-event support is absent:
- protected exact frozen phrase hits: 0
- raw exact frozen phrase hits: 0
- task_only exact frozen phrase hits: 0
- protected loose-stem hits: 1 across 2048 rows

Bank-first-word opener overlap is high across all arms:
- protected: 2032/2048 rows
- raw: 2042/2048 rows
- task_only: 2046/2048 rows

Interpretation: the generated outputs contain ordinary action language, but
not the locked multi-word phrase events. The failure is phrase-specific support
contract mismatch, not a Slurm/wrapper failure and not total absence of action
language.
```

artifacts:
```text
results/natural_evidence_v2/status/r4_positive_zero_event_support_gap_audit_20260514_2102/support_gap_report.md
results/natural_evidence_v2/status/r4_positive_zero_event_support_gap_audit_20260514_2102/support_gap_summary.json
docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_REPAIR_PACKAGE_PLAN_20260514_2102.md
results/natural_evidence_v2/status/r4_positive_support_repair_package_plan_20260514_2102/plan_summary.json
docs/natural_evidence_v2/CURRENT_STATE.md
```

next_allowed_action:
```text
Implement artifact-only support-repair contract/extractor/static fixture package.
No Slurm, generation, model scoring, training, Llama, FAR, sanitizer, payload
diversity, or paper-facing claim is unlocked.
```

