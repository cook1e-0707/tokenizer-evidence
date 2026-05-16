# Hermes sync: R4 reliability dev generation 867621 reviewed

phase:
`V2_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_FAILED_POSITIVE_GATE_ARTIFACT_REVIEW`

summary:
```text
Job 867621 completed cleanly on H200/pomplun, but failed the protected positive gate.

protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw/task-only/wrong-key/wrong-payload accepts: 0/32 each
protected forbidden public surface count: 0
coordinate-unique selected surface matches in protected: 0
```

artifact-only failure analysis:
```text
root_cause: free_generation_transfer_failure_surface_absent
protected coordinate-unique bank surface matches: 0
protected rows with any coordinate-unique bank surface: 0
protected duplicate response hash rows: 508
protected max duplicate response hash count: 27
protected rows with repeated sentence/clause units: 2001/2048
Create a plan protected occurrences: 44500
Prepare a schedule protected occurrences: 4234
Prepare a budget protected occurrences: 11709
Prepare a plan protected occurrences: 11691
```

recorded:
```text
docs/natural_evidence_v2/R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_review/
results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_failure_analysis_20260516/
results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_review_allowlist_safety_20260516.json
```

next_allowed_action:
Artifact-only repair or pivot route planning only. Do not rerun this route unchanged, lower gates, add 867621-observed phrases to the bank, submit new Slurm, start training, Llama, sanitizer, FAR, payload diversity, or paper-facing claims until a new reviewed route decision and fresh preflight are recorded.
