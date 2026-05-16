# Hermes/Codex Sync: 868212 reliability/duplicate repair preflight failed closed

phase:
`V2_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_FAILED_NO_SUBMIT`

summary:
```text
Codex implemented and ran the artifact-only reliability/duplicate repair
preflight required after the 868212 diagnostic.

Preflight artifact:
- results/natural_evidence_v2/status/r4_after_868212_reliability_duplicate_repair_preflight_20260516/

Status:
- FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT

Codebook failures:
- bit 1 active=[26], below min active coordinate count
- bit 1 active=[26], coordinate 26 is sole active coordinate
- bit 3 active=[19], below min active coordinate count
- bit 5 active=[8], below min active coordinate count
- bit 6 active=[4], below min active coordinate count

Duplicate taxonomy:
- duplicate hash groups: 2908
- duplicate extra rows: 4424
- cross-arm duplicate groups: 1621
- cross-shard duplicate groups: 2141

Tests:
- 14 passed, 1 skipped
```

next_allowed_action:
Artifact-only repaired codebook/duplicate-policy construction or reviewed pivot
route. Do not submit another Slurm generation/scoring/training job until the
repair preflight passes or a new reviewed pivot supersedes it.
