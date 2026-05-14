# Hermes/Codex Sync: R4 Positive 859277 Failure Analysis Recorded

phase:
`V2_R4_POSITIVE_DEV_DIAGNOSTIC_859277_FAILURE_ANALYSIS_RECORDED_NO_RESUBMIT`

blocker:
`BLOCK_R4_POSITIVE_859277_ZERO_EVENT_SUPPORT_FAILURE_ANALYSIS_RECORDED`

summary:
```text
Job 859277 completed cleanly on H200/pomplun, but the positive event-bank
diagnostic failed. Protected accepts were 0/32 under format_scrub=all and 0/32
under no-scrub. Raw/task-only/wrong-key/wrong-payload controls were also 0/32.
The decisive failure is zero extracted frozen phrase events in every block, so
support, distinct coordinates, keyed score, and margins are all zero.

Artifact-only failure analysis has been recorded. Do not resubmit the same
859277 route unchanged. A new reviewed repair or pivot route is required before
any new generation or Slurm submission.
```

artifacts:
```text
results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_review/review.md
results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_failure_analysis/failure_analysis.md
results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277_failure_analysis/failure_analysis_summary.json
docs/natural_evidence_v2/CURRENT_STATE.md
```

verification:
```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_evidence_contract.py \
  tests/natural_evidence_v2/test_r4_keyed_correlation_decoder.py \
  tests/natural_evidence_v2/test_r4_positive_event_bank_precommit.py \
  tests/natural_evidence_v2/test_r4_positive_dev_diagnostic_route.py \
  tests/natural_evidence_v2/test_r4_positive_phrase_event_extractor.py \
  tests/natural_evidence_v2/test_r4_positive_keyed_correlation_decode.py -q

28 passed
```

next_allowed_action:
Record a new reviewed repair/pivot route before any new generation or Slurm
submission. No positive claim, Llama, FAR, sanitizer, payload diversity, or
same-family claim is unlocked by this artifact.

