# Hermes/Codex Sync: R4 After 868212 Repaired First-Token Plan Validated

timestamp_utc: 2026-05-16T23:58:37Z

phase:
V2_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATED_NO_SUBMIT

summary:
```text
Artifact-only repair advanced past the failed 12-coordinate duplicate/codebook preflight.

Completed:
- Built a full16 duplicate-safe row allocation from the reviewed reliability score rows.
- Precommitted a repaired first-token event codebook/decoder/duplicate policy.
- Validated the repaired plan with no Slurm submission and no generation/model scoring/training.
- Tests: 17 passed, 1 skipped.

Key facts:
- source diagnostic job: 868212 remains diagnostic-only, not reclassified.
- allocation rows: 4096.
- shards: 4.
- rows per coordinate per shard: 64.
- selected coordinates: 16.
- min active coordinates per bit: 2.
- coordinate 26 is no longer a singleton payload-bit coordinate.
- locked-scale global duplicate gate is fixed at 0.
```

artifacts:
```text
full16 allocation:
  results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/
repaired precommit:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/
plan validation:
  results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_plan_validation_20260516/
current state:
  docs/natural_evidence_v2/CURRENT_STATE.md
gate status:
  results/natural_evidence_v1/status/gate_status.json
  results/natural_evidence_v2/status/gate_status.json
```

next_allowed_action:
```text
Artifact-only update/validate the full generation/decode wrapper, config, route
validator, and allowlist preflight so they consume the repaired full-16
precommit/allocation instead of the superseded 12-coordinate pivot.

No Slurm submission is allowed until that reviewed wrapper route validation
passes.
```

not_started:
```text
Slurm submission
generation
model scoring
training
Llama
same-family null
sanitizer
FAR aggregation
paper-facing positive claim
```
