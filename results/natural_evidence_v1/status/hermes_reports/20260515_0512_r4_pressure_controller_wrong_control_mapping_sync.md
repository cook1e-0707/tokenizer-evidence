# Hermes Sync: R4 Pressure-Controller Wrong-Control Mapping

Timestamp UTC: `2026-05-15T05:12:00Z`

Status:
`PASS_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_AND_FULL_WRAPPER_REVIEW_NO_SUBMIT`

Codex completed the artifact-only wrong-control mapping and full wrapper review:

```text
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_REVIEW_20260515_0512.md
results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_wrong_control_mapping_20260515_0512/wrong_control_mapping_summary.json
```

Mapping:

```text
controlled_protected -> committed controller target ids
wrong_payload_controlled -> complement controller target ids
wrong_key_controlled -> coordinate_hash_v1 using salt r4_wrong_key_controller_v1
scorer/verifier target remains committed target ids in all conditions
no transcript-conditioned mapping
no post-hoc key/payload remap
```

Validation:

```text
pytest: 20 passed, 2 skipped
route validator: PASS
wrapper plan-only: PASS
full-mode guard without ALLOW_PRESSURE_CONTROLLER_SCORING=1: exit code 2
allowlist safety: PASS with zero enabled entries
Hermes TG/email notification: SENT_ALL_REQUIRED_CHANNELS
```

Current phase:

```text
V2_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_AND_FULL_WRAPPER_REVIEW_PASS_NO_SUBMIT
```

Current blocker:

```text
BLOCK_R4_PRESSURE_CONTROLLER_REMOTE_PREFLIGHT_NEXT
```

Next allowed action:

```text
Remote sync and remote preflight only: remote wrapper plan-only validation,
local/remote hash preflight, remote zero-enabled allowlist safety, and active-job preflight.
```

No Slurm/model scoring/generation/training/Llama/null/sanitizer/FAR/payload-diversity/paper-claim action is unlocked by this sync.
