# Hermes / Codex Sync: R4 After-868151 First-Token Event Route Validated

phase:
`V2_R4_AFTER_868151_FIRST_TOKEN_EVENT_ROUTE_PLAN_VALIDATION_PASS_DECODER_IMPL_NEXT`

summary:
```text
Codex recorded the first-token / lemma event-channel pivot after the failed
868151 full-phrase generation diagnostic. The route is artifact-only and
validated locally. No Slurm, generation, model scoring, training, Llama, FAR,
sanitizer, or paper-claim action was started.
```

artifacts:
```text
event-channel spec:
  docs/natural_evidence_v2/R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_SPEC_20260516.md
route config:
  configs/natural_evidence_v2/r4_after_868151_first_token_event_channel.yaml
route validator:
  scripts/natural_evidence_v2/validate_r4_after_868151_first_token_event_channel_route.py
event decoder spec:
  results/natural_evidence_v2/precommit/r4_after_868151_first_token_event_channel_precommit_20260516/decoder_spec.json
validation:
  results/natural_evidence_v2/status/r4_after_868151_first_token_event_channel_route_validation_20260516/validation_summary.json
```

validation:
```text
status: PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_CHANNEL_ROUTE_VALIDATION_NO_SUBMIT
tests: 1 passed, 1 skipped
local allowlist safety: PASS zero-enabled
remote allowlist safety: PASS zero-enabled
```

next_allowed_action:
```text
Artifact-only first-token event decoder/extractor implementation and tests.
No Slurm submission until the implementation and wrapper/preflight are
reviewed and recorded.
```
