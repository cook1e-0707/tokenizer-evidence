# Hermes sync: R4 after-868299 dev route local preflight

phase:
V2_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_LOCAL_PREFLIGHT_PASSED_REMOTE_PREFLIGHT_NEXT

summary:
```text
Recorded the reviewed 32-block first-token event dev diagnostic route after the
868299 quality-repair confirmation pass.

Local route validation: PASS.
Wrapper plan-only smoke: PASS.
Allowlist state: zero enabled entries locally.

Route scope:
- Qwen only
- provider-side keyed first-token event trace channel
- 32 blocks / 32 shards
- protected/raw/task-only generation arms
- wrong-key and wrong-payload decode controls
- duplicate-safe generation policy v2
- contextual forbidden policy v2
- trace binding required

Allocation caveat:
The current reviewed full16 row bank supports four fully unique 1024-row shards.
The 32-block dev diagnostic therefore precommits cyclic reuse of the reviewed
four-shard allocation. This is dev diagnostic only and must not be described as
locked-scale independent evidence.

Gates:
- protected strict accepts >= 28/32
- protected accepts ignoring quality >= 30/32
- raw/task-only/wrong-key/wrong-payload accepts = 0/32 each
- global exact response duplicate = 0
- technical forbidden public surface = 0
- trace binding validity = 100%
- full-phrase decoder remains report-only, not a text-only success claim
```

next_allowed_action:
Run local/remote hash and allowlist preflight for this reviewed route. If both
pass, enable exactly `v2_r4_after_868299_first_token_event_dev_diagnostic_h200`,
submit one H200 array, and immediately disable the allowlist entry.

not_unlocked:
training; Llama; same-family null; sanitizer; FAR; payload diversity; locked
scale claim; paper-facing positive claim.
