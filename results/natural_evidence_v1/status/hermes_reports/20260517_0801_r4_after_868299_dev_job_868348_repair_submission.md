# Hermes sync: R4 after-868299 dev repair job submitted

phase:
`V2_R4_AFTER_868299_DEV_DIAGNOSTIC_REPAIRED_JOB_868348_SUBMITTED_MONITOR_ONLY`

summary:

```text
Patched the 868313 runtime allowlist-race failure:
- submission preflight allowlist checks remain strict;
- runtime shard validation now skips only allowlist enabled-state checks;
- runtime still verifies the reviewed allowlist entry and command pattern.

Local tests:
- test_r4_after_868299_first_token_event_dev_diagnostic_route.py: 5 passed
- related route/binding tests: 12 passed
- local zero-enabled allowlist safety: PASS
- local exactly-one submission preflight: PASS

Remote preflights:
- remote zero-enabled allowlist safety: PASS
- remote route validation: PASS
- remote wrapper plan-only smoke: PASS
- remote exactly-one submission preflight: PASS

Submitted repaired replacement:
- job_id: 868348
- job_name: nat-ev-v2-r4dev
- array: 0-31%4
- H200: pomplun / pomplun / cs_yinxin.wan

Post-submit:
- local allowlist safety: PASS
- remote allowlist safety: PASS
```

next_allowed_action:

```text
Monitor Slurm array 868348. After terminal completion, sync artifacts and run
the first-token event dev diagnostic review. Do not adopt partial 868313 output.
```

not_allowed:

```text
paper-facing positive claim
training
Llama
same-family null
sanitizer
FAR
payload diversity claim
```
