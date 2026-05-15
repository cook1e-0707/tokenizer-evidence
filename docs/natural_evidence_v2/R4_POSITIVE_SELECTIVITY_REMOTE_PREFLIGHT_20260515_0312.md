# R4 Positive Selectivity Remote Preflight

Date: 2026-05-15T03:12:00Z

## Decision

Status:

```text
PASS_R4_POSITIVE_SELECTIVITY_REMOTE_PREFLIGHT_NO_SUBMIT
```

This preflight synchronized the reviewed selectivity route files to Chimera and
validated the wrapper remotely in plan-only mode. No Slurm job was submitted, no
allowlist entry was enabled, no generation was started, and no claim is
unlocked.

## Remote Plan-Only Result

Remote output directory:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_dev_wrapper_remote_plan_smoke_20260515_0312
```

Local synced copy:

```text
results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_remote_plan_smoke_20260515_0312
```

Wrapper plan-only status:

```text
PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY
```

Toy decode result:

```text
protected accepts = 1
wrong-key accepts = 0
wrong-payload accepts = 0
```

Remote zero-enabled allowlist safety:

```text
PASS
```

Active Chimera jobs for the user during preflight:

```text
none observed
```

## Local/Remote Hash Check

The following local and remote hashes matched:

| File | SHA256 |
| --- | --- |
| `configs/natural_evidence_v2/run_allowlist.yaml` | `0fdfde71b2ce59e0d922e5e4c72432f31159cb377655f03f1829fa08eb6d1bc1` |
| `configs/natural_evidence_v2/r4_positive_selectivity_dev_diagnostic_route.yaml` | `ca590a6057b33e063497a07e1ca52035a7522426d2d72b24378bb0b7e6d92060` |
| `docs/natural_evidence_v2/CURRENT_STATE.md` | `bbc4172acf83b6f7fa38c2dd1b0a566d2268d6672e7abd4a1ca23272b9d78926` |
| `scripts/natural_evidence_v2/decode_r4_positive_support_window_correlation.py` | `e7e4b210896b2e78390f8baaf7a0045a78900b58a8223884eda1dfade2c76ca4` |
| `scripts/natural_evidence_v2/validate_r4_positive_selectivity_dev_diagnostic_route.py` | `74b485c12e2ce700c8125f516a5ed5497e7c92ff35f163002fa056d977783626` |
| `scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch` | `2ae033cf2190884cb5a38cd09ecbb0c1d6aeb3ef313352e7e4bbd5e07c595807` |
| `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158/event_window_bank.json` | `0b3624281bce0637667e629b7d940eaae579c00efadce8be017fd03784a646f6` |
| `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/dev_prompts.jsonl` | `c22134d3f3d8510a07ca6104a1278abb64f94247c40758a14b6e97fbbdb856d6` |
| `results/natural_evidence_v2/status/gate_status.json` | `da571e81e6313ee7e0f65ded937ff6da8f2f7763c74c110295cadd8e6c73d01d` |
| `results/natural_evidence_v1/status/gate_status.json` | `8e6471bdd2fbf87bfd329b5a143cc89510323ead5f58a663f69e8b275f1200b4` |

## Next Allowed Action

Record the single-submission route, send Hermes TG/email pre-submit
notification, enable exactly
`v2_r4_positive_selectivity_dev_diagnostic_h200`, submit exactly one H200/pomplun
Slurm array job, and disable the allowlist entry immediately after `sbatch`
returns.

