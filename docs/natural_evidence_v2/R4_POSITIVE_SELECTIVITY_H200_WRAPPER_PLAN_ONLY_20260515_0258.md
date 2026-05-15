# R4 Positive Selectivity H200 Wrapper Plan-Only Review

Date: 2026-05-15T02:58:00Z

## Decision

Status:

```text
PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY
```

This is still a no-submit review. No Slurm job was submitted, no allowlist entry
was enabled, no generation was started, no training was started, and no positive
claim is unlocked.

## Scope

Wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch
```

Disabled allowlist entry added for a future reviewed single-submission route:

```text
v2_r4_positive_selectivity_dev_diagnostic_h200
```

The wrapper is bound to:

- route config:
  `configs/natural_evidence_v2/r4_positive_selectivity_dev_diagnostic_route.yaml`
- selectivity package:
  `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158`
- prompt-policy package:
  `results/natural_evidence_v2/prompts/r4_positive_selectivity_prompt_policy_20260515_0242/dev_prompts.jsonl`
- H200 policy: `pomplun`, account `cs_yinxin.wan`, QoS `pomplun`,
  `--gres=gpu:h200:1`, `--time=30-00:00:00`
- dev shards: array `0-3`, 512 prompts per shard
- primary decode: `format_scrub=all`
- decoder controls: wrong-key and wrong-payload over protected transcripts

The wrapper requires explicit `ALLOW_STATIC_DEV_KEYS=1` because this diagnostic
uses the static dev key material in the precommitted selectivity package.

## Local Validation

Commands:

```text
bash -n scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch
REPO_HOME=/Users/guanjie/Documents/tokenizer_alignment RUN_ROOT=results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/runroot OUTPUT_DIR=results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258 VALIDATE_PLAN_ONLY=1 ALLOW_STATIC_DEV_KEYS=1 PYTHON_BIN=/Users/guanjie/Documents/tokenizer_alignment/.venv/bin/python bash scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch
uv run python scripts/natural_evidence_v2/check_allowlist_safety.py --require-zero-enabled --output-json results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/local_allowlist_safety.json
```

Results:

```text
bash -n: pass
wrapper plan-only: PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY
local zero-enabled allowlist safety: PASS
```

Plan-only artifacts:

- `results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/plan_validation/wrapper_plan_only_summary.json`
- `results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/plan_validation/generation_plan/r4_generation_plan_summary.json`
- `results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/plan_validation/toy_decode/decode_summary.json`
- `results/natural_evidence_v2/status/r4_positive_selectivity_dev_wrapper_plan_smoke_20260515_0258/local_allowlist_safety.json`

Plan-only toy decode:

```text
protected accepts = 1
wrong-key accepts = 0
wrong-payload accepts = 0
```

## Next Allowed Action

Remote sync and remote preflight only:

```text
Synchronize the reviewed wrapper/config/scripts/artifacts to Chimera, rerun
remote wrapper plan-only validation, local/remote hash preflight, remote
zero-enabled allowlist safety, and active-job preflight.
```

No Slurm submission is unlocked by this review. A later single-submission route
record must explicitly enable exactly one allowlist entry, submit exactly one
H200/pomplun job, and immediately disable the entry after `sbatch` returns.

