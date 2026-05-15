# R4 Positive Selectivity Single-Submission Route

Date: 2026-05-15T03:20:00Z

## Decision

This route authorizes exactly one H200/pomplun Slurm array submission for the
reviewed R4 positive selectivity small dev diagnostic.

Authorized allowlist entry:

```text
v2_r4_positive_selectivity_dev_diagnostic_h200
```

Authorized command:

```text
sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch
```

## Preconditions

All required preconditions have passed:

- route scope review:
  `PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT`
- local wrapper plan-only:
  `PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY`
- remote wrapper plan-only:
  `PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_WRAPPER_PLAN_ONLY`
- local/remote hash mismatch count: `0`
- remote zero-enabled allowlist safety: `PASS`
- active Chimera jobs before route record: none observed
- Hermes TG/email pre-submit notification: required before `sbatch`

## Scope

- Wrapper:
  `scripts/natural_evidence_v2/slurm/r4_positive_selectivity_dev_diagnostic_h200.sbatch`
- Partition/QoS/account/GPU:
  `pomplun` / `pomplun` / `cs_yinxin.wan` / `gpu:h200:1`
- Time limit: `30-00:00:00`
- Array: `0-3%4`
- Conditions generated: `protected`, `raw`, `task_only`
- Decoder controls: `wrong_key`, `wrong_payload`
- Primary decode: `format_scrub=all`
- Contract: same-contract `a55e`
- Payload diversity tested: `false`
- Llama tested: `false`
- Paper-facing claim: `false`

## Required Submission Procedure

1. Send Hermes TG/email pre-submit notification.
2. Enable exactly `v2_r4_positive_selectivity_dev_diagnostic_h200`.
3. Verify exactly one enabled allowlist entry locally and remotely.
4. Submit exactly one H200/pomplun Slurm array job.
5. Disable the allowlist entry immediately after `sbatch` returns.
6. Verify local and remote post-submit allowlist safety with zero enabled entries.
7. Record the Slurm job id and expected output directory.

## Stop Rules

Stop before submission if any of the following occurs:

- any active Chimera job appears that conflicts with this route;
- allowlist enabled entries are not exactly the authorized entry;
- local/remote hashes diverge;
- Hermes TG/email pre-submit notification fails;
- wrapper or route config differs from reviewed artifacts.

