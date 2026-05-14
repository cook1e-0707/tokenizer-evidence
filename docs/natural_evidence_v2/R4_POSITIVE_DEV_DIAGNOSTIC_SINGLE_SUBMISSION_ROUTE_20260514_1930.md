# R4 Positive Dev Diagnostic Single Submission Route

Timestamp: `2026-05-14T19:30:51Z`

## Decision

Status: `PASS_R4_POSITIVE_DEV_DIAGNOSTIC_SINGLE_SUBMISSION_ROUTE_REVIEWED`

This route authorizes exactly one H200/pomplun Slurm array submission for the
R4 positive event-bank dev diagnostic.

## Scope

- allowlist entry: `v2_r4_positive_dev_diagnostic_h200`
- command:
  `sbatch --export=ALL,ALLOW_STATIC_DEV_KEYS=1 scripts/natural_evidence_v2/slurm/r4_positive_dev_diagnostic_h200.sbatch`
- partition/qos/account: `pomplun` / `pomplun` / `cs_yinxin.wan`
- GPU: `gpu:h200:1`
- time limit: `30-00:00:00`
- array: `0-3%4`
- model family: Qwen only
- prompts: dev prompt bank only
- blocks: `32`
- prompts per block: `64`
- generation arms: `protected`, `raw`, `task_only`
- decoder controls: `wrong_key`, `wrong_payload`
- primary decode: `format_scrub=all`
- secondary decode: `format_scrub=none`
- same-contract only: `a55e`
- payload diversity tested: `false`
- paper-facing claim allowed: `false`

The route uses `ALLOW_STATIC_DEV_KEYS=1` because the current dev precommit
package is a diagnostic keyed-correlation package with committed static dev key
materials in code and commitments in the precommit manifest. This does not
create a paper-facing or production keying claim.

## Prerequisites Already Satisfied

- Full generation/decode wrapper implemented and locally reviewed.
- Focused pytest passed: `13` tests.
- Local plan-only wrapper validation passed.
- Static keyed-decoder fixture passed: protected `1/1`, wrong-key `0/1`,
  wrong-payload `0/1`.
- Remote plan-only wrapper validation passed.
- Remote zero-enabled allowlist safety passed.
- Local/remote hash preflight passed.
- Active-job preflight passed: no active Chimera jobs.
- Required remote precommit artifacts, dev prompt bank, protected adapter, and
  task-only adapter are present.

## Submission Rules

1. Send Hermes TG/email pre-submit notification.
2. Re-run final local allowlist safety before enablement.
3. Enable exactly `v2_r4_positive_dev_diagnostic_h200`.
4. Sync allowlist to Chimera.
5. Re-run remote exactly-one-enabled preflight.
6. Submit exactly one Slurm array job.
7. Immediately disable the allowlist entry locally and remotely after `sbatch`
   returns.
8. Record the job id, command, output directory, and post-submit allowlist
   safety.

## Still Not Unlocked

This route does not unlock training, Llama, same-family null, sanitizer, FAR
aggregation, payload diversity, or paper-facing positive claims.

