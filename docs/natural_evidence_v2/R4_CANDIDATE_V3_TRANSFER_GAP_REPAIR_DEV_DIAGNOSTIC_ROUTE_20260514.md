# R4 Candidate v3 Transfer-Gap Repair Dev Diagnostic Route

Timestamp UTC: 2026-05-14T07:15:00Z

## Route

Submit one H200/pomplun Slurm array job for a small Qwen dev generation
diagnostic using the transfer-gap repaired prompt bank.

This route is diagnostic only. It does not unlock a paper-facing positive claim.

## Inputs

- Source protected adapter: pressure-relaxation grid job `857764`, arm
  `B_ceiling_lambda_0_5`.
- Repaired prompt bank:
  `results/natural_evidence_v2/prompts/r4_candidate_v3_transfer_gap_repaired_dev_prompts_20260514_0710/dev_prompts.jsonl`.
- Wrapper:
  `scripts/natural_evidence_v2/slurm/r4_candidate_v3_transfer_gap_repair_dev_diagnostic_h200.sbatch`.
- Contract: same-contract `a55e`.
- Primary decoder scrub mode: `format_scrub=all`.

## Scope

Allowed after preflights pass:

- exactly one H200/pomplun Slurm array job;
- Qwen only;
- dev prompts only;
- protected/raw/task-only generation;
- wrong-key and wrong-payload decode controls.

Still not allowed by this route:

- training;
- Qwen E2E rerun outside this diagnostic wrapper;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- payload-diversity claim;
- paper-facing positive claim.

## Preflight Evidence

Local preflight passed before this route was recorded:

- transfer-gap repair plan validation: `PASS`;
- artifact-only repair package: `PASS`;
- repaired prompt bank: `PASS`, `2048` prompts, forbidden violations `0`;
- wrapper syntax: `bash -n` passed;
- all four local plan-only shards: `PLAN_ONLY_PASS`;
- local allowlist safety: `PASS` with zero enabled entries.

Required before submission:

- remote sync/hash preflight for wrapper, config, and prompt bank;
- remote all-shard plan-only validation;
- remote zero-enabled allowlist safety;
- active-job preflight;
- Hermes TG/email pre-submit notification;
- exactly one allowlist entry enabled:
  `v2_r4_candidate_v3_transfer_gap_repair_dev_diagnostic_h200`;
- immediate allowlist disablement after `sbatch`.

## Pass/Fail Review Gates

The eventual result must be reviewed before any downstream unlock:

- protected accepts under `format_scrub=all`;
- protected accepts under no scrub;
- raw/task-only/wrong-key/wrong-payload accepts;
- prefix-family hit rates;
- target-vs-other phrase support;
- forbidden technical public-surface count;
- contextual `bucket`/`coordinate` counts;
- protected-vs-raw shallow structural AUC.

If protected remains `0/32`, record a transfer-gap failure and do not rerun
blindly.
