# R4 Candidate v3 Gain-4 Dev Generation Diagnostic Route

Timestamp UTC: 2026-05-13T23:59Z

## Decision

The R4 candidate v3 adapter-gain sweep showed that the fixed prefix-native
candidate passes teacher-forced gates when the protected adapter gain is strong
enough. The first gain passing the main gate is `2.0`; the first gain passing
the main gate plus per-prefix lift protection is `4.0`.

This route prepares the next diagnostic: a small Qwen dev generation diagnostic
using protected adapter gain `4.0`, same-contract `a55e`, and the existing R4
format-scrub/null-control decoder path.

This route is diagnostic only. It does not authorize training, Llama,
same-family nulls, sanitizer benchmark, FAR aggregation, payload-diversity
evaluation, or paper-facing positive claims.

## Scope

- Model family: Qwen only.
- Split: dev prompts only.
- Contract: same-contract `a55e`; payload diversity is not tested.
- Protected generation uses `PROTECTED_ADAPTER_GAIN=4.0`.
- Raw and task-only conditions are not gain-scaled.
- Decode modes: `format_scrub=all` and `format_scrub=none`.
- Controls: protected, raw, task-only, wrong-key, wrong-payload.

## Reviewed Wrapper

Allowlist entry:

```text
v2_r4_candidate_v3_gain4_dev_diagnostic_h200
```

Slurm command:

```text
sbatch --export=ALL,PROTECTED_ADAPTER_GAIN=4.0 scripts/natural_evidence_v2/slurm/r4_cover_natural_dev_diagnostic_h200.sbatch
```

The wrapper uses `pomplun` / `cs_yinxin.wan` / `gpu:h200:1` with time limit
`30-00:00:00`. It is a single Slurm array submission (`0-3%4`), producing four
H200 shard tasks for the 2048-prompt dev diagnostic.

## Validation Completed

- Generator `py_compile`: pass.
- H200 wrapper `bash -n`: pass.
- Plan-only generation smoke with `--protected-adapter-gain 4.0`: pass.
- Zero-enabled allowlist safety: pass.

Plan-only smoke artifact:

```text
results/natural_evidence_v2/status/r4_candidate_v3_gain4_dev_diagnostic_plan_smoke_20260513/r4_generation_plan_summary.json
```

## Submission Rules

Before submission:

1. Send Hermes TG/email notification.
2. Re-run local allowlist safety and confirm zero enabled entries.
3. Sync reviewed files to Chimera and confirm remote file hashes.
4. Enable exactly one allowlist entry:
   `v2_r4_candidate_v3_gain4_dev_diagnostic_h200`.
5. Submit exactly one H200 Slurm array job.

Immediately after `sbatch` returns:

1. Disable the allowlist entry.
2. Re-run local and remote allowlist safety and require zero enabled entries.
3. Record submission artifacts with the array job id.

## Review Gate After Completion

This route can only pass as a dev diagnostic if review finds:

- protected accepts with `format_scrub=all` `>= 26/32`;
- raw accepts `0/32`;
- task-only accepts `0/32`;
- wrong-key accepts `0/32`;
- wrong-payload accepts `0/32`;
- forbidden public technical surface count `0`;
- duplicate generated-output hashes `0`;
- duplicate decode-row hashes `0`;
- protected-vs-raw shallow structural classifier AUC `<= 0.60`.

If these do not pass, do not proceed to locked-scale, training claims, FAR,
Llama, sanitizer, or paper-facing claims.

## Status

`READY_FOR_SINGLE_H200_GAIN4_DEV_GENERATION_DIAGNOSTIC_SUBMISSION_AFTER_NOTIFICATION_AND_REMOTE_HASH_PREFLIGHT`
