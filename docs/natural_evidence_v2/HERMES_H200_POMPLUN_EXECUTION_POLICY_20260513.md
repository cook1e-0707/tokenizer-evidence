# Hermes/Codex H200 Pomplun Execution Policy

Recorded: 2026-05-13T04:56:12Z

## Decision

For upcoming `natural_evidence_v2` GPU work, Hermes and Codex should not submit new A100/DGXA100 jobs unless a later explicit route decision supersedes this policy.

The default Chimera GPU execution target is:

- host: `chimera` via `ssh chimera`;
- partition: `pomplun`;
- account: `cs_yinxin.wan`;
- QoS: `pomplun`;
- GPU: `h200`;
- requested time limit: highest current route limit, `30-00:00:00`.

This policy does not unlock generation, training, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing claims. Those remain gate-controlled and may run only after their prerequisite route decisions, allowlist entries, wrapper reviews, and preflights pass.

## Required H200 Sbatch Header

Use this header pattern for reviewed H200 GPU jobs:

```bash
#SBATCH --partition=pomplun
#SBATCH --account=cs_yinxin.wan
#SBATCH --qos=pomplun
#SBATCH --gres=gpu:h200:1
#SBATCH --time=30-00:00:00
```

## Submission Pattern

Submit through Chimera Slurm only:

```bash
ssh chimera 'cd ~/tokenizer-evidence && sbatch scripts/natural_evidence_v2/slurm/<reviewed_h200_wrapper>.sbatch'
```

Do not run tokenizer/model scoring, generation, or training directly on the login node.

## Reviewed H200 Wrapper Examples

Current H200 wrappers that demonstrate the required resource pattern:

- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_shard_array_h200.sbatch`
- `scripts/natural_evidence_v2/slurm/r4_cover_natural_dev_diagnostic_h200.sbatch`
- `scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch`
- `scripts/natural_evidence_v2/slurm/r4_prefix_native_surface_mass_score_h200.sbatch`

The current active route remains blocked on artifact-only R4 prefix-native scorer/candidate boundary repair after job `853894`; no new scoring job is authorized by this execution policy alone.
