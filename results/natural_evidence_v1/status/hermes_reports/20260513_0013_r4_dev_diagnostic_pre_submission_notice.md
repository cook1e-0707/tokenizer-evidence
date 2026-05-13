# Hermes sync: R4 dev diagnostic pre-submission notice

Phase: `V2_R4_DEV_DIAGNOSTIC_REMOTE_SYNC_AND_SINGLE_JOB_PREFLIGHT`

Codex is proceeding under the user's standing approval for the R4 cover-natural
ECC dev diagnostic.

Preflight status:

- Local zero-enabled allowlist safety: PASS.
- Remote zero-enabled allowlist safety on Chimera: PASS.
- Local/remote control-plane hash comparison: PASS.
- Chimera active jobs for this user: none.
- Required WP5-R2 protected/task-only adapter configs: present.
- R4 H200 wrapper plan-only smoke: PASS for shards `shard_00` through
  `shard_03`.
- Wrapper now records explicit `replicate_group_id` per shard, preventing
  duplicate decode block ids across shards.

Next Codex action:

- Enable exactly one reviewed allowlist entry:
  `v2_r4_cover_natural_dev_diagnostic_h200`.
- Run local and remote single-enabled allowlist preflight.
- Submit exactly one Chimera Slurm H200 array job if those checks pass.
- Immediately disable the allowlist entry after `sbatch` returns.

Still not unlocked:

- training;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- payload-diversity claims;
- paper-facing positive claims.
