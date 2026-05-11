# R3.2 Submission Blocker: Chimera Preflight Still Unsafe

timestamp_utc:
2026-05-11T09:01:00Z

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

status:
BLOCK_R3_2_SUBMISSION_PREFLIGHT_UNSAFE_NO_SLURM

controlling_next_allowed_action:
Enable the existing R3.2 allowlist entry for exactly one reviewed Slurm command
only after the next required TG/email notification path is satisfied, then
submit exactly one allowlisted Chimera Slurm job.

notification_gate:
PASS. The Hermes 2026-05-11 09:01 notification artifact reports Telegram and
email both sent successfully:
`results/natural_evidence_v1/status/hermes_reports/20260511_0901_scheduled_tick_notification.json`.

blocker:
The notification gate is satisfied, but the Chimera submission path is still
not safe. The remote checkout at `$HOME/tokenizer-evidence` is missing the
reviewed R3.2 files required for the approved wrapper:

- `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
- `scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`
- `scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py`
- `configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml`
- `docs/natural_evidence_v2/CURRENT_STATE.md`

The remote allowlist also does not contain the required
`v2_r3_2_qwen_locked_scale_eval` entry, and still has forbidden Llama entries
enabled while `llama_allowed=false`:

- `build_llama_v2_bucket_bank`
- `llama_v2_wp5_train_and_score`

The local checkout is also not submission-safe because it still has
`llama_v2_wp6_e2e_eval` enabled while `llama_allowed=false`.

state_changing_action:
Recorded this R3.2 submission preflight blocker only. No allowlist entry was
enabled, no Slurm job was submitted, and no generation or Qwen E2E rerun was
started.

forbidden_actions_confirmed:
No training, Llama, same-family null, sanitizer benchmark, FAR aggregation,
paper-facing positive claim, generation, Qwen E2E rerun, or Chimera login-node
CPU/GPU work was started by this tick.

next_allowed_action:
Repair the Chimera submission preflight without running CPU/GPU work: reconcile
the remote checkout to the already reviewed R3.2 files, add/verify the disabled
R3.2 allowlist entry on Chimera, and ensure both local and remote allowlists
have no enabled forbidden Llama/sanitizer/FAR entries. After that, a later
notified tick may enable only `v2_r3_2_qwen_locked_scale_eval`, submit exactly
one reviewed Slurm job, and disable the entry after submission.
