# R3.2 submission preflight blocker

timestamp_utc = 2026-05-11T12:04:54Z

phase = V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

The 2026-05-11T12:02 Hermes TG/email notification path passed:

- notification_json = results/natural_evidence_v1/status/hermes_reports/20260511_1202_scheduled_tick_notification.json
- notification_status = SENT_ALL_REQUIRED_CHANNELS

The requested next action was not safe to execute. No allowlist entry was
enabled, no Slurm job was submitted, and no generation/Qwen E2E work was
started.

Blocking preflight findings:

- Local `configs/natural_evidence_v2/run_allowlist.yaml` still has enabled
  forbidden entry `llama_v2_wp6_e2e_eval` while `llama_allowed=false`.
- Remote Chimera checkout is missing or has empty reviewed R3.2 files:
  - `scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
  - `scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`
  - `scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py`
  - `configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml`
  - `docs/natural_evidence_v2/CURRENT_STATE.md`
- Remote `configs/natural_evidence_v2/run_allowlist.yaml` lacks
  `v2_r3_2_qwen_locked_scale_eval`.
- Remote allowlist still has enabled forbidden Llama entries:
  `build_llama_v2_bucket_bank` and `llama_v2_wp5_train_and_score`.

Required next action:

Repair the Chimera submission preflight without running CPU/GPU work: reconcile
the remote checkout to the already reviewed R3.2 files, add/verify the disabled
R3.2 allowlist entry on Chimera, and ensure both local and remote allowlists
have no enabled forbidden Llama/sanitizer/FAR entries. After that, a later
notified tick may enable only `v2_r3_2_qwen_locked_scale_eval`, submit exactly
one reviewed Slurm job, and disable the entry after submission.
