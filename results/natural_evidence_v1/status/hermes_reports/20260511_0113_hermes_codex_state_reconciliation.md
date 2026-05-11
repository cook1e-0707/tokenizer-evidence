# Hermes/Codex state reconciliation

timestamp_utc:
2026-05-11T01:13:43Z

phase:
V2_WP6_R2_OPTION_B_SCALE_GATE_852426_REVIEWED_PASS_HOLD_FOR_NEXT_ROUTE

canonical_state:
WP6-R2 Option B job `852426` remains the latest formally reviewed route result.
The Qwen robust-block scale gate passed, and the canonical next action remains
to stop until a new route is explicitly recorded.

conflict_observed:
Read-only Chimera `sacct` inspection found out-of-band Llama-related jobs after
`852426`: `852810`, `852811`, `852844`, `852853`, and `852881`. Remote
artifacts exist under
`/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/llama_migration/`.
Local worktree inspection also found noncanonical FAR/Llama/sanitizer artifacts
and scripts.

resolution:
- wrote `docs/natural_evidence_v2/HERMES_CODEX_STATE_RECONCILIATION_20260511.md`
- disabled the `build_llama_v2_bucket_bank` allowlist entry
- synchronized v1/v2 gate booleans so Llama, same-family nulls, sanitizer, FAR
  aggregation, and paper claims remain disabled
- marked out-of-band Llama/FAR/sanitizer artifacts as quarantined and
  noncanonical

actions_taken:
- read Hermes reports, v1/v2 gate state, allowlist, and Chimera `sacct`
- updated control-plane state only

forbidden_actions_not_taken:
- no Slurm submission
- no training
- no generation
- no Qwen E2E rerun
- no Llama route adoption
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claim
- no artifact overwrite

next_allowed_action:
Stop until a new route is explicitly recorded. Do not use out-of-band
Llama/FAR/sanitizer artifacts for formal progress or claims without a separate
route decision, provenance review, allowlist review, and Hermes/Codex state
update.
