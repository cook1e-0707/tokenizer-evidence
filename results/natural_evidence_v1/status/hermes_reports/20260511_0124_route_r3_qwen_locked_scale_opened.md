# Route R3 Qwen locked-scale route opened

timestamp_utc:
2026-05-11T01:24:00Z

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

canonical_route_decision:
Route R3 is opened as the Qwen v2 controlled-natural micro-slot
paper-readiness route. This does not start Llama, same-family nulls, sanitizer,
FAR aggregation, or paper-facing claims.

actions_taken:
- Wrote `docs/natural_evidence_v2/WP6_R2_OPTION_B_852426_CANONICAL_REVIEW.md`.
- Wrote `results/natural_evidence_v2/status/wp6_r2_option_b_852426_canonical_summary.json`.
- Wrote `docs/natural_evidence_v2/REPEATED_COORDINATE_DECODER_SPEC.md`.
- Updated v1/v2 gate state to `V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED`.

canonical_852426_status:
- Qwen-only positive diagnostic.
- Protected robust-block accepts at budget 64: `7/8`.
- Raw/task-only/wrong-key/wrong-payload robust-block accepts at budget 64:
  `0/8` each.
- Accepted-block minimum support: `26`.
- Accepted-block minimum majority margin: `5`.
- Forbidden public surface count: `0`.

claim_control:
- `qwen_v2_scale_allowed=true`
- `llama_allowed=false`
- `same_family_null_allowed=false`
- `sanitizer_allowed=false`
- `far_aggregation_allowed=false`
- `paper_claim_allowed=false`

forbidden_actions_not_taken:
- no Slurm submission
- no training
- no generation
- no Qwen E2E rerun
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claim
- no adoption of out-of-band Llama artifacts

next_allowed_action:
Prepare the Route R3.2 Qwen locked-scale package and wrapper review only.
The intended R3.2 package is payloads `P00/P01/P02/P03`, seeds `17/23/29`,
`8` blocks per cell, arms `protected/raw/task_only/wrong_key/wrong_payload`,
primary budget `64` with `16/32` diagnostics, protected accept gate `>=80/96`,
each null arm `0/96`, support `>=16`, majority margin `>=3`, and forbidden
public surface count `0`.

Do not submit Slurm until the R3.2 wrapper, allowlist, precommit, notification,
and gate review are recorded.
