# R3.2 full-wrapper blocker: payload semantics

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

controlling_tick:
results/natural_evidence_v1/status/hermes_reports/20260511_0545_scheduled_tick.md

files_checked:
- docs/natural_evidence_v2/CURRENT_STATE.md
- results/natural_evidence_v1/status/gate_status.json
- results/natural_evidence_v2/status/gate_status.json
- docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
- docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
- docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md
- docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_WRAPPER_REVIEW_20260511.md
- scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
- scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
- scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py
- scripts/natural_evidence_v2/decode_wp6_payload.py
- scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py

nearest_controlling_spec:
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md

decision:
Do not upgrade or submit the R3.2 full locked-scale generation/eval wrapper in this tick.

blocker:
The next action asks for a reviewed full R3.2 locked-scale generation/eval wrapper,
but the required payload semantics are not unambiguous enough to implement safely.

The recorded R3.2 package scope requires:
- payload grid: P00, P01, P02, P03
- seeds: 17, 23, 29
- 12 cells and 96 protected blocks

The available reviewed generation/decode path is built around the WP5-R2 fixed
prompt-local contract:
- source contract:
  results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/wp4_prompt_local_payload_contract.json
- payload_plus_checksum_hex: a55e
- generator script: scripts/natural_evidence_v2/generate_wp6_e2e_outputs.py
- decoder scripts:
  scripts/natural_evidence_v2/decode_wp6_payload.py
  scripts/natural_evidence_v2/decode_wp6_r1_scale_blocks.py

Those scripts accept a single WP4 contract and do not encode a reviewed mapping
from the R3.2 P00/P01/P02/P03 cell labels to distinct payload bytes/contracts.
Implementing a full wrapper by simply looping the same a55e contract across
P00/P01/P02/P03 would risk recording a misleading reviewed payload grid. Creating
new per-payload contracts would risk changing the locked protocol without a
recorded decision and without a reviewed trainer/adapter relationship for those
payloads.

hard_constraints_observed:
- no training
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claims
- no generation
- no Qwen E2E rerun
- any Chimera CPU/GPU work must use Slurm
- do not run CPU work directly on the Chimera login node
- do not overwrite existing artifacts

state_changing_action:
Recorded this blocker report only. No wrapper was changed, no allowlist entry was
enabled, no notification was sent by this Codex worker, and no Slurm job was
submitted.

next_safe_resolution_needed:
Record an explicit R3.2 payload semantics decision before full-wrapper upgrade:
either R3.2 P00/P01/P02/P03 are cell labels that all intentionally reuse the
fixed a55e WP5-R2 contract, or they are distinct payload contracts with reviewed
contract paths, expected bytes/checksums, and adapter compatibility.

status:
BLOCK_R3_2_FULL_WRAPPER_PAYLOAD_SEMANTICS_AMBIGUOUS_NO_SLURM
