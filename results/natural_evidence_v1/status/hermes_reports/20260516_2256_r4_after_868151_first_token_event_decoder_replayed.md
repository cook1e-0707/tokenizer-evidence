# Hermes/Codex Sync: R4 First-Token Event Decoder Replayed

phase: `V2_R4_AFTER_868151_FIRST_TOKEN_EVENT_DECODER_REPLAYED_QUALITY_REPAIR_NEXT`
status: `FIRST_TOKEN_EVENT_DECODE_RECORDED_ARTIFACT_ONLY_NOT_POSITIVE`

Summary:
- decoder implemented: `scripts/natural_evidence_v2/decode_r4_after_868151_first_token_event_channel.py`
- event rows extracted: `9216`
- event sources: `{'text_fallback_old_transcript': 9216}`
- event statuses: `{'erasure': 8289, 'other': 82, 'target': 845}`
- protected accepts with quality gates: `0/4`
- protected accepts ignoring quality: `4/4`
- raw/task-only/wrong-key/wrong-payload accepts ignoring quality: `0/4` each
- protected forbidden public surface count: `6`
- protected duplicate response hash count: `755`

Interpretation:
- This remains artifact-only and does not reclassify `868151` as positive.
- Future positives require token-id event traces, not old text fallback.
- Next work is quality repair planning before any Slurm submission.

next_allowed_action: Artifact-only first-token event quality-repair planning: technical literal audit, duplicate-output repair, token-id event trace wrapper planning. No Slurm until reviewed route/preflight passes.
