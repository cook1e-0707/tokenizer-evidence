# Hermes sync: R4 dev diagnostic 853691 reviewed

Phase: `V2_R4_DEV_853691_COMPLETED_GATE_FAIL_ARTIFACT_ONLY_REPAIR_PLANNING`

Slurm job `853691` completed cleanly:

- all 4 H200 array shards completed;
- all task exit codes were `0:0`;
- output synced locally to
  `results/natural_evidence_v2/status/r4_cover_natural_dev_diagnostic_h200_array_853691/`.

Gate review:

- protected accepts, no scrub: `0/32`, required `>=28/32`;
- protected accepts, `format_scrub=all`: `0/32`, required `>=26/32`;
- raw accepts: `0/32`;
- task-only accepts: `0/32`;
- wrong-key accepts: `0/32`;
- wrong-payload accepts: `0/32`;
- forbidden public surface total across decode rows: `237`, required `0`;
- protected-vs-raw shallow structural AUC: `0.624`, gate `<=0.60`.

Interpretation:

This is a clean Slurm completion and a failed R4 dev diagnostic. It is not a
provider/model crash. It is a positive-channel failure: phrase-surface matches
exist, but the observed coordinate polarities do not align with the protected
`a55e` codeword.

Artifacts:

- review summary:
  `results/natural_evidence_v2/status/r4_cover_natural_dev_diagnostic_h200_array_853691/review/r4_dev_diagnostic_853691_review_summary.json`;
- failure attribution:
  `results/natural_evidence_v2/status/r4_cover_natural_dev_diagnostic_h200_array_853691/failure_attribution/failure_attribution_summary.json`.

Next allowed action:

Artifact-only R4 repair planning: design a trainable cover-natural surface
target path and an R4 teacher-forced target-mass probe before any further
generation Slurm job.

Still not unlocked:

- locked-scale generation;
- Llama;
- same-family null;
- sanitizer;
- FAR aggregation;
- payload-diversity claims;
- paper-facing positive claims.
