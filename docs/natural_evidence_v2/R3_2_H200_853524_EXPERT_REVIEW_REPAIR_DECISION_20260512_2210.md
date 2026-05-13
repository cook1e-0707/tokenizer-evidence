# R3.2 H200 853524 expert review and repair decision

Date: 2026-05-12

## Scope

This is the expert review / artifact-only repair decision for completed H200
array job `853524`, following the recorded failure attribution and repair
decision package.

No generation, rerun, training, Llama, same-family null, sanitizer benchmark,
FAR aggregation, payload-diversity claim, or paper-facing positive claim was
started or unlocked by this review.

## Reviewed artifacts

- `docs/natural_evidence_v2/R3_2_H200_853524_REPAIR_DECISION_PACKAGE_20260512.md`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_h200_853524_repair_decision_package.json`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution.md`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/failure_attribution/r3_2_853524_failure_attribution_summary.json`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/r3_2_gate_review.json`

## Decision

`853524` is not artifact-only repairable into a passing locked-scale result.

The failure is not a missing artifact, Slurm failure, duplicate prompt-window
control-plane defect, or local aggregation defect. The run completed cleanly,
used unique prompt windows and blocks, and still produced only `6/96`
protected coordinate-majority accepts at budget 64 against a required `80/96`.

The reviewed attribution supports a real same-contract locked-scale stability
failure under the expanded prompt distribution:

- diagnostic null accepts remained clean at `0/96` for raw, task-only,
  wrong-key, and wrong-payload;
- protected majority margins were too weak, with median margin `1` and accepted
  blocks below support/margin gates;
- first-word drift dominated erasures with `30,021` observations;
- Step-slot malformed output was secondary but material, with `1,093`
  missing/out-of-order slots and `485` duplicate slots;
- forbidden public surface hits totaled `23` under the precommitted literal
  matcher and require separate matcher-semantics audit, but cannot rescue the
  protected positive gate.

Therefore the repair decision package is accepted as a negative-result package:
preserve `853524` as a failed locked-scale artifact and do not rerun it under
the same contract.

## Artifact-only repair boundary

Allowed as future planning only after explicit route review:

- design a new artifact-only protocol/prompt-bank repair plan;
- audit forbidden-surface matcher semantics as a separate diagnostic;
- compare negative `853524` against prior R3.2 artifacts to document scale
  instability.

Not allowed from this decision:

- resubmitting R3.2 or Qwen E2E;
- changing thresholds, slots, payload, key, or bucket policy after seeing
  transcripts and treating the same run as positive;
- unlocking Llama, same-family null, sanitizer, FAR aggregation, payload
  diversity, or paper-facing positive claims.

## Next allowed action

Hold for human/expert route decision, or perform artifact-only planning for a
new protocol/prompt-bank repair package if explicitly requested. Do not submit
a rerun or unlock downstream gates from `853524`.
