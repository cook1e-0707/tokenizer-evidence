# R3.2 payload semantics decision

Timestamp UTC: `2026-05-11T06:01:06Z`

## Decision

`P00/P01/P02/P03` are R3.2 locked-scale cell labels that intentionally reuse
the fixed WP5-R2 `a55e` contract. They are not distinct reviewed payload
contracts.

Decision record:

```text
docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION_20260511.md
```

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_payload_semantics_decision_20260511_0601.json
```

## Guardrails

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

## Next allowed action

Upgrade or review the R3.2 full locked-scale wrapper using `P00/P01/P02/P03`
as cell labels over the fixed `a55e` contract. Do not submit Slurm until wrapper
review, allowlist enablement, TG/email notification, and the exactly-one-job
submission gate are complete.

## Status

```text
R3_2_PAYLOAD_SEMANTICS_RESOLVED_A55E_CELL_LABELS_NO_SLURM
```
