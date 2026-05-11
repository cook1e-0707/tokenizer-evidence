# Hermes/Codex state sync and compact-state update

Timestamp UTC: `2026-05-11T05:45Z`

## Current phase

`V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED`

## Latest Hermes status

Hermes ticks at 05:15 and 05:30 ran, but both correctly blocked R3.2 Slurm
submission because the tick prompt still carried old hard constraints:

```text
no generation
no Qwen E2E rerun
```

That conflicted with the approved R3.2 route, which requires reviewed Qwen
locked-scale generation/eval.

## Sync action

The Hermes prompt template was updated so the R3.2 context no longer forbids
all generation. It now permits only this narrow class:

```text
reviewed R3.2 Qwen locked-scale generation/eval
via one enabled allowlist entry
after TG/email notification
as exactly one Chimera Slurm job
```

Training, Llama, same-family null, sanitizer, FAR aggregation, paper-facing
positive claims, unreviewed/non-allowlisted generation, and Chimera login-node
CPU/GPU work remain forbidden.

## Token reduction

Created compact canonical state:

```text
docs/natural_evidence_v2/CURRENT_STATE.md
```

Future Hermes/Codex ticks should read this compact file first and only consult
the long historical state files if the compact state is ambiguous.

## Current blocker

The R3.2 Slurm wrapper is still plan-only:

```text
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
```

It must be upgraded/reviewed before any full locked-scale Slurm submission.

## Next allowed action

Finish or upgrade the R3.2 wrapper from plan-only to reviewed full
locked-scale generation/eval, validate locally, record wrapper review, enable
one allowlist entry, notify configured channels, then submit exactly one
Chimera Slurm job. No further user approval is needed for this same R3 route.

