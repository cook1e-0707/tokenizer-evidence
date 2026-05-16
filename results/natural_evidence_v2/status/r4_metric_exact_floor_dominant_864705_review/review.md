# R4 Floor-Dominant Micro-Overfit 864705 Review

status: `FAIL_R4_METRIC_EXACT_FLOOR_DOMINANT_864705_TEACHER_FORCED_GATE`

## Slurm Completion

- job_id: `864705`
- job_name: `nat-ev-v2-r4mof`
- partition: `pomplun`
- node: `chimera21`
- elapsed: `00:04:40`
- exit code: `0:0`

This was a clean Slurm completion, not a scheduler, wrapper, tokenizer, or CUDA failure.

## Route Scope

This route used the reviewed floor-dominant metric-exact repair:

```text
SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus
TASK_CE_WEIGHT=0.0
TARGET_MASS_FLOOR=0.20
TARGET_MASS_FLOOR_LAMBDA=50.0
TARGET_MASS_CEILING=0.45
TARGET_MASS_CEILING_LAMBDA=5.0
MARGIN_LAMBDA=1.0
MAX_STEPS=128
LEARNING_RATE=1e-4
```

No generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, payload-diversity claim, or paper-facing claim was started.

## Gate Result

The teacher-forced surface-mass gate failed.

| metric | observed | gate |
| --- | ---: | ---: |
| protected mean target mass | `0.0847697` | diagnostic input |
| protected lift vs base | `+0.0799378` | `>= +0.15` |
| protected lift vs task-only | `+0.0830972` | `>= +0.10` |
| protected rank1 rate | `1.000000` | `>= 0.70` |
| protected median margin | `+0.0772580` | `> 0` |
| task-only lift vs base | `-0.0031594` | diagnostic cap passed |

This is a real improvement over job `864332`, but still below the training gate.

## Comparison To 864332

```text
864332 protected mass: 0.0179803, lift vs base +0.0131485, rank1 0.980469
864705 protected mass: 0.0847697, lift vs base +0.0799378, rank1 1.000000
```

The improvement in protected target mass is `+0.0667894`. The remaining odds-equivalent boost needed to hit the lift-vs-base gate is about `0.68` nats.

## Stratum Evidence

All four prefix/surface strata improved relative to earlier failures:

```text
A practical next move is: / Create a short summary   protected mass 0.0797753
A practical next move is: / Create a simple timeline protected mass 0.0612945
A useful action is: / Prepare a note                 protected mass 0.109776
A useful action is: / Prepare questions              protected mass 0.0882329
```

The previous weak `A useful action is:` / `Prepare...` strata are no longer near-zero, which supports the floor-dominant repair direction.

## Training Diagnostics

The final train summary shows:

```text
final_floor_loss: 0.0
final_margin_loss: 0.0298137
final_ce_loss: 4.57238
```

Unlike `864332`, the target-mass floor term can be satisfied under this objective. The remaining failure is now coverage/strength: the score-set aggregate mass is still too low for the committed gate.

## Artifact Policy

The remote adapter weight is retained on Chimera only. Local review records the remote adapter hash:

```text
7fc8225b722cb472dc412ff6d58a518078444053ffc94b67450562fdcf64e76d  protected_micro_overfit_train/adapter/adapter_model.safetensors
```

The local review artifacts include summary JSON, train summary, score summary, and derived stratification CSVs. The large adapter weight and raw score rows are not committed.

## Next Allowed Action

Artifact-only route decision only. A plausible next route is coverage-scale / stronger-floor training, but it must be separately recorded and preflighted before any new Slurm submission.
