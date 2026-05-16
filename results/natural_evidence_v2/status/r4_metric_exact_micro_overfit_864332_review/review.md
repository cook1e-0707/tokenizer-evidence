# R4 Metric-Exact Micro-Overfit 864332 Review

status: `FAIL_R4_METRIC_EXACT_MICRO_OVERFIT_864332_TEACHER_FORCED_GATE`

## Slurm Completion

- job_id: `864332`
- job_name: `nat-ev-v2-r4mof`
- partition: `pomplun`
- node: `chimera21`
- elapsed: `00:04:08`
- exit code: `0:0`

This was a clean Slurm completion, not a scheduler, wrapper, tokenizer, or CUDA failure.

## Route Scope

This route ran the reviewed metric-exact micro-overfit wrapper with:

- `SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus`
- `MAX_TRAIN_ROWS=512`
- `MAX_SCORE_ROWS=8192`
- `MAX_STEPS=64`
- `TARGET_MASS_FLOOR=0.20`
- `TARGET_MASS_FLOOR_LAMBDA=5.0`
- `TARGET_MASS_CEILING=0.45`
- `STRATUM_WEIGHTING_MODE=r4_candidate_v3_failed_surface`

No generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, payload-diversity claim, or paper-facing claim was started.

## Gate Result

The teacher-forced surface-mass gate failed.

| metric | observed | gate |
| --- | ---: | ---: |
| protected mean target mass | `0.0179803` | diagnostic input |
| protected lift vs base | `+0.0131485` | `>= +0.15` |
| protected lift vs task-only | `+0.0163079` | `>= +0.10` |
| protected rank1 rate | `0.980469` | `>= 0.70` |
| protected median margin | `+0.0059255` | `> 0` |
| task-only lift vs base | `-0.0031594` | diagnostic cap passed |

The adapter strongly improved rank ordering, but did not put enough absolute probability mass on the target first-token set.

## Comparison To Candidate V3 856453

Candidate v3 job `856453` had:

- protected mean target mass: `0.0577269`
- protected lift vs base: `+0.0528950`
- protected rank1 rate: `0.654297`

Job `864332` changed the failure mode:

- rank1 improved from `0.654297` to `0.980469`;
- absolute protected target mass dropped from `0.0577269` to `0.0179803`;
- lift vs base dropped from `+0.0528950` to `+0.0131485`.

So this patch did not simply strengthen the existing candidate-v3 signal. It made target tokens consistently outrank the immediate alternatives while leaving total target mass too low.

## Stratum Evidence

The strongest stratum remains `A practical next move is:` with `Create a short summary`, but even there mass is far below the gate:

- `Create a short summary`: protected mass `0.0468844`, lift vs base `+0.0359355`
- `Create a simple timeline`: protected mass `0.0194917`, lift vs base `+0.0137774`
- `Prepare a note`: protected mass `0.0034117`, lift vs base `+0.0021919`
- `Prepare questions`: protected mass `0.0021335`, lift vs base `+0.0006891`

The weak `A useful action is:` / `Prepare...` strata remain effectively low-mass.

## Training Diagnostics

The final train summary shows:

- final CE loss: `2.20671`
- final margin loss: `0.001611`
- final target-mass floor loss: `0.196320`
- final ceiling loss: `0.0`

Because the floor is `0.20`, a floor loss near `0.196` means the target-mass floor remained almost completely unsatisfied at the end of training. The metric-exact mode affected rank/margin, but did not create the required absolute target-mass pressure.

At the observed protected mean target mass `0.0179803`, reaching the +0.15 lift-vs-base gate would require protected target mass about `0.154832`, an additional odds-equivalent boost of roughly `2.30` nats from the observed result.

## Artifact Policy

The remote adapter weight is retained on Chimera only. Local review records the remote adapter hash:

```text
37223193f8ce01fb820c20d10c6bcdbbc0776c18813b8be3d0ca88989a02f6cd  protected_micro_overfit_train/adapter/adapter_model.safetensors
```

The local review artifacts include summary JSON, train summary, score summary, and derived stratification CSVs. The large adapter weight is not committed.

## Next Allowed Action

Artifact-only failure analysis and a reviewed repair or pivot route must be recorded before any new Slurm submission, generation, Llama, same-family null, sanitizer, FAR aggregation, payload-diversity claim, or paper-facing positive claim.
