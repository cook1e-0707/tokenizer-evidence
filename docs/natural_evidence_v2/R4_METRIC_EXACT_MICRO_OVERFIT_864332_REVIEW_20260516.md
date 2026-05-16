# R4 Metric-Exact Micro-Overfit 864332 Review

Canonical status: `FAIL_R4_METRIC_EXACT_MICRO_OVERFIT_864332_TEACHER_FORCED_GATE`

Job `864332` completed cleanly on H200/pomplun:

```text
job_name: nat-ev-v2-r4mof
partition: pomplun
node: chimera21
elapsed: 00:04:08
exit_code: 0:0
```

The run used the reviewed `logsumexp_softplus` metric-exact surface-margin mode. It did not run generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, payload-diversity, or paper-claim work.

## Result

The teacher-forced surface-mass gate failed:

```text
protected mean target mass:      0.0179803
base mean target mass:           0.00483184
task-only mean target mass:      0.00167243
protected lift vs base:         +0.0131485   required >= +0.15
protected lift vs task-only:    +0.0163079   required >= +0.10
protected rank1 rate:            0.980469    required >= 0.70
protected median margin:        +0.0059255   required > 0
```

The important interpretation is that rank ordering improved, but absolute target-token probability mass did not.

Compared with candidate-v3 scoring job `856453`, this new micro-overfit adapter had lower protected mass and lower lift:

```text
856453 protected mass: 0.0577269, lift vs base +0.0528950, rank1 0.654297
864332 protected mass: 0.0179803, lift vs base +0.0131485, rank1 0.980469
```

## Failure Shape

The best stratum remains too weak:

```text
A practical next move is: / Create a short summary
protected mass 0.0468844, lift vs base +0.0359355
```

The weak strata remain very low-mass:

```text
A useful action is: / Prepare a note       protected mass 0.0034117
A useful action is: / Prepare questions   protected mass 0.0021335
```

The train summary also shows the target-mass floor stayed unsatisfied:

```text
target_mass_floor:       0.20
final_floor_loss:        0.196320
final_margin_loss:       0.001611
final_ce_loss:           2.206710
```

So the metric-exact patch did not produce the intended target-mass pressure under the current 512-row, 64-step micro-overfit route.

## Artifacts

Detailed review:

```text
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_864332_review/
```

Run summaries:

```text
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864332/protected_micro_overfit_train/wp5_micro_slot_lora_train_summary.json
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864332/teacher_forced_score/r4_teacher_forced_surface_mass_summary.json
```

Remote adapter hash:

```text
37223193f8ce01fb820c20d10c6bcdbbc0776c18813b8be3d0ca88989a02f6cd
```

## Next State

No further compute route is unlocked by this failed gate. The next allowed action is artifact-only failure analysis and a reviewed repair or pivot route before any new Slurm submission.
