# R4 Metric-Exact Floor-Dominant 864705 Review

Canonical status: `FAIL_R4_METRIC_EXACT_FLOOR_DOMINANT_864705_TEACHER_FORCED_GATE`

Job `864705` completed cleanly:

```text
job_name: nat-ev-v2-r4mof
partition: pomplun
node: chimera21
elapsed: 00:04:40
exit_code: 0:0
```

The run used task CE disabled with stronger target-mass floor pressure. It did
not run generation, Qwen E2E, Llama, same-family null, sanitizer, FAR,
payload-diversity, or paper-claim work.

Audit note: the reviewed submission environment set `TASK_CE_WEIGHT=0.0`, and
the wrapper passed `--task-ce-weight 0.0`. The original train summary did not
record this field; the trainer has been patched so future train summaries
record `task_ce_weight` directly.

## Result

The teacher-forced gate still failed:

```text
protected mean target mass:      0.0847697
base mean target mass:           0.00483184
task-only mean target mass:      0.00167243
protected lift vs base:         +0.0799378   required >= +0.15
protected lift vs task-only:    +0.0830972   required >= +0.10
protected rank1 rate:            1.000000    required >= 0.70
protected median margin:        +0.0772580   required > 0
```

This is a significant improvement over `864332`, but not a pass.

```text
864332 protected mass: 0.0179803, lift vs base +0.0131485, rank1 0.980469
864705 protected mass: 0.0847697, lift vs base +0.0799378, rank1 1.000000
```

The result changes the diagnosis: floor-dominant pressure works directionally,
but the aggregate score-set mass is still too low. The remaining gap to the
lift-vs-base gate is about `0.68` nats odds-equivalent boost.

## Strata

The previously weak `A useful action is:` / `Prepare...` strata improved:

```text
A practical next move is: / Create a short summary   0.0797753
A practical next move is: / Create a simple timeline 0.0612945
A useful action is: / Prepare a note                 0.109776
A useful action is: / Prepare questions              0.0882329
```

The route still fails because none of this reaches the aggregate target mass
needed for the committed gate.

## Artifacts

```text
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864705/
results/natural_evidence_v2/status/r4_metric_exact_floor_dominant_864705_review/
```

Remote adapter hash:

```text
7fc8225b722cb472dc412ff6d58a518078444053ffc94b67450562fdcf64e76d
```

## Next State

No generation or downstream route is unlocked. The next action is artifact-only
route decision, likely coverage-scale or stronger-floor training, before any new
Slurm submission.
