# R4 After-864832 Two-Sided Gain-Sweep Failure Pivot Decision

Status: `ARTIFACT_ONLY_PIVOT_RECORDED_NO_COMPUTE`

Job `865252` showed that the repaired two-sided cover-natural bank is tokenizer/scorer compatible but does not pass the teacher-forced surface-mass gate at adapter gain 1.0. Job `865289` then tested scalar protected-adapter gain values `0.0,0.5,1.0,1.5,2.0,3.0,4.0`; no gain passed the gate.

## Facts

```text
865252 protected lift vs base: +0.047113
865252 protected lift vs task-only: +0.056045
865252 protected rank1: 0.437500
865252 protected median margin: -0.012231

865289 best gain by mean target mass: protected_gain_1
865289 best mean target mass: 0.066522
865289 best lift vs base: +0.047113
865289 best lift vs task-only: +0.056045
865289 rank1 for all nonzero gains: 0.437500
865289 any gain passed: false
```

## Decision

Do not run generation from this branch. Do not train from this branch without a new objective route. The immediate next route must be one of:

1. `controller_only_teacher_forced_scoring`: provider-side soft logit controller, no protected adapter dependence, no generation.
2. `metric_exact_objective_repair`: training-objective repair plan, no Slurm training until code review and micro-overfit route are recorded.

The fastest discriminator is `controller_only_teacher_forced_scoring`, because it tests whether the two-sided cover-natural bank can be made measurable at all under controlled logits. If controller-only teacher-forced scoring also fails, the bank/codebook itself is likely mismatched. If it passes, generation remains locked until a separate small dev generation route is recorded.

## Not Unlocked

```text
generation
training
Llama
same-family null
sanitizer
FAR
payload diversity
paper-facing positive claims
```

