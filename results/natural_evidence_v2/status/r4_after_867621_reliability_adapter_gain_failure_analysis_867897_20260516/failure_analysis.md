# R4 After 867621 Adapter-Gain Failure Analysis

status: `FAIL_R4_AFTER_867621_RELIABILITY_ADAPTER_GAIN_SWEEP_NO_GENERATION`

Job `867897` completed cleanly, but no protected-adapter gain satisfied the
teacher-forced surface-mass gate.

Best protected gain by mean target mass:

```text
condition: protected_gain_0_5
mean target mass: 0.019240
lift vs base: 0.008797
lift vs task_only: 0.013716
rank1 rate: 0.509277
median target margin: 0.00010527
```

Gate targets were `+0.15` lift vs base, `+0.10` lift vs task-only, rank1
`>=0.75`, and median target margin `>0`. The best observed lift vs base was
only `0.008797`.

Interpretation:

```text
adapter_gain_rescued_gate: false
generation_unlocked: false
training_unlocked: false
```

Increasing the protected adapter gain did not produce monotonic improvement.
The best point was `protected_gain_0_5`; larger gains reduced target mass and
rank1. This points away from a simple insufficient-scale explanation and toward
an adapter-direction or objective/surface mismatch. This result must not unlock
generation.
