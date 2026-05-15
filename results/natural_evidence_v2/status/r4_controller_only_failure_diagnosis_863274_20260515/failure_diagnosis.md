# R4 Controller-Only 863274 Failure Diagnosis

Status: `FAIL_R4_CONTROLLER_ONLY_TOO_WEAK_NO_GENERATION`

Job `863274` fixed the wrong-control semantics tested in `859672`: wrong-key and wrong-payload controlled-base arms are now `0/72` for the basic gate. The route still fails because no controlled-base grid passes the teacher-forced positive gate.

## Key Numbers

| Metric | Value |
| --- | ---: |
| controlled basic gate pass | 0/72 |
| overall selective gate pass | 0/72 |
| wrong-key basic gate pass | 0/72 |
| wrong-payload basic gate pass | 0/72 |
| best controlled mean target mass | 0.0202354971 |
| base mean target mass | 0.0048318370 |
| best controlled lift vs base | 0.0154036601 |
| best controlled rank1 | 0.4980468750 |
| best controlled median margin | -0.0001098111 |
| target mass needed for +0.15 lift | 0.1548318370 |
| extra boost from best mass to gate target | 2.1827 nats |

## Interpretation

The controller-only design is selective enough to reject wrong controls under the current basic gate, but the positive pressure is too weak. The best grid uses `bonus_nats=1.5`, `penalty_nats=0.25`, `max_target_mass=0.25`, and `max_kl_budget=0.1`, yet reaches only `+0.0154` lift vs base. Reaching the +0.15 lift target from the best observed mass would require roughly `2.18` additional logit-odds nats, assuming the same target bucket and no distributional side effects.

## Selected Grid Cap Diagnostics

A row-level artifact-only probe of the best grids (`61`, `67`, `71`) shows that the positive controlled-base rows are not broadly capped by the current safety constraints:

```text
grid 67 controlled_base rows: 8192
controller_scale median: 1.0
controller_scale mean: 0.9988
max_kl_budget cap rows: 128/8192
uncapped rows: 8064/8192
mean KL to base: 0.0150
max KL to base: 0.1000
mean target mass: 0.0202354971
median target mass: 0.0074881483
rank1 rate: 0.498046875
```

This means the observed failure is not mainly caused by the KL or target-mass cap suppressing most controlled-base rows. Under the tested grid, the additive controller itself does not provide enough average pressure over the committed target cylinders.

Full selected-grid diagnostics are recorded in:

```text
results/natural_evidence_v2/status/r4_controller_only_failure_diagnosis_863274_20260515/selected_grid_cap_diagnostics.jsonl
```

This artifact does not unlock generation, training, Llama, null expansion, sanitizer, FAR, payload diversity, or paper-facing claims. The next valid step is artifact-only repair-route planning for a stronger and still selective pressure mechanism.
