# R4 Safety-Bound Controller 864117 Failure Diagnosis

Status: `FAIL_R4_CONTROLLER_ONLY_SAFETY_BOUND_TOO_WEAK_NO_GENERATION`

Job `864117` completed cleanly and kept wrong controls clean, but still failed the positive teacher-forced selective gate.

## Key Numbers

| Metric | Value |
| --- | ---: |
| controlled basic gate pass | 0/24 |
| overall selective gate pass | 0/24 |
| wrong-key basic gate pass | 0/24 |
| wrong-payload basic gate pass | 0/24 |
| best controlled mean target mass | 0.0317901568 |
| base mean target mass | 0.0048318370 |
| best controlled lift vs base | 0.0269583198 |
| best controlled rank1 | 0.6015625000 |
| best controlled median margin | 0.0033881384 |
| target mass needed for +0.15 lift | 0.1548318370 |
| extra boost from best mass to gate target | 1.7191 nats |

## Interpretation

The safety-bound grid improved the best lift from the `863274` maximum of roughly `+0.0154` to `+0.0270`, and median target margin became positive. This is still far below the `+0.15` lift and `0.75` rank1 requirements. Wrong-key and wrong-payload controls remain clean, which means selectivity is not the limiting failure in this run.

The current evidence is that a simple additive controller within the reviewed `bonus<=2.0`, `KL<=0.20`, and `max_target_mass<=0.50` envelope is not strong enough for candidate-v3's surface cylinders. This does not unlock generation, training, Llama, null expansion, sanitizer, FAR, payload diversity, or paper-facing claims.

Selected grid cap diagnostics are recorded in:

```text
results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/selected_grid_cap_diagnostics.jsonl
```

The next valid action is artifact-only pivot planning: either a more expressive row-adaptive controller, a metric-exact objective/training route, or a new surface/channel design.
