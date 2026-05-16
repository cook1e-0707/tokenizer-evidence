# R4 After 868016 Reliability Coordinate Pivot Route

Status: `ROUTE_RECORDED_ARTIFACT_ONLY_NO_SLURM`

Job `868016` completed cleanly but failed the teacher-forced controller
feasibility gate. The best grid was close but still below the precommitted
protected lift-vs-base threshold:

```text
best grid: 26
bonus_nats: 4.0
penalty_nats: 0.5
max_target_mass: 0.5
max_kl_budget: 0.5
controlled lift vs base: +0.121635
controlled lift vs task_only: +0.126555
controlled rank1: 0.942139
wrong-control failures: 0
passing grids: 0/30
```

Row-level artifact-only failure analysis found that the mass-lift deficit is
not uniform. It is concentrated in a small set of weak coordinates. A post-hoc
diagnostic exclusion of coordinates `3,10,20,24` would have crossed the
teacher-forced mass gate on the reviewed rows, but that does **not** reclassify
`868016` as positive because the exclusion was derived after seeing the run.

## New Candidate

This route freezes a new coordinate-filtered candidate for a future tokenizer
preflight. It uses a held-out prompt slice relative to the rows used in
`868016`.

```text
contract_id: a55e
source candidate: after-867621 coordinate-unique reliability surface bank
excluded coordinates: 3,10,20,24
selected coordinates: 6,22,26,1,17,19,15,31,8,4,7,23
prompt_offset: 256
selected prompts: 256
row_count: 3072
split: dev held-out slice, not the 868016 prompt slice
```

New row artifacts:

```text
rows: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl
row summary: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows_summary.json
static preflight: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_static_boundary_preflight_20260516/r4_prefix_native_tokenizer_boundary_preflight_summary.json
```

## Immediate Route Scope

The only compute action prepared by this route is actual-Qwen tokenizer boundary
preflight on H200/pomplun. It is tokenizer-only:

```text
model forward: forbidden
teacher-forced scoring: forbidden until tokenizer preflight passes and a later scoring route is recorded
generation: forbidden
training: forbidden
Llama/same-family null/sanitizer/FAR/paper claims: forbidden
```

The route must use:

```text
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
gpu: h200
time_limit: 30-00:00:00
```

## Tokenizer Gate

Actual Qwen tokenizer preflight passes only if:

```text
checked_rows == 3072
failed_rows == 0
empty_target_id_row_count == 0
empty_other_id_row_count == 0
target_other_overlap_row_count == 0
```

If this tokenizer preflight fails, no scoring route may be submitted. If it
passes, the next allowed action is a separately reviewed teacher-forced
controller scoring route for the same coordinate-filtered rows.

## Claim Control

This route is not a positive result and does not modify paper-facing claims. It
is a precondition check for a new candidate derived from `868016` failure
analysis.
