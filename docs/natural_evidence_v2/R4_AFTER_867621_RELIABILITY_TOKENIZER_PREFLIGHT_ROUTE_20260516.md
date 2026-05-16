# R4 After 867621 Reliability Tokenizer Preflight Route

Status: `ARTIFACT_ONLY_ROUTE_RECORDED_NO_SUBMIT`

## Context

Job `867621` completed cleanly but failed the R4 reliability dev-generation
positive gate. The protected arm had `0/32` accepts under `format_scrub=all`,
and artifact-only failure analysis found no selected coordinate-unique surface
matches in protected outputs. The generator instead repeated older
candidate-v3 `Create/Prepare/Plan` phrases.

Root cause:

```text
free_generation_transfer_failure_surface_absent
```

This route does not reclassify `867621`, rerun generation, or unlock a paper
claim. It prepares the next prerequisite check for the coordinate-unique
reliability surfaces: actual Qwen tokenizer boundary validation.

## Prepared Artifacts

```text
score rows:
results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl

row summary:
results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows_summary.json

static boundary preflight:
results/natural_evidence_v2/status/r4_after_867621_reliability_static_boundary_preflight_20260516/

route config:
configs/natural_evidence_v2/r4_after_867621_reliability_tokenizer_preflight_route.yaml

Slurm wrapper:
scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200.sbatch

route validator:
scripts/natural_evidence_v2/validate_r4_after_867621_reliability_tokenizer_route.py
```

Prepared rows:

```text
rows: 4096
selected prompts: 256
selected coordinates: 16
surface entries: 128
current two-way scorer compatible: true
```

Static boundary contract preflight passed:

```text
status: PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING
checked rows: 4096
failed rows: 0
```

## Compute Scope

The next compute action, if separately submitted after local/remote preflight,
is tokenizer-only:

```text
entry: v2_r4_after_867621_reliability_qwen_tokenizer_boundary_preflight_h200
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
model forward: false
teacher-forced scoring: false
generation: false
training: false
```

The wrapper must run the existing tokenizer-boundary preflight with
`--run-qwen-tokenizer` and fail closed on any empty target/other token set,
target/other overlap, prefix instability, empty delta, or missing boundary row.

## Gate

Actual Qwen tokenizer boundary preflight must pass:

```text
checked_rows = 4096
failed_rows = 0
empty_target_id_row_count = 0
empty_other_id_row_count = 0
target_other_overlap_row_count = 0
```

Only after this pass and review may a new H200 teacher-forced surface-mass
scoring route be recorded. This route package itself does not unlock scoring,
generation, training, Llama, same-family null, sanitizer, FAR, payload
diversity, or paper-facing claims.
