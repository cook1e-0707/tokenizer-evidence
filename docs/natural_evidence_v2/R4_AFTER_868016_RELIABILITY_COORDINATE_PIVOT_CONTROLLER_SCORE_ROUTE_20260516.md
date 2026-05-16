# R4 After 868016 Reliability Coordinate Pivot Controller Score Route

Status: `ROUTE_RECORDED_ARTIFACT_ONLY_NO_SLURM`

The coordinate-filtered pivot passed actual Qwen tokenizer boundary preflight in
job `868103`:

```text
checked rows: 3072
failed rows: 0
empty target id rows: 0
empty other id rows: 0
target/other overlap rows: 0
```

This route prepares the next teacher-forced scoring-only check for the same
coordinate-filtered rows. It does not run generation and does not support any
paper-facing claim.

## Scope

```text
contract_id: a55e
model_family: qwen_only
score rows: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl
row_count: 3072
conditions: base, task_only, controlled_base, wrong_key_controlled_base, wrong_payload_controlled_base
controller condition set: controller_only_controls
protected adapter in controller arms: false
```

The grid is intentionally narrow around the best `868016` diagnostic region:

```text
bonus_nats: [3.5, 4.0]
penalty_nats: [0.5, 1.0]
max_target_mass: [0.5]
max_kl_budget: [0.5]
grid cells: 4
```

## Gate

The route passes only if at least one grid cell satisfies:

```text
controlled lift vs base >= +0.15
controlled lift vs task_only >= +0.10
controlled rank1 >= 0.75
wrong-key accepts = 0
wrong-payload accepts = 0
target/other overlap rate = 0
scorer boundary failures = 0
```

If the teacher-forced gate passes, the next action is route planning for a
small generation diagnostic. If it fails, return to artifact-only channel
repair. No generation is unlocked by route submission or by partial metrics.

## Control Plane

The job must use H200/pomplun:

```text
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
gpu: h200
time_limit: 30-00:00:00
```

The allowlist entry must be enabled only for submission and disabled
immediately after `sbatch` returns.
