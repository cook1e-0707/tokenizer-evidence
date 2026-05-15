# Hermes Sync: R4 Controller-Only Job 863274 Submitted

Phase:

```text
V2_R4_CONTROLLER_ONLY_H200_JOB_863274_RUNNING
```

Job:

```text
863274
```

Command:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1,CONTROLLER_CONDITION_SET=controller_only_controls,ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Post-submit control-plane status:

```text
local allowlist safety: PASS
remote allowlist safety: PASS
enabled entries after submission: []
```

First observed state:

```text
863274_0 RUNNING on chimera21
863274_1 RUNNING on chimera21
863274_2 RUNNING on chimera21
863274_[3-71%4] PENDING (Resources)
```

Expected output:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_pressure_controller_score_863274
```

Next allowed action:

```text
Monitor job 863274; after terminal completion, sync and review controller-only scoring artifacts before any downstream action.
```

Do not submit another controller-only scoring job before `863274` is reviewed.

