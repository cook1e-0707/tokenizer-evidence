# Qwen Perinucleus Candidate Utility Sanity

This is a utility sanity check for selected Qwen Perinucleus capacity-sweep adapters. It does not retrain and does not authorize a final matrix by itself.

## Decision

`QWEN_CANDIDATE_UTILITY_PASS: freeze the selected candidate before final protocol runs.`

## Utility Results

| kind | arm | exact | utility | base utility | abs drop | pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| base | base |  | 0.6035317339934293 | 0.6035317339934293 | 0.0 | True |
| adapter | qv_r64_fp16_e30 | 1.0 | 0.607548613797407 | 0.6035317339934293 | -0.004016879803977691 | True |
| adapter | all_linear_r16_fp16_e30 | 1.0 | 0.6172300852420466 | 0.6035317339934293 | -0.013698351248617291 | True |
| adapter | all_linear_r64_fp64_e80 | 1.0 | 0.6191832009104691 | 0.6035317339934293 | -0.01565146691703978 | True |

## Selected Candidate

{
  "absolute_drop": -0.01565146691703978,
  "adapter_path": "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/arms/all_linear_r64_fp64_e80/adapter_final",
  "arm_id": "all_linear_r64_fp64_e80",
  "base_total_accuracy": 0.6035317339934293,
  "epochs_run": 80,
  "error": "",
  "eval_results_path": "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_candidate_utility/runs/manual_20260430T014700Z/eval_results/all_linear_r64_fp64_e80_eval_results.json",
  "exact_accuracy": 1.0,
  "kind": "adapter",
  "lora_rank": 64,
  "missing_metrics": [],
  "num_fingerprints": 64,
  "relative_drop": -0.02593313000043537,
  "target_modules_label": "all_linear",
  "target_probability_mean": 0.9999891147017479,
  "target_rank_mean": 1.0,
  "total_accuracy": 0.6191832009104691,
  "train_ce_mean": 1.0892617459568044e-05,
  "utility_pass": true,
  "utility_status": "completed"
}

## Notes

- Adapter utility is evaluated as `pretrained=Qwen/Qwen2.5-7B-Instruct,peft=<adapter_path>`.
- `apply_chat_template=true` is used for the instruct backbone.
- The selected candidate must be frozen before any final matrix.

## Output Files

- Table: `/home/guanjie.lin001/tokenizer-evidence/results/tables/baseline_perinucleus_qwen_candidate_utility.csv`
- Summary: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_summary.json`
- Compute: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_compute.json`
