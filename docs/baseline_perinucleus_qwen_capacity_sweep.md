# Qwen Perinucleus Capacity Sweep

This is a diagnostic capacity sweep after the official-code forensic replay, Qwen overfit gate, and Llama anchor. It is not a final comparison matrix.

## Decision

`QWEN_CAPACITY_SWEEP_CANDIDATE_FOUND: run utility sanity for the selected candidate before any final matrix.`

## Sweep Arms

| arm | fingerprints | target modules | rank | epochs | exact accuracy | mean rank | mean probability | utility |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qv_r64_fp16_e30 | 16 | qv | 64 | 30 | 1.0 | 1.0 | 0.998759288340807 | pending_candidate_utility |
| qkvo_r64_fp16_e30 | 16 | qkvo | 64 | 30 | 1.0 | 1.0 | 0.9989882186055183 | pending_candidate_utility |
| all_linear_r64_fp16_e30 | 16 | all_linear | 64 | 30 | 1.0 | 1.0 | 0.9995247237384319 | pending_candidate_utility |
| all_linear_r16_fp16_e30 | 16 | all_linear | 16 | 30 | 1.0 | 1.0 | 0.9993154965341091 | pending_candidate_utility |
| all_linear_r64_fp64_e40 | 64 | all_linear | 64 | 40 | 1.0 | 1.0 | 0.9999511335045099 | pending_candidate_utility |
| qkvo_r64_fp64_e40 | 64 | qkvo | 64 | 40 | 1.0 | 1.0 | 0.9998351959511638 | pending_candidate_utility |
| all_linear_r64_fp64_e80 | 64 | all_linear | 64 | 80 | 1.0 | 1.0 | 0.9999891147017479 | pending_candidate_utility |

## Recommended Candidate

{
  "adapter_path": "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/arms/all_linear_r64_fp64_e80/adapter_final",
  "arm_id": "all_linear_r64_fp64_e80",
  "candidate_pass": true,
  "epochs_run": 80,
  "final": {
    "adapter_path": "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/arms/all_linear_r64_fp64_e80/adapter_final",
    "arm_id": "all_linear_r64_fp64_e80",
    "base_target_probability_mean": 0.022290308815968274,
    "base_vs_adapter_logit_delta_max": 24.25,
    "epoch": 80,
    "exact_accuracy": 1.0,
    "exact_count": 64,
    "lora_alpha_ratio": 2.0,
    "lora_max_norm": 4.723266124725342,
    "lora_nonzero_norm_count": 392,
    "lora_parameter_count": 392,
    "lora_rank": 64,
    "lora_total_norm": 960.1186808645725,
    "max_epochs": 80,
    "mismatch_examples": [],
    "num_fingerprints": 64,
    "rank1_accuracy": 1.0,
    "rank1_count": 64,
    "regularization_label": "diagnostic_off",
    "stage": "all_linear_r64_fp64_e80",
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "target_modules_label": "all_linear",
    "target_probability_mean": 0.9999891147017479,
    "target_probability_min": 0.9999240636825562,
    "target_rank_max": 1.0,
    "target_rank_mean": 1.0,
    "train_ce_mean": 1.0892617459568044e-05,
    "train_step_loss_mean": 1.1276321232323028e-05,
    "utility_status": "pending_candidate_utility"
  },
  "fingerprints_file": "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/generated/fingerprints_64/fingerprints-perinucleus-Qwen-Qwen2.5-7B-Instruct-nucleus_threshold-0.8-response_length-1-use_chat_template-True.json",
  "lora_alpha_ratio": 2.0,
  "lora_rank": 64,
  "max_epochs": 80,
  "num_fingerprints": 64,
  "pass": true,
  "pass_reason": "at least one exact fingerprint",
  "regularization_label": "diagnostic_off",
  "seconds": 1802.0044560432434,
  "stage": "all_linear_r64_fp64_e80",
  "strong_candidate": true,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "target_modules_label": "all_linear",
  "utility_status": "pending_candidate_utility"
}

## Fidelity Notes

- Fingerprint generation uses the official Scalable Fingerprinting repository at the recorded commit.
- Training uses the same adapted diagnostic LoRA loop that passed the single-fingerprint overfit gate.
- This sweep intentionally freezes a small diagnostic arm list before running; it must not be expanded post hoc using final-matrix feedback.
- Utility sanity is not treated as complete unless the selected candidate is evaluated separately and recorded before final-matrix use.

## Output Files

- Table: `/home/guanjie.lin001/tokenizer-evidence/results/tables/baseline_perinucleus_qwen_capacity_sweep.csv`
- Summary: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_summary.json`
- Compute: `/home/guanjie.lin001/tokenizer-evidence/results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_compute.json`
