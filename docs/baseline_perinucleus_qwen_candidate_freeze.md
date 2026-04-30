# Qwen Perinucleus Candidate Freeze

Generated at: `2026-04-30T03:50:41Z`
Decision: `QWEN_PERINUCLEUS_CANDIDATE_FROZEN: final protocol may use only this frozen adapter/config unless superseded before final launch.`

This freezes the Qwen-adapted official Scalable Fingerprinting / Perinucleus candidate for downstream final-protocol use. It is a candidate freeze, not a final matrix result.

## Frozen Candidate

- Arm: `all_linear_r64_fp64_e80`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter path: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/arms/all_linear_r64_fp64_e80/adapter_final`
- Fingerprints file: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_qwen_capacity_sweep/runs/manual_20260429T182756Z/generated/fingerprints_64/fingerprints-perinucleus-Qwen-Qwen2.5-7B-Instruct-nucleus_threshold-0.8-response_length-1-use_chat_template-True.json`
- Fingerprints: `64`
- Target modules: `all_linear` / `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`
- LoRA rank: `64`
- Epochs run: `80`
- Regularization label: `diagnostic_off`

## Gate Evidence

| gate | pass | evidence |
| --- | --- | --- |
| Llama anchor | True | `docs/baseline_perinucleus_llama_anchor_result.md` |
| Qwen capacity sweep | True | exact=1.0, rank_mean=1.0, prob_mean=0.9999891147017479 |
| Qwen utility sanity | True | utility=0.6191832009104691, base=0.6035317339934293, drop=-0.01565146691703978 |

## Final-Protocol Constraints

- Final Perinucleus Qwen runs must use this exact adapter path unless this freeze is superseded before any final launch.
- Do not use final-matrix feedback to change LoRA rank, target modules, epochs, fingerprints, or selected adapter.
- The baseline must be described as an adapted Qwen LoRA reproduction of official Scalable/Perinucleus, not as an unmodified full fine-tune.
- Utility drops are signed as `base_total_accuracy - adapter_total_accuracy`; negative values mean the adapter scored higher than the base on this tinyBenchmarks sanity.

## Source Artifacts

- Utility summary: `results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_summary.json`
- Capacity summary: `results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_summary.json`
- Llama anchor doc: `docs/baseline_perinucleus_llama_anchor_result.md`
- Freeze summary: `results/processed/paper_stats/baseline_perinucleus_qwen_candidate_freeze_summary.json`
- Freeze table: `results/tables/baseline_perinucleus_qwen_candidate_freeze.csv`
- Freeze config: `configs/experiment/baselines/perinucleus_official/qwen_frozen_candidate__baseline_perinucleus_official.yaml`
