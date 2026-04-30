# Qwen Perinucleus Candidate Utility Sanity

Status: pending execution.

This package evaluates utility for selected Qwen Perinucleus capacity-sweep adapters. It does not retrain and does not authorize final baseline matrices by itself.

## Candidates

The configured candidates are:

- `qv_r64_fp16_e30`: minimal target modules, rank 64, 16 fingerprints.
- `all_linear_r16_fp16_e30`: all-linear target modules, lower rank, 16 fingerprints.
- `all_linear_r64_fp64_e80`: strongest capacity-sweep candidate, 64 fingerprints.

## Utility Contract

Adapters are PEFT LoRA adapters, not standalone Hugging Face model directories. Utility evaluation must load them as:

```text
pretrained=Qwen/Qwen2.5-7B-Instruct,peft=<adapter_path>
```

The instruct backbone uses `apply_chat_template=true`.

Each model/adapter utility evaluation is launched in a separate Python
subprocess so that CUDA memory from `lm_eval` is released between candidates.
The configured evaluation batch size is `1` for 40GB GPU compatibility; base and
adapter utility are evaluated with the same batch size.

## Gate

A candidate passes utility sanity only if:

- the tinyBenchmarks evaluation completes;
- exact fingerprint accuracy from the capacity sweep is retained in the record;
- absolute utility drop relative to the base Qwen utility is at most `0.05`.

If multiple candidates pass, the selected candidate is the one with the smallest absolute utility drop.

## Execution

Submit from `chimerahead`:

```bash
bash scripts/submit_perinucleus_qwen_candidate_utility.sh
```

For an interactive GPU allocation:

```bash
python3 scripts/run_perinucleus_qwen_candidate_utility.py \
  --config configs/experiment/baselines/perinucleus_official/qwen_candidate_utility__baseline_perinucleus_official.yaml \
  --force
```

## Expected Outputs

- `docs/baseline_perinucleus_qwen_candidate_utility.md`
- `results/tables/baseline_perinucleus_qwen_candidate_utility.csv`
- `results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_compute.json`
