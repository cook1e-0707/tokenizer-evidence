# Qwen Perinucleus Capacity Sweep

Status: pending execution.

This package prepares the next diagnostic step after the Scalable/Perinucleus forensic replay, Qwen overfit gate, and Llama anchor. It is not a final comparison matrix and must not be used as paper evidence until a candidate also passes utility sanity and the final protocol is frozen.

## Goal

Find a Qwen-compatible Perinucleus insertion setting without post-hoc tuning on final comparison results.

## Preconditions

- Forensic replay must show no unresolved evaluator or prompt-template mismatch.
- Single-fingerprint overfit gate must pass.
- Llama anchor must be reviewed as passed or explicitly accepted as sufficient for proceeding.

## Planned Sweep

The configured arm list covers the required diagnostic dimensions without running a full Cartesian grid:

- Target modules: `qv`, `qkvo`, and `all_linear`.
- LoRA rank: `16` and `64`.
- Epoch budgets: `30`, `40`, and `80`.
- Fingerprint counts: `16` and `64`.
- Regularization: `diagnostic_off`; utility/regularized candidate validation remains a separate gate.

## Gate

A Qwen candidate is considered diagnostic-pass only if exact fingerprint accuracy is above base, target probability improves, adapter logits differ from base, and LoRA weights are nonzero. A strong candidate additionally needs exact accuracy at least `0.5` and mean target rank at most `2`.

Even if a strong candidate is found, final-matrix use remains blocked until utility sanity is run and recorded for the selected candidate.

## Execution

```bash
python3 scripts/run_perinucleus_qwen_capacity_sweep.py \
  --config configs/experiment/baselines/perinucleus_official/qwen_capacity_sweep__baseline_perinucleus_official.yaml \
  --force
```

## Expected Outputs

- `docs/baseline_perinucleus_qwen_capacity_sweep.md`
- `results/tables/baseline_perinucleus_qwen_capacity_sweep.csv`
- `results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_compute.json`
