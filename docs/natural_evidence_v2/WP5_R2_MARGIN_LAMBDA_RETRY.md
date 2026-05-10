# WP5-R2: Margin Lambda Sweep Retry

## Root Cause

WP5 (job 851373) failed the teacher-forced gate because margin loss signal was
too weak relative to CE loss:

- `margin_loss_weighted / total_loss ≈ 7.4%`
- `margin_raw ≈ 0.0164` per sample
- CE loss dominated gradient → adapter optimized LM quality, not mass shift
- Result: rank-1 improved on easy surfaces (+536 net) but mass collapsed
  (protected < base on 512/512 prompts)

## Changes for WP5-R2

| Parameter | WP5 | WP5-R2 | Rationale |
|---|---|---|---|
| `--margin-lambda` | 5.0 | 30.0 | Increase margin signal to ~40% of total loss |
| `--max-steps` | 64 | 256 | More epochs (512 rows × 32 epochs vs 8) |
| `--learning-rate` | 5e-5 | 1e-4 | Faster convergence with more steps |
| `--max-rows` | 512 | 512 | Keep same training data |
| `--lora-r` | 32 | 32 | Keep same capacity |
| `--lora-alpha` | 64 | 64 | Keep same scaling |
| `--task-ce-weight` | 1.0 | 1.0 | Keep CE as base objective |
| `--margin-tau` | 0.15 | 0.15 | Keep same margin threshold |

## Expected Outcomes

With margin_lambda=30.0 and CE≈0.7:
- Expected weighted margin ≈ 30 × 0.05 = 1.5 → margin/total ≈ 68%
- This should force the adapter to prioritize mass shift over LM quality
- Risk: CE loss degrades too much → output quality suffers
- Mitigation: 256 steps gives time for CE to stabilize first

## Gate Targets (same as WP5)

- `protected_target_bucket_mass_lift_vs_base >= +0.15`
- `protected_target_bucket_mass_lift_vs_task_only >= +0.10`
- `target_bucket_rank1_rate >= 0.70`
- `protected_median_target_margin > 0`

## Fallback

If WP5-R2 also fails (margin too strong → CE collapse):
- Try lambda=15.0 with 512 steps (gentler but longer)
- Or try two-phase training: phase 1 CE-only (128 steps), phase 2 CE+margin (128 steps)
