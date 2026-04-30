# Baseline Paper Registry

Generated at: `20260430T060230399389Z`

## Decision

The paper now has one paper-ready external active ownership baseline: `scalable_fingerprinting_perinucleus_official_qwen_final`. It must be reported with the `Qwen-adapted official Scalable/Perinucleus baseline` label.

The legacy adapted `baseline_perinucleus` artifacts remain excluded from Scalable Fingerprinting claims.

## Registry

| method_id | main_table_status | target | success | rate | paper_ready | required_label |
|---|---|---:|---:|---:|---|---|
| fixed_representative | appendix_or_internal_ablation | 12 | 12 | 1.000 | True |  |
| uniform_bucket | appendix_or_internal_ablation | 12 | 12 | 1.000 | True |  |
| english_random_active_fingerprint | appendix_negative_diagnostic | 12 | 0 | 0.000 | True |  |
| kgw_provenance_control | excluded_task_mismatched_control | 0 | 0 | 0.000 | False | task-mismatched provenance control |
| scalable_fingerprinting_perinucleus_official_qwen_final | eligible_with_adaptation_label | 48 | 48 | 1.000 | True | Qwen-adapted official Scalable/Perinucleus baseline |
| chain_hash_qwen_v1 | not_eligible_pending_execution | 48 | 0 | 0.000 | False |  |
| legacy_adapted_perinucleus_diagnostic | excluded_do_not_use_for_scalable_claim | 0 | 0 | 0.000 | False | not Scalable Fingerprinting |

## Guardrails

- Do not use `results/tables/baseline_perinucleus.csv` as the successful Scalable/Perinucleus result.
- Use `results/tables/baseline_perinucleus_official_qwen_final.csv` for the official Qwen-adapted Perinucleus result.
- Keep valid method failures in denominators; do not convert failures into exclusions.
- KGW/PostMark-style provenance controls must stay task-mismatched controls, not ownership baselines.
