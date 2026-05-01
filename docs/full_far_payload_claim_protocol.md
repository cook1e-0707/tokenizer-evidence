# Full FAR And Payload-Claim Benchmark Protocol

Status: prepared for review. No fresh model inference has been launched.

## Purpose

Clean Qwen verification is tied:

- ours: `48/48`;
- Qwen-adapted official Scalable/Perinucleus: `48/48`.

The next benchmark separates three quantities that must not be conflated:

- clean correct-claim acceptance;
- structured wrong-payload / wrong-owner claim accept rate;
- full null FAR under base/null models and non-owner or organic prompts.

The existing artifact-backed wrong-payload result is only a claim-acceptance
subset. This protocol upgrades it into a pre-registered full benchmark plan.

## Methods

| Method | Native verification object | Required label |
|---|---|---|
| `ours_compiled_ownership` | structured payload claim verification | Ours compiled tokenizer-aligned ownership |
| `scalable_fingerprinting_perinucleus_official_qwen_final` | binary key-response fingerprint detection | Qwen-adapted official Scalable/Perinucleus |

Original Perinucleus must not be described as a structured payload verifier. If
its wrong-payload claim accept rate is high, the correct interpretation is that
the original binary detector does not bind decoded payload claims.

## Frozen Split

Positive split:

- true payloads: `U00,U03,U12,U15`;
- seeds: `17,23,29`;
- query budgets: `M=1,3,5,10`;
- true owner: `owner_qwen_final`.

Wrong-payload claims:

- claim labels: all `U00` to `U15` labels except the true payload;
- cases per method: `4 true payloads x 15 wrong labels x 3 seeds x 4 budgets = 720`;
- report as `wrong_payload_claim_accept_rate`, not full FAR.

Wrong-owner claims:

- wrong owner IDs: `pseudo_owner_00` to `pseudo_owner_09`;
- report separately as `wrong_owner_claim_accept_rate`.

Full null FAR:

- required null model: base `Qwen/Qwen2.5-7B-Instruct`;
- optional null models: cached non-Qwen Llama 3.1 8B and unprotected Qwen fine-tune;
- null prompt/probe sets: base-model registered probes, non-owner probes, organic prompts;
- organic prompts: at least `1000` trials.

## Threshold Rules

- Thresholds must be frozen before final evaluation.
- No final-row feedback may change thresholds or probe ordering.
- All valid failed rows remain in the denominator.
- Report Wilson intervals for all binary rates.

## Required Outputs After Execution

- `results/tables/full_far_payload_claim.csv`
- `results/tables/full_far_payload_claim.tex`
- `results/processed/paper_stats/full_far_payload_claim_summary.json`
- `figures/far_roc_curves.pdf`
- `figures/payload_claim_heatmap.pdf`

Plan-only outputs:

- `results/tables/full_far_payload_claim_plan.csv`
- `results/processed/paper_stats/full_far_payload_claim_plan_summary.json`

## Runner State

```yaml
runner: scripts/run_full_far_payload_claim_benchmark.py
fresh_inference_backend: not_enabled_in_plan_mode
plan_generation: supported
launch_allowed_now: false
```

The first implementation step is to generate and review the case plan:

```bash
python3 scripts/run_full_far_payload_claim_benchmark.py \
  --config configs/experiment/comparison/full_far_payload_claim.yaml \
  --dry-run

python3 scripts/run_full_far_payload_claim_benchmark.py \
  --config configs/experiment/comparison/full_far_payload_claim.yaml \
  --write-plan \
  --force
```

Optional Slurm plan generation on Chimera H200:

```bash
RUN_MODE=write-plan \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_full_far_payload_claim_benchmark.sh
```

## Gate

Do not move to final manuscript integration until:

- full FAR and wrong-claim rates are separately reported;
- original Perinucleus task mismatch is explicitly labeled;
- payload-adapted Perinucleus is implemented or explicitly deferred;
- no post-hoc threshold tuning is used.
