# Full FAR And Payload-Claim Benchmark Protocol

Status: artifact-backed claim subset executed on Chimera H200. The registered
base-Qwen fresh-null backend is implemented and ready to run; organic and
non-owner prompt-bank inference remains pending.

## Purpose

Clean Qwen verification is tied:

- ours: `48/48`;
- Qwen-adapted official Scalable/Perinucleus: `48/48`.

The next benchmark separates three quantities that must not be conflated:

- clean correct-claim acceptance;
- structured wrong-payload / wrong-owner claim accept rate;
- full null FAR under base/null models and non-owner or organic prompts.

The artifact-backed wrong-payload result is a claim-acceptance subset. It is
useful for distinguishing structured payload-claim verification from binary
fingerprint detection, but it is not a full FAR/null calibration.

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
- registered-probe null rows are claim-conditioned over all `U00` to `U15`
  claim payload labels and seeds `17,23,29`.

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
artifact_claim_subset_backend: executed
fresh_registered_probe_backend: implemented
fresh_prompt_bank_backend: pending
plan_generation: supported
claim_rows_complete: true
full_far_complete: false
```

The plan generation commands are:

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

Artifact-backed claim subset execution on Chimera H200:

```bash
RUN_MODE=execute \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_full_far_payload_claim_benchmark.sh
```

Fresh base-Qwen registered-probe null execution on Chimera H200:

```bash
RUN_MODE=execute-registered-null \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_full_far_payload_claim_benchmark.sh
```

## Executed Artifact Subset

The current H200 execution produced:

```text
status = completed_artifact_subset
full_far_complete = False
claim_rows_complete = True
```

Completed rows:

| Row status | Count |
|---|---:|
| `completed_artifact_replay` | 48 |
| `completed_artifact_replay_budget_projection` | 768 |
| `completed_artifact_replay_task_mismatch_binary_detector` | 720 |

Pending rows:

| Row status | Count |
|---|---:|
| `not_executed_fresh_null_inference_required` | 11200 |
| `not_executed_owner_claim_not_encoded` | 480 |
| `not_executed_owner_claim_not_supported_by_binary_detector` | 480 |

Key artifact-backed claim metrics:

| Method | Metric | M | Trials | Accepts | Rate | 95% CI |
|---|---|---:|---:|---:|---:|---|
| Ours | clean correct claim acceptance | 1/3/5/10 | 12 each | 12 each | 1.0 | [0.758, 1.0] |
| Ours | wrong-payload claim acceptance | 1/3/5/10 | 180 each | 0 each | 0.0 | [0.0, 0.0209] |
| Original Perinucleus | clean correct claim acceptance | 1/3/5/10 | 12 each | 12 each | 1.0 | [0.758, 1.0] |
| Original Perinucleus | wrong-payload claim acceptance | 1/3/5/10 | 180 each | 180 each | 1.0 | [0.979, 1.0] |

Interpretation:

- Clean binary ownership success is tied in this artifact subset.
- Ours rejects wrong decoded payload claims under the artifact-backed structured verifier.
- Original Perinucleus accepts wrong-payload claims here because it is a binary fingerprint detector, not because a calibrated full FAR test failed.
- These rows must be reported as `wrong_payload_claim_accept_rate`, not as full FAR.

## Next Fresh-Null Slice

The `execute-registered-null` mode runs only the required base-Qwen registered
probe null rows:

```text
methods: 2
claim payloads: 16
seeds: 3
query budgets: 4
required null model: base_qwen
fresh registered-probe rows: 384
```

Expected status after this slice:

| Row status | Expected count |
|---|---:|
| `completed_fresh_registered_null` | 384 |
| `not_executed_optional_null_model_not_enabled` | 768 |
| `not_executed_fresh_null_inference_required` | 10048 |

`full_far_complete` must remain `False` until organic and non-owner prompt-bank
rows are implemented and executed.

## Gate

Do not move to final manuscript integration until:

- full FAR and wrong-claim rates are separately reported;
- original Perinucleus task mismatch is explicitly labeled;
- payload-adapted Perinucleus is implemented or explicitly deferred;
- no post-hoc threshold tuning is used.
