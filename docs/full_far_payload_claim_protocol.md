# Full FAR And Payload-Claim Benchmark Protocol

Status: artifact-backed claim subset and required base-Qwen registered-probe
null slice executed on Chimera H200. The required base-Qwen organic prompt-bank
backend is implemented next; non-owner prompt-bank and optional null-model
inference remain pending.

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
fresh_registered_probe_backend: executed_for_required_base_qwen
fresh_organic_prompt_backend: implemented_for_required_base_qwen
fresh_non_owner_prompt_backend: pending
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

Fresh base-Qwen organic prompt-bank null execution is two-stage. Stage 1 runs on
GPUs and writes prompt-level cache rows. Stage 2 runs offline from that cache and
expands the full organic FAR row shards. Do not use the older
`execute-organic-null-array` row-level path unless intentionally debugging it.

Parallel Stage 1 organic prompt-cache generation on 4 H200s plus 6 A100s uses
one global 10-shard split and two separate Slurm arrays:

```bash
SCR=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim
CACHE_DIR=$SCR/shards/organic-prompt-cache-10way
ROW_SHARD_DIR=$SCR/shards/organic-prompts-10way-from-cache
mkdir -p "$CACHE_DIR" "$ROW_SHARD_DIR"

# Shards 0-3 on H200 / pomplun.
GLOBAL_SHARD_COUNT=10 \
LOCAL_SHARD_COUNT=4 \
SHARD_OFFSET=0 \
MAX_PARALLEL=4 \
CACHE_OUTPUT_DIR=$CACHE_DIR \
PARTITION=pomplun \
ACCOUNT=cs_yinxin.wan \
QOS=pomplun \
GRES=gpu:h200:1 \
TIME_LIMIT=30-00:00:00 \
CHECKPOINT_INTERVAL=1 \
RUN_MODE=generate-organic-cache-array \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_full_far_payload_claim_benchmark_array.sh

# Shards 4-9 on A100.
GLOBAL_SHARD_COUNT=10 \
LOCAL_SHARD_COUNT=6 \
SHARD_OFFSET=4 \
MAX_PARALLEL=6 \
CACHE_OUTPUT_DIR=$CACHE_DIR \
PARTITION=DGXA100 \
ACCOUNT=pi_yinxin.wan \
QOS=scavenger_unlim \
GRES=gpu:A100:1 \
TIME_LIMIT=30-00:00:00 \
CHECKPOINT_INTERVAL=1 \
RUN_MODE=generate-organic-cache-array \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_full_far_payload_claim_benchmark_array.sh
```

Do not set `TIME_LIMIT=""` for long jobs. Omitting `sbatch --time` can fall back
to a short partition default even when the partition maximum is much larger.
Use an explicit limit at or below the partition/QOS maximum. Verify with:

```bash
sinfo -p pomplun,scavenger,DGXA100 -o "%P %G %l %D %t %N"
for p in pomplun scavenger DGXA100; do scontrol show partition "$p"; done | egrep 'PartitionName=|MaxTime=|DefaultTime=|AllowQos=|State=|TRES=|Gres='
sacctmgr -p show qos pomplun,scavenger,scavenger_unlim format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU | column -ts '|'
```

Stage 1 array jobs write prompt-cache shard files under:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim/shards/organic-prompt-cache-10way/
```

Progress check:

```bash
find "$CACHE_DIR" -maxdepth 1 \
  -name 'full_far_payload_claim_organic_prompt_cache_shard_*_of_010.csv' \
  -print -exec wc -l {} \;
```

After all prompt-cache shards finish, run Stage 2 as a CPU array. Do not submit
this step to `pomplun`: it is CPU-only post-processing and does not request
`--gres`, so it should use a CPU partition such as `Intel6240` or `Intel6326`.
The array parallelizes by output row shard; increasing `--cpus-per-task` alone
does not speed up the serial Python loop much.

```bash
PARTITION=Intel6240 \
ARRAY=1 \
MAX_PARALLEL=10 \
CPUS_PER_TASK=16 \
MEM=120G \
TIME_LIMIT=4-00:00:00 \
CACHE_DIR=$CACHE_DIR \
ROW_SHARD_DIR=$ROW_SHARD_DIR \
FULL_FAR_CONFIG=configs/experiment/comparison/full_far_payload_claim.yaml \
bash scripts/submit_build_full_far_organic_from_cache.sh
```

Stage 2 writes row shards compatible with the existing shard aggregator. They
must not write `results/tables/full_far_payload_claim.csv` directly. Aggregate
after Stage 2:

```bash
python3 scripts/aggregate_full_far_payload_claim_shards.py \
  --config configs/experiment/comparison/full_far_payload_claim.yaml \
  --shard-dir "$ROW_SHARD_DIR" \
  --fresh-null-mode organic-prompts \
  --expected-shard-count 10 \
  --force
```

If the registered-probe slice needs to be recomputed together with organic
prompts:

```bash
RUN_MODE=execute-registered-and-organic-null \
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

## Executed Registered-Probe Fresh-Null Slice

The `execute-registered-null` mode ran only the required base-Qwen registered
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

Observed registered-probe null metrics after the H200 run:

| Method | M | Trials | False accepts | Rate | 95% CI high |
|---|---:|---:|---:|---:|---:|
| Ours | 1/3/5/10 | 48 each | 0 each | 0.0 | 0.0741 |
| Original Perinucleus | 1/3/5/10 | 48 each | 0 each | 0.0 | 0.0741 |

This is a small registered-probe null slice, not a complete FAR result.

## Next Organic Prompt-Bank Slice

Organic prompt-bank execution now uses the two-stage cache backend. The older
`execute-organic-null` and `execute-organic-null-array` row-level modes are
guarded because they repeat too much per-row work; they require
`ALLOW_SLOW_ORGANIC_ROW_EXECUTION=1` to run.

```text
methods: 2
organic prompts per budget per method: 1000
query budgets: 4
prompt-cache rows: 1000
fresh organic rows after offline expansion: 8000
required null model: base_qwen
```

Decision rules:

- Ours: a row is a false accept if any prompt in the row's query-budget window
  decodes the claimed structured payload under the deterministic verifier.
- Original Perinucleus: a row is a false accept if all selected fingerprint
  responses match on the organic prompt window.

Expected status after this slice, assuming the registered slice is preserved:

| Row status | Expected count |
|---|---:|
| `completed_fresh_organic_null` | 8000 |
| `completed_fresh_registered_null` | 384 |
| `not_executed_optional_null_model_not_enabled` | 768 |
| `not_executed_fresh_null_inference_required` | 2048 |

## Gate

Do not move to final manuscript integration until:

- full FAR and wrong-claim rates are separately reported;
- original Perinucleus task mismatch is explicitly labeled;
- payload-adapted Perinucleus is implemented or explicitly deferred;
- no post-hoc threshold tuning is used.
