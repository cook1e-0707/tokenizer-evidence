# Official Scalable Fingerprinting / Perinucleus Protocol

Status: Step 1 clone and inspection completed locally on 2026-04-28. Smoke,
anchor, and matched Qwen runs are not yet executed.

## Rule

The existing no-train Perinucleus v0 package is renamed
`perinucleus_no_train_diagnostic`. It must not be called a Scalable
Fingerprinting baseline and must not enter the main comparison table.

The trained baseline described here is the only path by which the paper can
claim comparison to Scalable Fingerprinting / Perinucleus.

## Official Repository

| Field | Value |
|---|---|
| paper | `Scalable Fingerprinting of Large Language Models` |
| paper URL | `https://arxiv.org/abs/2502.07760` |
| OpenReview | `https://openreview.net/forum?id=CRyOyiVvvJ` |
| official repository | `https://github.com/SewoongLab/scalable-fingerprinting-of-llms` |
| local clone | `external_baselines/scalable_fingerprinting_official` |
| commit hash | `fdceaba14bd3e89340916a6a40e27c945d48460e` |
| license | MIT |
| dependency file | `requirements.txt` |
| main scripts inspected | `generate_finetuning_data.py`, `finetune_multigpu.py`, `check_fingerprints.py`, `eval_utility.py`, `fingerprint_models.sh` |
| README commands run | no; local machine lacks `torch` and CUDA |
| smoke config | `configs/experiment/baselines/perinucleus_official/smoke__baseline_perinucleus_official_qwen.yaml` |
| smoke manifest | `manifests/baseline_perinucleus_official_smoke/smoke_manifest.json` |

The official dependency stack pins `torch==2.3.1`, `transformers==4.44.2`,
`deepspeed==0.14.5`, `accelerate==0.32.1`, `datasets==2.20.0`, `peft==0.11.1`,
and `lm_eval==0.4.3`, among many other packages.

## Required Pipeline Stages

A faithful baseline must include all of the following:

1. Fingerprint key generation.
2. Perinucleus response selection using the base-model next-token distribution.
3. Fingerprint insertion by fine-tuning.
4. Fingerprint checking on the fingerprinted model.
5. Utility evaluation.
6. Query-budget and FAR evaluation, or compatible reporting with explicit
   differences from the project B0 protocol.

The official code contains direct entry points for these stages:

- `generate_finetuning_data.py` supports `--key_response_strategy perinucleus`,
  `--perinucleus_model`, `--nucleus_t`, `--nucleus_k`, and `--use_chat_template`.
- `finetune_multigpu.py` fine-tunes the target model on generated fingerprints
  and writes `results/saved_models/<config_hash>/final_model`.
- `check_fingerprints.py` loads the fingerprinting config and checks the
  fingerprinted model.
- `eval_utility.py` evaluates model utility.

## Smoke Gate

No full Chimera run is allowed until this smoke gate passes.

Smoke test:

- model: smallest feasible model, or Qwen/Qwen2.5-7B-Instruct if the official
  code path requires the target tokenizer/chat template;
- `num_fingerprints`: 8 or 16;
- `response_length`: 1;
- one seed;
- `key_response_strategy=perinucleus`;
- `nucleus_t=0.8`;
- `nucleus_k=3`;
- `use_chat_template=true` for instruct models;
- run fingerprint generation, fine-tuning, fingerprint checking, and cheap
  utility if available.

Smoke success requires:

- fingerprint checking accuracy is above random chance;
- the fingerprinted model differs from the base model on the fingerprint
  responses;
- no causal prompt/template mismatch is observed;
- the generated fingerprints, train config, final model path, check output, and
  utility output are all recorded under scratch.

## Chimera Smoke Commands

These commands submit exactly one smoke job. They must be run before any anchor
or final matrix.

```bash
cd "$REPO_HOME"
git pull --ff-only origin main

export EXP_SCRATCH=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence
mkdir -p "$EXP_SCRATCH/baselines/perinucleus_official_smoke"

python3 scripts/submit_slurm.py \
  --manifest manifests/baseline_perinucleus_official_smoke/smoke_manifest.json \
  --registry manifests/baseline_perinucleus_official_smoke/smoke_job_registry.jsonl \
  --all-pending \
  --submit
```

After the job exits, rebuild/check the smoke status by reading the generated
summary:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("results/processed/paper_stats/baseline_perinucleus_official_smoke_summary.json")
d = json.loads(p.read_text())
print("smoke_pass =", d["smoke_pass"])
print("decision =", d["decision"])
print("base_acc =", d["base_fingerprint_accuracy"])
print("trained_acc =", d["trained_fingerprint_accuracy"])
print("base_prob =", d["base_mean_first_token_probability"])
print("trained_prob =", d["trained_mean_first_token_probability"])
print("utility_status =", d["utility_status"])
print("run_root =", d["run_root"])
PY
```

The smoke job writes large files only under:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_official_smoke/
```

Repo-local smoke outputs are limited to:

- `docs/baseline_perinucleus_official_smoke_result.md`
- `results/tables/baseline_perinucleus_official_smoke.csv`
- `results/processed/paper_stats/baseline_perinucleus_official_smoke_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_official_smoke_compute.json`

The smoke runner validates the fixed official commit, then runs:

```text
generate_finetuning_data.py
-> check_fingerprints.py on the base model
-> response-probability scoring on the base model
-> finetune_multigpu.py
-> check_fingerprints.py on the fingerprinted model
-> response-probability scoring on the fingerprinted model
-> eval_utility.py
```

The smoke passes only if fingerprint accuracy and expected response probability
increase over the base model, the trained accuracy is above random-chance proxy,
chat-template mode is active, utility sanity completes, and all official stages
complete.

Manual fallback, if the project manifest machinery is unavailable:

```bash
cd "$REPO_HOME"
export EXP_SCRATCH=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence
export OFFICIAL_REPO="$REPO_HOME/external_baselines/scalable_fingerprinting_official"

if [ ! -d "$OFFICIAL_REPO/.git" ]; then
  mkdir -p "$REPO_HOME/external_baselines"
  git clone https://github.com/SewoongLab/scalable-fingerprinting-of-llms.git "$OFFICIAL_REPO"
fi
git -C "$OFFICIAL_REPO" checkout fdceaba14bd3e89340916a6a40e27c945d48460e

python3 scripts/run_perinucleus_official_smoke.py \
  --config configs/experiment/baselines/perinucleus_official/smoke__baseline_perinucleus_official_qwen.yaml \
  --override runtime.run_id=manual_smoke \
  --override runtime.output_dir="$EXP_SCRATCH/baselines/perinucleus_official_smoke/manual_smoke"
```

If this fails due dependency drift, record the failing package and stack trace in
`docs/baseline_perinucleus_official_smoke_result.md` before changing any code.

## Anchor Reproduction

After smoke passes, run one official-code anchor before Qwen matched adaptation:

- preferred: Llama-family setting consistent with official scripts;
- fallback: Qwen with official `--model_path` flags if Llama access is not
  feasible;
- `key_response_strategy=perinucleus`;
- `nucleus_t=0.8`;
- `nucleus_k=3`;
- official `check_fingerprints.py` must be used.

Anchor pass is required to promote this baseline beyond grade C.

## Matched Qwen Adaptation

Only after smoke and anchor pass:

- backbone: `Qwen/Qwen2.5-7B-Instruct`;
- `use_chat_template=true`;
- seeds: `17`, `23`, `29`;
- query budgets: `1`, `3`, `5`, `10`;
- train budget: matched to the primary method or explicitly documented;
- report FAR, utility, clean verification, base-vs-fingerprinted response
  delta, and all failure cases.

## Paper Rule

If only the no-train diagnostic exists, the paper must not claim comparison to
Scalable Fingerprinting. If the official trained baseline succeeds or fails,
report it with this fidelity record and all protocol differences.
