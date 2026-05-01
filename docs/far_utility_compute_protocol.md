# FAR / Utility / Compute Comparison Protocol

Status: terminology-audited on 2026-04-30. No fresh FAR/null jobs were launched.

## Purpose

The clean Qwen ownership result is tied:

- Ours compiled ownership: `48/48`.
- Qwen-adapted official Scalable/Perinucleus: `48/48`.

Therefore, the next differentiating comparison must audit full null false
accepts, structured claim accept rates, utility, compute, and query budget under
a matched protocol.

## Completeness Audit

| Requirement | Ours | Official Perinucleus | Status |
|---|---|---|---|
| Clean success | G1 `48/48`, success rate `1.0` | Final matrix `48/48`, success rate `1.0` | complete |
| Query budgets `M=1,3,5,10` | Missing; current paper protocol is B0 `M=4` exact-slot verifier | Present for clean success only: each budget has `12/12` | incomplete |
| FAR under null models | Missing for main method | Missing | incomplete |
| Wrong-payload claim accept-rate subset | Artifact-backed subset exists: `0.0` at `M=1,3,5,10` | Artifact-backed original-binary-detector subset exists: `1.0` at `M=1,3,5,10` | partial, not full FAR |
| FAR under wrong owner / non-owner / organic prompts | Missing | Missing | incomplete |
| Utility score on same benchmark | TinyBench sanity exists: adapter mean `0.6080` | TinyBench sanity exists: adapter `0.6192` | complete sanity only |
| Utility drop vs base model | Base `0.6035`, adapter mean `0.6080`, signed drop `-0.00451` | Base `0.6035`, adapter `0.6192`, signed drop `-0.01565` | complete sanity only |
| Train wall-clock | Missing normalized final value; requested GPU-hours exist | Capacity-sweep selected arm seconds `1802.0` exist | incomplete |
| GPU hours | Requested accounting exists for ours; not normalized to Perinucleus | Partial final eval seconds and capacity-sweep seconds exist | incomplete |
| Trainable parameters | Missing in matched comparison table | LoRA parameter metadata exists in capacity sweep but not normalized | incomplete |
| Number of training examples | Exists implicitly in contracts/summaries but not normalized | `num_fingerprints=64` exists | incomplete |
| Eval query cost | Missing for ours over `M=1,3,5,10` | Final eval forwards total `228` exists | incomplete |
| Confidence intervals | Clean success CIs exist | Clean success CIs exist | incomplete for FAR/utility/compute |

Decision: the matched FAR / utility / compute comparison is not complete. The
existing wrong-payload result is a structured payload-claim accept-rate subset,
not a full FAR/null calibration and not a general failure of Perinucleus as a
binary fingerprint detector.

## Protocol

### Methods

Compare exactly two paper-facing methods:

1. `ours_compiled_ownership`
2. `scalable_fingerprinting_perinucleus_official_qwen_final`

The Perinucleus row must be labeled:

```text
Qwen-adapted official Scalable/Perinucleus baseline
```

### Final Positive Split

Use the same positive matrix for both methods:

- model family: `Qwen/Qwen2.5-7B-Instruct`
- payloads: `U00`, `U03`, `U12`, `U15`
- seeds: `17`, `23`, `29`
- positive cases: `12`
- query budgets: `M=1,3,5,10`

### Null FAR And Structured Claim Checks

Use fixed thresholds selected before final evaluation.

Required null sources:

- `base_qwen`: unadapted `Qwen/Qwen2.5-7B-Instruct`
- `wrong_payload_null`: adapted run tested against the wrong claimed payload;
  report this as a wrong-payload claim accept rate unless it is pooled with a
  full frozen null calibration
- `wrong_owner_null`: valid evidence tested under the wrong owner claim
- `non_owner_probe_null`: registered non-owner probes
- `organic_prompt_null`: organic prompts with no owner payload

Optional null sources:

- `unprotected_qwen_finetuned`, only if such a checkpoint already exists
- `non_qwen_llama3_1_8b`, only if cached and authorized

Report full FAR for each method and query budget only after the required null
sets are generated:

- false accept count
- negative count
- observed FAR
- Wilson interval
- per-null-set FAR
- pooled FAR

Report structured claim checks separately:

- wrong-payload claim accept count
- wrong-payload claim negative count
- wrong-payload claim accept rate
- wrong-payload claim Wilson interval
- wrong-owner claim accept rate, once owner IDs are available

Do not write "Perinucleus FAR=1" from the wrong-payload subset. The original
Perinucleus baseline is a binary fingerprint detector; the subset tests whether
it binds a decoded payload claim, which is not its native verification object.

### Utility

Use the same utility benchmark for both methods.

Default:

```text
tinyBenchmarks
```

Reason: TinyBench has already run for the Perinucleus frozen candidate and is
the lowest-cost benchmark currently wired in the repository. If OpenLLM is
required, that must be a separate reviewed escalation.

Report:

- base model utility
- method utility
- absolute drop vs base
- confidence interval if available
- benchmark/task-level metrics

Ours must be evaluated on the same benchmark before any utility comparison can
be stated. Perinucleus existing TinyBench outputs may be reused only if the base
model, tasks, evaluator version, and prompt settings match the ours run.

Prepared ours utility runner:

```bash
python3 scripts/run_ours_tinybench_utility.py \
  --config configs/experiment/comparison/ours_tinybench_utility.yaml \
  --dry-run
```

The full utility run evaluates the base model once and all 12 final positive
case adapters. It requires a GPU allocation.

### Compute

Report both requested and observed compute separately.

Required fields:

- GPU type / partition
- number of GPUs
- CPU count
- memory
- train wall-clock seconds
- eval wall-clock seconds
- utility wall-clock seconds
- requested GPU-hours
- trainable parameters
- number of training examples
- eval query count
- model forward count, if available
- adapter artifact size

Do not claim compute efficiency unless these fields are available for both
methods under the same accounting convention.

## Prepared Configs

- `configs/experiment/comparison/far_utility_compute_ours.yaml`
- `configs/experiment/comparison/far_utility_compute_perinucleus.yaml`
- `configs/experiment/comparison/ours_tinybench_utility.yaml`

These files are job contracts, not launched runs.

## Expected Outputs

Per-method outputs:

- `results/processed/paper_stats/matched_far_utility_compute_ours_summary.json`
- `results/tables/matched_far_utility_compute_ours_far_cases.csv`
- `results/tables/matched_far_utility_compute_ours_utility.csv`
- `results/processed/paper_stats/matched_far_utility_compute_ours_compute.json`
- `results/processed/paper_stats/matched_far_utility_compute_perinucleus_summary.json`
- `results/tables/matched_far_utility_compute_perinucleus_far_cases.csv`
- `results/tables/matched_far_utility_compute_perinucleus_utility.csv`
- `results/processed/paper_stats/matched_far_utility_compute_perinucleus_compute.json`

Final aggregation outputs:

- `results/tables/matched_comparison_far_utility_compute.csv`
- `results/tables/matched_comparison_far_utility_compute.tex`
- `docs/matched_comparison_text.md`
- `results/processed/paper_stats/matched_comparison_far_utility_compute_summary.json`

Ours TinyBench utility outputs:

- `docs/ours_tinybench_utility.md`
- `results/tables/ours_tinybench_utility.csv`
- `results/processed/paper_stats/ours_tinybench_utility_summary.json`
- `results/processed/paper_stats/ours_tinybench_utility_compute.json`

## Final Decision

```yaml
run_required: true
```

Exact review commands to run before any execution-mode job:

```bash
cd "$REPO_HOME"

python3 scripts/run_matched_far_utility_compute.py \
  --config configs/experiment/comparison/far_utility_compute_ours.yaml \
  --dry-run

python3 scripts/run_matched_far_utility_compute.py \
  --config configs/experiment/comparison/far_utility_compute_perinucleus.yaml \
  --dry-run

python3 scripts/run_ours_tinybench_utility.py \
  --config configs/experiment/comparison/ours_tinybench_utility.yaml \
  --dry-run
```

First run ours TinyBench utility on a GPU allocation:

```bash
UTILITY_CONFIG=configs/experiment/comparison/ours_tinybench_utility.yaml \
bash scripts/submit_ours_tinybench_utility.sh
```

After `ours_tinybench_utility_summary.json` exists, rerun the artifact-backed
comparison. These commands do not run fresh FAR/null model inference and do not
complete full FAR/null calibration; unavailable null sets are explicitly marked
in the output. The wrong-payload column produced here is a claim accept-rate
subset.

```bash
python3 scripts/run_matched_far_utility_compute.py \
  --config configs/experiment/comparison/far_utility_compute_ours.yaml \
  --execute \
  --force

python3 scripts/run_matched_far_utility_compute.py \
  --config configs/experiment/comparison/far_utility_compute_perinucleus.yaml \
  --execute \
  --force
```

Optional Slurm submission for the same artifact-backed execution. The wrapper
defaults to the CPU `Intel` partition because this mode does not require GPU:

```bash
RUN_MODE=execute \
COMPARISON_CONFIG=configs/experiment/comparison/far_utility_compute_ours.yaml \
bash scripts/submit_matched_far_utility_compute.sh

RUN_MODE=execute \
COMPARISON_CONFIG=configs/experiment/comparison/far_utility_compute_perinucleus.yaml \
bash scripts/submit_matched_far_utility_compute.sh
```

Runner status:

```yaml
runner_exists: true
runner_mode_supported:
  dry_run: true
  write_plan: true
  execute: artifact_backed_partial
full_far_complete: false
runner_execute_scope: claim-conditioned wrong-payload claim accept rate from archived final artifacts, existing utility where available, partial compute normalization
runner_execute_missing: fresh base-Qwen/null-model outputs, wrong-owner identity protocol, non-owner probe outputs, organic prompt outputs
```

Expected output paths are the per-method and final aggregation paths listed
above.

Prompt T, the scrubbing/persistence differentiator, may proceed after the
matched comparison table records TinyBench utility for both ours and
Perinucleus and the protocol preserves this terminology. Full FAR can remain
marked incomplete, but the scrubbing protocol must not call the wrong-payload
claim subset a full FAR result.
