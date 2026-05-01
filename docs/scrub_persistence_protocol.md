# Scrubbing / Persistence Protocol

## Status

`prepared_for_review_do_not_launch`

This protocol prepares a model-side persistence comparison between our compiled
tokenizer-aligned evidence method and the Qwen-adapted official
Scalable/Perinucleus baseline. It must not be launched until the configs below
are reviewed and an execution runner is explicitly approved.

This is not an output-side formatting robustness experiment. Existing
truncate-tail and delimiter-scrub failures remain outside the current guarantee
and must not be reframed as successes.

## Preconditions

Satisfied:

- Ours clean Qwen final verification: `48/48`.
- Official Qwen-adapted Perinucleus final verification: `48/48`.
- Ours TinyBench utility: `utility_pass=true`, base `0.6035317339934293`,
  adapter mean `0.6080464224183832`, max absolute drop
  `0.003134674718503816`.
- Perinucleus TinyBench utility: `utility_pass=true`, base
  `0.6035317339934293`, adapter `0.6191832009104691`, signed drop
  `-0.01565146691703978`.
- Artifact-backed wrong-payload claim-acceptance subset exists for both
  methods.

Still incomplete:

- Full FAR/null calibration is not complete.
- Fresh base-Qwen, wrong-owner, non-owner-probe, and organic-prompt null sets are
  not available.
- Scrubbing execution backend is not implemented in this repository yet.

## Goal

Measure whether model-side post-training weakens ownership evidence differently
for:

- ours: structured tokenizer-aligned payload recovery;
- Perinucleus: key-response fingerprint detection.

The claim under test is persistence under mild model-side SFT, not broad
robustness to arbitrary output transformations.

## Methods

### Ours

Source artifacts:

- `results/tables/g1_payload_seed_scale.csv`
- `results/processed/paper_stats/g1_run_inclusion_list.json`
- `results/processed/paper_stats/ours_tinybench_utility_summary.json`

Adapter source policy:

- Use existing final positive case adapters only.
- Do not retrain or change payload training hyperparameters.
- Resolve each adapter from the final case train run:
  `train_summary_path/../checkpoints/hf_last`.

Metrics:

- decoded payload recovery;
- RS-aware recovery;
- slot bucket accuracy;
- exact representative accuracy;
- TinyBench retained after scrubbing;
- optional wrong-payload claim accept-rate subset if the same artifact replay
  remains valid.

### Perinucleus

Source artifacts:

- `configs/experiment/baselines/perinucleus_official/qwen_frozen_candidate__baseline_perinucleus_official.yaml`
- `results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json`
- `results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_summary.json`

Adapter source policy:

- Use the frozen Qwen-adapted official-code adapter only.
- Do not change LoRA rank, target modules, epoch count, fingerprint count, or
  selected candidate.

Metrics:

- fingerprint exact accuracy at query budgets `M=1,3,5,10`;
- target response probability/rank if available;
- TinyBench retained after scrubbing;
- optional wrong-payload claim accept-rate subset if the same artifact replay
  remains valid.

## Scrubbing Settings

### Stage 1: pilot generic SFT scrub

Run only after protocol review.

- cases: payloads `U00,U03,U12,U15`, seed `17`;
- scrub type: generic instruction LoRA SFT;
- dataset: existing cached benign/instruction data only; do not download during
  job execution unless explicitly approved;
- epochs: `1`;
- max training examples: `512`;
- LoRA rank: `16`;
- target modules: all linear Qwen modules if feasible, otherwise record exact
  modules;
- base: scrub the already fingerprinted/adapted model or adapter stack, not the
  clean base model;
- utility: run TinyBench after scrub.

Pass-to-expand criteria:

- both methods complete all pilot jobs;
- utility remains within `0.05` absolute drop from the same base TinyBench
  reference;
- at least one method retains measurable ownership evidence so the stress test
  is informative;
- no evaluator or adapter-loading mismatch appears.

### Stage 2: final split expansion

Run only if Stage 1 is reviewed.

- cases: payloads `U00,U03,U12,U15`, seeds `17,23,29`;
- same SFT scrub hyperparameters as Stage 1 unless the protocol is amended
  before launch;
- no final-matrix feedback may be used to change method hyperparameters.

### Optional settings

These are not part of the first launch:

- mismatched-target SFT;
- model merge;
- stronger multi-epoch scrub;
- non-Qwen scrub.

## Required Outputs

Prepared configs:

- `configs/experiment/robustness/scrub_ours.yaml`
- `configs/experiment/robustness/scrub_perinucleus.yaml`

Expected future outputs after runner implementation:

- `results/tables/scrub_persistence_ours.csv`
- `results/tables/scrub_persistence_perinucleus.csv`
- `results/tables/scrub_persistence_comparison.csv`
- `results/processed/paper_stats/scrub_persistence_ours_summary.json`
- `results/processed/paper_stats/scrub_persistence_perinucleus_summary.json`
- `results/processed/paper_stats/scrub_persistence_comparison_summary.json`
- `docs/scrub_persistence_result.md`

## Reporting Rules

Allowed after successful execution:

- "We evaluate model-side SFT persistence for both methods."
- "Under this bounded scrub setting, method X retains/fails ownership evidence."

Forbidden unless supported by additional experiments:

- "robust to truncation";
- "robust to delimiter destruction";
- "broadly robust to post-processing";
- "complete FAR-calibrated persistence";
- "Perinucleus has FAR=1 from the wrong-payload claim subset";
- "Perinucleus fails generally."

## Launch Decision

```yaml
run_required: true
runner_implemented: false
launch_allowed_now: false
next_step: implement and review scrub_persistence runner before submitting jobs
```
