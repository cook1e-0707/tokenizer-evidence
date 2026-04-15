# Tokenizer-Constrained Ownership Verification

This repository is a research-grade scaffold for tokenizer-constrained ownership evidence injection and verification in autoregressive LLMs. It is structured for reproducible experiments, baseline comparison, SLURM-based cluster execution, and later paper artifact preparation.

The current implementation covers:
- explicit config and manifest control planes
- schema-aware result aggregation
- tokenizer audit and bucket-spec validation
- canonical render -> parse -> verify integration
- a CPU-friendly real pilot eval path gated by frozen catalogs

Full training and paper-specific baseline integration remain follow-up work.

## Repository layout

- `configs/`: composable YAML configs for models, data, training, evaluation, attacks, sweeps, and named experiments.
- `src/`: core library code for tokenizer audits, bucket mapping, payload coding, parser/verifier flow, evaluation, and infrastructure utilities.
- `scripts/`: explicit CLIs for training, evaluation, calibration, attack runs, summarization, table/figure generation, and SLURM submission.
- `slurm/`: sbatch templates with configurable placeholders.
- `tests/`: local CPU-only tests, including synthetic parser/verifier smoke data.
- `results/`: canonical output tree for raw runs, processed aggregates, tables, and figures.
- `docs/`: concise operational documentation for experiment protocol, baselines, schemas, and Chimera workflow.

## Local workflow

1. Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Run the local smoke tests:

```bash
pytest
```

3. Launch a local training-shaped stub run:

```bash
python3 -m scripts.train --config configs/experiment/exp_alignment.yaml
```

4. Freeze a source carrier catalog before pilot manifest/eval:

```bash
python3 scripts/freeze_catalog.py \
  --source-catalog configs/data/real_pilot_catalog.yaml \
  --tokenizer-backend mock \
  --frozen-catalog-output artifacts/carrier_catalog_freeze_v1.yaml \
  --audit-report-output artifacts/tokenizer_audit_report_mock.json \
  --change-log-output artifacts/catalog_change_log.md \
  --data-config-output artifacts/real_pilot_frozen.yaml
```

5. If freeze fails, generate a remediation review instead of rewriting the catalog automatically:

```bash
python3 scripts/review_catalog_freeze.py \
  --audit-report results/processed/audits/tokenizer_audit_report__gpt2.json \
  --change-log docs/catalog_freezes/real_pilot_catalog__gpt2__v1.md \
  --output-table results/processed/audits/tokenizer_audit_remediation__gpt2.json \
  --output-review docs/catalog_freezes/real_pilot_catalog__gpt2__v1_review.md
```

6. Run a local real-mode pilot evaluation against a frozen-catalog overlay config:

```bash
python3 -m scripts.eval --config /path/to/generated_frozen_experiment_config.yaml
```

7. Run the tokenizer audit against the pilot carrier catalog.

If you want a real Hugging Face tokenizer, install `transformers` first. The mock path remains useful for local validation:

```bash
python3 scripts/tokenizer_audit.py \
  --config configs/experiment/exp_recovery.yaml \
  --tokenizer-backend mock \
  --no-strict
```

8. Run the parser/verifier synthetic smoke path:

```bash
python3 scripts/smoke_verify.py
```

## Config workflow

Experiment configs use YAML composition through an `includes:` list. Each script also supports repeated `--override key=value` arguments for explicit command-line overrides.

Example:

```bash
python3 -m scripts.train \
  --config configs/experiment/exp_alignment.yaml \
  --override run.seed=13 \
  --override model.name=tiny-debug-v2
```

Every run writes:

- `config.resolved.yaml`
- `environment.json`
- `run.log`
- optional `run.jsonl`
- one machine-readable summary JSON

Canonical eval runs also write:

- `rendered_evidence.txt`
- `rendered_evidence.json`
- `verifier_result.json`
- `eval_summary.json`

## Manifest and SLURM workflow

Generate a manifest from a sweep definition:

```bash
python3 -m scripts.make_manifest \
  --sweep configs/sweep/alignment_smoke.yaml \
  --output manifests/alignment_smoke.jsonl
```

Generate the stage-4 pilot manifest only from an experiment config that points to a frozen catalog:

```bash
python3 -m scripts.make_manifest \
  --config /path/to/generated_frozen_experiment_config.yaml \
  --dry-run
```

Dry-run SLURM rendering:

```bash
python3 -m scripts.submit_slurm \
  --manifest manifests/alignment_smoke.jsonl
```

Actual submission:

```bash
python3 -m scripts.submit_slurm \
  --manifest manifests/alignment_smoke.jsonl \
  --submit
```

## Results and aggregation

- Raw run artifacts live under `results/raw/<experiment_name>/<run_id>/`.
- Aggregated JSONL summaries are written to `results/processed/`.
- Publication tables are written to `results/tables/`.
- Lightweight SVG figures are written to `results/figures/`.

Aggregation flow:

```bash
python3 -m scripts.summarize
python3 -m scripts.make_table --input results/processed/comparison_rows.jsonl
python3 -m scripts.make_figures --input results/processed/comparison_rows.jsonl
```

For a pilot-only local aggregation:

```bash
python3 -m scripts.summarize \
  --results results/raw/exp_recovery \
  --output-dir results/processed
```

## Chimera notes

The repository is meant to be edited locally and executed mainly on Chimera. See `docs/chimera_runbook.md` for the local-to-cluster workflow, environment assumptions, smoke-test expectations, and logging locations.
