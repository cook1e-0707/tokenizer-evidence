# Experiment Protocol

## Config resolution

- The config system remains the single source of truth.
- Experiment YAML files compose through `includes:` and optional `--override key=value` arguments.
- Stage 4 uses explicit mode separation through `eval.verification_mode`:
  - `synthetic_fixture`: legacy smoke fixtures
  - `canonical_render`: real integration path for render -> parse -> verify

## Carrier audit workflow

- Real carrier catalogs live outside Python source, typically under `configs/data/*.yaml`.
- Raw source catalogs are not pilot-ready. Manifest generation and canonical-render eval now require a frozen catalog with strict-pass provenance.
- A carrier catalog is loaded as a `BucketLayout` with:
  - `field_name`
  - `field_type`
  - `buckets`
  - optional `notes`
  - optional `tags`
  - optional `disallowed_carriers`
- Run audit directly from an experiment config:

```bash
python scripts/tokenizer_audit.py \
  --config configs/experiment/exp_recovery.yaml \
  --tokenizer-backend mock \
  --no-strict
```

- For a real tokenizer backend, install `transformers` and omit the mock override.

## Catalog freeze workflow

1. Start from a raw source catalog.
2. Run `scripts/freeze_catalog.py`.
3. The workflow will:
   - run reporting-mode audit on the source catalog
   - drop carriers with explicit blocking reasons such as `multi_token`, `duplicate_normalized_form`, or `disallowed_carrier`
   - attempt to build a candidate frozen catalog
   - rerun strict audit on that candidate
4. Only a strict-passed frozen catalog may be used for pilot manifest/eval.

Example:

```bash
python scripts/freeze_catalog.py \
  --source-catalog configs/data/real_pilot_catalog.yaml \
  --tokenizer-backend mock \
  --frozen-catalog-output artifacts/carrier_catalog_freeze_v1.yaml \
  --audit-report-output artifacts/tokenizer_audit_report_mock.json \
  --change-log-output artifacts/catalog_change_log.md \
  --data-config-output artifacts/real_pilot_frozen.yaml
```

If freeze fails, generate a remediation review instead of modifying the source catalog:

```bash
python scripts/review_catalog_freeze.py \
  --audit-report results/processed/audits/tokenizer_audit_report__gpt2.json \
  --change-log docs/catalog_freezes/real_pilot_catalog__gpt2__v1.md \
  --output-table results/processed/audits/tokenizer_audit_remediation__gpt2.json \
  --output-review docs/catalog_freezes/real_pilot_catalog__gpt2__v1_review.md
```

## Canonical render format

- Stage 4 freezes one canonical structured evidence format: `canonical_v1`.
- Current contract:
  - one evidence block per line
  - fields rendered in catalog order
  - assignments rendered as `FIELD=value`
  - fields separated by `; `
- Example:

```text
SECTION=news; TONE=calm; TOPIC=market; REGION=urban
SECTION=report; TONE=clear; TOPIC=travel; REGION=rural
```

- The verifier now treats this as an explicit contract rather than a smoke-only convention.

## Eval and calibration contract

- `scripts/eval.py` always writes a fixed `eval_summary.json`.
- Canonical-render eval runs also write:
  - `rendered_evidence.txt`
  - `rendered_evidence.json`
  - `verifier_result.json`
- `scripts/calibrate.py` writes `calibration_summary.json`.
- `scripts/summarize.py` scans JSON files by schema, not by filename conventions.

## Foundation Gate F1

- `Qwen2.5-7B-Instruct` foundation bring-up is a mandatory promotion stage before reopening the main `canonical_render` clean eval path.
- `foundation_v1` is not itself a paper-facing clean acceptance result. It is a contract-validation stage for:
  - contextual next-token carrier alignment
  - deterministic field-wise decoding
  - deterministic render from slot outputs back into `canonical_v1`
- The main Qwen 7B clean eval path must remain blocked until a passing foundation eval summary is available and referenced through `data.foundation_eval_summary_path`.

### F1 Passing Criteria

- `field_valid_rate = 1.0`
- `bucket_correct_rate >= 0.95`
- `slot_exact_rate >= 0.95`
- `per_field_accuracy >= 0.95` for every active field
- deterministic render from foundation slot outputs to `canonical_v1` must produce:
  - `accepted = true`
  - `verifier_success = true`
- `git_commit != "nogit"`

### F1 Promotion Rule

- `canonical_render` clean eval for the Qwen 7B main path is not allowed unless:
  - a prior `foundation_gate` eval summary exists
  - that summary reports `foundation_gate_passed = true`
  - the summary also satisfies `accepted = true` and `verifier_success = true`
- Until F1 passes:
  - do not reopen Batch 3
  - do not open new model families
  - do not interpret Qwen 7B main-path failure as a paper-facing method result

## Pilot flow

1. Freeze the raw catalog.
2. Generate a new data-config or experiment overlay that points to the frozen catalog.
3. Dry-run the pilot manifest:

```bash
python scripts/make_manifest.py --config /path/to/generated_frozen_experiment_config.yaml --dry-run
```

4. Run the pilot eval locally:

```bash
python scripts/eval.py --config /path/to/generated_frozen_experiment_config.yaml
```

5. Aggregate summaries:

```bash
python scripts/summarize.py --results results/raw --output-dir results/processed
```

## Metadata storage

Each run directory stores:

- `config.resolved.yaml`
- `environment.json`
- `run.log`
- optional `run.jsonl`
- `eval_summary.json` or `calibration_summary.json`
- `submission.json` when launched through the manifest/SLURM control plane
