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
