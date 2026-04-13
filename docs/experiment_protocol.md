# Experiment Protocol

## Config resolution

- The config system remains the single source of truth.
- Experiment YAML files compose through `includes:` and optional `--override key=value` arguments.
- Stage 4 uses explicit mode separation through `eval.verification_mode`:
  - `synthetic_fixture`: legacy smoke fixtures
  - `canonical_render`: real integration path for render -> parse -> verify

## Carrier audit workflow

- Real carrier catalogs live outside Python source, typically under `configs/data/*.yaml`.
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

1. Validate the catalog locally.
2. Dry-run the pilot manifest:

```bash
python scripts/make_manifest.py --config configs/experiment/exp_recovery.yaml --dry-run
```

3. Run the pilot eval locally:

```bash
python scripts/eval.py --config configs/experiment/exp_recovery.yaml
```

4. Aggregate summaries:

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
