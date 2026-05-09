from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.natural_evidence_v1.common import read_jsonl, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_variable_arity_density_input_combiner_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine heldout and organic generated-output/compatibility inputs "
            "for a single variable-arity full-density audit. This is an indexing "
            "utility only: no model scoring, no training, no E2E, no FAR."
        )
    )
    parser.add_argument("--base-generated-outputs", required=True)
    parser.add_argument("--base-compatibility-by-entry-csv", required=True)
    parser.add_argument("--organic-generated-outputs", required=True)
    parser.add_argument("--organic-compatibility-by-entry-csv", required=True)
    parser.add_argument("--output-generated-outputs", required=True)
    parser.add_argument("--output-compatibility-by-entry-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _existing_outputs(*paths: Path) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def combine_density_inputs(
    *,
    base_generated_outputs: Path,
    base_compatibility_by_entry_csv: Path,
    organic_generated_outputs: Path,
    organic_compatibility_by_entry_csv: Path,
    output_generated_outputs: Path,
    output_compatibility_by_entry_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    existing = _existing_outputs(output_generated_outputs, output_compatibility_by_entry_csv, summary_json)
    if existing:
        raise FileExistsError("Refusing to overwrite combined density inputs: " + ", ".join(existing))

    base_generated_rows = read_jsonl(base_generated_outputs)
    organic_generated_rows = read_jsonl(organic_generated_outputs)
    base_fields, base_compat_rows = _read_csv(base_compatibility_by_entry_csv)
    organic_fields, organic_compat_rows = _read_csv(organic_compatibility_by_entry_csv)
    if not base_fields:
        raise ValueError(f"base compatibility CSV has no header: {base_compatibility_by_entry_csv}")
    if base_fields != organic_fields:
        raise ValueError("base and organic compatibility CSV fieldnames differ")
    if "generated_row_index" not in base_fields:
        raise ValueError("compatibility CSV missing generated_row_index")
    if "prompt_split" not in base_fields:
        raise ValueError("compatibility CSV missing prompt_split")

    offset = len(base_generated_rows)
    adjusted_organic_generated_rows: list[dict[str, Any]] = []
    for local_index, row in enumerate(organic_generated_rows):
        adjusted = dict(row)
        adjusted["generated_row_index"] = offset + local_index
        adjusted["prompt_split"] = "organic"
        adjusted["split"] = "organic"
        adjusted_organic_generated_rows.append(adjusted)

    adjusted_organic_compat_rows: list[dict[str, str]] = []
    for row in organic_compat_rows:
        adjusted = dict(row)
        adjusted["generated_row_index"] = str(offset + int(row.get("generated_row_index", "0")))
        adjusted["prompt_split"] = "organic"
        adjusted_organic_compat_rows.append(adjusted)

    write_jsonl(output_generated_outputs, [*base_generated_rows, *adjusted_organic_generated_rows])
    _write_csv(output_compatibility_by_entry_csv, [*base_compat_rows, *adjusted_organic_compat_rows], base_fields)
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_NOT_TRAINING",
        "base_generated_outputs": str(base_generated_outputs),
        "base_compatibility_by_entry_csv": str(base_compatibility_by_entry_csv),
        "organic_generated_outputs": str(organic_generated_outputs),
        "organic_compatibility_by_entry_csv": str(organic_compatibility_by_entry_csv),
        "output_generated_outputs": str(output_generated_outputs),
        "output_compatibility_by_entry_csv": str(output_compatibility_by_entry_csv),
        "base_generated_rows": len(base_generated_rows),
        "organic_generated_rows": len(organic_generated_rows),
        "combined_generated_rows": len(base_generated_rows) + len(organic_generated_rows),
        "base_compatibility_rows": len(base_compat_rows),
        "organic_compatibility_rows": len(organic_compat_rows),
        "combined_compatibility_rows": len(base_compat_rows) + len(organic_compat_rows),
        "organic_generated_row_index_offset": offset,
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "combined_density_inputs_not_payload_recovery",
    }
    write_json(summary_json, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = combine_density_inputs(
        base_generated_outputs=Path(args.base_generated_outputs),
        base_compatibility_by_entry_csv=Path(args.base_compatibility_by_entry_csv),
        organic_generated_outputs=Path(args.organic_generated_outputs),
        organic_compatibility_by_entry_csv=Path(args.organic_compatibility_by_entry_csv),
        output_generated_outputs=Path(args.output_generated_outputs),
        output_compatibility_by_entry_csv=Path(args.output_compatibility_by_entry_csv),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
