from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE_BANK = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_two_sided_cover_bank_20260516/surface_bank.json"
DEFAULT_CODEBOOK = ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/codebook.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_864832_reliability_surface_uniqueness_audit_20260516"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(surface_bank: Mapping[str, Any], codebook: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_coordinates = {int(item) for item in codebook.get("selected_coordinates", [])}
    phrase_map: dict[str, list[dict[str, Any]]] = {}
    for entry in surface_bank.get("entries", []):
        if not isinstance(entry, dict):
            continue
        aliases = [str(entry.get("canonical_lemma_or_phrase", ""))]
        aliases.extend(str(item) for item in entry.get("aliases", []))
        for alias in aliases:
            phrase = normalize(alias)
            if not phrase:
                continue
            phrase_map.setdefault(phrase, []).append(
                {
                    "coordinate_id": int(entry.get("coordinate_id", -1)),
                    "polarity_or_code_symbol": int(entry.get("polarity_or_code_symbol", -1)),
                    "surface_id": str(entry.get("surface_id", "")),
                    "phrase": phrase,
                }
            )

    rows: list[dict[str, Any]] = []
    selected_ambiguous = 0
    selected_opposite_polarity = 0
    for phrase, entries in sorted(phrase_map.items()):
        selected_entries = [entry for entry in entries if int(entry["coordinate_id"]) in selected_coordinates]
        selected_coordinate_polarities = sorted(
            {f"{entry['coordinate_id']}:{entry['polarity_or_code_symbol']}" for entry in selected_entries}
        )
        selected_polarities = {int(entry["polarity_or_code_symbol"]) for entry in selected_entries}
        ambiguous_selected = len(selected_coordinate_polarities) > 1
        opposite_selected = len(selected_polarities) > 1
        if ambiguous_selected:
            selected_ambiguous += 1
        if opposite_selected:
            selected_opposite_polarity += 1
        rows.append(
            {
                "phrase": phrase,
                "entry_count": len(entries),
                "selected_entry_count": len(selected_entries),
                "all_coordinate_polarities": ";".join(
                    sorted({f"{entry['coordinate_id']}:{entry['polarity_or_code_symbol']}" for entry in entries})
                ),
                "selected_coordinate_polarities": ";".join(selected_coordinate_polarities),
                "ambiguous_for_selected_coordinates": ambiguous_selected,
                "opposite_polarity_for_selected_coordinates": opposite_selected,
            }
        )

    summary = {
        "schema_name": "r4_after_864832_reliability_surface_uniqueness_audit_v1",
        "status": (
            "FAIL_R4_RELIABILITY_SURFACE_UNIQUENESS_SELECTED_COORDINATES_AMBIGUOUS"
            if selected_ambiguous or selected_opposite_polarity
            else "PASS_R4_RELIABILITY_SURFACE_UNIQUENESS"
        ),
        "surface_entries": len(surface_bank.get("entries", [])),
        "selected_coordinate_count": len(selected_coordinates),
        "unique_normalized_phrases": len(phrase_map),
        "phrases_ambiguous_for_selected_coordinates": selected_ambiguous,
        "phrases_with_opposite_polarity_for_selected_coordinates": selected_opposite_polarity,
        "generation_started": False,
        "model_scoring_started": False,
        "slurm_submitted": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit phrase uniqueness for the after-864832 reliability decoder.")
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    surface_bank = read_json(args.surface_bank if args.surface_bank.is_absolute() else ROOT / args.surface_bank)
    codebook = read_json(args.codebook if args.codebook.is_absolute() else ROOT / args.codebook)
    summary, rows = audit(surface_bank, codebook)
    write_json(output_dir / "surface_uniqueness_summary.json", summary)
    write_csv(output_dir / "surface_uniqueness_rows.csv", rows)
    report = [
        "# R4 Reliability Surface Uniqueness Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- surface entries: `{summary['surface_entries']}`",
        f"- selected coordinates: `{summary['selected_coordinate_count']}`",
        f"- unique normalized phrases: `{summary['unique_normalized_phrases']}`",
        f"- ambiguous selected phrases: `{summary['phrases_ambiguous_for_selected_coordinates']}`",
        f"- opposite-polarity selected phrases: `{summary['phrases_with_opposite_polarity_for_selected_coordinates']}`",
        "",
        "No Slurm submission, model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claim was started.",
    ]
    (output_dir / "surface_uniqueness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
