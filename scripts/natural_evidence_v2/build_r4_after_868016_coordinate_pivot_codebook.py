from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SOURCE_CODEBOOK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/codebook.json"
)
DEFAULT_SOURCE_DECODER = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_864832_reliability_weighted_codebook_precommit_20260516/decoder_spec.json"
)
DEFAULT_ROWS_SUMMARY = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    return str(resolve(path).relative_to(ROOT))


def build_filtered_codebook(source_codebook: Mapping[str, Any], rows_summary: Mapping[str, Any]) -> dict[str, Any]:
    selected_coordinates = [int(item) for item in rows_summary["selected_coordinates"]]
    selected_set = set(selected_coordinates)
    pair_mapping: list[dict[str, Any]] = []
    for item in sorted(source_codebook["pair_to_bit_mapping"], key=lambda row: int(row["bit_index"])):
        bit_index = int(item["bit_index"])
        source_coordinates = [int(coordinate) for coordinate in item["coordinates"]]
        kept_coordinates = [coordinate for coordinate in source_coordinates if coordinate in selected_set]
        if not kept_coordinates:
            raise ValueError(f"bit {bit_index} has no selected coordinate after filtering")
        pair_mapping.append(
            {
                "bit_index": bit_index,
                "coordinates": kept_coordinates,
                "source_coordinates": source_coordinates,
                "erased_source_coordinates": [coordinate for coordinate in source_coordinates if coordinate not in selected_set],
            }
        )
    seen_bits = {int(item["bit_index"]) for item in pair_mapping}
    if seen_bits != set(range(8)):
        raise ValueError(f"filtered pair mapping is incomplete: {sorted(seen_bits)}")
    return {
        "schema_name": "natural_evidence_v2_r4_after_868016_coordinate_pivot_codebook_v1",
        "status": "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_PRECOMMITTED",
        "contract_id": "a55e",
        "payload_bits": int(source_codebook.get("payload_bits", 4)),
        "checksum_bits": int(source_codebook.get("checksum_bits", 4)),
        "source_codebook_schema_name": source_codebook.get("schema_name", ""),
        "source_codebook_status": source_codebook.get("status", ""),
        "source_selected_coordinates": [int(item) for item in source_codebook.get("selected_coordinates", [])],
        "selected_coordinates": selected_coordinates,
        "excluded_coordinates": [int(item) for item in rows_summary.get("excluded_coordinates", [])],
        "selected_coordinate_count": len(selected_coordinates),
        "pair_to_bit_mapping": pair_mapping,
        "min_coordinates_per_bit": min(len(item["coordinates"]) for item in pair_mapping),
        "max_coordinates_per_bit": max(len(item["coordinates"]) for item in pair_mapping),
        "expected_codeword_bits": [int(bit) for bit in rows_summary["expected_codeword_bits"]],
        "prompt_offset": int(rows_summary.get("prompt_offset", -1)),
        "selected_prompt_count": int(rows_summary.get("selected_prompt_count", 0)),
        "no_posthoc_threshold_changes": True,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def build_decoder_spec(source_decoder: Mapping[str, Any], codebook: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_r4_after_868016_coordinate_pivot_decoder_spec_v1",
        "status": "PASS_R4_AFTER_868016_COORDINATE_PIVOT_DECODER_SPEC_PRECOMMITTED",
        "decoder": "pair_majority_then_checksum",
        "source_decoder_schema_name": source_decoder.get("schema_name", ""),
        "source_decoder_status": source_decoder.get("status", ""),
        "accept_rules": {
            "required_pairs": 8,
            "min_pair_support": 1,
            "checksum": "payload_bitwise_complement",
            "condition_codeword_must_match": True,
            "forbidden_public_surface_count": 0,
        },
        "selected_coordinates": codebook["selected_coordinates"],
        "pair_to_bit_mapping": codebook["pair_to_bit_mapping"],
        "erasure_policy": "missing_coordinate_allowed_if_pair_has_majority; missing_pair_rejects",
        "primary_format_scrub_mode": "all",
        "no_posthoc_threshold_changes": True,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 after-868016 coordinate-pivot filtered codebook.")
    parser.add_argument("--source-codebook", type=Path, default=DEFAULT_SOURCE_CODEBOOK)
    parser.add_argument("--source-decoder", type=Path, default=DEFAULT_SOURCE_DECODER)
    parser.add_argument("--rows-summary", type=Path, default=DEFAULT_ROWS_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_codebook_path = resolve(args.source_codebook)
    source_decoder_path = resolve(args.source_decoder)
    rows_summary_path = resolve(args.rows_summary)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing precommit dir: {output_dir}")
    source_codebook = read_json(source_codebook_path)
    source_decoder = read_json(source_decoder_path)
    rows_summary = read_json(rows_summary_path)
    if rows_summary.get("status") != "PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROWS_BUILT_ARTIFACT_ONLY":
        raise ValueError("rows summary is not the reviewed coordinate-pivot row artifact")
    if int(rows_summary.get("selected_coordinate_count", 0)) != 12:
        raise ValueError("coordinate-pivot codebook requires exactly 12 selected coordinates")

    codebook = build_filtered_codebook(source_codebook, rows_summary)
    decoder_spec = build_decoder_spec(source_decoder, codebook)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(output_dir / "codebook.json", codebook)
    write_json_new(output_dir / "decoder_spec.json", decoder_spec)
    hashes = {
        "codebook_sha256": sha256_file(output_dir / "codebook.json"),
        "decoder_spec_sha256": sha256_file(output_dir / "decoder_spec.json"),
    }
    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_868016_coordinate_pivot_precommit_manifest_v1",
        "status": "PASS_R4_AFTER_868016_COORDINATE_PIVOT_PRECOMMIT_RECORDED",
        "contract_id": "a55e",
        "precommit_dir": repo_rel(output_dir),
        "source_codebook": repo_rel(source_codebook_path),
        "source_codebook_sha256": sha256_file(source_codebook_path),
        "source_decoder": repo_rel(source_decoder_path),
        "source_decoder_sha256": sha256_file(source_decoder_path),
        "source_rows_summary": repo_rel(rows_summary_path),
        "source_rows_summary_sha256": sha256_file(rows_summary_path),
        "selected_coordinates": codebook["selected_coordinates"],
        "excluded_coordinates": codebook["excluded_coordinates"],
        "pair_to_bit_mapping": codebook["pair_to_bit_mapping"],
        "primary_format_scrub_mode": "all",
        "no_posthoc_threshold_changes": True,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        **hashes,
    }
    write_json_new(output_dir / "precommit_manifest.json", manifest)
    manifest_hash = sha256_file(output_dir / "precommit_manifest.json")
    manifest["precommit_manifest_file_sha256"] = manifest_hash
    # The self-hash is recorded in a sidecar because writing it into the manifest
    # would recursively change the manifest hash.
    write_json_new(output_dir / "precommit_hashes.json", {**hashes, "precommit_manifest_file_sha256": manifest_hash})
    (output_dir / "PRECOMMIT_REVIEW.md").write_text(
        "\n".join(
            [
                "# R4 After-868016 Coordinate-Pivot Codebook Precommit",
                "",
                f"Status: `{manifest['status']}`",
                "",
                f"- selected coordinates: `{codebook['selected_coordinates']}`",
                f"- excluded coordinates: `{codebook['excluded_coordinates']}`",
                f"- min coordinates per bit: `{codebook['min_coordinates_per_bit']}`",
                f"- max coordinates per bit: `{codebook['max_coordinates_per_bit']}`",
                f"- primary scrub mode: `{manifest['primary_format_scrub_mode']}`",
                "",
                "This artifact is derived from the reviewed after-868016 coordinate pivot rows.",
                "It does not start generation, training, Llama, FAR, sanitizer, or any paper-facing claim.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "output_dir": repo_rel(output_dir), **hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
