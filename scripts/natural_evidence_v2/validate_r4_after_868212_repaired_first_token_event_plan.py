from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import ROOT, read_json, read_jsonl, write_json_new, write_text_new


DEFAULT_PRECOMMIT = ROOT / "results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516"
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_plan_validation_20260516"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def singleton_failures(codebook: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in codebook.get("pair_to_bit_mapping", []):
        bit_index = int(item.get("bit_index", -1))
        coordinates = [int(coordinate) for coordinate in item.get("coordinates", [])]
        if len(coordinates) < 2:
            failures.append({"bit_index": bit_index, "coordinates": coordinates, "failure_reason": "singleton_bit"})
        if coordinates == [26]:
            failures.append(
                {"bit_index": bit_index, "coordinates": coordinates, "failure_reason": "coordinate_26_singleton"}
            )
    return failures


def allocation_failures(rows: list[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    shards = int(manifest.get("shards", 0))
    rows_per_coordinate_per_shard = int(manifest.get("rows_per_coordinate_per_shard", 0))
    coordinates = [int(coordinate) for coordinate in manifest.get("coordinates", [])]
    if shards <= 0:
        failures.append("allocation shards must be positive")
    if len(coordinates) != 16:
        failures.append("allocation must contain 16 coordinates")
    expected_total = shards * rows_per_coordinate_per_shard * len(coordinates)
    if len(rows) != expected_total:
        failures.append(f"allocation row count {len(rows)} != expected {expected_total}")
    by_shard_coord: Counter[tuple[int, int]] = Counter()
    shard_pairs: dict[int, set[str]] = {index: set() for index in range(shards)}
    duplicate_pairs: Counter[int] = Counter()
    for row in rows:
        shard = int(row.get("assigned_shard_index", -1))
        coordinate = int(row.get("coordinate_id", -1))
        by_shard_coord[(shard, coordinate)] += 1
        pair = f"{int(row.get('prompt_index', -1))}|{row.get('prefix_family_id', '')}"
        if pair in shard_pairs.setdefault(shard, set()):
            duplicate_pairs[shard] += 1
        shard_pairs[shard].add(pair)
    for shard in range(shards):
        for coordinate in coordinates:
            if by_shard_coord[(shard, coordinate)] != rows_per_coordinate_per_shard:
                failures.append(
                    f"shard {shard} coordinate {coordinate} rows {by_shard_coord[(shard, coordinate)]} != {rows_per_coordinate_per_shard}"
                )
        if duplicate_pairs[shard]:
            failures.append(f"shard {shard} duplicate prompt/prefix pairs: {duplicate_pairs[shard]}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repaired R4 after-868212 first-token event plan.")
    parser.add_argument("--precommit-dir", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    precommit_dir = resolve(args.precommit_dir)
    output_dir = resolve(args.output_dir)
    manifest = read_json(precommit_dir / "precommit_manifest.json")
    codebook = read_json(precommit_dir / "codebook.json")
    decoder_spec = read_json(precommit_dir / "decoder_spec.json")
    duplicate_policy = read_json(precommit_dir / "duplicate_policy.json")
    allocation_manifest_path = resolve(Path(str(manifest["allocation_manifest"])))
    allocation_rows_path = resolve(Path(str(manifest["allocation_rows"])))
    allocation_manifest = read_json(allocation_manifest_path)
    allocation_rows = read_jsonl(allocation_rows_path)

    errors: list[str] = []
    if manifest.get("reclassifies_868212") is not False:
        errors.append("precommit must not reclassify 868212")
    if codebook.get("min_active_coordinates_per_bit") != 2:
        errors.append("codebook min_active_coordinates_per_bit must be 2")
    errors.extend(f"{failure['failure_reason']}: bit {failure['bit_index']} {failure['coordinates']}" for failure in singleton_failures(codebook))
    if len(codebook.get("selected_coordinates", [])) != 16:
        errors.append("codebook must use 16 selected coordinates")
    if decoder_spec.get("future_positive_requires_token_id_trace") is not True:
        errors.append("decoder must require token-id traces")
    locked_scale = duplicate_policy.get("locked_scale_hard_fail", {})
    if int(locked_scale.get("global_duplicate_response_hash_count", -1)) != 0:
        errors.append("duplicate policy must require zero global duplicates for locked-scale claims")
    errors.extend(allocation_failures(allocation_rows, allocation_manifest))

    status = (
        "PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATION_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868212_repaired_first_token_event_plan_validation_v1",
        "status": status,
        "precommit_dir": str(precommit_dir.relative_to(ROOT)),
        "errors": errors,
        "selected_coordinate_count": len(codebook.get("selected_coordinates", [])),
        "min_active_coordinates_per_bit": codebook.get("min_active_coordinates_per_bit"),
        "allocation_rows": len(allocation_rows),
        "allocation_shards": allocation_manifest.get("shards"),
        "allocation_rows_per_coordinate_per_shard": allocation_manifest.get("rows_per_coordinate_per_shard"),
        "locked_scale_global_duplicate_gate": locked_scale.get("global_duplicate_response_hash_count"),
        "slurm_allowed": False,
        "slurm_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "plan_validation_summary.json", summary)
    write_text_new(
        output_dir / "plan_validation_report.md",
        "\n".join(
            [
                "# R4 After-868212 Repaired First-Token Event Plan Validation",
                "",
                f"Status: `{status}`",
                "",
                f"- selected coordinates: `{summary['selected_coordinate_count']}`",
                f"- min active coordinates per bit: `{summary['min_active_coordinates_per_bit']}`",
                f"- allocation rows: `{summary['allocation_rows']}`",
                f"- allocation shards: `{summary['allocation_shards']}`",
                f"- rows per coordinate per shard: `{summary['allocation_rows_per_coordinate_per_shard']}`",
                f"- errors: `{len(errors)}`",
                "",
                "This is plan-only validation. It does not submit Slurm or unlock claims.",
            ]
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
