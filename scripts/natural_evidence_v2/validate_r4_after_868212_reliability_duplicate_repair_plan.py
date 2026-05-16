from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


DEFAULT_DECODER_SPEC = Path("results/natural_evidence_v2/precommit/r4_after_868151_first_token_event_channel_precommit_20260516/decoder_spec.json")
DEFAULT_GENERATION_DIR = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212")
DEFAULT_OUTPUT = Path("results/natural_evidence_v2/status/r4_after_868212_reliability_duplicate_repair_preflight_20260516")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_generation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generated_path in sorted((path / "shards").glob("shard_*/r4_generated_outputs.jsonl")):
        for row in read_jsonl(generated_path):
            row = dict(row)
            row["shard_id"] = generated_path.parent.name
            rows.append(row)
    return rows


def singleton_bit_failures(decoder_spec: Mapping[str, Any], min_active_coordinates: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in decoder_spec.get("pair_to_bit_mapping", []):
        coordinates = list(item.get("coordinates", []))
        bit_index = int(item.get("bit_index", -1))
        if len(coordinates) < min_active_coordinates:
            failures.append(
                {
                    "bit_index": bit_index,
                    "active_coordinates": coordinates,
                    "active_coordinate_count": len(coordinates),
                    "erased_source_coordinates": list(item.get("erased_source_coordinates", [])),
                    "source_coordinates": list(item.get("source_coordinates", [])),
                    "failure_reason": "active_coordinate_count_below_minimum",
                }
            )
        if coordinates == [26]:
            failures.append(
                {
                    "bit_index": bit_index,
                    "active_coordinates": coordinates,
                    "active_coordinate_count": len(coordinates),
                    "erased_source_coordinates": list(item.get("erased_source_coordinates", [])),
                    "source_coordinates": list(item.get("source_coordinates", [])),
                    "failure_reason": "coordinate_26_is_sole_active_coordinate",
                }
            )
    return failures


def duplicate_taxonomy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("response_text_sha256", ""))
        if digest:
            by_hash[digest].append(row)

    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    taxonomy = {
        "generated_rows": len(rows),
        "unique_response_hashes": sum(1 for key in by_hash if key),
        "duplicate_hash_groups": len(duplicate_groups),
        "global_duplicate_response_hash_count": sum(len(group) - 1 for group in duplicate_groups),
        "global_duplicate_response_hash_max_group_size": max((len(group) for group in duplicate_groups), default=0),
        "within_arm_duplicate_groups": 0,
        "within_shard_duplicate_groups": 0,
        "cross_shard_duplicate_groups": 0,
        "cross_arm_duplicate_groups": 0,
        "per_block_duplicate_policy_present": True,
    }
    condition_sets: Counter[str] = Counter()
    shard_sets: Counter[str] = Counter()
    for group in duplicate_groups:
        conditions = sorted({str(row.get("generation_condition", "")) for row in group})
        shards = sorted({str(row.get("shard_id", "")) for row in group})
        condition_sets[",".join(conditions)] += 1
        shard_sets[",".join(shards)] += 1
        if len(conditions) == 1:
            taxonomy["within_arm_duplicate_groups"] += 1
        else:
            taxonomy["cross_arm_duplicate_groups"] += 1
        if len(shards) == 1:
            taxonomy["within_shard_duplicate_groups"] += 1
        else:
            taxonomy["cross_shard_duplicate_groups"] += 1
    taxonomy["duplicate_condition_sets"] = dict(condition_sets.most_common())
    taxonomy["duplicate_shard_sets_top"] = dict(shard_sets.most_common(12))
    return taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-868212 reliability/duplicate repair preconditions.")
    parser.add_argument("--decoder-spec", type=Path, default=DEFAULT_DECODER_SPEC)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-active-coordinates-per-bit", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decoder_spec = read_json(args.decoder_spec)
    rows = load_generation_rows(args.generation_dir)
    singleton_failures = singleton_bit_failures(decoder_spec, args.min_active_coordinates_per_bit)
    duplicates = duplicate_taxonomy(rows)

    pass_gate = not singleton_failures and duplicates["global_duplicate_response_hash_count"] == 0
    status = (
        "PASS_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT"
        if pass_gate
        else "FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868212_reliability_duplicate_repair_preflight_v1",
        "status": status,
        "decoder_spec": str(args.decoder_spec),
        "generation_dir": str(args.generation_dir),
        "min_active_coordinates_per_bit": int(args.min_active_coordinates_per_bit),
        "singleton_bit_failures": singleton_failures,
        "duplicate_taxonomy": duplicates,
        "slurm_allowed": False,
        "slurm_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "repair_preflight_summary.json", summary)
    report = [
        "# R4 After-868212 Reliability/Duplicate Repair Preflight",
        "",
        f"Status: `{status}`",
        "",
        "## Codebook Reliability",
        "",
        f"- min active coordinates per bit: `{args.min_active_coordinates_per_bit}`",
        f"- singleton/codebook failures: `{len(singleton_failures)}`",
        "",
        "## Duplicate Taxonomy",
        "",
        f"- generated rows: `{duplicates['generated_rows']}`",
        f"- duplicate groups: `{duplicates['duplicate_hash_groups']}`",
        f"- duplicate extra rows: `{duplicates['global_duplicate_response_hash_count']}`",
        f"- cross-arm duplicate groups: `{duplicates['cross_arm_duplicate_groups']}`",
        f"- cross-shard duplicate groups: `{duplicates['cross_shard_duplicate_groups']}`",
        "",
        "## Route Implication",
        "",
        "This preflight must pass before any next generation/scoring/training Slurm route is recorded.",
    ]
    if singleton_failures:
        report.extend(["", "## Singleton Failures", ""])
        for failure in singleton_failures:
            report.append(
                f"- bit `{failure['bit_index']}` active={failure['active_coordinates']} "
                f"source={failure['source_coordinates']} reason={failure['failure_reason']}"
            )
    (output_dir / "repair_preflight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
