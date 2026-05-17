from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_jsonl,
    sha256_file,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_text_new,
)
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (  # noqa: E402
    r4_row_surface_contract,
)


DEFAULT_STATUS_DIR = ROOT / "results/natural_evidence_v2/status"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868348_candidate_row_source_audit_20260517"
)

REQUIRED_GENERATION_FIELDS = (
    "assistant_prefix_before_surface",
    "bucket_0_surfaces",
    "bucket_1_surfaces",
    "contract_id",
    "coordinate_id",
    "prefix_family_id",
    "prompt_id",
    "prompt_index",
    "prompt_text",
    "row_key",
    "target_bit",
    "target_surface",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only audit of existing R4 row-source JSONL files after the "
            "868348 global duplicate gate failure. This does not build a new "
            "allocation, generate outputs, score a model, or submit Slurm."
        )
    )
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-blocks", type=int, default=32)
    parser.add_argument("--rows-per-block", type=int, default=1024)
    parser.add_argument("--rows-per-coordinate-per-block", type=int, default=64)
    return parser.parse_args()


def iter_row_files(status_dir: Path) -> list[Path]:
    patterns = ("*rows*.jsonl", "*row_allocation*.jsonl")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(status_dir.glob(f"**/{pattern}"))
    return sorted(path for path in paths if path.is_file())


def compact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_prompt_prefix_key(row: Mapping[str, Any]) -> str:
    return "::".join(
        [
            str(row.get("prompt_id", "")),
            str(row.get("prefix_family_id", "")),
            str(row.get("assistant_prefix_before_surface", "")),
        ]
    )


def row_content_prefix_key(row: Mapping[str, Any]) -> str:
    prompt_hash = str(row.get("prompt_text_sha256") or "")
    if not prompt_hash:
        prompt_hash = str(hash(str(row.get("prompt_text", ""))))
    return "::".join(
        [
            prompt_hash,
            str(row.get("prefix_family_id", "")),
            str(row.get("assistant_prefix_before_surface", "")),
        ]
    )


def technical_or_structural_prompt_failure(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt_text", ""))
    hits = technical_literal_hits(prompt)
    if hits:
        return f"technical_literal:{','.join(hits)}"
    lowered = prompt.lower()
    if "step " in lowered or "exactly 16" in lowered or "slot" in lowered:
        return "structural_prompt_literal"
    return ""


def row_compatibility_failure(row: Mapping[str, Any]) -> str:
    missing = [field for field in REQUIRED_GENERATION_FIELDS if field not in row]
    if missing:
        return "missing_fields:" + ",".join(missing)
    if str(row.get("contract_id")) != "a55e":
        return f"contract_not_a55e:{row.get('contract_id')}"
    if row.get("current_two_way_scorer_compatible") is False:
        return "current_two_way_scorer_compatible_false"
    prompt_failure = technical_or_structural_prompt_failure(row)
    if prompt_failure:
        return prompt_failure
    try:
        r4_row_surface_contract(row)
    except Exception as exc:  # noqa: BLE001 - artifact audit records closed failures.
        return f"surface_contract_error:{type(exc).__name__}:{exc}"
    return ""


def row_generation_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "generation_started": bool(row.get("generation_started", False)),
        "model_scoring_started": bool(row.get("model_scoring_started", False))
        or bool(row.get("model_generation_started", False)),
        "training_started": bool(row.get("training_started", False)),
        "slurm_submitted": bool(row.get("slurm_submitted", False)),
        "paper_claim_allowed": bool(row.get("paper_claim_allowed", False)),
    }


def summarize_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    failure_counts: Counter[str] = Counter()
    compatible_rows: list[dict[str, Any]] = []
    prompt_prefix_keys: Counter[str] = Counter()
    content_prefix_keys: Counter[str] = Counter()
    coordinates: Counter[int] = Counter()
    flags = Counter()

    for row in rows:
        failure = row_compatibility_failure(row)
        if failure:
            failure_counts[failure.split(":", 1)[0]] += 1
            continue
        row_flags = row_generation_flags(row)
        if any(row_flags.values()):
            failure_counts["non_artifact_only_row_flags"] += 1
            continue
        normalized = dict(row)
        compatible_rows.append(normalized)
        prompt_prefix_keys[row_prompt_prefix_key(row)] += 1
        content_prefix_keys[row_content_prefix_key(row)] += 1
        coordinates[int(row["coordinate_id"])] += 1
        for key, value in row_flags.items():
            if value:
                flags[key] += 1

    unique_prompt_prefix = len(prompt_prefix_keys)
    unique_content_prefix = len(content_prefix_keys)
    duplicate_prompt_prefix_extra = sum(count - 1 for count in prompt_prefix_keys.values() if count > 1)
    duplicate_content_prefix_extra = sum(count - 1 for count in content_prefix_keys.values() if count > 1)
    summary = {
        "path": compact_path(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "compatible_row_count": len(compatible_rows),
        "incompatible_row_count": len(rows) - len(compatible_rows),
        "failure_counts": dict(sorted(failure_counts.items())),
        "unique_prompt_prefix_pair_count": unique_prompt_prefix,
        "duplicate_prompt_prefix_pair_extra_rows": duplicate_prompt_prefix_extra,
        "unique_content_prefix_pair_count": unique_content_prefix,
        "duplicate_content_prefix_pair_extra_rows": duplicate_content_prefix_extra,
        "coordinate_count": len(coordinates),
        "coordinate_ids": ",".join(str(item) for item in sorted(coordinates)),
        "min_rows_per_coordinate": min(coordinates.values()) if coordinates else 0,
        "max_rows_per_coordinate": max(coordinates.values()) if coordinates else 0,
        "row_flag_counts": dict(sorted(flags.items())),
    }
    return summary, compatible_rows


def union_metrics(compatible_by_file: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for rows in compatible_by_file.values():
        all_rows.extend(rows)

    prompt_prefix_keys = Counter(row_prompt_prefix_key(row) for row in all_rows)
    content_prefix_keys = Counter(row_content_prefix_key(row) for row in all_rows)
    coordinates: dict[int, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        coordinates[int(row["coordinate_id"])][row_content_prefix_key(row)] += 1

    unique_content_by_coordinate = {
        str(coordinate): len(keys) for coordinate, keys in sorted(coordinates.items())
    }
    min_unique_content_by_coordinate = min(unique_content_by_coordinate.values()) if unique_content_by_coordinate else 0
    return {
        "compatible_source_file_count": sum(1 for rows in compatible_by_file.values() if rows),
        "compatible_row_count": len(all_rows),
        "unique_prompt_prefix_pair_count": len(prompt_prefix_keys),
        "duplicate_prompt_prefix_pair_extra_rows": sum(
            count - 1 for count in prompt_prefix_keys.values() if count > 1
        ),
        "unique_content_prefix_pair_count": len(content_prefix_keys),
        "duplicate_content_prefix_pair_extra_rows": sum(
            count - 1 for count in content_prefix_keys.values() if count > 1
        ),
        "coordinate_ids": [int(item) for item in sorted(coordinates)],
        "coordinate_count": len(coordinates),
        "unique_content_prefix_pairs_by_coordinate": unique_content_by_coordinate,
        "min_unique_content_prefix_pairs_by_coordinate": int(min_unique_content_by_coordinate),
    }


def write_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    inventory: Iterable[Mapping[str, Any]],
) -> None:
    rows = list(inventory)
    top_sources = sorted(rows, key=lambda row: int(row["compatible_row_count"]), reverse=True)[:12]
    top_source_lines = "\n".join(
        f"- `{row['path']}`: compatible_rows={row['compatible_row_count']}, "
        f"unique_content_pairs={row['unique_content_prefix_pair_count']}, "
        f"coordinates={row['coordinate_count']}"
        for row in top_sources
    )
    text = f"""# R4 After-868348 Candidate Row Source Audit

Date: 2026-05-17

## Status

`{summary['status']}`

This is an artifact-only audit. It does not reclassify `868348`, does not build
a new allocation, does not generate outputs, does not score a model, and does
not submit Slurm.

## Gate Context

The reviewed `868348` dev diagnostic had strong first-token event signal but
failed the strict global exact duplicate gate:

- protected strict accepts: `32/32`
- controls: `0/32` for raw/task-only/wrong-key/wrong-payload
- trace-binding invalid rows: `0`
- global exact duplicate extra rows: `2`
- duplicate attribution: task-only only

The immediate blocker is whether an existing reviewed row source can support a
32-block rerun without cyclic prompt/prefix reuse.

## Aggregate Inventory

- scanned row files: `{summary['scanned_row_file_count']}`
- compatible source files: `{summary['union']['compatible_source_file_count']}`
- compatible rows: `{summary['union']['compatible_row_count']}`
- unique content prompt/prefix pairs: `{summary['union']['unique_content_prefix_pair_count']}`
- required rows for 32 blocks: `{summary['required_rows_for_32_block_dev']}`
- required globally unique content prompt/prefix pairs: `{summary['required_unique_content_prompt_prefix_pairs']}`

## Top Compatible Sources

{top_source_lines if top_source_lines else '- none'}

## Interpretation

{summary['interpretation']}

## Next Allowed Action

{summary['next_allowed_action']}
"""
    write_text_new(output_dir / "row_source_audit.md", text)


def main() -> int:
    args = parse_args()
    status_dir = args.status_dir if args.status_dir.is_absolute() else ROOT / args.status_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    row_files = iter_row_files(status_dir)
    inventory: list[dict[str, Any]] = []
    compatible_by_file: dict[str, list[dict[str, Any]]] = {}
    for path in row_files:
        try:
            file_summary, compatible_rows = summarize_file(path)
        except Exception as exc:  # noqa: BLE001 - source inventory should survive bad legacy artifacts.
            file_summary = {
                "path": compact_path(path),
                "sha256": sha256_file(path),
                "row_count": "",
                "compatible_row_count": 0,
                "incompatible_row_count": "",
                "failure_counts": {"file_read_or_parse_error": f"{type(exc).__name__}:{exc}"},
                "unique_prompt_prefix_pair_count": 0,
                "duplicate_prompt_prefix_pair_extra_rows": 0,
                "unique_content_prefix_pair_count": 0,
                "duplicate_content_prefix_pair_extra_rows": 0,
                "coordinate_count": 0,
                "coordinate_ids": "",
                "min_rows_per_coordinate": 0,
                "max_rows_per_coordinate": 0,
                "row_flag_counts": {},
            }
            compatible_rows = []
        inventory.append(file_summary)
        compatible_by_file[file_summary["path"]] = compatible_rows

    union = union_metrics(compatible_by_file)
    required_rows = int(args.target_blocks) * int(args.rows_per_block)
    required_unique_pairs = required_rows
    min_per_coordinate_required = int(args.target_blocks) * int(args.rows_per_coordinate_per_block)

    enough_unique_pairs = int(union["unique_content_prefix_pair_count"]) >= required_unique_pairs
    enough_coordinate_coverage = (
        int(union["coordinate_count"]) >= 16
        and int(union["min_unique_content_prefix_pairs_by_coordinate"]) >= min_per_coordinate_required
    )
    if enough_unique_pairs and enough_coordinate_coverage:
        status = "PASS_R4_AFTER_868348_EXISTING_ROW_SOURCES_HAVE_NECESSARY_GLOBAL_UNIQUE_CAPACITY_NO_RERUN"
        interpretation = (
            "Existing compatible row sources meet the necessary aggregate count checks for a "
            "globally unique 32-block allocation. This is not yet a route approval; a "
            "deduplicating allocation builder, tokenizer/controller preflight, and reviewed "
            "Slurm route are still required."
        )
        next_allowed_action = (
            "Implement artifact-only deduplicating allocation construction and validation; "
            "do not submit Slurm until that allocation and the route preflight pass."
        )
    else:
        status = "FAIL_R4_AFTER_868348_EXISTING_ROW_SOURCES_INSUFFICIENT_FOR_GLOBAL_UNIQUE_32_BLOCK_ALLOCATION_NO_RERUN"
        interpretation = (
            "The existing compatible row sources do not meet the necessary global-unique "
            "capacity checks for a strict 32-block rerun. The project should not rerun the "
            "868348 route from the current row bank. The next repair must either build a "
            "larger reviewed prompt/row bank with tokenizer/controller preflight, or record "
            "a separate precommitted duplicate-gate semantics decision for future runs. "
            "Neither option can retroactively rescue 868348."
        )
        next_allowed_action = (
            "Artifact-only route planning for a larger reviewed row bank or a future-only "
            "duplicate-gate semantics package; no generation or Slurm submission from the "
            "current row bank."
        )

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868348_candidate_row_source_audit_v1",
        "status": status,
        "scanned_row_file_count": len(row_files),
        "target_blocks": int(args.target_blocks),
        "rows_per_block": int(args.rows_per_block),
        "rows_per_coordinate_per_block": int(args.rows_per_coordinate_per_block),
        "required_rows_for_32_block_dev": required_rows,
        "required_unique_content_prompt_prefix_pairs": required_unique_pairs,
        "required_unique_content_prompt_prefix_pairs_per_coordinate": min_per_coordinate_required,
        "union": union,
        "enough_unique_pairs_necessary_check": enough_unique_pairs,
        "enough_coordinate_coverage_necessary_check": enough_coordinate_coverage,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "interpretation": interpretation,
        "next_allowed_action": next_allowed_action,
    }

    write_json_new(output_dir / "row_source_audit_summary.json", summary)
    write_csv_new(
        output_dir / "row_source_inventory.csv",
        inventory,
        fieldnames=[
            "path",
            "sha256",
            "row_count",
            "compatible_row_count",
            "incompatible_row_count",
            "unique_prompt_prefix_pair_count",
            "duplicate_prompt_prefix_pair_extra_rows",
            "unique_content_prefix_pair_count",
            "duplicate_content_prefix_pair_extra_rows",
            "coordinate_count",
            "coordinate_ids",
            "min_rows_per_coordinate",
            "max_rows_per_coordinate",
            "failure_counts",
            "row_flag_counts",
        ],
    )
    write_report(output_dir, summary=summary, inventory=inventory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
