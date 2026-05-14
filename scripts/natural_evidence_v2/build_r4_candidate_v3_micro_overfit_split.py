from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/"
    "r4_prefix_native_surface_probe_rows_v3.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_split_20260514_codex"
)
EXPECTED_SOURCE_SHA256 = "d35e5483ce7f6d3d782ce17961b2c407909afc879a12917c5ccc27090f3c80b7"
SPLIT_PLAN_ID = "r4_candidate_v3_micro_overfit_row_split_20260514_0104"
CONTRACT_ARTIFACT = (
    "results/natural_evidence_v2/status/"
    "r4_candidate_v3_micro_overfit_row_split_contract_20260514_0104/row_split_contract.md"
)


REQUIRED_FIELDS = (
    "prompt_id",
    "prompt_index",
    "prompt_text",
    "coordinate_id",
    "prefix_family_id",
    "assistant_prefix_before_surface",
    "target_bit",
    "target_surface",
    "target_surface_id",
    "bucket_0_surfaces",
    "bucket_1_surfaces",
)


LOCKED_ACTIONS = {
    "allowlist_enablement_allowed": False,
    "far_aggregation_allowed": False,
    "generation_allowed": False,
    "llama_allowed": False,
    "paper_claim_allowed": False,
    "payload_diversity_allowed": False,
    "qwen_e2e_rerun_allowed": False,
    "remote_cpu_gpu_work_allowed": False,
    "same_family_null_allowed": False,
    "sanitizer_benchmark_allowed": False,
    "slurm_submission_allowed": False,
    "training_allowed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("prompt_id", "")),
        str(row.get("coordinate_id", "")),
        str(row.get("assistant_prefix_before_surface", "")),
        str(row.get("target_surface_id", "")),
    )


def stratum_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("prefix_family_id", "")),
        int(row.get("coordinate_id")),
        int(row.get("target_bit")),
    )


def ordering_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row.get("prefix_family_id", "")),
        int(row.get("coordinate_id")),
        int(row.get("target_bit")),
        int(row.get("prompt_index")),
        str(row.get("target_surface_id", "")),
    )


def validate_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("schema_name") != "natural_evidence_v2_r4_surface_teacher_forced_probe_row_v1":
        errors.append("schema_name_mismatch")
    if row.get("contract_id") != "a55e":
        errors.append("contract_id_mismatch")
    if row.get("split") != "dev":
        errors.append("split_not_dev")
    if row.get("score_objective") != "next_token_first_surface_cylinder_mass":
        errors.append("score_objective_mismatch")
    for field in ("generation_started", "training_started", "qwen_tokenizer_validation_started", "slurm_submitted"):
        if row.get(field) is not False:
            errors.append(f"{field}_not_false")
    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            errors.append(f"{field}_missing")
    target_bit = row.get("target_bit")
    if target_bit not in (0, 1):
        errors.append("target_bit_not_binary")
        return errors
    for field in ("bucket_0_surfaces", "bucket_1_surfaces"):
        value = row.get(field)
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            errors.append(f"{field}_invalid")
    bucket = row.get(f"bucket_{target_bit}_surfaces")
    if isinstance(bucket, list) and row.get("target_surface") not in bucket:
        errors.append("target_surface_not_in_target_bucket")
    return errors


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    duplicate_keys: list[tuple[str, str, str, str]] = []
    for index, row in enumerate(rows):
        row_errors = validate_row(row)
        if row_errors:
            errors.append({"row_index": index, "errors": row_errors})
        key = row_key(row)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    return {
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": duplicate_keys[:10],
        "row_error_count": len(errors),
        "row_errors": errors[:20],
    }


def take_round_robin(strata_rows: Mapping[tuple[str, int, int], Sequence[Mapping[str, Any]]], quota: int) -> list[Mapping[str, Any]]:
    queues = {key: deque(rows) for key, rows in sorted(strata_rows.items()) if rows}
    selected: list[Mapping[str, Any]] = []
    while len(selected) < quota and queues:
        progressed = False
        for key in list(queues):
            queue = queues[key]
            if not queue:
                del queues[key]
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
    return selected


def select_split(rows: Sequence[Mapping[str, Any]], *, train_count: int, heldout_count: int) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    by_stratum: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_key(row)].append(row)

    train_candidates: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    heldout_candidates: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for key, stratum_rows in by_stratum.items():
        ordered = sorted(stratum_rows, key=ordering_key)
        train_candidates[key] = ordered[0::2]
        heldout_candidates[key] = ordered[1::2]

    train_rows = take_round_robin(train_candidates, train_count)
    train_keys = {row_key(row) for row in train_rows}
    heldout_without_overlap = {
        key: [row for row in rows_for_key if row_key(row) not in train_keys]
        for key, rows_for_key in heldout_candidates.items()
    }
    heldout_rows = take_round_robin(heldout_without_overlap, heldout_count)

    heldout_keys = {row_key(row) for row in heldout_rows}
    if len(train_rows) != train_count:
        raise ValueError(f"could not select exact train quota: requested {train_count}, got {len(train_rows)}")
    if len(heldout_rows) != heldout_count:
        raise ValueError(f"could not select exact heldout quota: requested {heldout_count}, got {len(heldout_rows)}")
    overlap = train_keys & heldout_keys
    if overlap:
        raise ValueError(f"train/heldout overlap detected: {len(overlap)} keys")
    return list(train_rows), list(heldout_rows)


def annotate_row(row: Mapping[str, Any], split_name: str, source_sha256: str) -> dict[str, Any]:
    annotated = dict(row)
    annotated.update(
        {
            "micro_overfit_split": split_name,
            "row_split_contract_artifact": CONTRACT_ARTIFACT,
            "source_candidate_v3_row_key": "|".join(row_key(row)),
            "source_candidate_v3_rows_sha256": source_sha256,
            "split_plan_id": SPLIT_PLAN_ID,
        }
    )
    return annotated


def counts_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def stratum_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = "|".join(str(item) for item in stratum_key(row))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_split(
    *,
    source_path: Path,
    output_dir: Path,
    expected_source_sha256: str,
    train_count: int,
    heldout_count: int,
) -> dict[str, Any]:
    planned_outputs = [
        output_dir / "train_rows.jsonl",
        output_dir / "heldout_rows.jsonl",
        output_dir / "score_rows.jsonl",
        output_dir / "split_summary.json",
    ]
    existing = [str(path) for path in planned_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing split artifacts: {existing}")
    source_sha256 = sha256_file(source_path)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(f"candidate row sha256 mismatch: expected {expected_source_sha256}, observed {source_sha256}")

    rows = read_jsonl(source_path)
    validation = validate_rows(rows)
    if validation["duplicate_key_count"] or validation["row_error_count"]:
        raise ValueError(f"candidate row validation failed: {validation}")

    train_rows, heldout_rows = select_split(rows, train_count=train_count, heldout_count=heldout_count)
    train_keys = {row_key(row) for row in train_rows}
    heldout_keys = {row_key(row) for row in heldout_rows}
    overlap = train_keys & heldout_keys
    if overlap:
        raise ValueError(f"train/heldout overlap detected: {len(overlap)}")

    train_out = [annotate_row(row, "train", source_sha256) for row in train_rows]
    heldout_out = [annotate_row(row, "heldout", source_sha256) for row in heldout_rows]
    score_out = [annotate_row(row, "score_full_candidate_v3", source_sha256) for row in rows]

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(output_dir / "train_rows.jsonl", train_out)
    write_jsonl_new(output_dir / "heldout_rows.jsonl", heldout_out)
    write_jsonl_new(output_dir / "score_rows.jsonl", score_out)

    summary = {
        "schema_name": "natural_evidence_v2_r4_candidate_v3_micro_overfit_split_summary_v1",
        "status": "ARTIFACT_ONLY_R4_CANDIDATE_V3_MICRO_OVERFIT_SPLIT_BUILT_NO_COMPUTE",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "output_dir": str(output_dir),
        "train_rows_path": str(output_dir / "train_rows.jsonl"),
        "heldout_rows_path": str(output_dir / "heldout_rows.jsonl"),
        "score_rows_path": str(output_dir / "score_rows.jsonl"),
        "train_row_count": len(train_out),
        "heldout_row_count": len(heldout_out),
        "score_row_count": len(score_out),
        "duplicate_key_count": validation["duplicate_key_count"],
        "train_heldout_overlap_count": len(overlap),
        "train_counts_by_prefix_family_id": counts_by(train_out, "prefix_family_id"),
        "heldout_counts_by_prefix_family_id": counts_by(heldout_out, "prefix_family_id"),
        "train_counts_by_coordinate_id": counts_by(train_out, "coordinate_id"),
        "heldout_counts_by_coordinate_id": counts_by(heldout_out, "coordinate_id"),
        "train_counts_by_target_bit": counts_by(train_out, "target_bit"),
        "heldout_counts_by_target_bit": counts_by(heldout_out, "target_bit"),
        "train_counts_by_stratum": stratum_counts(train_out),
        "heldout_counts_by_stratum": stratum_counts(heldout_out),
        **LOCKED_ACTIONS,
    }
    write_json_new(output_dir / "split_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the artifact-only R4 candidate-v3 micro-overfit train/heldout "
            "split. This does not train, score, generate, submit Slurm, load "
            "tokenizers/models, or make paper claims."
        )
    )
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--heldout-count", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_split(
        source_path=args.source_rows,
        output_dir=args.output_dir,
        expected_source_sha256=str(args.expected_source_sha256),
        train_count=int(args.train_count),
        heldout_count=int(args.heldout_count),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
