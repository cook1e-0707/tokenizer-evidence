from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (  # noqa: E402
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
)
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (  # noqa: E402
    extract_support_window_events,
    load_event_window_bank,
)
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (  # noqa: E402
    decide_keyed_correlation,
)

DEFAULT_GENERATION_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277"
)
DEFAULT_PACKAGE_DIR = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_support_window_coverage_dry_run_20260514_2144"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        yield payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_generated_rows(generation_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(generation_dir.glob("shards/shard_*/r4_generated_outputs.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no generated output rows found under {generation_dir}")
    return rows


def _block_id(row: Mapping[str, Any], prompts_per_block: int) -> str:
    shard = str(row.get("replicate_group_id", "local"))
    prompt_index = int(row.get("prompt_index", 0))
    return f"{shard}_block_{prompt_index // prompts_per_block:02d}"


def _decision(
    events: list[Mapping[str, Any]],
    *,
    arm: str,
    contract: Mapping[str, Any],
    decoder_spec: Mapping[str, Any],
) -> Any:
    payload_id = str(contract["payload_id"])
    wrong_payload_id = "wrong_payload_a55e_control"
    audit_key = _DEV_AUDIT_KEY_MATERIAL
    wrong_key = _DEV_WRONG_KEY_MATERIAL
    if arm == "wrong_key":
        audit_key, wrong_key = wrong_key, audit_key
    if arm == "wrong_payload":
        payload_id, wrong_payload_id = wrong_payload_id, payload_id
    accept_requires = decoder_spec["accept_requires"]
    required = decoder_spec["required_before_accept"]
    return decide_keyed_correlation(
        events,
        audit_key=audit_key,
        payload_id=payload_id,
        wrong_audit_key=wrong_key,
        wrong_payload_id=wrong_payload_id,
        coordinate_count=int(contract["coordinate_count"]),
        min_observed_events=int(accept_requires["min_observed_events"]),
        min_distinct_coordinates=int(accept_requires["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(required["protected_keyed_correlation_score_min"]),
        min_specificity_margin=float(required["protected_minus_best_wrong_specificity_margin_min"]),
        max_wrong_score=float(required["wrong_key_correlation_score_max"]),
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def run_audit(
    *,
    generation_dir: Path,
    package_dir: Path,
    output_dir: Path,
    prompts_per_block: int,
    sample_events: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    generated_rows = _load_generated_rows(generation_dir)
    bank = load_event_window_bank(package_dir / "event_window_bank.json")
    contract = _read_json(package_dir / "contract.json")
    decoder_spec = _read_json(package_dir / "decoder_spec.json")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    condition_rows: Counter[str] = Counter()
    row_event_counts: dict[str, list[int]] = defaultdict(list)
    row_with_events: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sample_rows: list[dict[str, Any]] = []

    for row in generated_rows:
        condition = str(row.get("generation_condition", "unknown"))
        condition_rows[condition] += 1
        block_id = _block_id(row, prompts_per_block)
        events = extract_support_window_events(str(row.get("response_text", "")), bank, scrub_mode="all")
        row_event_counts[condition].append(len(events))
        if events:
            row_with_events[condition] += 1
        for event in events:
            event_payload = {
                **event,
                "block_id": block_id,
                "source_condition": condition,
                "generation_id": str(row.get("generation_id", "")),
                "prompt_id": str(row.get("prompt_id", "")),
            }
            grouped[(condition, block_id)].append(event_payload)
            family_counts[condition][str(event["surface_family"])] += 1
            if len(sample_rows) < sample_events:
                sample_rows.append(event_payload)

    decode_rows: list[dict[str, Any]] = []
    for condition in ("protected", "raw", "task_only"):
        block_ids = sorted(block_id for (source_condition, block_id) in grouped if source_condition == condition)
        if not block_ids:
            block_ids = sorted(
                {
                    _block_id(row, prompts_per_block)
                    for row in generated_rows
                    if str(row.get("generation_condition", "")) == condition
                }
            )
        for block_id in block_ids:
            events = grouped.get((condition, block_id), [])
            arms = [condition]
            if condition == "protected":
                arms.extend(["wrong_key", "wrong_payload"])
            for arm in arms:
                decision = _decision(events, arm=arm, contract=contract, decoder_spec=decoder_spec)
                decode_rows.append(
                    {
                        "block_id": block_id,
                        "arm": arm,
                        "source_condition": condition,
                        "dry_run_decoder_accept": decision.accept,
                        "keyed_correlation_score": decision.keyed_correlation_score,
                        "wrong_key_correlation_score": decision.wrong_key_correlation_score,
                        "wrong_payload_correlation_score": decision.wrong_payload_correlation_score,
                        "specificity_margin": decision.specificity_margin,
                        "observed_events": decision.observed_events,
                        "distinct_coordinates": decision.distinct_coordinates,
                        "format_scrub_mode": "all",
                    }
                )

    arm_summary: dict[str, dict[str, Any]] = {}
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decode_rows:
        by_arm[str(row["arm"])].append(row)
    for arm, rows in sorted(by_arm.items()):
        arm_summary[arm] = {
            "blocks": len(rows),
            "dry_run_accepts": sum(1 for row in rows if row["dry_run_decoder_accept"]),
            "observed_events_mean": sum(float(row["observed_events"]) for row in rows) / len(rows),
            "observed_events_median": _median([float(row["observed_events"]) for row in rows]),
            "distinct_coordinates_mean": sum(float(row["distinct_coordinates"]) for row in rows) / len(rows),
            "keyed_score_mean": sum(float(row["keyed_correlation_score"]) for row in rows) / len(rows),
            "specificity_margin_mean": sum(float(row["specificity_margin"]) for row in rows) / len(rows),
            "specificity_margin_min": min(float(row["specificity_margin"]) for row in rows),
        }

    condition_coverage_rows: list[dict[str, Any]] = []
    for condition in sorted(condition_rows):
        counts = row_event_counts[condition]
        total_events = sum(counts)
        condition_coverage_rows.append(
            {
                "condition": condition,
                "rows": condition_rows[condition],
                "rows_with_support_events": row_with_events[condition],
                "row_support_rate": row_with_events[condition] / condition_rows[condition],
                "total_support_events": total_events,
                "mean_events_per_row": total_events / condition_rows[condition],
                "median_events_per_row": _median([float(value) for value in counts]),
            }
        )

    family_rows: list[dict[str, Any]] = []
    for condition, counter in sorted(family_counts.items()):
        total = sum(counter.values())
        for family, count in counter.most_common():
            family_rows.append(
                {
                    "condition": condition,
                    "surface_family": family,
                    "event_count": count,
                    "fraction_within_condition": count / total if total else 0.0,
                }
            )

    protected_accepts = arm_summary.get("protected", {}).get("dry_run_accepts", 0)
    raw_accepts = arm_summary.get("raw", {}).get("dry_run_accepts", 0)
    task_only_accepts = arm_summary.get("task_only", {}).get("dry_run_accepts", 0)
    wrong_key_accepts = arm_summary.get("wrong_key", {}).get("dry_run_accepts", 0)
    wrong_payload_accepts = arm_summary.get("wrong_payload", {}).get("dry_run_accepts", 0)
    if raw_accepts or task_only_accepts or wrong_key_accepts or wrong_payload_accepts:
        interpretation = "Support-window contract is not selective enough in this dry-run because one or more null/control arms accept."
        status = "FAIL_SUPPORT_WINDOW_DRY_RUN_NULL_OR_CONTROL_ACCEPTS_NO_COMPUTE"
    elif not protected_accepts:
        interpretation = "Support-window contract increases support diagnostics but does not recover protected accept-like blocks in this dry-run."
        status = "FAIL_SUPPORT_WINDOW_DRY_RUN_NO_PROTECTED_ACCEPTS_NO_COMPUTE"
    else:
        interpretation = (
            "Support-window dry-run produces protected accept-like blocks with clean controls, "
            "but this remains a post-hoc diagnostic over 859277 and is not a positive claim."
        )
        status = "PASS_SUPPORT_WINDOW_DRY_RUN_DIAGNOSTIC_ONLY_NO_COMPUTE"

    summary = {
        "schema_name": "natural_evidence_v2_r4_positive_support_window_coverage_dry_run_v1",
        "status": status,
        "generation_dir": str(generation_dir),
        "package_dir": str(package_dir),
        "generated_rows": len(generated_rows),
        "condition_coverage": condition_coverage_rows,
        "arm_summary": arm_summary,
        "interpretation": interpretation,
        "diagnostic_only": True,
        "positive_reclassification_allowed": False,
        "post_hoc_phrase_mining_allowed": False,
        "slurm_allowed": False,
        "generation_allowed": False,
        "paper_claim_allowed": False,
        "audit_key_commitment": _sha256_text(_DEV_AUDIT_KEY_MATERIAL),
        "wrong_key_commitment": _sha256_text(_DEV_WRONG_KEY_MATERIAL),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "coverage_summary.json", summary)
    _write_csv(
        output_dir / "condition_coverage.csv",
        condition_coverage_rows,
        [
            "condition",
            "rows",
            "rows_with_support_events",
            "row_support_rate",
            "total_support_events",
            "mean_events_per_row",
            "median_events_per_row",
        ],
    )
    _write_csv(
        output_dir / "per_block_dry_run_decode.csv",
        decode_rows,
        [
            "block_id",
            "arm",
            "source_condition",
            "dry_run_decoder_accept",
            "keyed_correlation_score",
            "wrong_key_correlation_score",
            "wrong_payload_correlation_score",
            "specificity_margin",
            "observed_events",
            "distinct_coordinates",
            "format_scrub_mode",
        ],
    )
    _write_csv(
        output_dir / "event_family_counts.csv",
        family_rows,
        ["condition", "surface_family", "event_count", "fraction_within_condition"],
    )
    _write_jsonl(output_dir / "support_event_sample.jsonl", sample_rows)
    _write_report(output_dir / "coverage_report.md", summary)
    return summary


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    lines = [
        "# R4 Positive Support-Window Coverage Dry-Run",
        "",
        "## Verdict",
        "",
        f"`{summary['status']}`",
        "",
        summary["interpretation"],
        "",
        "This is a diagnostic-only dry-run over existing `859277` outputs. It does not",
        "reclassify `859277` as positive and does not unlock compute or claims.",
        "",
        "## Coverage By Condition",
        "",
        "| condition | rows | rows with events | support rate | total events | mean events/row |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["condition_coverage"]:
        lines.append(
            "| {condition} | {rows} | {rows_with_support_events} | {row_support_rate:.3f} | "
            "{total_support_events} | {mean_events_per_row:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Dry-Run Decoder Summary",
            "",
            "| arm | blocks | dry-run accepts | mean events | mean coords | mean keyed score | mean margin |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, row in sorted(summary["arm_summary"].items()):
        lines.append(
            "| {arm} | {blocks} | {dry_run_accepts} | {observed_events_mean:.2f} | "
            "{distinct_coordinates_mean:.2f} | {keyed_score_mean:.2f} | "
            "{specificity_margin_mean:.2f} |".format(arm=arm, **row)
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `coverage_summary.json`",
            "- `condition_coverage.csv`",
            "- `per_block_dry_run_decode.csv`",
            "- `event_family_counts.csv`",
            "- `support_event_sample.jsonl`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run R4 support-window coverage on existing 859277 outputs.")
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--sample-events", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_audit(
        generation_dir=args.generation_dir if args.generation_dir.is_absolute() else ROOT / args.generation_dir,
        package_dir=args.package_dir if args.package_dir.is_absolute() else ROOT / args.package_dir,
        output_dir=args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir,
        prompts_per_block=args.prompts_per_block,
        sample_events=args.sample_events,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
