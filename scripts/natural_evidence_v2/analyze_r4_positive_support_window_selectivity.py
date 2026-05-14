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

from scripts.natural_evidence_v2.audit_r4_positive_support_window_coverage import (  # noqa: E402
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
    _block_id,
    _load_generated_rows,
    _read_json,
)
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (  # noqa: E402
    extract_support_window_events,
    load_event_window_bank,
)
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (  # noqa: E402
    decide_keyed_correlation,
    map_surface_to_coordinate_and_polarity,
)

DEFAULT_GENERATION_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277"
)
DEFAULT_PACKAGE_DIR = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149"
)


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


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


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("inf") if numerator > 0 else 0.0
    return float(numerator / denominator)


def _finite_ratio(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.6f}"


def run_analysis(
    *,
    generation_dir: Path,
    package_dir: Path,
    output_dir: Path,
    prompts_per_block: int,
    accepted_null_top_k: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")

    generated_rows = _load_generated_rows(generation_dir)
    bank = load_event_window_bank(package_dir / "event_window_bank.json")
    contract = _read_json(package_dir / "contract.json")
    decoder_spec = _read_json(package_dir / "decoder_spec.json")
    coordinate_count = int(contract["coordinate_count"])
    payload_id = str(contract["payload_id"])

    condition_rows: Counter[str] = Counter()
    condition_row_ids_with_event: dict[str, set[str]] = defaultdict(set)
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)
    surface_positive_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_positive_counts: dict[str, Counter[str]] = defaultdict(Counter)
    coordinate_counts: dict[str, Counter[int]] = defaultdict(Counter)
    coordinate_positive_counts: dict[str, Counter[int]] = defaultdict(Counter)
    row_event_counts: dict[str, list[int]] = defaultdict(list)
    grouped_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    surface_family: dict[str, str] = {}
    surface_phrase: dict[str, str] = {}
    surface_coordinate: dict[str, int] = {}
    surface_polarity: dict[str, int] = {}

    for row in generated_rows:
        condition = str(row.get("generation_condition", "unknown"))
        generation_id = str(row.get("generation_id", ""))
        block_id = _block_id(row, prompts_per_block)
        condition_rows[condition] += 1
        events = extract_support_window_events(str(row.get("response_text", "")), bank, scrub_mode="all")
        row_event_counts[condition].append(len(events))
        if events:
            condition_row_ids_with_event[condition].add(generation_id)
        for event in events:
            surface_id = str(event["surface_id"])
            family = str(event["surface_family"])
            coordinate, polarity = map_surface_to_coordinate_and_polarity(
                audit_key=_DEV_AUDIT_KEY_MATERIAL,
                payload_id=payload_id,
                surface_id=surface_id,
                coordinate_count=coordinate_count,
            )
            payload = {
                **event,
                "block_id": block_id,
                "source_condition": condition,
                "generation_id": generation_id,
                "prompt_id": str(row.get("prompt_id", "")),
                "keyed_coordinate": coordinate,
                "keyed_polarity": polarity,
            }
            grouped_events[(condition, block_id)].append(payload)
            surface_counts[condition][surface_id] += 1
            family_counts[condition][family] += 1
            coordinate_counts[condition][coordinate] += 1
            if polarity > 0:
                surface_positive_counts[condition][surface_id] += 1
                family_positive_counts[condition][family] += 1
                coordinate_positive_counts[condition][coordinate] += 1
            surface_family[surface_id] = family
            surface_phrase[surface_id] = str(event["canonical_phrase"])
            surface_coordinate[surface_id] = coordinate
            surface_polarity[surface_id] = polarity

    decode_rows: list[dict[str, Any]] = []
    for condition in ("protected", "raw", "task_only"):
        block_ids = sorted(
            {
                _block_id(row, prompts_per_block)
                for row in generated_rows
                if str(row.get("generation_condition", "")) == condition
            }
        )
        for block_id in block_ids:
            events = grouped_events.get((condition, block_id), [])
            arms = [condition]
            if condition == "protected":
                arms.extend(["wrong_key", "wrong_payload"])
            for arm in arms:
                decision = _decision(events, arm=arm, contract=contract, decoder_spec=decoder_spec)
                positive_events = sum(1 for event in events if int(event["keyed_polarity"]) > 0)
                negative_events = sum(1 for event in events if int(event["keyed_polarity"]) < 0)
                decode_rows.append(
                    {
                        "block_id": block_id,
                        "arm": arm,
                        "source_condition": condition,
                        "dry_run_decoder_accept": decision.accept,
                        "observed_events": decision.observed_events,
                        "distinct_coordinates": decision.distinct_coordinates,
                        "positive_keyed_events": positive_events,
                        "negative_keyed_events": negative_events,
                        "keyed_correlation_score": decision.keyed_correlation_score,
                        "wrong_key_correlation_score": decision.wrong_key_correlation_score,
                        "wrong_payload_correlation_score": decision.wrong_payload_correlation_score,
                        "specificity_margin": decision.specificity_margin,
                    }
                )

    conditions = ["protected", "raw", "task_only"]
    surface_rows: list[dict[str, Any]] = []
    all_surface_ids = sorted(set().union(*(set(surface_counts[c]) for c in conditions)))
    for surface_id in all_surface_ids:
        counts = {condition: surface_counts[condition][surface_id] for condition in conditions}
        positive_counts = {condition: surface_positive_counts[condition][surface_id] for condition in conditions}
        rates = {condition: _rate(counts[condition], condition_rows[condition]) for condition in conditions}
        null_max_rate = max(rates["raw"], rates["task_only"])
        protected_enrichment = _ratio(rates["protected"], null_max_rate)
        surface_rows.append(
            {
                "surface_id": surface_id,
                "surface_family": surface_family.get(surface_id, ""),
                "canonical_phrase": surface_phrase.get(surface_id, ""),
                "keyed_coordinate": surface_coordinate.get(surface_id, ""),
                "keyed_polarity": surface_polarity.get(surface_id, ""),
                "protected_events": counts["protected"],
                "raw_events": counts["raw"],
                "task_only_events": counts["task_only"],
                "protected_positive_events": positive_counts["protected"],
                "raw_positive_events": positive_counts["raw"],
                "task_only_positive_events": positive_counts["task_only"],
                "protected_event_rate": rates["protected"],
                "raw_event_rate": rates["raw"],
                "task_only_event_rate": rates["task_only"],
                "protected_over_max_null_rate": _finite_ratio(protected_enrichment),
                "diagnostic_selective_under_859277": (
                    counts["protected"] >= 16
                    and counts["raw"] == 0
                    and counts["task_only"] == 0
                ),
            }
        )
    surface_rows.sort(
        key=lambda row: (
            str(row["diagnostic_selective_under_859277"]) != "True",
            -float(row["protected_event_rate"]),
            str(row["surface_id"]),
        )
    )

    family_rows: list[dict[str, Any]] = []
    all_families = sorted(set().union(*(set(family_counts[c]) for c in conditions)))
    for family in all_families:
        counts = {condition: family_counts[condition][family] for condition in conditions}
        positive_counts = {condition: family_positive_counts[condition][family] for condition in conditions}
        rates = {condition: _rate(counts[condition], condition_rows[condition]) for condition in conditions}
        positive_rates = {
            condition: _rate(positive_counts[condition], condition_rows[condition]) for condition in conditions
        }
        null_max_rate = max(rates["raw"], rates["task_only"])
        positive_null_max_rate = max(positive_rates["raw"], positive_rates["task_only"])
        family_rows.append(
            {
                "surface_family": family,
                "protected_events": counts["protected"],
                "raw_events": counts["raw"],
                "task_only_events": counts["task_only"],
                "protected_event_rate": rates["protected"],
                "raw_event_rate": rates["raw"],
                "task_only_event_rate": rates["task_only"],
                "protected_over_max_null_rate": _finite_ratio(_ratio(rates["protected"], null_max_rate)),
                "protected_positive_events": positive_counts["protected"],
                "raw_positive_events": positive_counts["raw"],
                "task_only_positive_events": positive_counts["task_only"],
                "protected_positive_rate": positive_rates["protected"],
                "raw_positive_rate": positive_rates["raw"],
                "task_only_positive_rate": positive_rates["task_only"],
                "protected_positive_over_max_null_positive_rate": _finite_ratio(
                    _ratio(positive_rates["protected"], positive_null_max_rate)
                ),
            }
        )
    family_rows.sort(key=lambda row: (-int(row["protected_events"]), str(row["surface_family"])))

    coordinate_rows: list[dict[str, Any]] = []
    for coordinate in range(coordinate_count):
        counts = {condition: coordinate_counts[condition][coordinate] for condition in conditions}
        positive_counts = {condition: coordinate_positive_counts[condition][coordinate] for condition in conditions}
        coordinate_rows.append(
            {
                "coordinate": coordinate,
                "protected_events": counts["protected"],
                "raw_events": counts["raw"],
                "task_only_events": counts["task_only"],
                "protected_positive_events": positive_counts["protected"],
                "raw_positive_events": positive_counts["raw"],
                "task_only_positive_events": positive_counts["task_only"],
                "protected_positive_minus_max_null_positive": positive_counts["protected"]
                - max(positive_counts["raw"], positive_counts["task_only"]),
            }
        )

    accepted_null_rows: list[dict[str, Any]] = []
    for row in decode_rows:
        if str(row["arm"]) not in {"raw", "task_only"} or not row["dry_run_decoder_accept"]:
            continue
        events = grouped_events[(str(row["source_condition"]), str(row["block_id"]))]
        positive_by_surface: Counter[str] = Counter()
        positive_by_family: Counter[str] = Counter()
        for event in events:
            if int(event["keyed_polarity"]) <= 0:
                continue
            positive_by_surface[str(event["surface_id"])] += 1
            positive_by_family[str(event["surface_family"])] += 1
        top_surfaces = [
            f"{surface_id}:{count}"
            for surface_id, count in positive_by_surface.most_common(accepted_null_top_k)
        ]
        top_families = [
            f"{family}:{count}"
            for family, count in positive_by_family.most_common(accepted_null_top_k)
        ]
        accepted_null_rows.append(
            {
                "block_id": row["block_id"],
                "arm": row["arm"],
                "observed_events": row["observed_events"],
                "distinct_coordinates": row["distinct_coordinates"],
                "positive_keyed_events": row["positive_keyed_events"],
                "negative_keyed_events": row["negative_keyed_events"],
                "keyed_correlation_score": row["keyed_correlation_score"],
                "specificity_margin": row["specificity_margin"],
                "top_positive_families": ";".join(top_families),
                "top_positive_surfaces": ";".join(top_surfaces),
            }
        )

    arm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decode_rows:
        arm_groups[str(row["arm"])].append(row)
    arm_summary: dict[str, dict[str, Any]] = {}
    for arm, rows in sorted(arm_groups.items()):
        arm_summary[arm] = {
            "blocks": len(rows),
            "accepts": sum(1 for row in rows if row["dry_run_decoder_accept"]),
            "mean_observed_events": sum(float(row["observed_events"]) for row in rows) / len(rows),
            "median_observed_events": _median([float(row["observed_events"]) for row in rows]),
            "mean_distinct_coordinates": sum(float(row["distinct_coordinates"]) for row in rows) / len(rows),
            "mean_positive_keyed_events": sum(float(row["positive_keyed_events"]) for row in rows) / len(rows),
            "mean_negative_keyed_events": sum(float(row["negative_keyed_events"]) for row in rows) / len(rows),
            "mean_keyed_score": sum(float(row["keyed_correlation_score"]) for row in rows) / len(rows),
            "min_specificity_margin": min(float(row["specificity_margin"]) for row in rows),
        }

    protected_accepts = arm_summary.get("protected", {}).get("accepts", 0)
    raw_accepts = arm_summary.get("raw", {}).get("accepts", 0)
    task_only_accepts = arm_summary.get("task_only", {}).get("accepts", 0)
    wrong_key_accepts = arm_summary.get("wrong_key", {}).get("accepts", 0)
    wrong_payload_accepts = arm_summary.get("wrong_payload", {}).get("accepts", 0)
    diagnostic_selective_surfaces = [
        row for row in surface_rows if row["diagnostic_selective_under_859277"]
    ]
    common_plan_fraction_raw = next(
        (
            float(row["raw_events"]) / max(1, sum(family_counts["raw"].values()))
            for row in family_rows
            if row["surface_family"] == "plan"
        ),
        0.0,
    )
    common_plan_fraction_task_only = next(
        (
            float(row["task_only_events"]) / max(1, sum(family_counts["task_only"].values()))
            for row in family_rows
            if row["surface_family"] == "plan"
        ),
        0.0,
    )

    status = "FAIL_SELECTIVITY_ANALYSIS_COMMON_SUPPORT_NO_COMPUTE"
    if raw_accepts == 0 and task_only_accepts == 0 and wrong_key_accepts == 0 and wrong_payload_accepts == 0:
        status = "PASS_SELECTIVITY_ANALYSIS_DIAGNOSTIC_ONLY_NO_COMPUTE"
    summary = {
        "schema_name": "natural_evidence_v2_r4_positive_support_window_selectivity_analysis_v1",
        "status": status,
        "diagnostic_only": True,
        "source_failed_job": "859277",
        "generation_dir": str(generation_dir),
        "package_dir": str(package_dir),
        "event_window_bank_sha256": _sha256_file(package_dir / "event_window_bank.json"),
        "generated_rows": len(generated_rows),
        "condition_rows": dict(condition_rows),
        "condition_rows_with_events": {
            condition: len(condition_row_ids_with_event[condition]) for condition in sorted(condition_rows)
        },
        "arm_summary": arm_summary,
        "protected_dry_run_accepts": protected_accepts,
        "raw_dry_run_accepts": raw_accepts,
        "task_only_dry_run_accepts": task_only_accepts,
        "wrong_key_dry_run_accepts": wrong_key_accepts,
        "wrong_payload_dry_run_accepts": wrong_payload_accepts,
        "accepted_null_blocks": len(accepted_null_rows),
        "diagnostic_selective_surface_count": len(diagnostic_selective_surfaces),
        "top_families_by_protected_events": family_rows[:5],
        "plan_fraction_within_raw_events": common_plan_fraction_raw,
        "plan_fraction_within_task_only_events": common_plan_fraction_task_only,
        "interpretation": (
            "Support-window events are broad task-language features, not a selective protected channel. "
            "Raw/task-only accepted blocks are driven by common positive-polarity support events under the same key."
        ),
        "positive_reclassification_allowed": False,
        "post_hoc_surface_bank_from_859277_allowed": False,
        "slurm_allowed": False,
        "generation_allowed": False,
        "model_scoring_allowed": False,
        "training_allowed": False,
        "paper_claim_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "selectivity_summary.json", summary)
    _write_csv(
        output_dir / "surface_selectivity.csv",
        surface_rows,
        [
            "surface_id",
            "surface_family",
            "canonical_phrase",
            "keyed_coordinate",
            "keyed_polarity",
            "protected_events",
            "raw_events",
            "task_only_events",
            "protected_positive_events",
            "raw_positive_events",
            "task_only_positive_events",
            "protected_event_rate",
            "raw_event_rate",
            "task_only_event_rate",
            "protected_over_max_null_rate",
            "diagnostic_selective_under_859277",
        ],
    )
    _write_csv(
        output_dir / "family_selectivity.csv",
        family_rows,
        [
            "surface_family",
            "protected_events",
            "raw_events",
            "task_only_events",
            "protected_event_rate",
            "raw_event_rate",
            "task_only_event_rate",
            "protected_over_max_null_rate",
            "protected_positive_events",
            "raw_positive_events",
            "task_only_positive_events",
            "protected_positive_rate",
            "raw_positive_rate",
            "task_only_positive_rate",
            "protected_positive_over_max_null_positive_rate",
        ],
    )
    _write_csv(
        output_dir / "coordinate_selectivity.csv",
        coordinate_rows,
        [
            "coordinate",
            "protected_events",
            "raw_events",
            "task_only_events",
            "protected_positive_events",
            "raw_positive_events",
            "task_only_positive_events",
            "protected_positive_minus_max_null_positive",
        ],
    )
    _write_csv(
        output_dir / "accepted_null_block_attribution.csv",
        accepted_null_rows,
        [
            "block_id",
            "arm",
            "observed_events",
            "distinct_coordinates",
            "positive_keyed_events",
            "negative_keyed_events",
            "keyed_correlation_score",
            "specificity_margin",
            "top_positive_families",
            "top_positive_surfaces",
        ],
    )
    _write_report(output_dir / "selectivity_report.md", summary)
    return summary


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    lines = [
        "# R4 Positive Support-Window Selectivity Analysis",
        "",
        "## Verdict",
        "",
        f"`{summary['status']}`",
        "",
        str(summary["interpretation"]),
        "",
        "This is artifact-only analysis over the already failed `859277` outputs. It does not",
        "reclassify that run and does not permit Slurm, generation, model scoring, training, or claims.",
        "",
        "## Dry-Run Arm Summary",
        "",
        "| arm | blocks | accepts | mean events | mean positive keyed events | mean keyed score | min margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, row in sorted(summary["arm_summary"].items()):
        lines.append(
            "| {arm} | {blocks} | {accepts} | {mean_observed_events:.2f} | "
            "{mean_positive_keyed_events:.2f} | {mean_keyed_score:.2f} | "
            "{min_specificity_margin:.2f} |".format(arm=arm, **row)
        )
    lines.extend(
        [
            "",
            "## Selectivity Diagnostics",
            "",
            f"- accepted null/control blocks: `{summary['accepted_null_blocks']}`",
            f"- diagnostic selective surfaces under 859277: `{summary['diagnostic_selective_surface_count']}`",
            f"- raw plan-family event fraction: `{summary['plan_fraction_within_raw_events']:.3f}`",
            f"- task-only plan-family event fraction: `{summary['plan_fraction_within_task_only_events']:.3f}`",
            "",
            "Top protected families:",
            "",
            "| family | protected events | raw events | task-only events | protected/max-null rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["top_families_by_protected_events"]:
        lines.append(
            "| {surface_family} | {protected_events} | {raw_events} | {task_only_events} | "
            "{protected_over_max_null_rate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `selectivity_summary.json`",
            "- `surface_selectivity.csv`",
            "- `family_selectivity.csv`",
            "- `coordinate_selectivity.csv`",
            "- `accepted_null_block_attribution.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze why R4 support-window dry-run accepts raw/task-only controls."
    )
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--accepted-null-top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_analysis(
        generation_dir=args.generation_dir if args.generation_dir.is_absolute() else ROOT / args.generation_dir,
        package_dir=args.package_dir if args.package_dir.is_absolute() else ROOT / args.package_dir,
        output_dir=args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir,
        prompts_per_block=args.prompts_per_block,
        accepted_null_top_k=args.accepted_null_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
