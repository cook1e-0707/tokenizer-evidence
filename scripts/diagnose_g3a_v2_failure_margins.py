from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.infrastructure.paths import current_timestamp, discover_repo_root


FAILURE_CASE_IDS = ("B1_U03_s23", "B1_U12_s23", "B1_U15_s23", "B4_U12_s23")

SLOT_MARGIN_FIELDS = [
    "anchor_failure_case_id",
    "comparison_role",
    "neighbor_relation",
    "case_id",
    "result_class",
    "block_count",
    "seed",
    "payload",
    "block_index",
    "field_name",
    "expected_bucket",
    "decoded_bucket",
    "bucket_correct",
    "exact_token_correct",
    "target_bucket_logmass",
    "strongest_wrong_bucket",
    "strongest_wrong_bucket_logmass",
    "bucket_margin",
    "target_bucket_rank",
    "target_token_probability",
    "generated_token",
    "top_5_bucket_logmasses",
    "top_5_tokens",
]

NEIGHBOR_FIELDS = [
    "failure_case_id",
    "neighbor_case_id",
    "neighbor_relation",
    "failure_result_class",
    "neighbor_result_class",
    "failure_block_count",
    "neighbor_block_count",
    "failure_seed",
    "neighbor_seed",
    "failure_payload",
    "neighbor_payload",
    "failure_slot_bucket_accuracy",
    "neighbor_slot_bucket_accuracy",
    "slot_bucket_accuracy_delta_neighbor_minus_failure",
    "failure_symbol_error_count",
    "neighbor_symbol_error_count",
    "failure_wrong_fields",
    "neighbor_wrong_fields",
    "failure_normalized_L_set_mean",
    "neighbor_normalized_L_set_mean",
    "failure_slot_margin_min_final",
    "neighbor_slot_margin_min_final",
    "failure_checkpoint_selection_best_step",
    "neighbor_checkpoint_selection_best_step",
    "failure_checkpoint_selection_best_metric_value",
    "neighbor_checkpoint_selection_best_metric_value",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose G3a-v2 valid method failures.")
    parser.add_argument("--run-table", default="results/tables/g3a_v2_block_scale.csv")
    parser.add_argument("--slot-table", default="results/tables/g3a_v2_slot_diagnostics.csv")
    parser.add_argument("--symbol-table", default="results/tables/g3a_v2_symbol_diagnostics.csv")
    parser.add_argument(
        "--slot-margin-out",
        default="results/tables/g3a_v2_failure_slot_margin.csv",
    )
    parser.add_argument(
        "--neighbor-out",
        default="results/tables/g3a_v2_failure_neighbor_comparison.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="results/processed/paper_stats/g3a_v2_failure_margin_summary.json",
    )
    parser.add_argument("--report-out", default="docs/g3a_v2_failure_root_cause.md")
    return parser.parse_args()


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _case_key(block_count: int, payload: str, seed: int) -> str:
    variant = {1: "B1", 2: "B2", 4: "B4"}[int(block_count)]
    return f"{variant}_{payload}_s{seed}"


def _neighbor_cases(failure: dict[str, str], rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    failure_id = failure["case_id"]
    block_count = failure["block_count"]
    payload = failure["payload"]
    seed = failure["seed"]
    output: list[tuple[str, dict[str, str]]] = [("anchor_failure", failure)]
    for row in rows:
        if row["case_id"] == failure_id:
            continue
        relation = ""
        if row["block_count"] == block_count and row["seed"] == seed and row["payload"] != payload:
            relation = "same_block_count_same_seed_different_payload"
        elif row["block_count"] == block_count and row["payload"] == payload and row["seed"] != seed:
            relation = "same_block_count_same_payload_different_seed"
        elif row["seed"] == seed and row["payload"] == payload and row["block_count"] != block_count:
            relation = "same_seed_same_payload_different_block_count"
        if relation:
            output.append((relation, row))
    return output


def _wrong_fields(case_id: str, slot_by_case: dict[str, list[dict[str, str]]]) -> list[str]:
    return [
        row["field_name"]
        for row in slot_by_case.get(case_id, [])
        if not _is_true(row.get("bucket_correct"))
    ]


def _slot_margin_rows(
    failure_rows: list[dict[str, str]],
    all_run_rows: list[dict[str, str]],
    slot_by_case: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for failure in failure_rows:
        for relation, case in _neighbor_cases(failure, all_run_rows):
            role = "failure" if relation == "anchor_failure" else "neighbor"
            for slot in slot_by_case.get(case["case_id"], []):
                key = (failure["case_id"], relation, case["case_id"], slot["slot_index"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "anchor_failure_case_id": failure["case_id"],
                        "comparison_role": role,
                        "neighbor_relation": relation,
                        "case_id": case["case_id"],
                        "result_class": case["result_class"],
                        "block_count": case["block_count"],
                        "seed": case["seed"],
                        "payload": case["payload"],
                        "block_index": slot["block_index"],
                        "field_name": slot["field_name"],
                        "expected_bucket": slot["expected_bucket"],
                        "decoded_bucket": slot["decoded_bucket"],
                        "bucket_correct": slot["bucket_correct"],
                        "exact_token_correct": slot["exact_token_correct"],
                        "target_bucket_logmass": "missing",
                        "strongest_wrong_bucket": "missing",
                        "strongest_wrong_bucket_logmass": "missing",
                        "bucket_margin": "missing",
                        "target_bucket_rank": "missing",
                        "target_token_probability": "missing",
                        "generated_token": slot["generated_token"],
                        "top_5_bucket_logmasses": "missing",
                        "top_5_tokens": "missing",
                    }
                )
    return rows


def _neighbor_comparison_rows(
    failure_rows: list[dict[str, str]],
    all_run_rows: list[dict[str, str]],
    slot_by_case: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for failure in failure_rows:
        failure_wrong = _wrong_fields(failure["case_id"], slot_by_case)
        for relation, neighbor in _neighbor_cases(failure, all_run_rows):
            if relation == "anchor_failure":
                continue
            key = (failure["case_id"], relation, neighbor["case_id"])
            if key in seen:
                continue
            seen.add(key)
            neighbor_wrong = _wrong_fields(neighbor["case_id"], slot_by_case)
            rows.append(
                {
                    "failure_case_id": failure["case_id"],
                    "neighbor_case_id": neighbor["case_id"],
                    "neighbor_relation": relation,
                    "failure_result_class": failure["result_class"],
                    "neighbor_result_class": neighbor["result_class"],
                    "failure_block_count": failure["block_count"],
                    "neighbor_block_count": neighbor["block_count"],
                    "failure_seed": failure["seed"],
                    "neighbor_seed": neighbor["seed"],
                    "failure_payload": failure["payload"],
                    "neighbor_payload": neighbor["payload"],
                    "failure_slot_bucket_accuracy": failure["slot_bucket_accuracy"],
                    "neighbor_slot_bucket_accuracy": neighbor["slot_bucket_accuracy"],
                    "slot_bucket_accuracy_delta_neighbor_minus_failure": (
                        _float(neighbor["slot_bucket_accuracy"]) - _float(failure["slot_bucket_accuracy"])
                    ),
                    "failure_symbol_error_count": failure["symbol_error_count"],
                    "neighbor_symbol_error_count": neighbor["symbol_error_count"],
                    "failure_wrong_fields": ";".join(failure_wrong),
                    "neighbor_wrong_fields": ";".join(neighbor_wrong),
                    "failure_normalized_L_set_mean": failure["normalized_L_set_mean"],
                    "neighbor_normalized_L_set_mean": neighbor["normalized_L_set_mean"],
                    "failure_slot_margin_min_final": failure["slot_margin_min"],
                    "neighbor_slot_margin_min_final": neighbor["slot_margin_min"],
                    "failure_checkpoint_selection_best_step": failure["checkpoint_selection_best_step"],
                    "neighbor_checkpoint_selection_best_step": neighbor["checkpoint_selection_best_step"],
                    "failure_checkpoint_selection_best_metric_value": failure[
                        "checkpoint_selection_best_metric_value"
                    ],
                    "neighbor_checkpoint_selection_best_metric_value": neighbor[
                        "checkpoint_selection_best_metric_value"
                    ],
                }
            )
    return rows


def _read_train_metrics(run_row: dict[str, str]) -> dict[str, Any]:
    train_summary_path = Path(run_row.get("train_summary_path", ""))
    metrics_path = train_summary_path.parent / "train_metrics.jsonl" if train_summary_path else Path("")
    if not metrics_path.exists():
        return {
            "available": False,
            "train_metrics_path": str(metrics_path),
            "missing_reason": "train_metrics.jsonl_not_available_in_local_workspace_or_not_saved",
            "train_loss_curve": "missing",
            "L_set_mean_curve": "missing",
            "L_margin_curve": "missing",
            "target_bucket_mass_mean_curve": "missing",
            "target_bucket_mass_min_curve": "missing",
            "slot_margin_mean_curve": "missing",
            "slot_margin_min_curve": "missing",
            "selected_checkpoint_epoch": "missing",
            "final_checkpoint_epoch": "missing",
            "best_checkpoint_metric": run_row.get("checkpoint_selection_best_metric_value", "missing"),
            "final_checkpoint_metric": run_row.get("normalized_L_set_mean", "missing"),
            "would_have_passed_earlier_checkpoint": "unknown_no_per_checkpoint_eval",
        }
    metrics = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            metrics.append(json.loads(line))
    def curve(key: str) -> list[float]:
        return [_float(item[key]) for item in metrics if key in item]
    epochs = [_int(item.get("epoch")) for item in metrics if item.get("epoch") is not None]
    return {
        "available": True,
        "train_metrics_path": str(metrics_path),
        "point_count": len(metrics),
        "train_loss_curve": curve("loss"),
        "L_set_mean_curve": curve("normalized_L_set_mean"),
        "L_margin_curve": "missing",
        "target_bucket_mass_mean_curve": curve("target_bucket_mass_mean"),
        "target_bucket_mass_min_curve": curve("target_bucket_mass_min"),
        "slot_margin_mean_curve": curve("slot_margin_mean"),
        "slot_margin_min_curve": curve("slot_margin_min"),
        "selected_checkpoint_epoch": "missing_step_to_epoch_mapping_not_recorded",
        "final_checkpoint_epoch": max(epochs) if epochs else "missing",
        "best_checkpoint_metric": run_row.get("checkpoint_selection_best_metric_value", "missing"),
        "final_checkpoint_metric": run_row.get("normalized_L_set_mean", "missing"),
        "would_have_passed_earlier_checkpoint": "unknown_no_per_checkpoint_eval",
    }


def _summary_payload(
    run_by_case: dict[str, dict[str, str]],
    slot_by_case: dict[str, list[dict[str, str]]],
    neighbor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [run_by_case[case_id] for case_id in FAILURE_CASE_IDS]
    failure_counter_by_seed = Counter(row["seed"] for row in failures)
    failure_counter_by_payload = Counter(row["payload"] for row in failures)
    failure_counter_by_variant = Counter(row["variant_id"] for row in failures)
    wrong_field_counter = Counter()
    for row in failures:
        wrong_field_counter.update(_wrong_fields(row["case_id"], slot_by_case))
    especially = {
        "B1_U03_s23_vs_other_seeds": [
            run_by_case[_case_key(1, "U03", seed)]["result_class"] for seed in (17, 29)
        ],
        "B1_U12_s23_vs_other_seeds": [
            run_by_case[_case_key(1, "U12", seed)]["result_class"] for seed in (17, 29)
        ],
        "B1_U15_s23_vs_other_seeds": [
            run_by_case[_case_key(1, "U15", seed)]["result_class"] for seed in (17, 29)
        ],
        "B4_U12_s23_vs_other_seeds": [
            run_by_case[_case_key(4, "U12", seed)]["result_class"] for seed in (17, 29)
        ],
        "U12_s23_across_blocks": {
            "B1": run_by_case[_case_key(1, "U12", 23)]["result_class"],
            "B2": run_by_case[_case_key(2, "U12", 23)]["result_class"],
            "B4": run_by_case[_case_key(4, "U12", 23)]["result_class"],
        },
    }
    training_dynamics = {
        case_id: _read_train_metrics(run_by_case[case_id])
        for case_id in FAILURE_CASE_IDS
    }
    aggregate_training_metric_observations = {
        "B1_seed23_shared_training_metrics": {
            case_id: {
                "result_class": run_by_case[case_id]["result_class"],
                "normalized_L_set_mean": run_by_case[case_id]["normalized_L_set_mean"],
                "slot_margin_min_final": run_by_case[case_id]["slot_margin_min"],
                "target_bucket_mass_min": run_by_case[case_id]["target_bucket_mass_min"],
                "checkpoint_selection_best_step": run_by_case[case_id]["checkpoint_selection_best_step"],
            }
            for case_id in ("B1_U00_s23", "B1_U03_s23", "B1_U12_s23", "B1_U15_s23")
        },
        "U12_seed23_across_blocks": {
            case_id: {
                "result_class": run_by_case[case_id]["result_class"],
                "normalized_L_set_mean": run_by_case[case_id]["normalized_L_set_mean"],
                "slot_margin_min_final": run_by_case[case_id]["slot_margin_min"],
                "target_bucket_mass_min": run_by_case[case_id]["target_bucket_mass_min"],
                "checkpoint_selection_best_step": run_by_case[case_id]["checkpoint_selection_best_step"],
            }
            for case_id in ("B1_U12_s23", "B2_U12_s23", "B4_U12_s23")
        },
    }
    missing_instrumentation = [
        "per-slot target_bucket_logmass at generation/evaluation time",
        "per-slot strongest wrong bucket id and logmass",
        "per-slot target bucket rank",
        "per-slot target token probability",
        "top-5 bucket logmasses per slot",
        "top-5 token probabilities per slot",
        "per-checkpoint verifier replay or saved generated_text per checkpoint",
        "explicit checkpoint step-to-epoch mapping in training_health.json",
        "per-slot L_margin curve; current training metrics only provide aggregate slot_margin_mean/min when raw train_metrics is accessible",
    ]
    root_cause_ranking = [
        {
            "rank": 1,
            "candidate": "seed-specific optimization instability",
            "evidence": "All four method failures occur at seed 23; the specified same-payload same-block neighbors at seeds 17 and 29 pass. The pattern is not global seed collapse because some seed-23 neighbors pass.",
            "confidence": "moderate",
        },
        {
            "rank": 2,
            "candidate": "payload/bucket hardness",
            "evidence": "Within B1 seed 23, U00 passes while U03/U12/U15 fail under the same aggregate training metrics. U12 fails in B1 and B4 at seed 23 but passes in B2.",
            "confidence": "moderate",
        },
        {
            "rank": 3,
            "candidate": "insufficient target-vs-wrong bucket margin",
            "evidence": "Failures are slot/symbol bucket substitutions, but target-vs-wrong logmass margins were not saved. Aggregate training margins are not decisive: B4_U12_s23 has high positive aggregate slot_margin_min_final, and B1_U00_s23 passes despite sharing the negative B1 seed-23 aggregate margin.",
            "confidence": "plausible_unmeasured",
        },
        {
            "rank": 4,
            "candidate": "checkpoint drift",
            "evidence": "No per-checkpoint eval or saved per-checkpoint generated text is available to determine whether an earlier checkpoint passed.",
            "confidence": "inconclusive",
        },
        {
            "rank": 5,
            "candidate": "verifier/RS logic bug",
            "evidence": "Contract hashes match, failures decompose to symbol errors with no erasures, and nearest neighbors pass under the same verifier.",
            "confidence": "unlikely",
        },
    ]
    return {
        "schema_name": "g3a_v2_failure_margin_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "failure_case_ids": list(FAILURE_CASE_IDS),
        "failure_count": len(FAILURE_CASE_IDS),
        "failure_counter_by_seed": dict(failure_counter_by_seed),
        "failure_counter_by_payload": dict(failure_counter_by_payload),
        "failure_counter_by_variant": dict(failure_counter_by_variant),
        "wrong_field_counter": dict(wrong_field_counter),
        "required_comparisons": especially,
        "neighbor_comparison_count": len(neighbor_rows),
        "logit_margin_fields_available": False,
        "training_dynamics": training_dynamics,
        "aggregate_training_metric_observations": aggregate_training_metric_observations,
        "instrumentation_required_for_v3": missing_instrumentation,
        "root_cause_ranking": root_cause_ranking,
        "primary_root_cause": "F. still inconclusive due to missing instrumentation",
    }


def _report(summary: dict[str, Any], neighbor_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# G3a-v2 Failure Root Cause",
        "",
        "This report analyzes the four valid G3a-v2 method failures without rerunning training or evaluation.",
        "",
        "## Accounting",
        "",
        f"- Failures: `{', '.join(summary['failure_case_ids'])}`",
        f"- Failure count by seed: `{summary['failure_counter_by_seed']}`",
        f"- Failure count by payload: `{summary['failure_counter_by_payload']}`",
        f"- Failure count by variant: `{summary['failure_counter_by_variant']}`",
        f"- Wrong fields among failed slots: `{summary['wrong_field_counter']}`",
        "",
        "## Neighbor Comparisons",
        "",
        "- B1 U03/U12/U15 seed 23 all pass at seeds 17 and 29 under the same block count and payload.",
        "- B4 U12 seed 23 passes at seeds 17 and 29 under the same block count and payload.",
        "- U12 seed 23 fails for B1 and B4 but passes for B2, so the pattern is not a pure payload-only failure.",
        f"- Total neighbor comparison rows written: `{len(neighbor_rows)}`.",
        "",
        "## Slot And Margin Evidence",
        "",
        "The committed slot diagnostics identify wrong decoded buckets and generated representatives, but do not include logits or bucket logmasses. Therefore the requested per-slot margin fields are recorded as `missing` in `results/tables/g3a_v2_failure_slot_margin.csv`.",
        "",
        "Available evidence shows valid semantic bucket substitutions rather than parser or RS failures: all four failures are valid completed runs with matching contracts, no erasures, and one symbol error each.",
        "",
        "## Training Dynamics",
        "",
        "The committed paper-facing artifacts include final aggregate training metrics, selected checkpoint metadata, and paths to raw training metrics. The raw `train_metrics.jsonl` files are not present in this local workspace. The diagnostic script records full curves if run on a machine where the scratch paths exist; otherwise it marks curves as `missing`.",
        "",
        "Aggregate training metrics are not sufficient to explain the failures. B1 seed-23 cases share the same aggregate training metrics: `B1_U00_s23` passes, while `B1_U03_s23`, `B1_U12_s23`, and `B1_U15_s23` fail. Conversely, `B4_U12_s23` fails despite high aggregate target-bucket mass and positive aggregate slot margin. This points to seed-specific payload/bucket interactions at evaluation slots, not a confirmed global loss or aggregate-margin failure.",
        "",
        "No per-checkpoint verifier replay or per-checkpoint generated text is available, so the current artifacts cannot determine whether an earlier checkpoint would have passed.",
        "",
        "## Root-Cause Ranking",
        "",
    ]
    for item in summary["root_cause_ranking"]:
        lines.append(f"{item['rank']}. {item['candidate']}: {item['evidence']} Confidence: `{item['confidence']}`.")
    lines.extend(
        [
            "",
            "## V3 Instrumentation Required",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["instrumentation_required_for_v3"])
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The strongest observed pattern is seed-specific optimization instability at seed 23, with possible insufficient target-vs-wrong bucket margin. However, the margin-level root cause cannot be confirmed because per-slot logits, bucket logmasses, ranks, and per-checkpoint verifier replay were not saved.",
            "",
            "F. still inconclusive due to missing instrumentation",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    run_rows = _read_csv(_resolve(repo_root, args.run_table))
    slot_rows = _read_csv(_resolve(repo_root, args.slot_table))
    _read_csv(_resolve(repo_root, args.symbol_table))
    run_by_case = {row["case_id"]: row for row in run_rows}
    missing = [case_id for case_id in FAILURE_CASE_IDS if case_id not in run_by_case]
    if missing:
        raise ValueError(f"Missing required failure case rows: {missing}")
    slot_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in slot_rows:
        slot_by_case[row["case_id"]].append(row)
    failure_rows = [run_by_case[case_id] for case_id in FAILURE_CASE_IDS]
    slot_margin_rows = _slot_margin_rows(failure_rows, run_rows, slot_by_case)
    neighbor_rows = _neighbor_comparison_rows(failure_rows, run_rows, slot_by_case)
    summary = _summary_payload(run_by_case, slot_by_case, neighbor_rows)

    _write_csv(_resolve(repo_root, args.slot_margin_out), slot_margin_rows, SLOT_MARGIN_FIELDS)
    _write_csv(_resolve(repo_root, args.neighbor_out), neighbor_rows, NEIGHBOR_FIELDS)
    _write_json(_resolve(repo_root, args.summary_out), summary)
    report_path = _resolve(repo_root, args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary, neighbor_rows), encoding="utf-8")
    print(f"wrote {args.slot_margin_out}")
    print(f"wrote {args.neighbor_out}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
