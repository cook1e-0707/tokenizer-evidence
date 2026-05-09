from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import write_csv, write_json


SCHEMA_NAME = "natural_evidence_qwen_846699_r1_selector_contract_analysis_v1"
PAIRWISE_FIELDS = [
    "payload_id",
    "seed",
    "match_policy",
    "query_budget",
    "protected_target_hit_rate",
    "raw_target_hit_rate",
    "task_only_target_hit_rate",
    "protected_minus_raw_target_hit_rate",
    "protected_minus_task_only_target_hit_rate",
    "protected_compatible_hit_rate",
    "raw_compatible_hit_rate",
    "task_only_compatible_hit_rate",
    "protected_minus_raw_compatible_hit_rate",
    "protected_minus_task_only_compatible_hit_rate",
    "protected_prefix_match_rate",
    "raw_prefix_match_rate",
    "task_only_prefix_match_rate",
    "protected_target_coordinate_count",
    "raw_target_coordinate_count",
    "task_only_target_coordinate_count",
    "protected_max_target_slots_per_frame",
    "raw_max_target_slots_per_frame",
    "task_only_max_target_slots_per_frame",
    "decision",
]
SUMMARY_FIELDS = [
    "match_policy",
    "query_budget",
    "comparison_rows",
    "mean_protected_target_hit_rate",
    "mean_raw_target_hit_rate",
    "mean_task_only_target_hit_rate",
    "mean_protected_minus_raw_target_hit_rate",
    "mean_protected_minus_task_only_target_hit_rate",
    "positive_vs_raw_rows",
    "positive_vs_task_only_rows",
    "max_protected_target_hit_rate",
    "min_raw_target_hit_rate",
    "max_task_only_target_hit_rate",
    "selector_contract_status",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only R1 interpretation and selector-contract repair "
            "analysis. Reads the R1 replay outputs and reports protected-vs-raw "
            "and protected-vs-task-only coordinate lift without training, "
            "generation, E2E rerun, payload recovery, or FAR claims."
        )
    )
    parser.add_argument("--r1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--by-condition-csv", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float:
    try:
        if value is None or str(value) == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        if value is None or str(value) == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _index_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str, str, int], Mapping[str, str]]:
    index: dict[tuple[str, str, str, str, str, int], Mapping[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("model_condition", "")),
            str(row.get("payload_id", "")),
            str(row.get("seed", "")),
            str(row.get("expected_payload_id", "")),
            str(row.get("match_policy", "")),
            _as_int(row.get("query_budget", 0)),
        )
        index[key] = row
    return index


def _rate(row: Mapping[str, str] | None, field: str) -> float:
    return _as_float(row.get(field, "")) if row is not None else 0.0


def _count(row: Mapping[str, str] | None, field: str) -> int:
    return _as_int(row.get(field, "")) if row is not None else 0


def build_pairwise_lift(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    index = _index_rows(rows)
    output: list[dict[str, Any]] = []
    payload_ids = sorted(
        {
            str(row.get("expected_payload_id", ""))
            for row in rows
            if str(row.get("expected_payload_id", "")) not in {"", "ALL"}
        }
    )
    seeds = sorted(
        {
            str(row.get("seed", ""))
            for row in rows
            if str(row.get("model_condition", "")) == "protected_trained" and str(row.get("seed", ""))
        }
    )
    policies = sorted({str(row.get("match_policy", "")) for row in rows if str(row.get("match_policy", ""))})
    budgets = sorted({_as_int(row.get("query_budget", 0)) for row in rows if _as_int(row.get("query_budget", 0)) > 0})
    for payload_id in payload_ids:
        for seed in seeds:
            for policy in policies:
                for budget in budgets:
                    protected = index.get(("protected_trained", payload_id, seed, payload_id, policy, budget))
                    task_only = index.get(("task_only_lora", payload_id, seed, payload_id, policy, budget))
                    raw = index.get(("raw", "", "", payload_id, policy, budget))
                    if protected is None or task_only is None or raw is None:
                        continue
                    protected_target = _rate(protected, "target_hit_rate")
                    raw_target = _rate(raw, "target_hit_rate")
                    task_target = _rate(task_only, "target_hit_rate")
                    protected_compatible = _rate(protected, "compatible_hit_rate")
                    raw_compatible = _rate(raw, "compatible_hit_rate")
                    task_compatible = _rate(task_only, "compatible_hit_rate")
                    delta_raw = protected_target - raw_target
                    delta_task = protected_target - task_target
                    if delta_raw > 0.0 and delta_task > 0.0:
                        decision = "PASS_COORDINATE_LIFT"
                    elif delta_raw <= 0.0 and delta_task <= 0.0:
                        decision = "FAIL_BELOW_RAW_AND_TASK_ONLY"
                    elif delta_raw <= 0.0:
                        decision = "FAIL_BELOW_RAW"
                    else:
                        decision = "FAIL_BELOW_TASK_ONLY"
                    output.append(
                        {
                            "payload_id": payload_id,
                            "seed": seed,
                            "match_policy": policy,
                            "query_budget": budget,
                            "protected_target_hit_rate": protected_target,
                            "raw_target_hit_rate": raw_target,
                            "task_only_target_hit_rate": task_target,
                            "protected_minus_raw_target_hit_rate": delta_raw,
                            "protected_minus_task_only_target_hit_rate": delta_task,
                            "protected_compatible_hit_rate": protected_compatible,
                            "raw_compatible_hit_rate": raw_compatible,
                            "task_only_compatible_hit_rate": task_compatible,
                            "protected_minus_raw_compatible_hit_rate": protected_compatible - raw_compatible,
                            "protected_minus_task_only_compatible_hit_rate": protected_compatible - task_compatible,
                            "protected_prefix_match_rate": _rate(protected, "prefix_match_rate"),
                            "raw_prefix_match_rate": _rate(raw, "prefix_match_rate"),
                            "task_only_prefix_match_rate": _rate(task_only, "prefix_match_rate"),
                            "protected_target_coordinate_count": _count(protected, "target_coordinate_count"),
                            "raw_target_coordinate_count": _count(raw, "target_coordinate_count"),
                            "task_only_target_coordinate_count": _count(task_only, "target_coordinate_count"),
                            "protected_max_target_slots_per_frame": _count(protected, "max_target_slots_per_frame"),
                            "raw_max_target_slots_per_frame": _count(raw, "max_target_slots_per_frame"),
                            "task_only_max_target_slots_per_frame": _count(task_only, "max_target_slots_per_frame"),
                            "decision": decision,
                        }
                    )
    return output


def summarize_pairwise(pairwise_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        grouped[(str(row["match_policy"]), int(row["query_budget"]))].append(row)
    output: list[dict[str, Any]] = []
    for (policy, budget), rows in sorted(grouped.items()):
        mean_delta_raw = mean(float(row["protected_minus_raw_target_hit_rate"]) for row in rows)
        mean_delta_task = mean(float(row["protected_minus_task_only_target_hit_rate"]) for row in rows)
        positive_vs_raw = sum(1 for row in rows if float(row["protected_minus_raw_target_hit_rate"]) > 0.0)
        positive_vs_task = sum(1 for row in rows if float(row["protected_minus_task_only_target_hit_rate"]) > 0.0)
        if positive_vs_raw == len(rows) and positive_vs_task == len(rows):
            status = "PASS_ALL_SLICES_HAVE_PROTECTED_LIFT"
        elif positive_vs_raw == 0 and positive_vs_task == 0:
            status = "FAIL_NO_SLICE_HAS_PROTECTED_LIFT"
        else:
            status = "FAIL_MIXED_OR_INSUFFICIENT_PROTECTED_LIFT"
        output.append(
            {
                "match_policy": policy,
                "query_budget": budget,
                "comparison_rows": len(rows),
                "mean_protected_target_hit_rate": mean(float(row["protected_target_hit_rate"]) for row in rows),
                "mean_raw_target_hit_rate": mean(float(row["raw_target_hit_rate"]) for row in rows),
                "mean_task_only_target_hit_rate": mean(float(row["task_only_target_hit_rate"]) for row in rows),
                "mean_protected_minus_raw_target_hit_rate": mean_delta_raw,
                "mean_protected_minus_task_only_target_hit_rate": mean_delta_task,
                "positive_vs_raw_rows": positive_vs_raw,
                "positive_vs_task_only_rows": positive_vs_task,
                "max_protected_target_hit_rate": max(float(row["protected_target_hit_rate"]) for row in rows),
                "min_raw_target_hit_rate": min(float(row["raw_target_hit_rate"]) for row in rows),
                "max_task_only_target_hit_rate": max(float(row["task_only_target_hit_rate"]) for row in rows),
                "selector_contract_status": status,
            }
        )
    return output


def _budget512_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in summary_rows if int(row["query_budget"]) == 512]


def _write_markdown(path: Path, summary: Mapping[str, Any], by_policy: Sequence[Mapping[str, Any]]) -> None:
    rows512 = _budget512_rows(by_policy)
    lines = [
        "# R1 selector-contract repair analysis",
        "",
        "This is an artifact-only interpretation of the Phase R1 prefix-conditioned selector replay. It is not payload recovery, not FAR, and not a paper-facing positive claim.",
        "",
        "## Decision",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "The prefix-conditioned replay confirms that strict token-index anchoring was discarding many observable prefix-conditioned events. It does not produce an ownership-specific signal because protected rows do not beat raw or task-only null behavior.",
        "",
        "## Budget 512 lift summary",
        "",
        "| Policy | mean protected target | mean raw target | mean task-only target | protected-raw delta | protected-task delta | positive vs raw | positive vs task-only | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows512:
        lines.append(
            "| {policy} | {pt:.6f} | {rt:.6f} | {tt:.6f} | {dr:.6f} | {dt:.6f} | {pr}/{n} | {ptask}/{n} | {status} |".format(
                policy=row["match_policy"],
                pt=float(row["mean_protected_target_hit_rate"]),
                rt=float(row["mean_raw_target_hit_rate"]),
                tt=float(row["mean_task_only_target_hit_rate"]),
                dr=float(row["mean_protected_minus_raw_target_hit_rate"]),
                dt=float(row["mean_protected_minus_task_only_target_hit_rate"]),
                pr=int(row["positive_vs_raw_rows"]),
                ptask=int(row["positive_vs_task_only_rows"]),
                n=int(row["comparison_rows"]),
                status=row["selector_contract_status"],
            )
        )
    lines.extend(
        [
            "",
            "## Repair implications",
            "",
            "- Do not use R1 target-hit counts as verifier accepts.",
            "- Do not select match policy, payload, seed, or threshold post hoc from this replay.",
            "- Freeze any future selector contract before generation, including match policy, reference tokenizer/model, bucket policy, thresholds, query budget, prompt split, key, and payload.",
            "- Repair training targets before any new Qwen run: current protected outputs do not preserve or create the prefix-conditioned target event distribution better than nulls.",
            "- Treat branch-aware or regenerated-suffix training data as a required preflight candidate, not an optional cleanup.",
            "- Keep sparse coordinate-level coding as the likely decoder-side repair after coordinate survival shows owner-specific lift.",
            "",
            "## Next allowed action",
            "",
            "Artifact-only selector-contract precommit design and branch-aware/regenerated-suffix training-target preflight. No training, generation, E2E rerun, Llama, same-family null, sanitizer, or manuscript positive claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    *,
    r1_dir: Path,
    output_dir: Path,
    summary_json: Path | None = None,
    by_condition_csv: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    summary_path = summary_json or r1_dir / "prefix_conditioned_selector_replay_summary.json"
    by_condition_path = by_condition_csv or r1_dir / "prefix_conditioned_selector_replay_by_condition.csv"
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing R1 summary: {summary_path}")
    if not by_condition_path.is_file() or by_condition_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing R1 by-condition CSV: {by_condition_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary_path = output_dir / "r1_selector_contract_repair_summary.json"
    if output_summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing analysis: {output_summary_path}")

    r1_summary = _read_json(summary_path)
    by_condition_rows = _read_csv(by_condition_path)
    pairwise_rows = build_pairwise_lift(by_condition_rows)
    by_policy_rows = summarize_pairwise(pairwise_rows)
    all_policy_statuses = {str(row["selector_contract_status"]) for row in by_policy_rows}
    positive_vs_raw_total = sum(int(row["positive_vs_raw_rows"]) for row in by_policy_rows)
    positive_vs_task_total = sum(int(row["positive_vs_task_only_rows"]) for row in by_policy_rows)
    if positive_vs_raw_total == 0 and positive_vs_task_total == 0:
        status = "COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS"
    elif "PASS_ALL_SLICES_HAVE_PROTECTED_LIFT" in all_policy_statuses:
        status = "COMPLETE_R1_ANALYSIS_MIXED_REQUIRES_LOCKBOX_REPLAY"
    else:
        status = "COMPLETE_R1_ANALYSIS_INSUFFICIENT_PROTECTED_LIFT"
    output = {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "r1_selector_contract_analysis_not_payload_recovery_not_far",
        },
        "inputs": {
            "r1_summary_json": str(summary_path),
            "r1_by_condition_csv": str(by_condition_path),
            "r1_status": r1_summary.get("status", ""),
            "r1_claim_control": r1_summary.get("claim_control", {}),
        },
        "comparison_rows": len(pairwise_rows),
        "policy_budget_rows": len(by_policy_rows),
        "positive_vs_raw_rows_total": positive_vs_raw_total,
        "positive_vs_task_only_rows_total": positive_vs_task_total,
        "budget512_summary": _budget512_rows(by_policy_rows),
        "selector_contract_decision": {
            "direct_replay_verifier_allowed": False,
            "training_allowed": False,
            "e2e_rerun_allowed": False,
            "reason": (
                "R1 rediscovered prefix-conditioned coordinates, but protected "
                "target-hit rates do not exceed raw/task-only null behavior."
            ),
        },
        "next_allowed_action": (
            "Artifact-only selector-contract precommit design and branch-aware/"
            "regenerated-suffix training-target preflight; no training or E2E."
        ),
        "forbidden_claims_remain": [
            "natural-output success",
            "payload recovery",
            "full FAR",
            "cross-family generality",
            "robustness",
            "sanitizer resistance",
            "superiority over Scalable/Perinucleus",
            "24,576 fingerprints",
        ],
    }
    write_json(output_summary_path, output)
    write_csv(output_dir / "r1_selector_contract_pairwise_lift.csv", pairwise_rows, PAIRWISE_FIELDS)
    write_csv(output_dir / "r1_selector_contract_by_policy_budget.csv", by_policy_rows, SUMMARY_FIELDS)
    _write_markdown(output_dir / "r1_selector_contract_repair_analysis.md", output, by_policy_rows)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        r1_dir=_resolve(args.r1_dir),
        output_dir=_resolve(args.output_dir),
        summary_json=_resolve(args.summary_json) if args.summary_json else None,
        by_condition_csv=_resolve(args.by_condition_csv) if args.by_condition_csv else None,
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
