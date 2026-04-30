from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and plan the matched FAR/utility/compute comparison. "
            "This runner is intentionally non-executing until method-specific "
            "FAR and utility backends are reviewed."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writing outputs.")
    parser.add_argument("--write-plan", action="store_true", help="Write plan-only output files from the config.")
    parser.add_argument("--execute", action="store_true", help="Reserved for reviewed GPU execution; currently blocked.")
    parser.add_argument("--force", action="store_true", help="Overwrite plan-only outputs.")
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite plan output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite plan output")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status_kind(status: str | None) -> str:
    if not status:
        return "missing"
    if status == "complete" or status.startswith("complete_"):
        return "complete"
    if status.startswith("partial"):
        return "partial"
    if status == "missing":
        return "missing"
    return status


def _source_artifact_rows(repo_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, raw_path in (cfg.get("source_artifacts") or {}).items():
        path = _resolve(repo_root, raw_path)
        rows.append(
            {
                "artifact_key": key,
                "path": str(raw_path),
                "resolved_path": str(path) if path else "",
                "exists": bool(path and path.exists()),
            }
        )
    return rows


def _audit_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    evidence = cfg.get("existing_evidence") or {}

    def ev_status(key: str) -> str:
        value = evidence.get(key) or {}
        return str(value.get("status", "missing"))

    utility_status = ev_status("utility")
    compute_status = ev_status("compute")
    query_status = ev_status("query_budget_grid")
    clean_status = ev_status("clean_success")
    far_status = ev_status("far_null_calibration")

    rows = [
        {
            "requirement_id": "clean_success",
            "requirement": "clean positive ownership success",
            "status": clean_status,
            "evidence": json.dumps(evidence.get("clean_success", {}), sort_keys=True),
            "missing_evidence": "" if _status_kind(clean_status) == "complete" else "positive split success counts",
            "next_action": "reuse existing clean artifact" if _status_kind(clean_status) == "complete" else "run clean final positive split",
        },
        {
            "requirement_id": "query_budgets",
            "requirement": "query budgets M=1,3,5,10 for both positive and negative decisions",
            "status": "incomplete_for_far" if query_status == "complete_for_clean_success_only" else query_status,
            "evidence": json.dumps(evidence.get("query_budget_grid", {}), sort_keys=True),
            "missing_evidence": "negative/null outcomes per query budget" if query_status != "missing" else "query-budget grid",
            "next_action": "run matched FAR/null grid with frozen thresholds",
        },
        {
            "requirement_id": "far_null_models",
            "requirement": "FAR under required null models",
            "status": far_status,
            "evidence": json.dumps(evidence.get("far_null_calibration", {}), sort_keys=True),
            "missing_evidence": "base-Qwen null and optional non-Qwen/unprotected nulls",
            "next_action": "execute FAR verifier backend after review",
        },
        {
            "requirement_id": "far_wrong_payload_owner",
            "requirement": "FAR under wrong payload, wrong owner, non-owner probes, and organic prompts",
            "status": far_status,
            "evidence": json.dumps(evidence.get("far_null_calibration", {}), sort_keys=True),
            "missing_evidence": "per-null-set false accept counts and Wilson intervals",
            "next_action": "execute FAR verifier backend after review",
        },
        {
            "requirement_id": "utility_score",
            "requirement": "utility score on the matched TinyBench benchmark",
            "status": utility_status,
            "evidence": json.dumps(evidence.get("utility", {}), sort_keys=True),
            "missing_evidence": "" if _status_kind(utility_status) == "complete" else "TinyBench utility for this method",
            "next_action": "reuse existing TinyBench only if evaluator matches; otherwise run utility backend",
        },
        {
            "requirement_id": "utility_drop",
            "requirement": "utility drop versus base model on the same benchmark",
            "status": utility_status,
            "evidence": json.dumps(evidence.get("utility", {}), sort_keys=True),
            "missing_evidence": "" if _status_kind(utility_status) == "complete" else "base-vs-method TinyBench paired comparison",
            "next_action": "run paired utility benchmark",
        },
        {
            "requirement_id": "train_wall_clock",
            "requirement": "training wall-clock seconds under matched accounting",
            "status": compute_status,
            "evidence": json.dumps(evidence.get("compute", {}), sort_keys=True),
            "missing_evidence": "normalized train wall-clock" if _status_kind(compute_status) != "complete" else "",
            "next_action": "extract actual run seconds or mark requested-only fallback",
        },
        {
            "requirement_id": "gpu_hours",
            "requirement": "requested and observed GPU-hours",
            "status": compute_status,
            "evidence": json.dumps(evidence.get("compute", {}), sort_keys=True),
            "missing_evidence": "observed GPU seconds and normalized requested GPU-hours",
            "next_action": "normalize compute accounting across methods",
        },
        {
            "requirement_id": "trainable_parameters",
            "requirement": "trainable parameter count",
            "status": "missing" if cfg.get("method_id") == "ours_compiled_ownership" else "partial",
            "evidence": json.dumps(evidence.get("compute", {}), sort_keys=True),
            "missing_evidence": "paper-row trainable parameter count",
            "next_action": "extract adapter/train summary metadata",
        },
        {
            "requirement_id": "training_examples",
            "requirement": "number of training examples",
            "status": "partial",
            "evidence": json.dumps(cfg.get("method_contract", {}), sort_keys=True),
            "missing_evidence": "normalized numeric field in matched table",
            "next_action": "derive from method contract and case summaries",
        },
        {
            "requirement_id": "eval_query_cost",
            "requirement": "evaluation query cost per budget and method",
            "status": "missing" if query_status == "missing" else "partial",
            "evidence": json.dumps(evidence.get("query_budget_grid", {}), sort_keys=True),
            "missing_evidence": "query counts for negative/null grids",
            "next_action": "record model forwards during FAR execution",
        },
        {
            "requirement_id": "confidence_intervals",
            "requirement": "confidence intervals for FAR, utility, and clean success",
            "status": "partial",
            "evidence": json.dumps(evidence.get("confidence_interval", {}), sort_keys=True),
            "missing_evidence": "FAR and utility CIs",
            "next_action": "compute Wilson intervals for FAR and paired bootstrap/CI for utility",
        },
    ]
    return rows


def _planned_far_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    method_id = str(cfg.get("method_id", "unknown"))
    budgets = cfg.get("query_budget_protocol", {}).get("budgets", [])
    far_protocol = cfg.get("far_protocol") or {}
    null_models = far_protocol.get("null_models") or []
    null_probe_sets = far_protocol.get("null_probe_sets") or []
    rows: list[dict[str, Any]] = []
    for split_name in ["calibration_split", "final_split"]:
        split = far_protocol.get(split_name) or {}
        payloads = split.get("payloads") or []
        seeds = split.get("seeds")
        if seeds is None:
            seeds = [split.get("seed", "")]
        if not isinstance(seeds, list):
            seeds = [seeds]
        planned_positive_cases = len(payloads) * max(1, len([seed for seed in seeds if seed != ""]))
        for budget in budgets:
            for null_model in null_models:
                for null_probe_set in null_probe_sets:
                    rows.append(
                        {
                            "method_id": method_id,
                            "split": split_name,
                            "query_budget": budget,
                            "null_model_id": null_model.get("id", ""),
                            "null_model": null_model.get("model", ""),
                            "null_model_status": null_model.get("status", ""),
                            "null_probe_set": null_probe_set,
                            "planned_positive_cases": planned_positive_cases,
                            "status": "planned_not_run",
                            "false_accept_count": "",
                            "negative_count": "",
                            "observed_far": "",
                            "wilson_low": "",
                            "wilson_high": "",
                        }
                    )
    return rows


def _utility_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    method_id = str(cfg.get("method_id", "unknown"))
    utility = (cfg.get("existing_evidence") or {}).get("utility") or {}
    protocol = cfg.get("utility_protocol") or {}
    benchmark = protocol.get("benchmark", "")
    status = str(utility.get("status", "missing"))
    if _status_kind(status) == "complete":
        return [
            {
                "method_id": method_id,
                "benchmark": benchmark,
                "row_kind": "existing_method_utility",
                "status": status,
                "base_total_accuracy": utility.get("base_total_accuracy", ""),
                "method_total_accuracy": utility.get("adapter_total_accuracy", ""),
                "absolute_drop": utility.get("signed_absolute_drop", ""),
                "ci95_low": "",
                "ci95_high": "",
                "source": "existing_evidence.utility",
            }
        ]
    return [
        {
            "method_id": method_id,
            "benchmark": benchmark,
            "row_kind": "planned_base_utility",
            "status": "planned_not_run",
            "base_total_accuracy": "",
            "method_total_accuracy": "",
            "absolute_drop": "",
            "ci95_low": "",
            "ci95_high": "",
            "source": "",
        },
        {
            "method_id": method_id,
            "benchmark": benchmark,
            "row_kind": "planned_method_utility",
            "status": "planned_not_run",
            "base_total_accuracy": "",
            "method_total_accuracy": "",
            "absolute_drop": "",
            "ci95_low": "",
            "ci95_high": "",
            "source": "",
        },
    ]


def _compute_payload(cfg: dict[str, Any], source_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_name": "matched_far_utility_compute_method_compute_plan",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "method_id": cfg.get("method_id"),
        "status": "plan_only_not_executed",
        "existing_compute_evidence": (cfg.get("existing_evidence") or {}).get("compute", {}),
        "requested_resources": (cfg.get("compute_protocol") or {}).get("requested_resources", {}),
        "required_report_fields": (cfg.get("compute_protocol") or {}).get("report", []),
        "source_artifacts": source_artifacts,
        "note": "This file is a plan/audit artifact. It is not a completed compute comparison.",
    }


def _build_plan(repo_root: Path, config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    source_artifacts = _source_artifact_rows(repo_root, cfg)
    audit_rows = _audit_rows(cfg)
    incomplete = [
        row["requirement_id"]
        for row in audit_rows
        if _status_kind(row["status"]) != "complete"
    ]
    outputs = cfg.get("repo_outputs") or {}
    return {
        "schema_name": "matched_far_utility_compute_method_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "method_id": cfg.get("method_id"),
        "display_name": cfg.get("display_name"),
        "status": "plan_only_not_executed",
        "run_required": bool((cfg.get("decision") or {}).get("run_required", True)),
        "execute_supported": False,
        "execute_blocker": (
            "Method-specific FAR/null verifier and paired utility execution backends are not "
            "implemented in this runner yet. Use --write-plan for review only."
        ),
        "source_artifacts": source_artifacts,
        "audit_rows": audit_rows,
        "missing_or_partial_requirements": incomplete,
        "planned_far_case_count": len(_planned_far_rows(cfg)),
        "planned_utility_rows": len(_utility_rows(cfg)),
        "outputs": outputs,
        "launch_guard": (cfg.get("decision") or {}).get("launch_guard", ""),
    }


def _write_plan_outputs(repo_root: Path, cfg: dict[str, Any], plan: dict[str, Any], *, force: bool) -> dict[str, str]:
    outputs = cfg.get("repo_outputs") or {}
    summary_path = _resolve(repo_root, outputs.get("summary"))
    far_path = _resolve(repo_root, outputs.get("far_cases"))
    utility_path = _resolve(repo_root, outputs.get("utility"))
    compute_path = _resolve(repo_root, outputs.get("compute"))
    if not all([summary_path, far_path, utility_path, compute_path]):
        raise ValueError("Config repo_outputs must define summary, far_cases, utility, and compute paths.")

    _write_json(summary_path, plan, force=force)
    _write_csv(far_path, _planned_far_rows(cfg), force=force)
    _write_csv(utility_path, _utility_rows(cfg), force=force)
    _write_json(
        compute_path,
        _compute_payload(cfg, plan.get("source_artifacts", [])),
        force=force,
    )
    return {
        "summary": str(summary_path),
        "far_cases": str(far_path),
        "utility": str(utility_path),
        "compute": str(compute_path),
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    config_path = _resolve(repo_root, args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = _load_yaml(config_path)
    if cfg.get("schema_name") != "matched_far_utility_compute_method_plan":
        raise ValueError(f"Unexpected config schema_name: {cfg.get('schema_name')}")

    plan = _build_plan(repo_root, config_path, cfg)

    if args.execute:
        raise RuntimeError(
            "EXECUTE_BACKEND_NOT_IMPLEMENTED: this runner currently supports dry-run "
            "and plan-only artifacts only. Do not submit matched FAR/utility GPU jobs "
            "until method-specific verifier and TinyBench execution backends are reviewed."
        )

    if args.write_plan:
        paths = _write_plan_outputs(repo_root, cfg, plan, force=args.force)
        print(json.dumps({"status": "plan_written", "paths": paths, "missing_or_partial_requirements": plan["missing_or_partial_requirements"]}, indent=2))
        return 0

    # Default to dry-run behavior to make accidental local execution safe.
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
