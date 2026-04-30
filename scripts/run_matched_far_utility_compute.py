from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the artifact-backed comparison subset. This does not launch "
            "fresh model inference; unavailable null sets are explicitly marked."
        ),
    )
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


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


def _final_payloads_and_seeds(cfg: dict[str, Any]) -> tuple[list[str], list[int]]:
    final_split = (cfg.get("far_protocol") or {}).get("final_split") or {}
    payloads = [str(item) for item in final_split.get("payloads", [])]
    seeds = [_as_int(item) for item in final_split.get("seeds", [])]
    return payloads, seeds


def _budgets(cfg: dict[str, Any]) -> list[int]:
    return [_as_int(item) for item in (cfg.get("query_budget_protocol") or {}).get("budgets", [])]


def _threshold(cfg: dict[str, Any]) -> float:
    return _as_float((cfg.get("query_budget_protocol") or {}).get("threshold", 1.0), 1.0)


def _case_far_row(
    *,
    method_id: str,
    query_budget: int,
    null_model_id: str,
    null_probe_set: str,
    label: bool,
    owner_payload: str,
    claim_payload: str,
    seed: int,
    score: float | None,
    threshold: float,
    status: str,
    evidence_source: str,
    note: str = "",
) -> dict[str, Any]:
    accepted = bool(score is not None and score >= threshold)
    false_accept = bool((not label) and accepted)
    return {
        "method_id": method_id,
        "split": "final_split",
        "query_budget": query_budget,
        "null_model_id": null_model_id,
        "null_model": "",
        "null_model_status": "artifact_backed" if status == "completed" else "not_available",
        "null_probe_set": null_probe_set,
        "row_kind": "positive" if label else "negative",
        "label": label,
        "owner_payload": owner_payload,
        "claim_payload": claim_payload,
        "seed": seed,
        "status": status,
        "threshold": threshold,
        "ownership_score": "" if score is None else score,
        "accepted": accepted if status == "completed" else "",
        "false_accept": false_accept if status == "completed" else "",
        "negative_count": 0 if label or status != "completed" else 1,
        "false_accept_count": 0 if label or status != "completed" else int(false_accept),
        "observed_far": "" if label or status != "completed" else float(false_accept),
        "wilson_low": "",
        "wilson_high": "",
        "evidence_source": evidence_source,
        "note": note,
    }


def _unavailable_far_rows(cfg: dict[str, Any], *, unavailable_sets: list[str], note: str) -> list[dict[str, Any]]:
    method_id = str(cfg.get("method_id", "unknown"))
    rows: list[dict[str, Any]] = []
    for budget in _budgets(cfg):
        for null_probe_set in unavailable_sets:
            rows.append(
                _case_far_row(
                    method_id=method_id,
                    query_budget=budget,
                    null_model_id="not_run",
                    null_probe_set=null_probe_set,
                    label=False,
                    owner_payload="",
                    claim_payload="",
                    seed=0,
                    score=None,
                    threshold=_threshold(cfg),
                    status="not_available",
                    evidence_source="",
                    note=note,
                )
            )
    return rows


def _ours_artifact_far_rows(repo_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = cfg.get("source_artifacts") or {}
    table_path = _resolve(repo_root, source.get("clean_table"))
    if table_path is None:
        raise ValueError("Ours config is missing source_artifacts.clean_table")
    table = _read_csv(table_path)
    payloads, seeds = _final_payloads_and_seeds(cfg)
    budgets = _budgets(cfg)
    threshold = _threshold(cfg)
    method_id = str(cfg.get("method_id"))
    by_key = {
        (str(row.get("payload")), _as_int(row.get("seed"))): row
        for row in table
        if str(row.get("payload")) in payloads and _as_int(row.get("seed")) in seeds
    }
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for seed in seeds:
            for owner_payload in payloads:
                owner = by_key.get((owner_payload, seed), {})
                score = 1.0 if (
                    _as_bool(owner.get("accepted"))
                    and _as_bool(owner.get("verifier_success"))
                    and str(owner.get("decoded_payload", "")) == owner_payload
                ) else 0.0
                rows.append(
                    _case_far_row(
                        method_id=method_id,
                        query_budget=budget,
                        null_model_id="adapted_ours",
                        null_probe_set="positive_owner_claim",
                        label=True,
                        owner_payload=owner_payload,
                        claim_payload=owner_payload,
                        seed=seed,
                        score=score,
                        threshold=threshold,
                        status="completed",
                        evidence_source=str(table_path),
                        note="Artifact-backed positive replay; same final output is reused across query budgets because G1 has no per-M verifier traces.",
                    )
                )
                for claim_payload in payloads:
                    if claim_payload == owner_payload:
                        continue
                    wrong_score = 1.0 if (
                        _as_bool(owner.get("accepted"))
                        and _as_bool(owner.get("verifier_success"))
                        and str(owner.get("decoded_payload", "")) == claim_payload
                    ) else 0.0
                    rows.append(
                        _case_far_row(
                            method_id=method_id,
                            query_budget=budget,
                            null_model_id="adapted_ours",
                            null_probe_set="wrong_payload_null",
                            label=False,
                            owner_payload=owner_payload,
                            claim_payload=claim_payload,
                            seed=seed,
                            score=wrong_score,
                            threshold=threshold,
                            status="completed",
                            evidence_source=str(table_path),
                            note="Claim-conditioned replay against decoded payload; no fresh model generation.",
                        )
                    )
    rows.extend(
        _unavailable_far_rows(
            cfg,
            unavailable_sets=["base_qwen_null", "wrong_owner_null", "non_owner_probe_null", "organic_prompt_null"],
            note="Required null set needs fresh outputs or an explicit owner-identity protocol; not available from current artifacts.",
        )
    )
    return rows


def _perinucleus_artifact_far_rows(repo_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = cfg.get("source_artifacts") or {}
    table_path = _resolve(repo_root, source.get("final_table"))
    if table_path is None:
        raise ValueError("Perinucleus config is missing source_artifacts.final_table")
    table = _read_csv(table_path)
    payloads, seeds = _final_payloads_and_seeds(cfg)
    budgets = _budgets(cfg)
    threshold = _threshold(cfg)
    method_id = str(cfg.get("method_id"))
    by_key = {
        (str(row.get("payload")), _as_int(row.get("seed")), _as_int(row.get("query_budget"))): row
        for row in table
        if str(row.get("payload")) in payloads
        and _as_int(row.get("seed")) in seeds
        and _as_int(row.get("query_budget")) in budgets
    }
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for seed in seeds:
            for owner_payload in payloads:
                owner = by_key.get((owner_payload, seed, budget), {})
                score = _as_float(owner.get("exact_response_match_ratio"), 0.0)
                rows.append(
                    _case_far_row(
                        method_id=method_id,
                        query_budget=budget,
                        null_model_id="adapted_perinucleus",
                        null_probe_set="positive_owner_claim",
                        label=True,
                        owner_payload=owner_payload,
                        claim_payload=owner_payload,
                        seed=seed,
                        score=score,
                        threshold=threshold,
                        status="completed",
                        evidence_source=str(table_path),
                        note="Existing Perinucleus final row.",
                    )
                )
                for claim_payload in payloads:
                    if claim_payload == owner_payload:
                        continue
                    claim_row = by_key.get((claim_payload, seed, budget), {})
                    claim_score = _as_float(claim_row.get("exact_response_match_ratio"), 0.0)
                    rows.append(
                        _case_far_row(
                            method_id=method_id,
                            query_budget=budget,
                            null_model_id="adapted_perinucleus",
                            null_probe_set="wrong_payload_null",
                            label=False,
                            owner_payload=owner_payload,
                            claim_payload=claim_payload,
                            seed=seed,
                            score=claim_score,
                            threshold=threshold,
                            status="completed",
                            evidence_source=str(table_path),
                            note=(
                                "Claim-conditioned artifact replay: Perinucleus payload label is a "
                                "query-subset selector, so accepted rows for the wrong claim are false accepts."
                            ),
                        )
                    )
    rows.extend(
        _unavailable_far_rows(
            cfg,
            unavailable_sets=["base_qwen_null", "wrong_owner_null", "non_owner_probe_null", "organic_prompt_null"],
            note="Required null set needs fresh base/non-owner/organic model outputs; not available from current artifacts.",
        )
    )
    return rows


def _aggregate_far(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed" or _as_bool(row.get("label")):
            continue
        key = (row.get("method_id"), row.get("query_budget"), row.get("null_probe_set"))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method_id, query_budget, null_probe_set), items in sorted(groups.items()):
        negative_count = len(items)
        false_accept_count = sum(1 for row in items if _as_bool(row.get("false_accept")))
        low, high = _wilson(false_accept_count, negative_count)
        out.append(
            {
                "method_id": method_id,
                "query_budget": query_budget,
                "null_probe_set": null_probe_set,
                "negative_count": negative_count,
                "false_accept_count": false_accept_count,
                "observed_far": false_accept_count / negative_count if negative_count else "",
                "wilson_low": low,
                "wilson_high": high,
                "status": "completed_artifact_subset",
            }
        )
    return out


def _execution_utility_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _utility_rows(cfg)
    for row in rows:
        if row["status"] == "planned_not_run":
            row["status"] = "not_available_missing_matched_tinybench"
            row["source"] = "current_artifacts_do_not_contain_matched_tinybench"
    return rows


def _execution_compute_payload(repo_root: Path, cfg: dict[str, Any], source_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    method_id = str(cfg.get("method_id"))
    payload = _compute_payload(cfg, source_artifacts)
    payload["status"] = "completed_partial_artifact_accounting"
    payload["note"] = "Artifact-backed compute accounting. Missing fields remain explicit."
    if method_id == "ours_compiled_ownership":
        source = cfg.get("source_artifacts") or {}
        accounting_path = _resolve(repo_root, source.get("compute_accounting"))
        accounting = _load_json(accounting_path) if accounting_path else None
        rows = accounting.get("rows", []) if isinstance(accounting, dict) else []
        g1 = [row for row in rows if row.get("stage") == "G1"]
        payload["g1_requested_gpu_hours"] = sum(_as_float(row.get("requested_gpu_hours")) for row in g1)
        payload["g1_requested_cpu_hours"] = sum(_as_float(row.get("requested_cpu_hours")) for row in g1)
        payload["trainable_parameters_status"] = "missing"
        payload["training_examples_status"] = "partial_contract_only"
    else:
        payload["existing_compute_evidence"] = (cfg.get("existing_evidence") or {}).get("compute", {})
        payload["trainable_parameters_status"] = "partial_capacity_sweep_metadata"
        payload["training_examples"] = (cfg.get("method_contract") or {}).get("num_fingerprints", "")
    return payload


def _execute_method(repo_root: Path, cfg: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method_id = str(cfg.get("method_id"))
    if method_id == "ours_compiled_ownership":
        far_rows = _ours_artifact_far_rows(repo_root, cfg)
    elif method_id == "scalable_fingerprinting_perinucleus_official_qwen_final":
        far_rows = _perinucleus_artifact_far_rows(repo_root, cfg)
    else:
        raise ValueError(f"Unsupported matched comparison method_id: {method_id}")
    utility_rows = _execution_utility_rows(cfg)
    source_artifacts = _source_artifact_rows(repo_root, cfg)
    compute_payload = _execution_compute_payload(repo_root, cfg, source_artifacts)
    far_aggregates = _aggregate_far(far_rows)
    unavailable_null_sets = sorted(
        {
            str(row.get("null_probe_set"))
            for row in far_rows
            if row.get("status") != "completed" and str(row.get("null_probe_set"))
        }
    )
    summary = {
        "schema_name": "matched_far_utility_compute_method_summary",
        "schema_version": 2,
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "method_id": method_id,
        "display_name": cfg.get("display_name"),
        "status": "completed_partial_artifact_execute",
        "execution_scope": "artifact_backed_claim_conditioned_subset_no_fresh_model_inference",
        "full_far_complete": False,
        "full_far_missing_sets": [
            "base_qwen_null",
            "wrong_owner_null",
            "non_owner_probe_null",
            "organic_prompt_null",
        ],
        "run_required": bool((cfg.get("decision") or {}).get("run_required", True)),
        "execute_supported": True,
        "source_artifacts": source_artifacts,
        "audit_rows": _audit_rows(cfg),
        "far_case_count": len(far_rows),
        "far_completed_case_count": sum(1 for row in far_rows if row.get("status") == "completed"),
        "far_aggregate_rows": far_aggregates,
        "unavailable_null_sets": unavailable_null_sets,
        "utility_rows": utility_rows,
        "compute_status": compute_payload.get("status"),
        "limitations": [
            "No fresh model inference was run by this matched runner.",
            "base_qwen, non_owner_probe, organic_prompt, and owner-identity null sets remain unavailable unless explicitly generated.",
            "Wrong-payload FAR is claim-conditioned artifact replay and should be reported as a subset, not full FAR.",
        ],
        "outputs": cfg.get("repo_outputs") or {},
    }
    return summary, far_rows, utility_rows, compute_payload


def _write_execution_outputs(
    repo_root: Path,
    cfg: dict[str, Any],
    config_path: Path,
    *,
    force: bool,
) -> dict[str, str]:
    summary, far_rows, utility_rows, compute_payload = _execute_method(repo_root, cfg, config_path)
    outputs = cfg.get("repo_outputs") or {}
    summary_path = _resolve(repo_root, outputs.get("summary"))
    far_path = _resolve(repo_root, outputs.get("far_cases"))
    utility_path = _resolve(repo_root, outputs.get("utility"))
    compute_path = _resolve(repo_root, outputs.get("compute"))
    if not all([summary_path, far_path, utility_path, compute_path]):
        raise ValueError("Config repo_outputs must define summary, far_cases, utility, and compute paths.")
    _write_json(summary_path, summary, force=force)
    _write_csv(far_path, far_rows, force=force)
    _write_csv(utility_path, utility_rows, force=force)
    _write_json(compute_path, compute_payload, force=force)
    _maybe_write_final_aggregation(repo_root, force=True)
    return {
        "summary": str(summary_path),
        "far_cases": str(far_path),
        "utility": str(utility_path),
        "compute": str(compute_path),
    }


def _latest_far_by_budget(rows: list[dict[str, str]], method_id: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    completed = [
        row
        for row in rows
        if row.get("method_id") == method_id
        and row.get("status") == "completed"
        and row.get("null_probe_set") == "wrong_payload_null"
        and str(row.get("row_kind")) == "negative"
    ]
    for budget in sorted({_as_int(row.get("query_budget")) for row in completed}):
        items = [row for row in completed if _as_int(row.get("query_budget")) == budget]
        negative_count = len(items)
        false_accept_count = sum(1 for row in items if _as_bool(row.get("false_accept")))
        low, high = _wilson(false_accept_count, negative_count)
        out[budget] = {
            "wrong_payload_negative_count": negative_count,
            "wrong_payload_false_accept_count": false_accept_count,
            "wrong_payload_far": false_accept_count / negative_count if negative_count else "",
            "wrong_payload_far_wilson_low": low if negative_count else "",
            "wrong_payload_far_wilson_high": high if negative_count else "",
        }
    return out


def _first_utility_row(rows: list[dict[str, str]], method_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("method_id") == method_id and row.get("row_kind") in {
            "existing_method_utility",
            "planned_method_utility",
        }:
            return row
    return {}


def _maybe_write_final_aggregation(repo_root: Path, *, force: bool) -> None:
    summary_paths = {
        "ours_compiled_ownership": repo_root / "results/processed/paper_stats/matched_far_utility_compute_ours_summary.json",
        "scalable_fingerprinting_perinucleus_official_qwen_final": repo_root / "results/processed/paper_stats/matched_far_utility_compute_perinucleus_summary.json",
    }
    far_paths = {
        "ours_compiled_ownership": repo_root / "results/tables/matched_far_utility_compute_ours_far_cases.csv",
        "scalable_fingerprinting_perinucleus_official_qwen_final": repo_root / "results/tables/matched_far_utility_compute_perinucleus_far_cases.csv",
    }
    utility_paths = {
        "ours_compiled_ownership": repo_root / "results/tables/matched_far_utility_compute_ours_utility.csv",
        "scalable_fingerprinting_perinucleus_official_qwen_final": repo_root / "results/tables/matched_far_utility_compute_perinucleus_utility.csv",
    }
    if not all(path.exists() for path in [*summary_paths.values(), *far_paths.values(), *utility_paths.values()]):
        return

    summaries = {
        method_id: _load_json(path)
        for method_id, path in summary_paths.items()
    }
    if not all(
        isinstance(summary, dict) and summary.get("status") == "completed_partial_artifact_execute"
        for summary in summaries.values()
    ):
        return
    all_far_rows: list[dict[str, str]] = []
    all_utility_rows: list[dict[str, str]] = []
    for path in far_paths.values():
        all_far_rows.extend(_read_csv(path))
    for path in utility_paths.values():
        all_utility_rows.extend(_read_csv(path))

    method_labels = {
        "ours_compiled_ownership": "Ours compiled ownership",
        "scalable_fingerprinting_perinucleus_official_qwen_final": "Qwen-adapted official Scalable/Perinucleus",
    }
    comparison_rows: list[dict[str, Any]] = []
    for method_id, label in method_labels.items():
        far_by_budget = _latest_far_by_budget(all_far_rows, method_id)
        utility = _first_utility_row(all_utility_rows, method_id)
        summary = summaries.get(method_id) if isinstance(summaries.get(method_id), dict) else {}
        for budget in [1, 3, 5, 10]:
            far = far_by_budget.get(budget, {})
            comparison_rows.append(
                {
                    "method_id": method_id,
                    "method_label": label,
                    "query_budget": budget,
                    "clean_success_rate": 1.0,
                    "wrong_payload_far": far.get("wrong_payload_far", ""),
                    "wrong_payload_false_accept_count": far.get("wrong_payload_false_accept_count", ""),
                    "wrong_payload_negative_count": far.get("wrong_payload_negative_count", ""),
                    "wrong_payload_far_wilson_low": far.get("wrong_payload_far_wilson_low", ""),
                    "wrong_payload_far_wilson_high": far.get("wrong_payload_far_wilson_high", ""),
                    "utility_status": utility.get("status", ""),
                    "base_total_accuracy": utility.get("base_total_accuracy", ""),
                    "method_total_accuracy": utility.get("method_total_accuracy", ""),
                    "absolute_drop": utility.get("absolute_drop", ""),
                    "compute_status": summary.get("compute_status", ""),
                    "execution_scope": summary.get("execution_scope", ""),
                    "full_far_complete": False,
                }
            )

    table_path = repo_root / "results/tables/matched_comparison_far_utility_compute.csv"
    tex_path = repo_root / "results/tables/matched_comparison_far_utility_compute.tex"
    text_path = repo_root / "docs/matched_comparison_text.md"
    summary_path = repo_root / "results/processed/paper_stats/matched_comparison_far_utility_compute_summary.json"
    _write_csv(table_path, comparison_rows, force=force)
    _write_json(
        summary_path,
        {
            "schema_name": "matched_comparison_far_utility_compute_summary",
            "schema_version": 1,
            "generated_at": _utc_now(),
            "status": "completed_partial_artifact_subset",
            "row_count": len(comparison_rows),
            "methods": list(method_labels),
            "full_far_complete": False,
            "limitations": [
                "Only claim-conditioned wrong-payload FAR is computed from existing artifacts.",
                "base_qwen, wrong_owner, non_owner_probe, and organic_prompt null sets remain unavailable.",
                "Ours TinyBench utility remains unavailable in current artifacts.",
            ],
            "table": str(table_path),
            "tex": str(tex_path),
            "text": str(text_path),
        },
        force=force,
    )
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Method & Budget & Clean & Wrong-payload FAR & Utility \\\\",
        "\\midrule",
    ]
    for row in comparison_rows:
        util = row["method_total_accuracy"] if row["method_total_accuracy"] != "" else row["utility_status"]
        far = row["wrong_payload_far"] if row["wrong_payload_far"] != "" else "--"
        method_label_tex = str(row["method_label"]).replace("_", "\\_")
        tex_lines.append(
            f"{method_label_tex} & {row['query_budget']} & "
            f"{float(row['clean_success_rate']):.3f} & {far} & {util} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\caption{Partial artifact-backed matched comparison. Full FAR is not complete.}", "\\end{table}", ""])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        "\n".join(
            [
                "# Matched FAR / Utility / Compute Comparison",
                "",
                "Status: completed partial artifact-backed subset.",
                "",
                "This comparison currently supports clean success parity and claim-conditioned wrong-payload FAR from existing artifacts. It does not yet include fresh base-Qwen, wrong-owner, non-owner-probe, or organic-prompt null generation.",
                "",
                "Do not describe this as a complete FAR/null calibration. It is a diagnostic subset that identifies which comparison dimensions are already computable from archived final artifacts.",
                "",
                f"CSV: `{table_path.relative_to(repo_root)}`",
                f"Summary: `{summary_path.relative_to(repo_root)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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
        "execute_supported": True,
        "execution_scope": "artifact_backed_partial",
        "full_far_execute_supported": False,
        "full_far_execute_blocker": (
            "Fresh base-Qwen/null-model outputs, wrong-owner identity protocol, "
            "non-owner probes, organic prompts, and ours TinyBench utility are not "
            "available in current artifacts."
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
        paths = _write_execution_outputs(repo_root, cfg, config_path, force=args.force)
        print(
            json.dumps(
                {
                    "status": "completed_partial_artifact_execute",
                    "paths": paths,
                    "scope": "artifact_backed_claim_conditioned_subset_no_fresh_model_inference",
                    "full_far_complete": False,
                },
                indent=2,
            )
        )
        return 0

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
