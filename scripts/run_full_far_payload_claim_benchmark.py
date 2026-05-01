from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and execute the full FAR / payload-claim benchmark plan."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the currently supported artifact-backed claim subset and write final outputs. "
            "Rows that require fresh null-model/prompt-bank inference are retained as pending."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve(repo_root: Path, raw: str | Path | None) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, payload: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "success"}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        if raw is None or str(raw).strip() == "":
            return default
        return int(float(str(raw)))
    except (TypeError, ValueError):
        return default


def _wilson_interval(
    success_count: int,
    total_count: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total_count <= 0:
        return (math.nan, math.nan)
    phat = success_count / total_count
    denom = 1.0 + z * z / total_count
    center = (phat + z * z / (2.0 * total_count)) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total_count)) / total_count)
        / denom
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.12g}"


def _methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    methods = cfg.get("methods") or []
    if not methods:
        raise ValueError("Config must define at least one method.")
    return [dict(method) for method in methods]


def _positive_payloads(cfg: dict[str, Any]) -> list[str]:
    return [str(item) for item in (cfg.get("positive_split") or {}).get("payloads", [])]


def _seeds(cfg: dict[str, Any]) -> list[int]:
    return [int(item) for item in (cfg.get("positive_split") or {}).get("seeds", [])]


def _budgets(cfg: dict[str, Any]) -> list[int]:
    return [int(item) for item in (cfg.get("positive_split") or {}).get("query_budgets", [])]


def _all_payload_labels(cfg: dict[str, Any]) -> list[str]:
    labels = [
        str(item)
        for item in (cfg.get("payload_claim_protocol") or {}).get("all_payload_labels", [])
    ]
    if not labels:
        raise ValueError("payload_claim_protocol.all_payload_labels must be non-empty.")
    return labels


def _wrong_owner_ids(cfg: dict[str, Any]) -> list[str]:
    return [
        str(item)
        for item in (cfg.get("owner_claim_protocol") or {}).get("wrong_owner_ids", [])
    ]


def _null_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in (cfg.get("null_protocol") or {}).get("null_models", [])]


def _row(
    *,
    cfg: dict[str, Any],
    method: dict[str, Any],
    case_index: int,
    evaluation_type: str,
    true_payload: str = "",
    claim_payload: str = "",
    true_owner: str = "",
    claim_owner: str = "",
    seed: int = 0,
    query_budget: int = 0,
    null_model_id: str = "",
    null_probe_set: str = "",
    organic_prompt_id: str = "",
    status: str = "planned",
    notes: str = "",
) -> dict[str, Any]:
    method_id = str(method["method_id"])
    case_id_parts = [
        "fullfar",
        method_id.replace("_", "-"),
        evaluation_type.replace("_", "-"),
        true_payload.lower() or "null",
        claim_payload.lower() or "none",
        claim_owner.replace("_", "-") if claim_owner and claim_owner != true_owner else "",
        f"s{seed}" if seed else "sna",
        f"m{query_budget}" if query_budget else "mna",
        null_model_id,
        null_probe_set,
        organic_prompt_id,
        "registered" if not (null_model_id or null_probe_set or organic_prompt_id) else "",
    ]
    return {
        "case_index": case_index,
        "case_id": "-".join(part for part in case_id_parts if part),
        "method_id": method_id,
        "method_display_name": method.get("display_name", method_id),
        "native_verifier_object": method.get("verifier_object", ""),
        "evaluation_type": evaluation_type,
        "true_payload": true_payload,
        "claim_payload": claim_payload,
        "true_owner": true_owner,
        "claim_owner": claim_owner,
        "seed": seed,
        "query_budget": query_budget,
        "null_model_id": null_model_id,
        "null_probe_set": null_probe_set,
        "organic_prompt_id": organic_prompt_id,
        "score_name": "ownership_score",
        "threshold_policy": (cfg.get("payload_claim_protocol") or {}).get("threshold_policy", ""),
        "target_far": (cfg.get("payload_claim_protocol") or {}).get("target_far", ""),
        "requires_fresh_model_inference": True,
        "status": status,
        "notes": notes,
    }


def build_plan_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    methods = _methods(cfg)
    payloads = _positive_payloads(cfg)
    seeds = _seeds(cfg)
    budgets = _budgets(cfg)
    all_payloads = _all_payload_labels(cfg)
    true_owner = str(
        (cfg.get("owner_claim_protocol") or {}).get("true_owner_id")
        or (cfg.get("positive_split") or {}).get("owner_id", "owner")
    )
    wrong_owner_ids = _wrong_owner_ids(cfg)
    organic_count = int((cfg.get("null_protocol") or {}).get("organic_prompt_count", 0))
    non_owner_probe_count = int((cfg.get("null_protocol") or {}).get("non_owner_probe_count", 0))
    null_models = _null_models(cfg)

    rows: list[dict[str, Any]] = []
    case_index = 0

    def add(**kwargs: Any) -> None:
        nonlocal case_index
        case_index += 1
        rows.append(_row(cfg=cfg, case_index=case_index, **kwargs))

    for method in methods:
        for payload in payloads:
            for seed in seeds:
                for budget in budgets:
                    add(
                        method=method,
                        evaluation_type="clean_correct_claim",
                        true_payload=payload,
                        claim_payload=payload,
                        true_owner=true_owner,
                        claim_owner=true_owner,
                        seed=seed,
                        query_budget=budget,
                        notes="positive clean correct owner/payload claim",
                    )
                    for claim_payload in all_payloads:
                        if claim_payload == payload:
                            continue
                        add(
                            method=method,
                            evaluation_type="wrong_payload_claim",
                            true_payload=payload,
                            claim_payload=claim_payload,
                            true_owner=true_owner,
                            claim_owner=true_owner,
                            seed=seed,
                            query_budget=budget,
                            null_probe_set="wrong_payload_claim",
                            notes="structured claim check; not full FAR by itself",
                        )
                    for wrong_owner in wrong_owner_ids:
                        add(
                            method=method,
                            evaluation_type="wrong_owner_claim",
                            true_payload=payload,
                            claim_payload=payload,
                            true_owner=true_owner,
                            claim_owner=wrong_owner,
                            seed=seed,
                            query_budget=budget,
                            null_probe_set="wrong_owner_claim",
                            notes="owner identity claim check",
                        )

        for budget in budgets:
            for model in null_models:
                model_id = str(model.get("id", "unknown_null_model"))
                model_status = str(model.get("status", "required"))
                add(
                    method=method,
                    evaluation_type="null_model_registered_probes",
                    query_budget=budget,
                    null_model_id=model_id,
                    null_probe_set="base_model_registered_probes",
                    status="planned" if model_status == "required" else "optional_planned",
                    notes=f"null model status={model_status}",
                )
            for index in range(organic_count):
                add(
                    method=method,
                    evaluation_type="organic_prompt_null",
                    query_budget=budget,
                    null_model_id="base_qwen",
                    null_probe_set="organic_prompt_null",
                    organic_prompt_id=f"organic_{index:04d}",
                    notes="organic prompt null trial",
                )
            for index in range(non_owner_probe_count):
                add(
                    method=method,
                    evaluation_type="non_owner_probe_null",
                    query_budget=budget,
                    null_model_id="base_qwen",
                    null_probe_set="non_owner_probe_null",
                    organic_prompt_id=f"non_owner_probe_{index:04d}",
                    notes="registered non-owner probe null trial",
                )
    return rows


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, int]] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        method = str(row["method_id"])
        kind = str(row["evaluation_type"])
        by_method.setdefault(method, {})
        by_method[method][kind] = by_method[method].get(kind, 0) + 1
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "case_count": len(rows),
        "by_method": by_method,
        "by_type": by_type,
    }


def _source_artifacts(repo_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _methods(cfg):
        for key, raw in (method.get("source_artifacts") or {}).items():
            path = _resolve(repo_root, raw)
            rows.append(
                {
                    "method_id": method["method_id"],
                    "artifact_key": key,
                    "path": str(raw),
                    "resolved_path": str(path) if path else "",
                    "exists": bool(path and path.exists()),
                }
            )
    return rows


def build_summary(
    repo_root: Path,
    cfg: dict[str, Any],
    config_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    count_payload = _counts(rows)
    wrong_payload_min = int(
        (cfg.get("payload_claim_protocol") or {}).get(
            "minimum_wrong_payload_claims_per_method",
            0,
        )
    )
    organic_min = int((cfg.get("null_protocol") or {}).get("minimum_null_prompt_trials", 0))
    gate_failures: list[str] = []
    for method_id, counts in count_payload["by_method"].items():
        wrong_payload_count = counts.get("wrong_payload_claim", 0)
        organic_count = counts.get("organic_prompt_null", 0)
        if wrong_payload_count < wrong_payload_min:
            gate_failures.append(
                f"{method_id}: wrong_payload_claim count "
                f"{wrong_payload_count} < {wrong_payload_min}"
            )
        if organic_count < organic_min:
            gate_failures.append(
                f"{method_id}: organic_prompt_null count {organic_count} < {organic_min}"
            )
    return {
        "schema_name": "full_far_payload_claim_plan_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "status": "plan_ready" if not gate_failures else "plan_has_gate_failures",
        "fresh_inference_backend_supported": False,
        "launch_allowed_now": False,
        "counts": count_payload,
        "gate_failures": gate_failures,
        "source_artifacts": _source_artifacts(repo_root, cfg),
        "required_outputs_after_execution": {
            key: value
            for key, value in (cfg.get("outputs") or {}).items()
            if key.startswith("final_") or key.endswith("_figure")
        },
        "plan_outputs": {
            "plan_summary": (cfg.get("outputs") or {}).get("plan_summary"),
            "plan_cases": (cfg.get("outputs") or {}).get("plan_cases"),
        },
        "terminology_rules": [
            "wrong_payload_claim rows report claim accept rate, not full FAR",
            "full FAR requires generated null-model/non-owner/organic rows under frozen thresholds",
            (
                "original Perinucleus is a binary fingerprint detector, "
                "not a structured payload verifier"
            ),
        ],
        "next_required_implementation": [
            "fresh inference backend for ours on base/null prompts",
            "fresh inference backend for original Perinucleus on base/null prompts",
            "payload-adapted Perinucleus baseline if making structured-claim superiority claims",
            "ROC and heatmap aggregation after fresh rows complete",
        ],
    }


OURS_METHOD_ID = "ours_compiled_ownership"
PERINUCLEUS_METHOD_ID = "scalable_fingerprinting_perinucleus_official_qwen_final"

NEGATIVE_EVALUATION_TYPES = {
    "wrong_payload_claim",
    "wrong_owner_claim",
    "null_model_registered_probes",
    "organic_prompt_null",
    "non_owner_probe_null",
}

NULL_EVALUATION_TYPES = {
    "null_model_registered_probes",
    "organic_prompt_null",
    "non_owner_probe_null",
}


def _method_artifact_path(
    repo_root: Path,
    cfg: dict[str, Any],
    method_id: str,
    artifact_key: str,
) -> Path | None:
    for method in _methods(cfg):
        if method.get("method_id") != method_id:
            continue
        raw = (method.get("source_artifacts") or {}).get(artifact_key)
        return _resolve(repo_root, raw)
    return None


def _load_artifact_indexes(repo_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    ours_rows: dict[tuple[str, int], dict[str, str]] = {}
    perinucleus_rows: dict[tuple[str, int, int], dict[str, str]] = {}
    artifact_status: list[dict[str, Any]] = []

    ours_clean_table = _method_artifact_path(repo_root, cfg, OURS_METHOD_ID, "clean_table")
    artifact_status.append(
        {
            "method_id": OURS_METHOD_ID,
            "artifact_key": "clean_table",
            "resolved_path": str(ours_clean_table) if ours_clean_table else "",
            "exists": bool(ours_clean_table and ours_clean_table.exists()),
        }
    )
    if ours_clean_table and ours_clean_table.exists():
        for row in _read_csv(ours_clean_table):
            if not _truthy(row.get("included", "True")):
                continue
            ours_rows[(str(row.get("payload", "")), _safe_int(row.get("seed")))] = row

    perinucleus_final_table = _method_artifact_path(
        repo_root,
        cfg,
        PERINUCLEUS_METHOD_ID,
        "final_table",
    )
    artifact_status.append(
        {
            "method_id": PERINUCLEUS_METHOD_ID,
            "artifact_key": "final_table",
            "resolved_path": str(perinucleus_final_table) if perinucleus_final_table else "",
            "exists": bool(perinucleus_final_table and perinucleus_final_table.exists()),
        }
    )
    if perinucleus_final_table and perinucleus_final_table.exists():
        for row in _read_csv(perinucleus_final_table):
            key = (
                str(row.get("payload", "")),
                _safe_int(row.get("seed")),
                _safe_int(row.get("query_budget")),
            )
            perinucleus_rows[key] = row

    return {
        "ours_clean_rows": ours_rows,
        "perinucleus_final_rows": perinucleus_rows,
        "artifact_status": artifact_status,
    }


def _execution_base_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["plan_status"] = output.get("status", "")
    output.update(
        {
            "status": "not_executed",
            "execution_backend": "",
            "execution_scope": "",
            "claim_accept": "",
            "ownership_score": "",
            "false_accept": "",
            "true_accept": "",
            "source_artifact": "",
            "source_status": "",
            "failure_reason": "",
            "full_far_component_complete": False,
            "fresh_model_inference_performed": False,
        }
    )
    return output


def _set_completed(
    output: dict[str, Any],
    *,
    claim_accept: bool,
    status: str,
    execution_scope: str,
    source_artifact: str,
    source_status: str,
    notes_suffix: str,
) -> dict[str, Any]:
    evaluation_type = str(output.get("evaluation_type", ""))
    output["status"] = status
    output["execution_backend"] = "artifact_replay_v1"
    output["execution_scope"] = execution_scope
    output["claim_accept"] = claim_accept
    output["ownership_score"] = 1.0 if claim_accept else 0.0
    output["source_artifact"] = source_artifact
    output["source_status"] = source_status
    output["full_far_component_complete"] = evaluation_type not in NULL_EVALUATION_TYPES
    if evaluation_type == "clean_correct_claim":
        output["true_accept"] = claim_accept
        output["false_accept"] = False
    elif evaluation_type in NEGATIVE_EVALUATION_TYPES:
        output["true_accept"] = False
        output["false_accept"] = claim_accept
    if notes_suffix:
        output["notes"] = f"{output.get('notes', '')}; {notes_suffix}".strip("; ")
    return output


def _set_pending(
    output: dict[str, Any],
    *,
    status: str,
    reason: str,
    execution_scope: str = "fresh_inference_pending",
) -> dict[str, Any]:
    output["status"] = status
    output["execution_backend"] = "pending_backend_v1"
    output["execution_scope"] = execution_scope
    output["failure_reason"] = reason
    output["notes"] = f"{output.get('notes', '')}; {reason}".strip("; ")
    return output


def _execute_ours_artifact_row(
    row: dict[str, Any],
    indexes: dict[str, Any],
    repo_root: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    output = _execution_base_row(row)
    evaluation_type = str(row.get("evaluation_type", ""))
    source_path = _method_artifact_path(repo_root, cfg, OURS_METHOD_ID, "clean_table")

    if evaluation_type == "wrong_owner_claim":
        return _set_pending(
            output,
            status="not_executed_owner_claim_not_encoded",
            reason=(
                "current artifact verifies decoded payload, but no explicit owner-id "
                "field is encoded in the claim artifact"
            ),
            execution_scope="owner_claim_protocol_pending",
        )

    if evaluation_type in NULL_EVALUATION_TYPES:
        return _set_pending(
            output,
            status="not_executed_fresh_null_inference_required",
            reason="fresh null-model/prompt-bank inference is required for full FAR",
        )

    if evaluation_type not in {"clean_correct_claim", "wrong_payload_claim"}:
        return _set_pending(
            output,
            status="not_executed_unknown_evaluation_type",
            reason=f"unsupported evaluation_type={evaluation_type}",
        )

    key = (str(row.get("true_payload", "")), _safe_int(row.get("seed")))
    source = indexes["ours_clean_rows"].get(key)
    if source is None:
        return _set_pending(
            output,
            status="not_executed_missing_ours_clean_artifact",
            reason=f"missing G1 artifact row for payload={key[0]} seed={key[1]}",
            execution_scope="source_artifact_missing",
        )

    verifier_success = _truthy(source.get("accepted")) and _truthy(source.get("verifier_success"))
    decoded_payload = str(source.get("decoded_payload", ""))
    claim_payload = str(row.get("claim_payload", ""))
    claim_accept = bool(verifier_success and decoded_payload == claim_payload)
    status = "completed_artifact_replay_budget_projection"
    scope = "structured_payload_claim_from_g1_clean_artifact_no_fresh_inference"
    notes = (
        "budget projection from deterministic decoded payload artifact; "
        "not a substitute for fresh null-model FAR"
    )
    return _set_completed(
        output,
        claim_accept=claim_accept,
        status=status,
        execution_scope=scope,
        source_artifact=str(source_path) if source_path else "",
        source_status=str(source.get("status", "")),
        notes_suffix=notes,
    )


def _execute_perinucleus_artifact_row(
    row: dict[str, Any], indexes: dict[str, Any], repo_root: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    output = _execution_base_row(row)
    evaluation_type = str(row.get("evaluation_type", ""))
    source_path = _method_artifact_path(repo_root, cfg, PERINUCLEUS_METHOD_ID, "final_table")

    if evaluation_type == "wrong_owner_claim":
        return _set_pending(
            output,
            status="not_executed_owner_claim_not_supported_by_binary_detector",
            reason=(
                "original Perinucleus is a binary fingerprint detector and has no "
                "owner-id claim field in this benchmark"
            ),
            execution_scope="owner_claim_protocol_not_applicable_original_binary_detector",
        )

    if evaluation_type in NULL_EVALUATION_TYPES:
        return _set_pending(
            output,
            status="not_executed_fresh_null_inference_required",
            reason="fresh null-model/prompt-bank inference is required for full FAR",
        )

    if evaluation_type == "clean_correct_claim":
        payload_for_probe = str(row.get("claim_payload", ""))
    elif evaluation_type == "wrong_payload_claim":
        # The original Perinucleus verifier is binary: it detects whether the
        # fingerprinted model is present, not whether a decoded payload label is
        # bound to the claim. Therefore a wrong-payload claim is accepted when
        # the true embedded fingerprint is detected. Payload-specific rejection
        # belongs to the payload-adapted Perinucleus baseline, not this original
        # binary detector row.
        payload_for_probe = str(row.get("true_payload", ""))
    else:
        return _set_pending(
            output,
            status="not_executed_unknown_evaluation_type",
            reason=f"unsupported evaluation_type={evaluation_type}",
        )

    key = (payload_for_probe, _safe_int(row.get("seed")), _safe_int(row.get("query_budget")))
    source = indexes["perinucleus_final_rows"].get(key)
    if source is None:
        return _set_pending(
            output,
            status="not_executed_missing_perinucleus_claim_payload_artifact",
            reason=(
                "missing official final artifact for binary detector subset "
                f"payload={key[0]} seed={key[1]} budget={key[2]}"
            ),
            execution_scope="source_artifact_missing",
        )

    claim_accept = _truthy(source.get("success")) and _truthy(source.get("accepted"))
    if evaluation_type == "wrong_payload_claim":
        status = "completed_artifact_replay_task_mismatch_binary_detector"
        scope = "binary_fingerprint_detector_claim_subset_no_payload_binding"
        notes = (
            "original Perinucleus accepts fingerprint presence for the claimed trigger subset; "
            "this is a task-mismatch diagnostic, not a full FAR measurement"
        )
    else:
        status = "completed_artifact_replay"
        scope = "official_perinucleus_clean_budget_artifact_no_fresh_inference"
        notes = "clean binary fingerprint detection artifact replay"

    return _set_completed(
        output,
        claim_accept=claim_accept,
        status=status,
        execution_scope=scope,
        source_artifact=str(source_path) if source_path else "",
        source_status=str(source.get("status", "")),
        notes_suffix=notes,
    )


def execute_plan_rows(
    repo_root: Path,
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexes = _load_artifact_indexes(repo_root, cfg)
    executed: list[dict[str, Any]] = []
    for row in rows:
        method_id = str(row.get("method_id", ""))
        if method_id == OURS_METHOD_ID:
            executed.append(_execute_ours_artifact_row(row, indexes, repo_root, cfg))
        elif method_id == PERINUCLEUS_METHOD_ID:
            executed.append(_execute_perinucleus_artifact_row(row, indexes, repo_root, cfg))
        else:
            executed.append(
                _set_pending(
                    _execution_base_row(row),
                    status="not_executed_unknown_method",
                    reason=f"no execution backend registered for method_id={method_id}",
                    execution_scope="method_backend_missing",
                )
            )
    return executed


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _execution_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, dict[str, int]]] = {}
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        method = str(row.get("method_id", ""))
        evaluation_type = str(row.get("evaluation_type", ""))
        status = str(row.get("status", ""))
        by_method.setdefault(method, {}).setdefault(evaluation_type, {})
        previous = by_method[method][evaluation_type].get(status, 0)
        by_method[method][evaluation_type][status] = previous + 1
        by_type.setdefault(evaluation_type, {})
        by_type[evaluation_type][status] = by_type[evaluation_type].get(status, 0) + 1
    return {
        "case_count": len(rows),
        "by_method_type_status": by_method,
        "by_type_status": by_type,
        "by_status": _status_counts(rows),
    }


def _metric_name(evaluation_type: str) -> str:
    if evaluation_type == "clean_correct_claim":
        return "clean_correct_claim_accept_rate"
    if evaluation_type == "wrong_payload_claim":
        return "wrong_payload_claim_accept_rate"
    if evaluation_type == "wrong_owner_claim":
        return "wrong_owner_claim_accept_rate"
    return "false_accept_rate"


def _aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("claim_accept", "")) == "":
            continue
        key = (
            str(row.get("method_id", "")),
            str(row.get("evaluation_type", "")),
            _safe_int(row.get("query_budget")),
        )
        groups.setdefault(key, []).append(row)

    metrics: list[dict[str, Any]] = []
    for (method_id, evaluation_type, query_budget), group_rows in sorted(groups.items()):
        accept_count = sum(1 for row in group_rows if _truthy(row.get("claim_accept")))
        total_count = len(group_rows)
        ci_low, ci_high = _wilson_interval(accept_count, total_count)
        metrics.append(
            {
                "method_id": method_id,
                "evaluation_type": evaluation_type,
                "query_budget": query_budget,
                "metric_name": _metric_name(evaluation_type),
                "trial_count": total_count,
                "accept_count": accept_count,
                "accept_rate": _format_float(
                    accept_count / total_count if total_count else math.nan
                ),
                "ci95_low": _format_float(ci_low),
                "ci95_high": _format_float(ci_high),
                "scope": "completed_artifact_rows_only",
            }
        )
    return metrics


def _build_execution_summary(
    repo_root: Path,
    cfg: dict[str, Any],
    config_path: Path,
    plan_rows: list[dict[str, Any]],
    executed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _aggregate_metrics(executed_rows)
    full_far_complete = all(
        row.get("evaluation_type") not in NULL_EVALUATION_TYPES
        or str(row.get("claim_accept", "")) != ""
        for row in executed_rows
    )
    all_claim_rows_complete = all(
        row.get("evaluation_type") not in {"clean_correct_claim", "wrong_payload_claim"}
        or str(row.get("claim_accept", "")) != ""
        for row in executed_rows
    )
    status = "completed_full_far" if full_far_complete else "completed_artifact_subset"
    return {
        "schema_name": "full_far_payload_claim_execution_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "status": status,
        "execution_backend": "artifact_replay_v1_with_pending_fresh_far_rows",
        "fresh_model_inference_performed": False,
        "full_far_complete": full_far_complete,
        "claim_rows_complete": all_claim_rows_complete,
        "plan_counts": _counts(plan_rows),
        "execution_counts": _execution_counts(executed_rows),
        "metrics": metrics,
        "source_artifacts": _source_artifacts(repo_root, cfg),
        "artifact_indexes": _load_artifact_indexes(repo_root, cfg)["artifact_status"],
        "interpretation_rules": [
            "wrong_payload_claim_accept_rate is not full FAR",
            (
                "Perinucleus wrong-payload rows are task-mismatch diagnostics "
                "for the original binary detector"
            ),
            (
                "organic/non-owner/null-model rows remain pending until fresh "
                "inference prompt banks are implemented"
            ),
            "do not use this summary to claim complete FAR superiority",
        ],
        "next_required_implementation": [
            "fresh inference backend for organic_prompt_null and non_owner_probe_null",
            "fresh null-model registered-probe backend for base Qwen and optional null models",
            "owner-id encoding or a declared exclusion for wrong_owner_claim rows",
            "payload-adapted Perinucleus baseline before structured-claim superiority claims",
        ],
    }


def _latex_escape(raw: Any) -> str:
    text = str(raw)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _build_metrics_tex(metrics: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Method & Type & $M$ & Trials & Accepts & Rate & 95\% CI \\",
        r"\midrule",
    ]
    for row in metrics:
        ci = f"[{row['ci95_low']}, {row['ci95_high']}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["method_id"]),
                    _latex_escape(row["evaluation_type"]),
                    _latex_escape(row["query_budget"]),
                    _latex_escape(row["trial_count"]),
                    _latex_escape(row["accept_count"]),
                    _latex_escape(row["accept_rate"]),
                    _latex_escape(ci),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = _load_yaml(config_path)
    rows = build_plan_rows(cfg)
    summary = build_summary(repo_root, cfg, config_path, rows)

    if args.execute:
        executed_rows = execute_plan_rows(repo_root, cfg, rows)
        outputs = cfg.get("outputs") or {}
        summary_path = _resolve(repo_root, outputs.get("final_summary"))
        table_path = _resolve(repo_root, outputs.get("final_table"))
        tex_path = _resolve(repo_root, outputs.get("final_tex"))
        if summary_path is None or table_path is None or tex_path is None:
            raise ValueError(
                "outputs.final_summary, outputs.final_table, and outputs.final_tex are required."
            )
        execution_summary = _build_execution_summary(
            repo_root,
            cfg,
            config_path,
            rows,
            executed_rows,
        )
        _write_json(summary_path, execution_summary, force=args.force)
        _write_csv(table_path, executed_rows, force=args.force)
        _write_text(tex_path, _build_metrics_tex(execution_summary["metrics"]), force=args.force)
        print(
            json.dumps(
                {
                    "status": execution_summary["status"],
                    "summary": str(summary_path),
                    "table": str(table_path),
                    "tex": str(tex_path),
                    "case_count": len(executed_rows),
                    "full_far_complete": execution_summary["full_far_complete"],
                    "claim_rows_complete": execution_summary["claim_rows_complete"],
                    "fresh_model_inference_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.write_plan:
        outputs = cfg.get("outputs") or {}
        summary_path = _resolve(repo_root, outputs.get("plan_summary"))
        cases_path = _resolve(repo_root, outputs.get("plan_cases"))
        if summary_path is None or cases_path is None:
            raise ValueError("outputs.plan_summary and outputs.plan_cases are required.")
        _write_json(summary_path, summary, force=args.force)
        _write_csv(cases_path, rows, force=args.force)
        print(
            json.dumps(
                {
                    "status": "plan_written",
                    "summary": str(summary_path),
                    "cases": str(cases_path),
                    "case_count": len(rows),
                    "launch_allowed_now": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise SystemExit("Specify --dry-run, --write-plan, or --execute.")


if __name__ == "__main__":
    raise SystemExit(main())
