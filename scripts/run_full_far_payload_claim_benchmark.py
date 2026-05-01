from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the full FAR / payload-claim benchmark plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for the fresh-inference backend; currently blocked by design.",
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
    labels = [str(item) for item in (cfg.get("payload_claim_protocol") or {}).get("all_payload_labels", [])]
    if not labels:
        raise ValueError("payload_claim_protocol.all_payload_labels must be non-empty.")
    return labels


def _wrong_owner_ids(cfg: dict[str, Any]) -> list[str]:
    return [str(item) for item in (cfg.get("owner_claim_protocol") or {}).get("wrong_owner_ids", [])]


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
    true_owner = str((cfg.get("owner_claim_protocol") or {}).get("true_owner_id") or (cfg.get("positive_split") or {}).get("owner_id", "owner"))
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


def build_summary(repo_root: Path, cfg: dict[str, Any], config_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_payload = _counts(rows)
    wrong_payload_min = int((cfg.get("payload_claim_protocol") or {}).get("minimum_wrong_payload_claims_per_method", 0))
    organic_min = int((cfg.get("null_protocol") or {}).get("minimum_null_prompt_trials", 0))
    gate_failures: list[str] = []
    for method_id, counts in count_payload["by_method"].items():
        wrong_payload_count = counts.get("wrong_payload_claim", 0)
        organic_count = counts.get("organic_prompt_null", 0)
        if wrong_payload_count < wrong_payload_min:
            gate_failures.append(f"{method_id}: wrong_payload_claim count {wrong_payload_count} < {wrong_payload_min}")
        if organic_count < organic_min:
            gate_failures.append(f"{method_id}: organic_prompt_null count {organic_count} < {organic_min}")
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
            "original Perinucleus is a binary fingerprint detector, not a structured payload verifier",
        ],
        "next_required_implementation": [
            "fresh inference backend for ours on base/null prompts",
            "fresh inference backend for original Perinucleus on base/null prompts",
            "payload-adapted Perinucleus baseline if making structured-claim superiority claims",
            "ROC and heatmap aggregation after fresh rows complete",
        ],
    }


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
        raise RuntimeError(
            "Fresh model inference is intentionally blocked in this runner revision. "
            "Use --write-plan, review the plan, then implement the method-specific backends."
        )

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

    raise SystemExit("Specify --dry-run or --write-plan. --execute is not enabled yet.")


if __name__ == "__main__":
    raise SystemExit(main())
