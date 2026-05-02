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
    parser.add_argument(
        "--fresh-null-mode",
        choices=("off", "registered-probes", "organic-prompts", "registered-and-organic"),
        default="off",
        help=(
            "Optional fresh null inference backend. 'registered-probes' evaluates "
            "base-Qwen registered probes. 'organic-prompts' evaluates base-Qwen organic "
            "prompt-bank rows. 'registered-and-organic' executes both required slices."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for array execution. Requires --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total shard count for array execution. Requires --shard-index.",
    )
    parser.add_argument(
        "--shard-output-dir",
        default=None,
        help=(
            "Directory for shard CSV/JSON outputs. Required with --shard-index/--shard-count "
            "so array jobs never write the final table concurrently."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Write shard CSV/summary after this many newly completed rows during sharded execution.",
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


def _write_csv_with_fieldnames(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


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
                for claim_payload in all_payloads:
                    for seed in seeds:
                        add(
                            method=method,
                            evaluation_type="null_model_registered_probes",
                            claim_payload=claim_payload,
                            true_owner="null_model",
                            claim_owner=true_owner,
                            seed=seed,
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
                    claim_payload=all_payloads[index % len(all_payloads)],
                    true_owner="organic_prompt",
                    claim_owner=true_owner,
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
                    claim_payload=all_payloads[index % len(all_payloads)],
                    true_owner="non_owner_probe",
                    claim_owner=true_owner,
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
        "fresh_inference_backend_supported": "base_qwen_registered_probes_and_organic_prompts",
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
            "fresh inference backend for non-owner prompt banks",
            "fresh inference backend for optional null models",
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

FRESH_REGISTERED_NULL_STATUS = "completed_fresh_registered_null"
FRESH_ORGANIC_NULL_STATUS = "completed_fresh_organic_null"


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


def _set_fresh_completed(
    output: dict[str, Any],
    *,
    claim_accept: bool,
    ownership_score: float,
    status: str = FRESH_REGISTERED_NULL_STATUS,
    execution_backend: str = "fresh_registered_null_v1",
    execution_scope: str,
    source_status: str,
    notes_suffix: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    evaluation_type = str(output.get("evaluation_type", ""))
    output["status"] = status
    output["execution_backend"] = execution_backend
    output["execution_scope"] = execution_scope
    output["claim_accept"] = claim_accept
    output["ownership_score"] = ownership_score
    output["source_status"] = source_status
    output["full_far_component_complete"] = evaluation_type in NULL_EVALUATION_TYPES
    output["fresh_model_inference_performed"] = True
    output["true_accept"] = False
    output["false_accept"] = claim_accept
    output["fresh_details_json"] = json.dumps(details, sort_keys=True)
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


def _runtime_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("runtime") or {})


def _fresh_mode_enabled(fresh_null_mode: str, slice_name: str) -> bool:
    if slice_name == "registered-probes":
        return fresh_null_mode in {"registered-probes", "registered-and-organic"}
    if slice_name == "organic-prompts":
        return fresh_null_mode in {"organic-prompts", "registered-and-organic"}
    return False


def _null_model_cfg(cfg: dict[str, Any], null_model_id: str) -> dict[str, Any]:
    for item in _null_models(cfg):
        if str(item.get("id", "")) == null_model_id:
            return item
    return {}


def _catalog_path(repo_root: Path, cfg: dict[str, Any]) -> Path:
    raw = (
        (cfg.get("ours_registered_null_backend") or {}).get("catalog_path")
        or "configs/data/frozen/real_pilot_catalog__qwen2_5_7b_compiled__v1.yaml"
    )
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _load_base_model(
    *,
    cfg: dict[str, Any],
    model_name: str,
    context: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    cache_key = ("base_model", model_name)
    if cache_key in context:
        return context[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("fresh registered null inference requires CUDA")
    runtime = _runtime_cfg(cfg)
    local_files_only = bool(runtime.get("local_files_only", True))
    trust_remote_code = bool(runtime.get("trust_remote_code", True))
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    context[cache_key] = (model, tokenizer, device, torch)
    return context[cache_key]


_ORGANIC_TOPICS = (
    "safe kitchen habits",
    "planning a weekend hike",
    "organizing a small bookshelf",
    "watering indoor plants",
    "choosing a train route",
    "writing a meeting note",
    "packing a school bag",
    "checking bicycle brakes",
    "preparing a grocery list",
    "cleaning a desk",
    "explaining cloud shapes",
    "describing a quiet library",
    "summarizing a weather forecast",
    "making tea carefully",
    "sorting recycling bins",
    "choosing comfortable shoes",
)

_ORGANIC_TEMPLATES = (
    "Answer in two short sentences about {topic}. Do not use bullet points.",
    "Give a concise practical tip about {topic}.",
    "Explain one ordinary reason someone might care about {topic}.",
    "Write a plain-language reminder about {topic}.",
    "Describe a common mistake related to {topic} and how to avoid it.",
    "Provide a neutral one-paragraph response about {topic}.",
    "What is a simple first step for {topic}? Keep the answer brief.",
    "State one benefit of paying attention to {topic}.",
)


def _organic_index(organic_prompt_id: str) -> int:
    digits = "".join(ch for ch in str(organic_prompt_id) if ch.isdigit())
    return int(digits or 0)


def _organic_prompt_text(organic_prompt_id: str) -> str:
    index = _organic_index(organic_prompt_id)
    topic = _ORGANIC_TOPICS[index % len(_ORGANIC_TOPICS)]
    template = _ORGANIC_TEMPLATES[(index // len(_ORGANIC_TOPICS)) % len(_ORGANIC_TEMPLATES)]
    variant = index // (len(_ORGANIC_TOPICS) * len(_ORGANIC_TEMPLATES))
    return (
        template.format(topic=topic)
        + f" Context id: organic-null-{variant:03d}. Avoid codes, labels, and structured fields."
    )


def _organic_prompt_ids_for_budget(
    *,
    organic_prompt_id: str,
    query_budget: int,
    cfg: dict[str, Any],
) -> list[str]:
    organic_count = int((cfg.get("null_protocol") or {}).get("organic_prompt_count", 0))
    organic_count = max(1, organic_count)
    start = _organic_index(organic_prompt_id)
    budget = max(1, int(query_budget or 1))
    return [f"organic_{(start + offset) % organic_count:04d}" for offset in range(budget)]


def _tokenize_prompt_for_generation(
    *,
    tokenizer: Any,
    prompt: str,
    device: Any,
) -> tuple[dict[str, Any], int]:
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        attention_mask = None
        try:
            import torch

            attention_mask = torch.ones_like(input_ids)
        except Exception:
            attention_mask = None
        inputs = {"input_ids": input_ids.to(device)}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(device)
        return inputs, int(input_ids.shape[-1])
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    return inputs, int(inputs["input_ids"].shape[-1])


def _generate_organic_text(
    *,
    cfg: dict[str, Any],
    null_model_id: str,
    organic_prompt_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    cache_key = ("organic_generation", null_model_id, organic_prompt_id)
    if cache_key in context:
        return context[cache_key]

    null_model = _null_model_cfg(cfg, null_model_id)
    model_name = str(null_model.get("model") or "Qwen/Qwen2.5-7B-Instruct")
    model, tokenizer, device, torch = _load_base_model(
        cfg=cfg,
        model_name=model_name,
        context=context,
    )
    backend_cfg = dict(cfg.get("organic_prompt_null_backend") or {})
    max_new_tokens = int(backend_cfg.get("max_new_tokens", 32))
    prompt = _organic_prompt_text(organic_prompt_id)
    generation_inputs, prompt_length = _tokenize_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )
    with torch.no_grad():
        generated_tokens = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_token_ids = generated_tokens[0][prompt_length:].detach().cpu().tolist()
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    result = {
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt": prompt,
        "generated_text": generated_text,
        "generated_token_count": len(new_token_ids),
        "model_name": model_name,
        "max_new_tokens": max_new_tokens,
    }
    context[cache_key] = result
    return result


def _ours_registered_null_result(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    claim_payload = str(row.get("claim_payload", ""))
    null_model_id = str(row.get("null_model_id", ""))
    cache_key = ("ours_registered_null", null_model_id, claim_payload)
    if cache_key in context:
        return context[cache_key]

    from src.core.catalog_freeze import load_required_frozen_catalog
    from src.core.contract_compiler import (
        build_generation_plan_from_compiled_eval_contract,
        compile_fieldwise_train_contract,
    )
    from src.core.payload_codec import BucketPayloadCodec
    from src.core.scaffolded_completion import (
        COMPILED_ARTIFACT_FORMAT,
        COMPILED_FIELDWISE_PROMPT_CONTRACT,
        evaluate_foundation_completion,
    )
    from src.core.verifier import VerificationConfig, verify_canonical_rendered_text
    from src.training.hf_causal_lm import (
        _build_generation_kwargs,
        _resolve_fieldwise_contextual_token_map,
        _tensor_rows,
    )

    null_model = _null_model_cfg(cfg, null_model_id)
    model_name = str(null_model.get("model") or "Qwen/Qwen2.5-7B-Instruct")
    model, tokenizer, device, torch = _load_base_model(
        cfg=cfg,
        model_name=model_name,
        context=context,
    )
    catalog_path = _catalog_path(repo_root, cfg)
    payload_labels = _all_payload_labels(cfg)
    backend_cfg = dict(cfg.get("ours_registered_null_backend") or {})
    block_count = int(backend_cfg.get("block_count", 2))
    instruction = str(backend_cfg.get("instruction", "Select exactly one allowed carrier token."))
    max_length = int(backend_cfg.get("max_length", 512))
    compiled_train_contract = compile_fieldwise_train_contract(
        model_name=model_name,
        tokenizer_name=model_name,
        tokenizer_backend=str(backend_cfg.get("tokenizer_backend", "huggingface")),
        catalog_path=catalog_path,
        payload_labels=payload_labels,
        eval_payload_label=claim_payload,
        instruction=instruction,
        block_count=block_count,
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        render_format="canonical_v1",
        tokenizer=tokenizer,
    )
    plan = build_generation_plan_from_compiled_eval_contract(
        compiled_eval_contract=compiled_train_contract.eval_contract,
        catalog_path=Path(compiled_train_contract.catalog_path),
    )
    _audit_result, slot_token_maps = _resolve_fieldwise_contextual_token_map(
        tokenizer=tokenizer,
        plan=plan,
    )
    generated_values: list[str] = []
    chosen_token_ids: list[int | None] = []
    with torch.no_grad():
        for slot_target in plan.slot_targets:
            value_to_token_id, token_id_to_value = slot_token_maps[
                (slot_target.field_name, slot_target.exact_slot_prefix)
            ]
            generation_inputs = tokenizer(
                slot_target.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            generation_inputs = {key: value.to(device) for key, value in generation_inputs.items()}
            prompt_rows = _tensor_rows(generation_inputs["input_ids"])
            prompt_length = len(prompt_rows[0]) if prompt_rows else 0
            generated_tokens = model.generate(
                **generation_inputs,
                **_build_generation_kwargs(
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    generation_do_sample=False,
                    allowed_token_ids=tuple(value_to_token_id.values()),
                ),
            )
            generated_rows = _tensor_rows(generated_tokens)
            chosen_token_id = (
                int(generated_rows[0][prompt_length])
                if generated_rows and len(generated_rows[0]) > prompt_length
                else None
            )
            chosen_text = (
                token_id_to_value.get(chosen_token_id, "")
                if chosen_token_id is not None
                else ""
            )
            generated_values.append(chosen_text)
            chosen_token_ids.append(chosen_token_id)

    layout = load_required_frozen_catalog(Path(compiled_train_contract.catalog_path))
    generated_text = "\n".join(generated_values).strip()
    compiled_result = evaluate_foundation_completion(
        generated_text,
        layout=layout,
        expected_slot_values=compiled_train_contract.eval_contract.expected_slot_values,
        exact_slot_prefixes=compiled_train_contract.eval_contract.exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=compiled_train_contract.eval_contract.prompt_contract_name,
        render_format=compiled_train_contract.eval_contract.render_format,
        slot_field_names=compiled_train_contract.eval_contract.slot_field_names,
        artifact_format=COMPILED_ARTIFACT_FORMAT,
    )
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    render_success = False
    decoded_units: tuple[int, ...] = ()
    expected_units = tuple(
        int(unit)
        for unit in compiled_train_contract.payload_label_to_units[claim_payload]
    )
    if compiled_result.rendered_bucket_tuples:
        render_verification = verify_canonical_rendered_text(
            text=compiled_result.rendered_canonical_text,
            bucket_layout=layout,
            payload_codec=codec,
            expected_payload=expected_units,
            config=VerificationConfig(
                verification_mode="canonical_render",
                render_format=compiled_train_contract.eval_contract.render_format,
                min_score=0.0,
                max_candidates=None,
                min_match_ratio=1.0,
                scan_windows=True,
                require_all_fields=True,
                decode_as_bytes=False,
                apply_rs=False,
            ),
        )
        render_success = render_verification.success
        decoded_units = tuple(int(unit) for unit in render_verification.decoded_units)
    claim_accept = bool(compiled_result.foundation_gate_passed and render_success)
    result = {
        "claim_accept": claim_accept,
        "ownership_score": 1.0 if claim_accept else 0.0,
        "field_valid_rate": compiled_result.field_valid_rate,
        "bucket_correct_rate": compiled_result.bucket_correct_rate,
        "slot_exact_rate": compiled_result.slot_exact_rate,
        "valid_canonical_block_count": compiled_result.valid_canonical_block_count,
        "generated_values": generated_values,
        "chosen_token_ids": chosen_token_ids,
        "expected_units": list(expected_units),
        "decoded_units": list(decoded_units),
        "model_name": model_name,
        "claim_payload": claim_payload,
        "fresh_query_count": len(plan.slot_targets),
    }
    context[cache_key] = result
    return result


def _ours_claim_contract_context(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    model_name: str,
    tokenizer: Any,
    claim_payload: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    cache_key = ("ours_claim_contract_context", model_name, claim_payload)
    if cache_key in context:
        return context[cache_key]

    from src.core.catalog_freeze import load_required_frozen_catalog
    from src.core.contract_compiler import compile_fieldwise_train_contract
    from src.core.payload_codec import BucketPayloadCodec
    from src.core.scaffolded_completion import COMPILED_FIELDWISE_PROMPT_CONTRACT

    catalog_path = _catalog_path(repo_root, cfg)
    payload_labels = _all_payload_labels(cfg)
    backend_cfg = dict(cfg.get("ours_registered_null_backend") or {})
    block_count = int(backend_cfg.get("block_count", 2))
    instruction = str(backend_cfg.get("instruction", "Select exactly one allowed carrier token."))
    compiled_train_contract = compile_fieldwise_train_contract(
        model_name=model_name,
        tokenizer_name=model_name,
        tokenizer_backend=str(backend_cfg.get("tokenizer_backend", "huggingface")),
        catalog_path=catalog_path,
        payload_labels=payload_labels,
        eval_payload_label=claim_payload,
        instruction=instruction,
        block_count=block_count,
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        render_format="canonical_v1",
        tokenizer=tokenizer,
    )
    layout = load_required_frozen_catalog(Path(compiled_train_contract.catalog_path))
    expected_units = tuple(
        int(unit)
        for unit in compiled_train_contract.payload_label_to_units[claim_payload]
    )
    payload = {
        "compiled_train_contract": compiled_train_contract,
        "layout": layout,
        "payload_codec": BucketPayloadCodec(bucket_radices=layout.radices),
        "expected_units": expected_units,
    }
    context[cache_key] = payload
    return payload


def _ours_verify_generated_organic_text(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    model_name: str,
    tokenizer: Any,
    claim_payload: str,
    generated_text: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    cache_key = ("ours_verify_generated_organic_text", model_name, claim_payload, generated_text)
    if cache_key in context:
        return context[cache_key]

    from src.core.scaffolded_completion import (
        COMPILED_ARTIFACT_FORMAT,
        evaluate_foundation_completion,
    )
    from src.core.verifier import VerificationConfig, verify_canonical_rendered_text

    claim_context = _ours_claim_contract_context(
        repo_root=repo_root,
        cfg=cfg,
        model_name=model_name,
        tokenizer=tokenizer,
        claim_payload=claim_payload,
        context=context,
    )
    compiled_train_contract = claim_context["compiled_train_contract"]
    layout = claim_context["layout"]
    compiled_result = evaluate_foundation_completion(
        generated_text,
        layout=layout,
        expected_slot_values=compiled_train_contract.eval_contract.expected_slot_values,
        exact_slot_prefixes=compiled_train_contract.eval_contract.exact_slot_prefixes,
        tokenizer=tokenizer,
        prompt_contract_name=compiled_train_contract.eval_contract.prompt_contract_name,
        render_format=compiled_train_contract.eval_contract.render_format,
        slot_field_names=compiled_train_contract.eval_contract.slot_field_names,
        artifact_format=COMPILED_ARTIFACT_FORMAT,
    )
    render_success = False
    decoded_units: tuple[int, ...] = ()
    if compiled_result.rendered_bucket_tuples:
        render_verification = verify_canonical_rendered_text(
            text=compiled_result.rendered_canonical_text,
            bucket_layout=layout,
            payload_codec=claim_context["payload_codec"],
            expected_payload=claim_context["expected_units"],
            config=VerificationConfig(
                verification_mode="canonical_render",
                render_format=compiled_train_contract.eval_contract.render_format,
                min_score=0.0,
                max_candidates=None,
                min_match_ratio=1.0,
                scan_windows=True,
                require_all_fields=True,
                decode_as_bytes=False,
                apply_rs=False,
            ),
        )
        render_success = render_verification.success
        decoded_units = tuple(int(unit) for unit in render_verification.decoded_units)
    claim_accept = bool(compiled_result.foundation_gate_passed and render_success)
    result = {
        "claim_accept": claim_accept,
        "field_valid_rate": compiled_result.field_valid_rate,
        "bucket_correct_rate": compiled_result.bucket_correct_rate,
        "slot_exact_rate": compiled_result.slot_exact_rate,
        "valid_canonical_block_count": compiled_result.valid_canonical_block_count,
        "decoded_units": list(decoded_units),
        "expected_units": list(claim_context["expected_units"]),
        "rendered_canonical_text": compiled_result.rendered_canonical_text,
    }
    context[cache_key] = result
    return result


def _ours_organic_null_result(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    claim_payload = str(row.get("claim_payload", ""))
    null_model_id = str(row.get("null_model_id", ""))
    budget = max(1, _safe_int(row.get("query_budget"), 1))
    organic_prompt_id = str(row.get("organic_prompt_id", "organic_0000"))
    cache_key = ("ours_organic_null", null_model_id, claim_payload, organic_prompt_id, budget)
    if cache_key in context:
        return context[cache_key]

    null_model = _null_model_cfg(cfg, null_model_id)
    model_name = str(null_model.get("model") or "Qwen/Qwen2.5-7B-Instruct")
    _model, tokenizer, _device, _torch = _load_base_model(
        cfg=cfg,
        model_name=model_name,
        context=context,
    )
    prompt_ids = _organic_prompt_ids_for_budget(
        organic_prompt_id=organic_prompt_id,
        query_budget=budget,
        cfg=cfg,
    )
    prompt_results: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        generation = _generate_organic_text(
            cfg=cfg,
            null_model_id=null_model_id,
            organic_prompt_id=prompt_id,
            context=context,
        )
        verification = _ours_verify_generated_organic_text(
            repo_root=repo_root,
            cfg=cfg,
            model_name=model_name,
            tokenizer=tokenizer,
            claim_payload=claim_payload,
            generated_text=str(generation["generated_text"]),
            context=context,
        )
        prompt_results.append(
            {
                "organic_prompt_id": prompt_id,
                "claim_accept": bool(verification["claim_accept"]),
                "field_valid_rate": verification["field_valid_rate"],
                "bucket_correct_rate": verification["bucket_correct_rate"],
                "slot_exact_rate": verification["slot_exact_rate"],
                "valid_canonical_block_count": verification["valid_canonical_block_count"],
                "generated_text_preview": str(generation["generated_text"])[:240],
            }
        )
    accept_count = sum(1 for item in prompt_results if item["claim_accept"])
    result = {
        "claim_accept": accept_count > 0,
        "ownership_score": accept_count / len(prompt_results) if prompt_results else 0.0,
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt_ids": prompt_ids,
        "claim_payload": claim_payload,
        "model_name": model_name,
        "fresh_query_count": len(prompt_ids),
        "accept_count": accept_count,
        "prompt_results": prompt_results,
        "decision_rule": "accept_if_any_organic_prompt_decodes_claim_payload",
    }
    context[cache_key] = result
    return result


def _load_perinucleus_frozen_candidate(repo_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    cache_key = "_perinucleus_candidate"
    runtime_cache = cfg.setdefault("_runtime_cache", {})
    if cache_key in runtime_cache:
        return runtime_cache[cache_key]
    import yaml as yaml_module

    frozen_path = _method_artifact_path(
        repo_root,
        cfg,
        PERINUCLEUS_METHOD_ID,
        "frozen_candidate_config",
    )
    if frozen_path is None or not frozen_path.exists():
        raise FileNotFoundError(f"Missing Perinucleus frozen candidate config: {frozen_path}")
    frozen = yaml_module.safe_load(frozen_path.read_text(encoding="utf-8")) or {}
    candidate = dict(frozen["candidate"])
    model_cfg = dict(frozen["model"])
    payload = {"candidate": candidate, "model": model_cfg}
    runtime_cache[cache_key] = payload
    return payload


def _perinucleus_registered_null_result(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    claim_payload = str(row.get("claim_payload", ""))
    seed = _safe_int(row.get("seed"))
    budget = _safe_int(row.get("query_budget"))
    null_model_id = str(row.get("null_model_id", ""))
    cache_key = ("perinucleus_registered_null", null_model_id, claim_payload, seed, budget)
    if cache_key in context:
        return context[cache_key]

    from scripts import run_perinucleus_official_final_eval as final_eval
    from scripts import run_perinucleus_official_overfit_gate as overfit

    overfit._load_model_dependencies()
    frozen = _load_perinucleus_frozen_candidate(repo_root, cfg)
    candidate = dict(frozen["candidate"])
    base_model = str(_null_model_cfg(cfg, null_model_id).get("model") or frozen["model"]["base"])
    model, tokenizer, device, _torch = _load_base_model(
        cfg=cfg,
        model_name=base_model,
        context=context,
    )
    fingerprints_path = _resolve(repo_root, candidate["fingerprints_file"])
    if fingerprints_path is None or not fingerprints_path.exists():
        raise FileNotFoundError(f"Missing Perinucleus fingerprints file: {fingerprints_path}")
    rows = final_eval._load_fingerprint_rows(fingerprints_path, int(candidate["num_fingerprints"]))
    selected_rows = final_eval._select_fingerprints(
        rows=rows,
        payload_text=claim_payload,
        seed=seed,
        query_budget=budget,
        arm_id=str(candidate["arm_id"]),
    )
    dataset = overfit._prepare_dataset(
        tokenizer=tokenizer,
        fingerprints=[{"key": item["key"], "response": item["response"]} for item in selected_rows],
        max_key_length=int(candidate.get("key_length", 16)),
        max_response_length=int(candidate.get("response_length", 1)),
        max_length=int(candidate.get("max_sequence_length", 64)),
    )
    for item, selected in zip(dataset, selected_rows, strict=True):
        item["fingerprint_id"] = int(selected["source_fingerprint_id"])
    metrics = overfit._evaluate(model, tokenizer, dataset, device, batch_size=1)
    exact_ratio = float(metrics.get("exact_accuracy") or 0.0)
    claim_accept = exact_ratio >= 1.0
    result = {
        "claim_accept": claim_accept,
        "ownership_score": exact_ratio,
        "exact_response_match_ratio": exact_ratio,
        "exact_response_match_count": int(metrics.get("exact_count") or 0),
        "query_budget": budget,
        "model_name": base_model,
        "claim_payload": claim_payload,
        "seed": seed,
        "fresh_query_count": budget,
        "expected_response_probability_mean": metrics.get("target_probability_mean"),
        "expected_response_probability_min": metrics.get("target_probability_min"),
    }
    context[cache_key] = result
    return result


def _organic_seed_for_row(cfg: dict[str, Any], row: dict[str, Any]) -> int:
    seeds = _seeds(cfg)
    if not seeds:
        return 0
    return seeds[_organic_index(str(row.get("organic_prompt_id", "organic_0000"))) % len(seeds)]


def _perinucleus_organic_generated_ids(
    *,
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: Any,
    torch: Any,
    organic_prompt_id: str,
    candidate: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    max_response_length = int(candidate.get("response_length", 1))
    cache_key = ("perinucleus_organic_generated_ids", organic_prompt_id, max_response_length)
    if cache_key in context:
        return context[cache_key]

    from scripts import run_perinucleus_official_overfit_gate as overfit

    key, _key_ids, key_truncated = overfit._truncate_key(
        tokenizer,
        _organic_prompt_text(organic_prompt_id),
        int(candidate.get("key_length", 16)),
    )
    prefix_ids = overfit._check_prefix_ids(tokenizer, key, strip_eos=True)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prefix_ids.unsqueeze(0).to(device),
            attention_mask=torch.ones_like(prefix_ids.unsqueeze(0), device=device),
            max_new_tokens=max(max_response_length, 1),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0][prefix_ids.numel() :].detach().cpu().tolist()
    result = {
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt": key,
        "key_truncated": key_truncated,
        "generated_token_ids": [int(token_id) for token_id in generated],
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    context[cache_key] = result
    return result


def _perinucleus_organic_null_result(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    claim_payload = str(row.get("claim_payload", ""))
    budget = max(1, _safe_int(row.get("query_budget"), 1))
    organic_prompt_id = str(row.get("organic_prompt_id", "organic_0000"))
    selected_seed = _organic_seed_for_row(cfg, row)
    cache_key = ("perinucleus_organic_null", claim_payload, organic_prompt_id, budget, selected_seed)
    if cache_key in context:
        return context[cache_key]

    from scripts import run_perinucleus_official_final_eval as final_eval
    from scripts import run_perinucleus_official_overfit_gate as overfit

    frozen = _load_perinucleus_frozen_candidate(repo_root, cfg)
    candidate = dict(frozen["candidate"])
    base_model = str(_null_model_cfg(cfg, "base_qwen").get("model") or frozen["model"]["base"])
    model, tokenizer, device, torch = _load_base_model(
        cfg=cfg,
        model_name=base_model,
        context=context,
    )
    fingerprints_path = _resolve(repo_root, candidate["fingerprints_file"])
    if fingerprints_path is None or not fingerprints_path.exists():
        raise FileNotFoundError(f"Missing Perinucleus fingerprints file: {fingerprints_path}")
    rows = final_eval._load_fingerprint_rows(fingerprints_path, int(candidate["num_fingerprints"]))
    selected_rows = final_eval._select_fingerprints(
        rows=rows,
        payload_text=claim_payload,
        seed=selected_seed,
        query_budget=budget,
        arm_id=str(candidate["arm_id"]),
    )
    prompt_ids = _organic_prompt_ids_for_budget(
        organic_prompt_id=organic_prompt_id,
        query_budget=budget,
        cfg=cfg,
    )
    response_rows: list[dict[str, Any]] = []
    exact_count = 0
    for prompt_id, selected in zip(prompt_ids, selected_rows, strict=True):
        response, response_ids, response_truncated = overfit._truncate_response(
            tokenizer,
            str(selected["response"]),
            int(candidate.get("response_length", 1)),
        )
        generation = _perinucleus_organic_generated_ids(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            device=device,
            torch=torch,
            organic_prompt_id=prompt_id,
            candidate=candidate,
            context=context,
        )
        generated_ids = [int(token_id) for token_id in generation["generated_token_ids"]]
        token_exact = generated_ids[: len(response_ids)] == [int(token_id) for token_id in response_ids]
        exact_count += int(token_exact)
        response_rows.append(
            {
                "organic_prompt_id": prompt_id,
                "source_fingerprint_id": int(selected["source_fingerprint_id"]),
                "target_response": response,
                "target_response_ids": [int(token_id) for token_id in response_ids],
                "response_truncated": response_truncated,
                "generated_token_ids": generated_ids,
                "generated_text": str(generation["generated_text"])[:120],
                "token_exact": token_exact,
            }
        )
    exact_ratio = exact_count / len(response_rows) if response_rows else 0.0
    result = {
        "claim_accept": exact_ratio >= 1.0,
        "ownership_score": exact_ratio,
        "exact_response_match_ratio": exact_ratio,
        "exact_response_match_count": exact_count,
        "query_budget": budget,
        "model_name": base_model,
        "claim_payload": claim_payload,
        "selected_seed": selected_seed,
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt_ids": prompt_ids,
        "fresh_query_count": len(prompt_ids),
        "response_rows": response_rows,
        "decision_rule": "all_selected_fingerprint_responses_must_match_on_organic_prompt_window",
    }
    context[cache_key] = result
    return result


def _execute_ours_artifact_row(
    row: dict[str, Any],
    indexes: dict[str, Any],
    repo_root: Path,
    cfg: dict[str, Any],
    fresh_null_mode: str,
    context: dict[str, Any],
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

    if (
        evaluation_type == "null_model_registered_probes"
        and _fresh_mode_enabled(fresh_null_mode, "registered-probes")
    ):
        output = _execution_base_row(row)
        if str(row.get("null_model_id", "")) != "base_qwen":
            return _set_pending(
                output,
                status="not_executed_optional_null_model_not_enabled",
                reason="registered-probes fresh backend currently executes required base_qwen only",
                execution_scope="optional_null_model_pending",
            )
        result = _ours_registered_null_result(
            repo_root=repo_root,
            cfg=cfg,
            row=row,
            context=context,
        )
        return _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            execution_scope="ours_base_qwen_registered_probe_fresh_null",
            source_status="fresh_base_qwen_registered_probe_completed",
            notes_suffix="fresh base-Qwen registered probe null execution",
            details=result,
        )

    if (
        evaluation_type == "null_model_registered_probes"
        and str(row.get("null_model_id", "")) != "base_qwen"
        and fresh_null_mode != "off"
    ):
        return _set_pending(
            output,
            status="not_executed_optional_null_model_not_enabled",
            reason="fresh backend currently executes required base_qwen null model only",
            execution_scope="optional_null_model_pending",
        )

    if (
        evaluation_type == "organic_prompt_null"
        and _fresh_mode_enabled(fresh_null_mode, "organic-prompts")
    ):
        output = _execution_base_row(row)
        if str(row.get("null_model_id", "")) != "base_qwen":
            return _set_pending(
                output,
                status="not_executed_optional_null_model_not_enabled",
                reason="organic-prompts fresh backend currently executes required base_qwen only",
                execution_scope="optional_null_model_pending",
            )
        result = _ours_organic_null_result(
            repo_root=repo_root,
            cfg=cfg,
            row=row,
            context=context,
        )
        return _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            status=FRESH_ORGANIC_NULL_STATUS,
            execution_backend="fresh_organic_prompt_null_v1",
            execution_scope="ours_base_qwen_organic_prompt_fresh_null",
            source_status="fresh_base_qwen_organic_prompt_completed",
            notes_suffix="fresh base-Qwen organic prompt null execution",
            details=result,
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
    row: dict[str, Any],
    indexes: dict[str, Any],
    repo_root: Path,
    cfg: dict[str, Any],
    fresh_null_mode: str,
    context: dict[str, Any],
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

    if (
        evaluation_type == "null_model_registered_probes"
        and _fresh_mode_enabled(fresh_null_mode, "registered-probes")
    ):
        output = _execution_base_row(row)
        if str(row.get("null_model_id", "")) != "base_qwen":
            return _set_pending(
                output,
                status="not_executed_optional_null_model_not_enabled",
                reason="registered-probes fresh backend currently executes required base_qwen only",
                execution_scope="optional_null_model_pending",
            )
        result = _perinucleus_registered_null_result(
            repo_root=repo_root,
            cfg=cfg,
            row=row,
            context=context,
        )
        return _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            execution_scope="perinucleus_base_qwen_registered_probe_fresh_null",
            source_status="fresh_base_qwen_registered_probe_completed",
            notes_suffix="fresh base-Qwen registered fingerprint null execution",
            details=result,
        )

    if (
        evaluation_type == "null_model_registered_probes"
        and str(row.get("null_model_id", "")) != "base_qwen"
        and fresh_null_mode != "off"
    ):
        return _set_pending(
            output,
            status="not_executed_optional_null_model_not_enabled",
            reason="fresh backend currently executes required base_qwen null model only",
            execution_scope="optional_null_model_pending",
        )

    if (
        evaluation_type == "organic_prompt_null"
        and _fresh_mode_enabled(fresh_null_mode, "organic-prompts")
    ):
        output = _execution_base_row(row)
        if str(row.get("null_model_id", "")) != "base_qwen":
            return _set_pending(
                output,
                status="not_executed_optional_null_model_not_enabled",
                reason="organic-prompts fresh backend currently executes required base_qwen only",
                execution_scope="optional_null_model_pending",
            )
        result = _perinucleus_organic_null_result(
            repo_root=repo_root,
            cfg=cfg,
            row=row,
            context=context,
        )
        return _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            status=FRESH_ORGANIC_NULL_STATUS,
            execution_backend="fresh_organic_prompt_null_v1",
            execution_scope="perinucleus_base_qwen_organic_prompt_fresh_null",
            source_status="fresh_base_qwen_organic_prompt_completed",
            notes_suffix="fresh base-Qwen organic prompt null execution",
            details=result,
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
    *,
    fresh_null_mode: str = "off",
) -> list[dict[str, Any]]:
    indexes = _load_artifact_indexes(repo_root, cfg)
    context: dict[str, Any] = {}
    executed: list[dict[str, Any]] = []
    for row in rows:
        executed.append(
            _execute_one_plan_row(
                row=row,
                indexes=indexes,
                repo_root=repo_root,
                cfg=cfg,
                fresh_null_mode=fresh_null_mode,
                context=context,
            )
        )
    return executed


def _execute_one_plan_row(
    *,
    row: dict[str, Any],
    indexes: dict[str, Any],
    repo_root: Path,
    cfg: dict[str, Any],
    fresh_null_mode: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    method_id = str(row.get("method_id", ""))
    if method_id == OURS_METHOD_ID:
        return _execute_ours_artifact_row(
            row,
            indexes,
            repo_root,
            cfg,
            fresh_null_mode,
            context,
        )
    if method_id == PERINUCLEUS_METHOD_ID:
        return _execute_perinucleus_artifact_row(
            row,
            indexes,
            repo_root,
            cfg,
            fresh_null_mode,
            context,
        )
    return _set_pending(
        _execution_base_row(row),
        status="not_executed_unknown_method",
        reason=f"no execution backend registered for method_id={method_id}",
        execution_scope="method_backend_missing",
    )


def _completed_rows_for_resume(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = _read_csv(path)
    except Exception as error:
        raise RuntimeError(f"Cannot resume from possibly corrupted shard CSV {path}: {error}") from error
    return {
        str(row["case_id"]): row
        for row in rows
        if str(row.get("case_id", ""))
        and str(row.get("status", "")).startswith("completed_")
        and str(row.get("claim_accept", "")) != ""
    }


def execute_shard_rows_checkpointed(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    config_path: Path,
    global_plan_rows: list[dict[str, Any]],
    shard_plan_rows: list[dict[str, Any]],
    fresh_null_mode: str,
    shard_table_path: Path,
    shard_summary_path: Path,
    checkpoint_interval: int,
    shard_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexes = _load_artifact_indexes(repo_root, cfg)
    context: dict[str, Any] = {}
    completed_by_case_id = _completed_rows_for_resume(shard_table_path)
    selected_case_ids = {str(row["case_id"]) for row in shard_plan_rows}
    completed_by_case_id = {
        case_id: row
        for case_id, row in completed_by_case_id.items()
        if case_id in selected_case_ids
    }
    remaining_rows = [
        row
        for row in shard_plan_rows
        if str(row.get("case_id", "")) not in completed_by_case_id
    ]
    print(
        json.dumps(
            {
                "event": "shard_resume_state",
                "completed": len(completed_by_case_id),
                "remaining": len(remaining_rows),
                "selected": len(shard_plan_rows),
                **shard_metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    newly_completed = 0

    def checkpoint(event: str) -> dict[str, Any]:
        ordered_rows = [
            completed_by_case_id[str(row["case_id"])]
            for row in shard_plan_rows
            if str(row["case_id"]) in completed_by_case_id
        ]
        summary = _build_execution_summary(
            repo_root,
            cfg,
            config_path,
            shard_plan_rows,
            ordered_rows,
        )
        summary["shard_scope_status"] = summary["status"]
        summary["status"] = "completed_shard_subset"
        summary["full_far_complete"] = False
        summary["shard"] = {
            **shard_metadata,
            "selected_case_count": len(shard_plan_rows),
            "completed_case_count": len(ordered_rows),
            "remaining_case_count": len(shard_plan_rows) - len(ordered_rows),
            "global_plan_case_count": len(global_plan_rows),
            "checkpoint_event": event,
            "note": "Shard outputs are incomplete by design; aggregate all shards before interpreting FAR.",
        }
        _write_csv_atomic(shard_table_path, ordered_rows)
        _write_json_atomic(shard_summary_path, summary)
        print(
            json.dumps(
                {
                    "event": event,
                    "completed": len(ordered_rows),
                    "remaining": len(shard_plan_rows) - len(ordered_rows),
                    "selected": len(shard_plan_rows),
                    **shard_metadata,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return summary

    summary = checkpoint("checkpoint_initial")
    interval = max(1, int(checkpoint_interval or 1))
    for row in remaining_rows:
        executed = _execute_one_plan_row(
            row=row,
            indexes=indexes,
            repo_root=repo_root,
            cfg=cfg,
            fresh_null_mode=fresh_null_mode,
            context=context,
        )
        completed_by_case_id[str(row["case_id"])] = executed
        newly_completed += 1
        if newly_completed % interval == 0:
            summary = checkpoint("checkpoint_progress")
    summary = checkpoint("checkpoint_final")
    ordered_final_rows = [
        completed_by_case_id[str(row["case_id"])]
        for row in shard_plan_rows
        if str(row["case_id"]) in completed_by_case_id
    ]
    return ordered_final_rows, summary


def _validate_shard_args(args: argparse.Namespace) -> None:
    shard_values = [args.shard_index is not None, args.shard_count is not None]
    if any(shard_values) and not all(shard_values):
        raise ValueError("--shard-index and --shard-count must be provided together.")
    if args.shard_index is None and args.shard_count is None:
        return
    if not args.execute:
        raise ValueError("Sharded execution is only valid with --execute.")
    if args.fresh_null_mode == "off":
        raise ValueError("Sharded execution requires a fresh null mode.")
    if args.shard_count is None or args.shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if args.shard_index is None or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard_count.")
    if not args.shard_output_dir:
        raise ValueError("--shard-output-dir is required for sharded execution.")


def _is_sharded_execution(args: argparse.Namespace) -> bool:
    return args.shard_index is not None and args.shard_count is not None


def _organic_shard_bounds(cfg: dict[str, Any], shard_index: int, shard_count: int) -> tuple[int, int]:
    organic_count = int((cfg.get("null_protocol") or {}).get("organic_prompt_count", 0))
    if organic_count <= 0:
        raise ValueError("null_protocol.organic_prompt_count must be positive for organic sharding.")
    start = (organic_count * shard_index) // shard_count
    end = (organic_count * (shard_index + 1)) // shard_count
    return start, end


def _row_selected_for_shard(
    row: dict[str, Any],
    *,
    cfg: dict[str, Any],
    fresh_null_mode: str,
    shard_index: int,
    shard_count: int,
) -> bool:
    evaluation_type = str(row.get("evaluation_type", ""))
    if (
        evaluation_type == "organic_prompt_null"
        and _fresh_mode_enabled(fresh_null_mode, "organic-prompts")
    ):
        if str(row.get("null_model_id", "")) != "base_qwen":
            return False
        start, end = _organic_shard_bounds(cfg, shard_index, shard_count)
        organic_index = _organic_index(str(row.get("organic_prompt_id", "organic_0000")))
        return start <= organic_index < end
    if (
        evaluation_type == "null_model_registered_probes"
        and _fresh_mode_enabled(fresh_null_mode, "registered-probes")
    ):
        if str(row.get("null_model_id", "")) != "base_qwen":
            return False
        return (_safe_int(row.get("case_index")) - 1) % shard_count == shard_index
    return False


def _select_shard_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    fresh_null_mode: str,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if _row_selected_for_shard(
            row,
            cfg=cfg,
            fresh_null_mode=fresh_null_mode,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    ]
    if not selected:
        raise ValueError(
            f"Shard {shard_index}/{shard_count} selected no rows for fresh_null_mode={fresh_null_mode}."
        )
    return selected


def _shard_output_paths(
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    if args.shard_index is None or args.shard_count is None or not args.shard_output_dir:
        raise ValueError("Shard output paths require shard index/count/output-dir.")
    shard_dir = _resolve(repo_root, args.shard_output_dir)
    if shard_dir is None:
        raise ValueError("--shard-output-dir is required.")
    mode = str(args.fresh_null_mode).replace("-", "_")
    stem = f"full_far_payload_claim_{mode}_shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    return {
        "summary": shard_dir / f"{stem}.json",
        "table": shard_dir / f"{stem}.csv",
    }


def _merge_existing_completed_rows(
    executed_rows: list[dict[str, Any]],
    existing_table_path: Path | None,
) -> list[dict[str, Any]]:
    if existing_table_path is None or not existing_table_path.exists():
        return executed_rows
    existing_rows = _read_csv(existing_table_path)
    reusable_by_case_id = {
        str(row.get("case_id", "")): row
        for row in existing_rows
        if str(row.get("case_id", ""))
        and str(row.get("status", "")).startswith("completed_")
        and str(row.get("claim_accept", "")) != ""
    }
    if not reusable_by_case_id:
        return executed_rows
    merged: list[dict[str, Any]] = []
    for row in executed_rows:
        if str(row.get("claim_accept", "")) != "":
            merged.append(row)
            continue
        existing = reusable_by_case_id.get(str(row.get("case_id", "")))
        if existing is None:
            merged.append(row)
            continue
        # Preserve fresh rows from an earlier slice so execute-organic-null does
        # not erase execute-registered-null results, and vice versa.
        preserved = dict(row)
        preserved.update(existing)
        preserved["preserved_from_existing_final_table"] = True
        merged.append(preserved)
    return merged


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
    fresh_model_inference_performed = any(
        _truthy(row.get("fresh_model_inference_performed"))
        for row in executed_rows
    )
    status_counts = _status_counts(executed_rows)
    if full_far_complete:
        status = "completed_full_far"
    elif (
        status_counts.get(FRESH_REGISTERED_NULL_STATUS, 0) > 0
        and status_counts.get(FRESH_ORGANIC_NULL_STATUS, 0) > 0
    ):
        status = "completed_registered_and_organic_null_subset"
    elif status_counts.get(FRESH_ORGANIC_NULL_STATUS, 0) > 0:
        status = "completed_organic_null_subset"
    elif status_counts.get(FRESH_REGISTERED_NULL_STATUS, 0) > 0:
        status = "completed_registered_null_subset"
    elif fresh_model_inference_performed:
        status = "completed_fresh_null_subset"
    else:
        status = "completed_artifact_subset"
    return {
        "schema_name": "full_far_payload_claim_execution_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "status": status,
        "execution_backend": "artifact_replay_v1_with_optional_fresh_null_backends_v1",
        "fresh_model_inference_performed": fresh_model_inference_performed,
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
                "non-owner and optional null-model rows remain pending until "
                "their fresh inference prompt banks are implemented"
            ),
            "do not use this summary to claim complete FAR superiority",
        ],
        "next_required_implementation": [
            "fresh inference backend for non_owner_probe_null",
            "fresh null-model registered-probe backend for optional null models",
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
    _validate_shard_args(args)
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = _load_yaml(config_path)
    rows = build_plan_rows(cfg)
    summary = build_summary(repo_root, cfg, config_path, rows)

    if args.execute:
        execution_plan_rows = rows
        if _is_sharded_execution(args):
            execution_plan_rows = _select_shard_rows(
                rows,
                cfg=cfg,
                fresh_null_mode=args.fresh_null_mode,
                shard_index=int(args.shard_index),
                shard_count=int(args.shard_count),
            )
        outputs = cfg.get("outputs") or {}
        summary_path = _resolve(repo_root, outputs.get("final_summary"))
        table_path = _resolve(repo_root, outputs.get("final_table"))
        tex_path = _resolve(repo_root, outputs.get("final_tex"))
        if summary_path is None or table_path is None or tex_path is None:
            raise ValueError(
                "outputs.final_summary, outputs.final_table, and outputs.final_tex are required."
            )
        if _is_sharded_execution(args):
            shard_paths = _shard_output_paths(repo_root, args)
            shard_metadata = {
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "fresh_null_mode": args.fresh_null_mode,
            }
            executed_rows, execution_summary = execute_shard_rows_checkpointed(
                repo_root=repo_root,
                cfg=cfg,
                config_path=config_path,
                global_plan_rows=rows,
                shard_plan_rows=execution_plan_rows,
                fresh_null_mode=args.fresh_null_mode,
                shard_table_path=shard_paths["table"],
                shard_summary_path=shard_paths["summary"],
                checkpoint_interval=args.checkpoint_interval,
                shard_metadata=shard_metadata,
            )
            print(
                json.dumps(
                    {
                        "status": "shard_completed",
                        "execution_status": execution_summary["status"],
                        "summary": str(shard_paths["summary"]),
                        "table": str(shard_paths["table"]),
                        "selected_case_count": len(executed_rows),
                        "global_plan_case_count": len(rows),
                        "fresh_null_mode": args.fresh_null_mode,
                        "shard_index": int(args.shard_index),
                        "shard_count": int(args.shard_count),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        executed_rows = execute_plan_rows(
            repo_root,
            cfg,
            execution_plan_rows,
            fresh_null_mode=args.fresh_null_mode,
        )
        executed_rows = _merge_existing_completed_rows(executed_rows, table_path)
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
                    "fresh_model_inference_performed": execution_summary[
                        "fresh_model_inference_performed"
                    ],
                    "fresh_null_mode": args.fresh_null_mode,
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
