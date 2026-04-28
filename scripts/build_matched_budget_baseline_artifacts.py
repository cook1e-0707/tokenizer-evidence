from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_matched_budget_baselines import (
    _case_records,
    _calibration_rows,
    _load_frozen_thresholds,
)
from src.evaluation.report import EvalRunSummary, TrainRunSummary, maybe_load_result_json
from src.infrastructure.paths import current_timestamp, discover_repo_root


CONTRACT_HASH_FIELDS = [
    "model_id_hash",
    "tokenizer_id_hash",
    "block_count_hash",
    "fields_per_block_hash",
    "field_order_hash",
    "codebook_hash",
    "bucket_partition_hash",
    "payload_map_hash",
    "train_contract_hash",
    "eval_contract_hash",
    "prompt_family_hash",
    "generation_config_hash",
    "verifier_contract_hash",
    "rs_config_hash",
    "adapter_checkpoint_hash",
]

RUN_FIELDS = [
    "case_id",
    "b_stage",
    "method_id",
    "method_slug",
    "method_name",
    "display_name",
    "baseline_family",
    "baseline_role",
    "train_objective",
    "matched_budget_status",
    "requires_training",
    "requires_external_integration",
    "paper_ready_denominator",
    "block_count",
    "payload",
    "seed",
    "query_budget",
    "queries_used",
    "target_far",
    "frozen_threshold",
    "calibration_observed_far",
    "utility_acceptance_rate",
    "ownership_score",
    "accepted",
    "verifier_success",
    "decoded_payload",
    "status",
    "result_class",
    "failure_reasons",
    "valid_completed",
    "success",
    "method_failure",
    "invalid_excluded",
    "pending",
    "unavailable",
    "contract_hash_status",
    "contract_hash_missing_fields",
    "contract_hash_mismatch_fields",
    *CONTRACT_HASH_FIELDS,
    "run_dir",
    "case_root",
    "eval_summary_path",
    "train_summary_path",
    "config_path",
]

CALIBRATION_FIELDS = [
    "method_id",
    "method_slug",
    "method_name",
    "baseline_family",
    "baseline_role",
    "status",
    "score_name",
    "score_direction",
    "target_far",
    "frozen_threshold",
    "calibration_observed_far",
    "calibration_payloads",
    "calibration_seed",
    "negative_sets",
    "requires_external_integration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build B1/B2 matched-budget baseline artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/matched_budget_baselines_v1.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--case-root-base",
        help="Optional base directory for baseline case roots. Defaults to EXP_SCRATCH/matched_budget_baselines_v1.",
    )
    parser.add_argument(
        "--calibration-summary",
        default="results/processed/paper_stats/baseline_calibration_summary.json",
        help="Frozen B0 calibration summary used by final baseline artifacts.",
    )
    parser.add_argument("--audit-doc", default="docs/baseline_artifact_audit.md")
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str | None:
    if path.is_file():
        return _file_sha256(path)
    if not path.exists() or not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(file for file in path.rglob("*") if file.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _hash_fields_empty() -> dict[str, str]:
    return {field: "" for field in CONTRACT_HASH_FIELDS}


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    return prefix


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Success & Method fail & Invalid & Pending & FAR target \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        method = str(row["method_slug"]).replace("_", "\\_")
        lines.append(
            f"{method} & {row['success_count']} & {row['method_failure_count']} & "
            f"{row['invalid_excluded_count']} & {row['pending_count']} & "
            f"{float(row['target_far']):.2f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{B1/B2 matched-budget baseline package under frozen B0 calibration rules. Pending and unavailable rows are not reported as method failures.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_doc(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Artifact Audit",
        "",
        f"Generated at: `{summary['generated_at']}`.",
        "",
        "## Accounting",
        "",
        f"- Paper-ready denominator target cases: `{summary['target_count']}`.",
        f"- Reporting rows: `{summary['reporting_row_count']}`.",
        f"- Valid completed denominator cases: `{summary['valid_completed_count']}`.",
        f"- Successes: `{summary['success_count']}`.",
        f"- Method failures: `{summary['method_failure_count']}`.",
        f"- Invalid exclusions: `{summary['invalid_excluded_count']}`.",
        f"- Pending denominator cases: `{summary['pending_count']}`.",
        f"- Task-mismatched unavailable controls: `{summary['control_unavailable_count']}`.",
        f"- Paper ready: `{summary['paper_ready']}`.",
        "",
        "## Contract Hash Gate",
        "",
        f"- Contract hash status counts: `{summary['contract_hash_status_counts']}`.",
    ]
    if summary["invalid_excluded_case_ids"]:
        lines.append("- Invalid excluded cases:")
        lines.extend(f"  - `{case_id}`" for case_id in summary["invalid_excluded_case_ids"])
    else:
        lines.append("- No invalid exclusions are recorded.")
    lines.extend(
        [
            "",
            "## Control Semantics",
            "",
            "`kgw_provenance_control` is a task-mismatched provenance control and is not part of the paper-ready ownership denominator unless a real adapter is integrated and audited.",
            "",
            "`english_random_active_fingerprint` is part of the ownership denominator once its executable adapter eval summaries are present.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime if item.exists() else 0)
    return matches[-1] if matches else None


def _summary_paths(case_root: Path) -> tuple[Path | None, Path | None]:
    train_summary = _find_latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary = _find_latest(case_root, "runs/exp_eval/*/eval_summary.json")
    return train_summary, eval_summary


def _load_resolved_config_hash(run_dir: Path | None, section: str) -> str | None:
    if run_dir is None:
        return None
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        return None
    config = _load_yaml(config_path)
    section_payload = config.get(section, {}) if isinstance(config.get(section, {}), dict) else {}
    if section == "train":
        payload = {
            key: section_payload.get(key)
            for key in (
                "objective",
                "generation_prompt",
                "generation_do_sample",
                "generation_max_new_tokens",
                "generation_stop_strings",
                "generation_bad_words",
                "generation_suppress_tokens",
                "generation_sequence_bias",
            )
        }
    elif section == "eval":
        payload = {
            key: section_payload.get(key)
            for key in (
                "verification_mode",
                "render_format",
                "payload_text",
                "min_score",
                "target_far",
                "expected_payload_source",
            )
        }
    else:
        payload = section_payload
    return _stable_hash(payload)


def _bucket_partition_hash(train_contract: dict[str, Any]) -> str | None:
    samples = train_contract.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    buckets: dict[str, dict[str, list[int]]] = {}
    for item in samples:
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field_name", "missing"))
        raw = item.get("bucket_to_token_ids", {})
        if not isinstance(raw, dict):
            continue
        buckets.setdefault(field_name, {})
        for bucket_id, token_ids in raw.items():
            buckets[field_name][str(bucket_id)] = [int(token_id) for token_id in token_ids]
    return _stable_hash(buckets) if buckets else None


def _field_order(eval_contract: dict[str, Any]) -> list[str] | None:
    raw_fields = eval_contract.get("slot_field_names")
    if not isinstance(raw_fields, list) or not raw_fields:
        return None
    ordered: list[str] = []
    for field in raw_fields:
        value = str(field)
        if value not in ordered:
            ordered.append(value)
    return ordered


def _compare_equal(field: str, values: list[Any], mismatches: list[str]) -> None:
    comparable = [value for value in values if value is not None]
    if len(comparable) > 1 and any(value != comparable[0] for value in comparable[1:]):
        mismatches.append(field)


def _finalize_contract_audit(missing: list[str], mismatches: list[str], hashes: dict[str, str]) -> dict[str, Any]:
    missing_unique = sorted(set(missing))
    mismatches_unique = sorted(set(mismatches))
    status = "match"
    if missing_unique:
        status = "missing_hash"
    elif mismatches_unique:
        status = "mismatch"
    return {
        "contract_hash_status": status,
        "contract_hash_missing_fields": ";".join(missing_unique),
        "contract_hash_mismatch_fields": ";".join(mismatches_unique),
        **hashes,
    }


def _compiled_contract_audit(
    *,
    case: dict[str, Any],
    train_summary_path: Path | None,
    eval_summary_path: Path,
    eval_summary: EvalRunSummary,
) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: list[str] = []
    hashes = _hash_fields_empty()
    train_summary = maybe_load_result_json(train_summary_path) if train_summary_path else None
    if not isinstance(train_summary, TrainRunSummary):
        missing.append("train_summary")
        return _finalize_contract_audit(missing, mismatches, hashes)

    train_run_dir = train_summary_path.parent
    eval_run_dir = eval_summary_path.parent
    train_contract_path = train_run_dir / "compiled_train_contract.json"
    train_eval_contract_path = train_run_dir / "compiled_eval_contract.json"
    latest_eval_input_path = train_run_dir.parent / "latest_eval_input.json"
    if not latest_eval_input_path.exists():
        legacy_eval_input_path = train_run_dir / "latest_eval_input.json"
        if legacy_eval_input_path.exists():
            latest_eval_input_path = legacy_eval_input_path
    train_contract = _read_json(train_contract_path)
    train_eval_contract = _read_json(train_eval_contract_path)
    eval_input = _read_json(latest_eval_input_path)
    diagnostics = eval_summary.diagnostics if isinstance(eval_summary.diagnostics, dict) else {}
    eval_contract = diagnostics.get("compiled_eval_contract") if isinstance(diagnostics.get("compiled_eval_contract"), dict) else {}

    if not train_contract:
        missing.append("compiled_train_contract.json")
    if not train_eval_contract:
        missing.append("compiled_eval_contract.json")
    if not eval_input:
        missing.append("latest_eval_input.json")
    if not eval_contract:
        missing.append("eval_summary.diagnostics.compiled_eval_contract")

    model_id = train_contract.get("model_name") if train_contract else None
    tokenizer_id = train_contract.get("tokenizer_name") if train_contract else None
    block_count_values = [
        case["block_count"],
        train_contract.get("block_count") if train_contract else None,
        train_eval_contract.get("block_count") if train_eval_contract else None,
        eval_contract.get("block_count") if eval_contract else None,
    ]
    fields_per_block_values = [
        train_contract.get("fields_per_block") if train_contract else None,
        train_eval_contract.get("fields_per_block") if train_eval_contract else None,
        eval_contract.get("fields_per_block") if eval_contract else None,
    ]
    field_order_values = [
        _field_order(train_eval_contract) if train_eval_contract else None,
        _field_order(eval_contract) if eval_contract else None,
    ]
    prompt_contract_names = [
        train_contract.get("prompt_contract_name") if train_contract else None,
        train_eval_contract.get("prompt_contract_name") if train_eval_contract else None,
        eval_contract.get("prompt_contract_name") if eval_contract else None,
    ]
    _compare_equal("model_id", [model_id, train_summary.model_name, eval_summary.model_name], mismatches)
    _compare_equal("block_count", block_count_values, mismatches)
    _compare_equal("fields_per_block", fields_per_block_values, mismatches)
    _compare_equal("field_order", field_order_values, mismatches)
    _compare_equal("prompt_family", prompt_contract_names, mismatches)

    for field, value in {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "block_count": None if any(value is None for value in block_count_values) else block_count_values[0],
        "fields_per_block": (
            None if any(value is None for value in fields_per_block_values) else fields_per_block_values[0]
        ),
        "field_order": None if any(value is None for value in field_order_values) else field_order_values[0],
        "prompt_family": None if any(value is None for value in prompt_contract_names) else prompt_contract_names[0],
    }.items():
        if value is None:
            missing.append(field)

    codebook_hash = train_contract.get("catalog_sha256") if train_contract else None
    bucket_partition_hash = _bucket_partition_hash(train_contract) if train_contract else None
    payload_map = train_contract.get("payload_label_to_units") if train_contract else None
    payload_map_hash = _stable_hash(payload_map) if isinstance(payload_map, dict) else None
    if isinstance(payload_map, dict) and case["payload"] in payload_map and eval_contract:
        if list(payload_map[case["payload"]]) != list(eval_contract.get("payload_units", [])):
            mismatches.append("payload_map")
    elif payload_map is None:
        missing.append("payload_label_to_units")
    else:
        missing.append("payload_map_eval_payload")

    train_hash_values = [
        train_contract.get("contract_hash") if train_contract else None,
        eval_input.get("compiled_train_contract_hash") if eval_input else None,
        diagnostics.get("compiled_train_contract_hash") if diagnostics else None,
    ]
    eval_contract_hash_values = [
        _stable_hash(train_eval_contract) if train_eval_contract else None,
        _stable_hash(eval_input.get("compiled_eval_contract")) if isinstance(eval_input.get("compiled_eval_contract"), dict) else None,
        _stable_hash(eval_contract) if eval_contract else None,
    ]
    _compare_equal("train_contract_hash", train_hash_values, mismatches)
    _compare_equal("eval_contract_hash", eval_contract_hash_values, mismatches)
    if any(value is None for value in train_hash_values):
        missing.append("train_contract_hash")
    if any(value is None for value in eval_contract_hash_values):
        missing.append("eval_contract_hash")

    generation_config_hash = _load_resolved_config_hash(train_run_dir, "train")
    verifier_contract_hash = _stable_hash(
        {
            "verification_mode": eval_summary.verification_mode,
            "render_format": eval_summary.render_format,
            "threshold": eval_summary.threshold,
            "compiled_eval_contract": eval_contract,
        }
    ) if eval_contract else None
    rs_config_hash = _stable_hash({"apply_rs": False, "verification_mode": eval_summary.verification_mode})
    checkpoint_path_raw = eval_input.get("checkpoint_path") or diagnostics.get("checkpoint_path")
    checkpoint_path = Path(str(checkpoint_path_raw)) if checkpoint_path_raw else None
    adapter_checkpoint_hash = _path_sha256(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and not checkpoint_path.is_absolute():
        adapter_checkpoint_hash = _path_sha256(train_run_dir / checkpoint_path)

    for field, value in {
        "codebook_hash": codebook_hash,
        "bucket_partition_hash": bucket_partition_hash,
        "payload_map_hash": payload_map_hash,
        "generation_config_hash": generation_config_hash,
        "verifier_contract_hash": verifier_contract_hash,
        "rs_config_hash": rs_config_hash,
        "adapter_checkpoint_hash": adapter_checkpoint_hash,
    }.items():
        if value is None:
            missing.append(field)

    hashes.update(
        {
            "model_id_hash": _stable_hash(model_id) if model_id is not None else "",
            "tokenizer_id_hash": _stable_hash(tokenizer_id) if tokenizer_id is not None else "",
            "block_count_hash": _stable_hash(block_count_values[0]) if block_count_values[0] is not None else "",
            "fields_per_block_hash": (
                _stable_hash(fields_per_block_values[0]) if fields_per_block_values[0] is not None else ""
            ),
            "field_order_hash": _stable_hash(field_order_values[0]) if field_order_values[0] is not None else "",
            "codebook_hash": str(codebook_hash or ""),
            "bucket_partition_hash": str(bucket_partition_hash or ""),
            "payload_map_hash": str(payload_map_hash or ""),
            "train_contract_hash": str(train_hash_values[0] or ""),
            "eval_contract_hash": str(eval_contract_hash_values[0] or ""),
            "prompt_family_hash": str(train_contract.get("prompt_contract_hash", "") if train_contract else ""),
            "generation_config_hash": str(generation_config_hash or ""),
            "verifier_contract_hash": str(verifier_contract_hash or ""),
            "rs_config_hash": str(rs_config_hash or ""),
            "adapter_checkpoint_hash": str(adapter_checkpoint_hash or ""),
        }
    )
    if not hashes["prompt_family_hash"]:
        missing.append("prompt_family_hash")
    return _finalize_contract_audit(missing, mismatches, hashes)


def _baseline_adapter_contract_audit(
    *,
    case: dict[str, Any],
    eval_summary_path: Path,
    eval_summary: EvalRunSummary,
) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: list[str] = []
    hashes = _hash_fields_empty()
    diagnostics = eval_summary.diagnostics if isinstance(eval_summary.diagnostics, dict) else {}
    eval_run_dir = eval_summary_path.parent
    resolved_config = _load_yaml(eval_run_dir / "config.resolved.yaml") if (eval_run_dir / "config.resolved.yaml").exists() else {}
    model = resolved_config.get("model", {}) if isinstance(resolved_config.get("model", {}), dict) else {}
    eval_config = resolved_config.get("eval", {}) if isinstance(resolved_config.get("eval", {}), dict) else {}
    baseline_contract = diagnostics.get("baseline_contract") if isinstance(diagnostics.get("baseline_contract"), dict) else {}
    baseline_contract_hash = diagnostics.get("baseline_contract_hash")

    _compare_equal("method_name", [case["method_name"], eval_summary.method_name], mismatches)
    _compare_equal("seed", [case["seed"], eval_summary.seed], mismatches)
    _compare_equal("payload", [case["payload"], diagnostics.get("payload_text"), eval_config.get("payload_text")], mismatches)
    frozen_threshold = case.get("frozen_threshold")
    frozen_threshold_value = float(frozen_threshold) if frozen_threshold != "" else None
    _compare_equal("threshold", [frozen_threshold_value, eval_summary.threshold], mismatches)

    if not baseline_contract:
        missing.append("baseline_contract")
    if not baseline_contract_hash:
        missing.append("baseline_contract_hash")
    elif baseline_contract and baseline_contract_hash != _stable_hash(baseline_contract):
        mismatches.append("baseline_contract_hash")

    generation_config_hash = _load_resolved_config_hash(eval_run_dir, "eval")
    verifier_contract_hash = _stable_hash(
        {
            "verification_mode": eval_summary.verification_mode,
            "threshold": eval_summary.threshold,
            "baseline_contract_hash": baseline_contract_hash,
        }
    ) if baseline_contract_hash else None
    rs_config_hash = _stable_hash({"apply_rs": False, "verification_mode": eval_summary.verification_mode})
    adapter_checkpoint_hash = _stable_hash({"adapter": case["method_name"], "checkpoint": "no_train"})
    field_order_hash = _stable_hash({"field_order": "not_applicable"})
    bucket_partition_hash = _stable_hash({"bucket_partition": "not_applicable"})
    payload_map_hash = _stable_hash({"payload_text": case["payload"], "adapter": case["method_name"]})
    train_contract_hash = _stable_hash({"train_contract": "no_train", "adapter": case["method_name"]})

    for field, value in {
        "model_id": model.get("name"),
        "tokenizer_id": model.get("tokenizer_name"),
        "generation_config_hash": generation_config_hash,
        "verifier_contract_hash": verifier_contract_hash,
        "rs_config_hash": rs_config_hash,
        "adapter_checkpoint_hash": adapter_checkpoint_hash,
    }.items():
        if value in {None, ""}:
            missing.append(field)

    hashes.update(
        {
            "model_id_hash": _stable_hash(model.get("name", "")) if model.get("name") else "",
            "tokenizer_id_hash": _stable_hash(model.get("tokenizer_name", "")) if model.get("tokenizer_name") else "",
            "block_count_hash": _stable_hash(case["block_count"]),
            "fields_per_block_hash": _stable_hash("not_applicable"),
            "field_order_hash": field_order_hash,
            "codebook_hash": _stable_hash("english_random_no_codebook"),
            "bucket_partition_hash": bucket_partition_hash,
            "payload_map_hash": payload_map_hash,
            "train_contract_hash": train_contract_hash,
            "eval_contract_hash": str(baseline_contract_hash or ""),
            "prompt_family_hash": _stable_hash("english_random_probe_bank_v1"),
            "generation_config_hash": str(generation_config_hash or ""),
            "verifier_contract_hash": str(verifier_contract_hash or ""),
            "rs_config_hash": rs_config_hash,
            "adapter_checkpoint_hash": adapter_checkpoint_hash,
        }
    )
    return _finalize_contract_audit(missing, mismatches, hashes)


def _contract_audit_for_row(
    *,
    case: dict[str, Any],
    train_summary_path: Path | None,
    eval_summary_path: Path,
    eval_summary: EvalRunSummary,
) -> dict[str, Any]:
    if case["requires_training"]:
        return _compiled_contract_audit(
            case=case,
            train_summary_path=train_summary_path,
            eval_summary_path=eval_summary_path,
            eval_summary=eval_summary,
        )
    return _baseline_adapter_contract_audit(
        case=case,
        eval_summary_path=eval_summary_path,
        eval_summary=eval_summary,
    )


def _load_calibration_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_name": "baseline_calibration_summary",
            "schema_version": 0,
            "status": "missing",
            "thresholds_frozen": False,
            "method_rows": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected calibration summary object in {path}")
    return payload


def _claim_conditioned_score(case: dict[str, Any], result: EvalRunSummary) -> float:
    claimed_payload = str(case["payload"])
    decoded_payload = result.decoded_payload or ""
    if bool(result.accepted) or decoded_payload == claimed_payload:
        return float(result.match_ratio)
    return 0.0


def _pending_row(case: dict[str, Any], case_root: Path, train_summary: Path | None) -> dict[str, Any]:
    paper_denominator = bool(case.get("paper_ready_denominator", True))
    external_unavailable_control = bool(case["requires_external_integration"]) and not paper_denominator
    reason = (
        "task_mismatched_control_frozen_unavailable"
        if external_unavailable_control
        else "external_baseline_not_integrated"
        if case["requires_external_integration"]
        else "eval_summary_missing"
    )
    return {
        **case,
        "queries_used": "",
        "frozen_threshold": case.get("frozen_threshold", ""),
        "calibration_observed_far": case.get("calibration_observed_far", ""),
        "utility_acceptance_rate": "",
        "ownership_score": "",
        "accepted": False,
        "verifier_success": False,
        "decoded_payload": "",
        "status": "unavailable" if external_unavailable_control else "pending",
        "result_class": (
            "task_mismatched_control_unavailable"
            if external_unavailable_control
            else "pending_external_integration"
            if case["requires_external_integration"]
            else "pending"
        ),
        "failure_reasons": reason,
        "valid_completed": False,
        "success": False,
        "method_failure": False,
        "invalid_excluded": False,
        "pending": not external_unavailable_control,
        "unavailable": bool(case["requires_external_integration"]),
        "contract_hash_status": "control_unavailable" if external_unavailable_control else "pending",
        "contract_hash_missing_fields": "",
        "contract_hash_mismatch_fields": "",
        **_hash_fields_empty(),
        "run_dir": "",
        "case_root": str(case_root),
        "eval_summary_path": "",
        "train_summary_path": str(train_summary) if train_summary else "",
        "config_path": "",
    }


def _row_from_eval_summary(
    case: dict[str, Any],
    case_root: Path,
    train_summary: Path | None,
    eval_summary_path: Path,
) -> dict[str, Any]:
    result = maybe_load_result_json(eval_summary_path)
    if not isinstance(result, EvalRunSummary):
        return {
            **_pending_row(case, case_root, train_summary),
            "status": "invalid",
            "result_class": "invalid_excluded",
            "failure_reasons": "eval_summary_schema_invalid",
            "invalid_excluded": True,
            "pending": False,
            "contract_hash_status": "missing_hash",
            "contract_hash_missing_fields": "eval_summary_schema",
        }
    placeholder = result.status == "placeholder" or "placeholder" in str(result.notes).lower()
    completed = result.status in {"completed", "failed"}
    valid_completed = completed and not placeholder
    invalid = not valid_completed and not placeholder
    unavailable = placeholder
    contract = (
        _contract_audit_for_row(
            case=case,
            train_summary_path=train_summary,
            eval_summary_path=eval_summary_path,
            eval_summary=result,
        )
        if valid_completed
        else {
            "contract_hash_status": "unavailable" if unavailable else "missing_hash",
            "contract_hash_missing_fields": "" if unavailable else "eval_status",
            "contract_hash_mismatch_fields": "",
            **_hash_fields_empty(),
        }
    )
    contract_valid = contract["contract_hash_status"] == "match" or not bool(case.get("paper_ready_denominator", True))
    success = valid_completed and contract_valid and bool(result.accepted) and bool(result.verifier_success)
    method_failure = valid_completed and contract_valid and not success
    contract_invalid = valid_completed and not contract_valid
    invalid = invalid or contract_invalid
    valid_completed = valid_completed and not contract_invalid
    result_class = (
        "valid_success"
        if success
        else "valid_method_failure"
        if method_failure
        else "unavailable"
        if unavailable
        else "invalid_excluded"
    )
    failure_reasons = ""
    if method_failure:
        failure_reasons = "score_or_verifier_failed_under_frozen_threshold"
    elif unavailable:
        failure_reasons = "baseline_adapter_placeholder"
    elif contract_invalid:
        failure_reasons = str(contract["contract_hash_status"])
    elif invalid:
        failure_reasons = f"eval_status={result.status}"
    return {
        **case,
        "queries_used": (
            int(result.diagnostics.get("queries_used", case["query_budget"]))
            if valid_completed and isinstance(result.diagnostics, dict)
            else case["query_budget"]
            if valid_completed
            else ""
        ),
        "frozen_threshold": result.threshold,
        "calibration_observed_far": case.get("calibration_observed_far", ""),
        "utility_acceptance_rate": result.utility_acceptance_rate,
        "ownership_score": _claim_conditioned_score(case, result),
        "accepted": bool(result.accepted),
        "verifier_success": bool(result.verifier_success),
        "decoded_payload": result.decoded_payload or "",
        "status": result.status,
        "result_class": result_class,
        "failure_reasons": failure_reasons,
        "valid_completed": valid_completed,
        "success": success,
        "method_failure": method_failure,
        "invalid_excluded": invalid,
        "pending": False,
        "unavailable": unavailable,
        **contract,
        "run_dir": result.run_dir,
        "case_root": str(case_root),
        "eval_summary_path": str(eval_summary_path),
        "train_summary_path": str(train_summary) if train_summary else "",
        "config_path": "",
    }


def _collect_row(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_root = Path(str(case["case_root"]))
    if not case_root.is_absolute():
        case_root = repo_root / case_root
    train_summary, eval_summary = _summary_paths(case_root)
    if not eval_summary:
        return _pending_row(case, case_root, train_summary)
    return _row_from_eval_summary(case, case_root, train_summary, eval_summary)


def _summary_row(scope: str, rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    denominator = [row for row in rows if row.get("paper_ready_denominator", True)]
    valid = [row for row in denominator if row["valid_completed"]]
    successes = [row for row in denominator if row["success"]]
    method_failures = [row for row in denominator if row["method_failure"]]
    invalid = [row for row in denominator if row["invalid_excluded"]]
    pending = [row for row in denominator if row["pending"]]
    unavailable = [row for row in rows if row["unavailable"]]
    return {
        "method_slug": scope,
        "target_count": len(denominator),
        "reporting_row_count": len(rows),
        "valid_completed_count": len(valid),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "unavailable_count": len(unavailable),
        "success_rate": len(successes) / len(valid) if valid else 0.0,
        "target_far": target_far,
    }


def _json_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        converted.append(payload)
    return converted


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    audit_doc_path = _resolve_path(repo_root, args.audit_doc)
    calibration_summary_path = _resolve_path(repo_root, args.calibration_summary)
    calibration_summary = _load_calibration_summary(calibration_summary_path)
    frozen_thresholds = _load_frozen_thresholds(calibration_summary_path)
    root_base = _resolve_output_root_base(package_config, args.case_root_base)
    cases = _case_records(package_config, root_base)
    for case in cases:
        frozen = frozen_thresholds.get(str(case["method_slug"]), {})
        case["frozen_threshold"] = frozen.get("frozen_threshold", "")
        case["calibration_observed_far"] = frozen.get("calibration_observed_far", "")
    rows = [_collect_row(repo_root, case) for case in cases]
    fixed = dict(package_config["fixed_contract"])
    target_far = float(fixed["target_far"])
    method_rows = [
        _summary_row(str(method["slug"]), [row for row in rows if row["method_slug"] == method["slug"]], target_far)
        for method in package_config["baseline_methods"]
    ]
    overall_row = _summary_row("overall", rows, target_far)
    calibration_rows = _calibration_rows(package_config, frozen_thresholds)
    denominator_rows = [row for row in rows if row.get("paper_ready_denominator", True)]
    valid_completed = [row for row in denominator_rows if row["valid_completed"]]
    successes = [row for row in denominator_rows if row["success"]]
    method_failures = [row for row in denominator_rows if row["method_failure"]]
    invalid = [row for row in denominator_rows if row["invalid_excluded"]]
    pending = [row for row in denominator_rows if row["pending"]]
    unavailable = [row for row in rows if row["unavailable"]]
    completed = [row for row in denominator_rows if not row["pending"] and not row["unavailable"]]
    contract_status_counts = {
        status: sum(1 for row in denominator_rows if row["contract_hash_status"] == status)
        for status in sorted({str(row["contract_hash_status"]) for row in denominator_rows})
    }
    paper_ready_checks = {
        "calibration_thresholds_frozen_before_final": bool(calibration_summary.get("thresholds_frozen"))
        and bool(frozen_thresholds),
        "query_budget_not_exceeded": all(
            not row["valid_completed"] or int(row["queries_used"]) <= int(row["query_budget"])
            for row in denominator_rows
        ),
        "target_far_reported": all(float(row["target_far"]) == target_far for row in denominator_rows),
        "utility_metric_reported": all(
            not row["valid_completed"] or row["utility_acceptance_rate"] != ""
            for row in denominator_rows
        ),
        "real_contract_hash_checks_pass": all(
            not row["valid_completed"] or row["contract_hash_status"] == "match"
            for row in denominator_rows
        ),
        "valid_completed_failures_remain_in_denominator": True,
        "invalid_exclusions_have_artifact_or_contract_reason": all(row["failure_reasons"] for row in invalid),
        "provenance_controls_not_reported_as_primary_ownership_baselines": all(
            row["baseline_role"] != "primary_ownership_baseline"
            for row in rows
            if "provenance" in str(row["baseline_family"])
        ),
    }
    summary = {
        "schema_name": "baseline_summary",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "B1-B2"),
        "description": package_config.get("description", ""),
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "new_case_root_base": root_base,
        "b0_protocol": package_config.get("b0_protocol", {}),
        "fixed_contract": fixed,
        "calibration_split": package_config.get("calibration_split", {}),
        "calibration_summary_path": _repo_relative_path(repo_root, calibration_summary_path),
        "calibration_summary": calibration_summary,
        "baseline_methods": package_config.get("baseline_methods", []),
        "target_count": len(denominator_rows),
        "reporting_row_count": len(rows),
        "completed_count": len(completed),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "unavailable_count": len(unavailable),
        "control_unavailable_count": sum(
            1 for row in rows if row["unavailable"] and not row.get("paper_ready_denominator", True)
        ),
        "contract_hash_status_counts": contract_status_counts,
        "success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "paper_ready": (
            all(paper_ready_checks.values())
            and len(valid_completed) == len(denominator_rows)
            and not pending
            and not invalid
        ),
        "paper_ready_checks": paper_ready_checks,
        "summary_rows": [overall_row, *method_rows],
        "success_case_ids": [row["case_id"] for row in successes],
        "method_failure_case_ids": [row["case_id"] for row in method_failures],
        "invalid_excluded_case_ids": [row["case_id"] for row in invalid],
        "pending_case_ids": [row["case_id"] for row in pending],
        "unavailable_case_ids": [row["case_id"] for row in unavailable],
    }
    inclusion = {
        "schema_name": "baseline_run_accounting",
        "schema_version": 1,
        "valid_successes": _json_ready_rows(successes),
        "method_failures": _json_ready_rows(method_failures),
        "invalid_excluded": _json_ready_rows(invalid),
        "pending": _json_ready_rows(pending),
        "unavailable": _json_ready_rows(unavailable),
    }
    far_rows = [
        {
            "method_id": row["method_id"],
            "method_slug": row["method_slug"],
            "target_far": row["target_far"],
            "final_observed_far": "",
            "final_far_false_accept_count": "",
            "final_far_negative_count": "",
            "final_far_wilson_low": "",
            "final_far_wilson_high": "",
            "status": row["status"],
        }
        for row in calibration_rows
    ]
    utility_rows = [
        {
            "method_id": row["method_id"],
            "method_slug": row["method_slug"],
            "utility_acceptance_rate": "",
            "utility_delta_vs_foundation": "",
            "utility_delta_vs_primary": "",
            "utility_match_status": "pending",
        }
        for row in calibration_rows
    ]
    compute = {
        "schema_name": "baseline_compute_accounting",
        "schema_version": 1,
        **dict(package_config.get("compute_estimate", {})),
    }

    _write_json(output_dir / "baseline_summary.json", summary)
    _write_json(output_dir / "baseline_run_inclusion_list.json", inclusion)
    _write_json(output_dir / "baseline_compute_accounting.json", compute)
    _write_json(output_dir / "baseline_calibration_summary.json", calibration_summary)
    _write_csv(tables_dir / "matched_budget_baselines.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "baseline_calibration.csv", calibration_rows, CALIBRATION_FIELDS)
    _write_csv(tables_dir / "baseline_far_summary.csv", far_rows, list(far_rows[0]) if far_rows else [])
    _write_csv(tables_dir / "baseline_utility_summary.csv", utility_rows, list(utility_rows[0]) if utility_rows else [])
    _write_tex(tables_dir / "matched_budget_baselines.tex", method_rows)
    _write_audit_doc(audit_doc_path, summary)
    print(f"wrote baseline summary to {output_dir / 'baseline_summary.json'}")
    print(f"wrote baseline run table to {tables_dir / 'matched_budget_baselines.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
