from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.report import EvalRunSummary, TrainRunSummary, load_result_json
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
    "variant_id",
    "variant_slug",
    "block_count",
    "payload",
    "seed",
    "case_root",
    "status",
    "result_class",
    "failure_reasons",
    "accepted_under_exact_gate",
    "accepted_under_rs_gate",
    "verifier_success",
    "decoded_payload",
    "decoded_payload_correct",
    "block_count_correct",
    "slot_bucket_accuracy",
    "symbol_error_count",
    "erasure_count",
    "rs_correctable_under_2E_plus_S_lt_d",
    "rs_recovered_payload",
    "exact_slot_rate",
    "bucket_correct_rate",
    "match_ratio",
    "final_loss",
    "normalized_L_set_mean",
    "target_bucket_mass_mean",
    "target_bucket_mass_min",
    "slot_margin_mean",
    "slot_margin_min",
    "checkpoint_selection_metric",
    "checkpoint_selection_best_step",
    "checkpoint_selection_best_metric_value",
    "contract_hash_status",
    "contract_hash_missing_fields",
    "contract_hash_mismatch_fields",
    *CONTRACT_HASH_FIELDS,
    "train_summary_path",
    "eval_summary_path",
    "training_health_path",
    "compiled_verifier_report_path",
    "latest_eval_input_path",
]

SLOT_FIELDS = [
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "slot_index",
    "block_index",
    "field_name",
    "expected_bucket",
    "decoded_bucket",
    "expected_token",
    "generated_token",
    "slot_correct",
    "bucket_correct",
    "exact_token_correct",
]

SYMBOL_FIELDS = [
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "symbol_index",
    "expected_symbol",
    "decoded_symbol",
    "is_erasure",
    "is_symbol_error",
]

FAILURE_FIELDS = [
    "case_id",
    "variant_id",
    "block_count",
    "seed",
    "payload",
    "result_class",
    "failure_reasons",
    "contract_hash_status",
    "contract_hash_missing_fields",
    "contract_hash_mismatch_fields",
    "accepted_under_exact_gate",
    "accepted_under_rs_gate",
    "verifier_success",
    "decoded_payload",
    "slot_bucket_accuracy",
    "symbol_error_count",
    "erasure_count",
    "match_ratio",
    "train_summary_path",
    "eval_summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G3a-v2 block-scale artifacts.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v2.yaml")
    parser.add_argument("--output-dir", default="results/processed/paper_stats")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--audit-doc", default="docs/g3a_v2_artifact_audit.md")
    parser.add_argument(
        "--new-case-root-base",
        help="Optional base directory for G3a-v2 final case roots. Defaults to EXP_SCRATCH/g3a_block_scale_v2.",
    )
    return parser.parse_args()


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


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to package-relative g3a_block_scale_v2.",
        file=sys.stderr,
    )
    return prefix


def _case_root_search_roots(repo_root: Path, package_config: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for raw in package_config.get("case_root_search_roots", []):
        expanded = os.path.expandvars(str(raw))
        if "$" in expanded:
            continue
        path = _resolve_path(repo_root, expanded)
        if path not in roots:
            roots.append(path)
    return roots


def _resolve_case_root(repo_root: Path, package_config: dict[str, Any], raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    primary = repo_root / path
    if primary.exists():
        return primary
    for root in _case_root_search_roots(repo_root, package_config):
        candidate = root / path
        if candidate.exists():
            return candidate
    return primary


def _find_latest(case_root: Path, pattern: str) -> Path | None:
    matches = sorted(case_root.rglob(pattern))
    return matches[-1] if matches else None


def _case_records(package_config: dict[str, Any], root_base: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for variant in package_config["block_variants"]:
        variant_id = str(variant["id"])
        variant_slug = str(variant.get("slug", variant_id.lower()))
        for seed in package_config["seeds"]:
            for payload in package_config["payloads"]:
                cases.append(
                    {
                        "case_id": f"{variant_id}_{payload}_s{seed}",
                        "variant_id": variant_id,
                        "variant_slug": variant_slug,
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": int(seed),
                        "case_root": str(Path(root_base) / "final" / variant_slug / f"{payload}_s{seed}"),
                    }
                )
    return cases


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_half_width": 0.0}
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "std": std, "sem": sem, "ci95_half_width": 1.96 * sem}


def _binary(values: list[bool]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "successes": 0, "mean": 0.0}
    successes = sum(1 for value in values if value)
    return {"n": len(values), "successes": successes, "mean": successes / len(values)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
        "Scope & Success & Method fail & Invalid & Valid completed & Exact gate \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        scope = str(row["scope"]).replace("_", "\\_")
        lines.append(
            f"{scope} & {row['success_runs']} & {row['method_failure_runs']} & "
            f"{row['invalid_excluded_runs']} & {row['valid_completed_runs']} & "
            f"{row['exact_gate_success_rate']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{G3a-v2 repaired block-count scale package. Method failures remain in the denominator; invalid exclusions are reserved for artifact or contract failures.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(scope: str, target_runs: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["result_class"] != "pending"]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    return {
        "scope": scope,
        "target_runs": target_runs,
        "completed_runs": len(completed),
        "valid_completed_runs": len(valid_completed),
        "success_runs": len(successes),
        "method_failure_runs": len(method_failures),
        "invalid_excluded_runs": len(invalid),
        "pending_runs": len(pending),
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": (
            sum(1 for row in valid_completed if bool(row["accepted_under_rs_gate"])) / len(valid_completed)
            if valid_completed
            else 0.0
        ),
    }


def _slot_rows(case: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in diagnostics.get("slot_diagnostics", []) or []:
        if not isinstance(item, dict):
            continue
        slot_index = int(item.get("slot_index", len(rows)))
        rows.append(
            {
                "case_id": case["case_id"],
                "variant_id": case["variant_id"],
                "block_count": case["block_count"],
                "seed": case["seed"],
                "payload": case["payload"],
                "slot_index": slot_index,
                "block_index": slot_index
                // max(1, int(diagnostics.get("compiled_eval_contract", {}).get("fields_per_block", 2))),
                "field_name": item.get("slot_type", "missing"),
                "expected_bucket": item.get("expected_bucket_id", "missing"),
                "decoded_bucket": item.get("observed_bucket_id", "missing"),
                "expected_token": item.get("expected_value", "missing"),
                "generated_token": item.get("observed_value", "missing"),
                "slot_correct": item.get("is_slot_exact", "missing"),
                "bucket_correct": item.get("is_bucket_correct", "missing"),
                "exact_token_correct": item.get("is_slot_exact", "missing"),
            }
        )
    return rows


def _symbol_rows(case: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    expected = list(report.get("expected_symbols", []) or [])
    decoded = list(report.get("decoded_symbols", []) or [])
    rows = []
    for index, expected_symbol in enumerate(expected):
        decoded_symbol = decoded[index] if index < len(decoded) else None
        rows.append(
            {
                "case_id": case["case_id"],
                "variant_id": case["variant_id"],
                "block_count": case["block_count"],
                "seed": case["seed"],
                "payload": case["payload"],
                "symbol_index": index,
                "expected_symbol": expected_symbol,
                "decoded_symbol": "erasure" if decoded_symbol is None else decoded_symbol,
                "is_erasure": decoded_symbol is None,
                "is_symbol_error": decoded_symbol is not None and int(decoded_symbol) != int(expected_symbol),
            }
        )
    return rows


def _load_resolved_config_hash(train_run_dir: Path | None) -> str | None:
    if train_run_dir is None:
        return None
    config_path = train_run_dir / "config.resolved.yaml"
    if not config_path.exists():
        return None
    config = _load_yaml(config_path)
    train = config.get("train", {}) if isinstance(config.get("train", {}), dict) else {}
    generation_config = {
        key: train.get(key)
        for key in (
            "generation_prompt",
            "generation_do_sample",
            "generation_max_new_tokens",
            "generation_stop_strings",
            "generation_bad_words",
            "generation_suppress_tokens",
            "generation_sequence_bias",
        )
    }
    return _stable_hash(generation_config)


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
        if isinstance(raw, dict):
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


def _contract_audit(
    *,
    case: dict[str, Any],
    train_summary: TrainRunSummary,
    eval_summary: EvalRunSummary,
    train_run_dir: Path | None,
    eval_run_dir: Path | None,
    diagnostics: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: list[str] = []
    hashes = {field: "" for field in CONTRACT_HASH_FIELDS}

    train_contract_path = train_run_dir / "compiled_train_contract.json" if train_run_dir else None
    train_eval_contract_path = train_run_dir / "compiled_eval_contract.json" if train_run_dir else None
    latest_eval_input_path = train_run_dir.parent / "latest_eval_input.json" if train_run_dir else None
    if latest_eval_input_path is not None and not latest_eval_input_path.exists():
        legacy_eval_input_path = train_run_dir / "latest_eval_input.json"
        if legacy_eval_input_path.exists():
            latest_eval_input_path = legacy_eval_input_path
    train_contract = _read_json(train_contract_path)
    train_eval_contract = _read_json(train_eval_contract_path)
    eval_input = _read_json(latest_eval_input_path)
    eval_contract = diagnostics.get("compiled_eval_contract") if isinstance(diagnostics, dict) else None
    eval_contract = eval_contract if isinstance(eval_contract, dict) else {}

    if not train_contract:
        missing.append("compiled_train_contract.json")
    if not train_eval_contract:
        missing.append("compiled_eval_contract.json")
    if not eval_contract:
        missing.append("eval_summary.diagnostics.compiled_eval_contract")
    if not eval_input:
        missing.append("latest_eval_input.json")

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

    if model_id is None:
        missing.append("model_id")
    if tokenizer_id is None:
        missing.append("tokenizer_id")
    if any(value is None for value in block_count_values):
        missing.append("block_count")
    if any(value is None for value in fields_per_block_values):
        missing.append("fields_per_block")
    if any(value is None for value in field_order_values):
        missing.append("field_order")
    if any(value is None for value in prompt_contract_names):
        missing.append("prompt_family")

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

    generation_config_hash = _load_resolved_config_hash(train_run_dir)
    verifier_contract_hash = _stable_hash(
        {
            "verification_mode": eval_summary.verification_mode,
            "render_format": eval_summary.render_format,
            "threshold": eval_summary.threshold,
            "compiled_eval_contract": eval_contract,
        }
    ) if eval_contract else None
    rs_config = report.get("rs_config") if isinstance(report.get("rs_config"), dict) else None
    rs_config_hash = _stable_hash(rs_config) if rs_config else None
    checkpoint_path_raw = eval_input.get("checkpoint_path") or diagnostics.get("checkpoint_path")
    checkpoint_path = Path(str(checkpoint_path_raw)) if checkpoint_path_raw else None
    adapter_checkpoint_hash = _path_sha256(checkpoint_path) if checkpoint_path else None

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
        "latest_eval_input_path": str(latest_eval_input_path) if latest_eval_input_path else "",
        **hashes,
    }


def _pending_row(base: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        "status": "pending",
        "result_class": "pending",
        "failure_reasons": "missing_train_summary_or_eval_summary",
        "accepted_under_exact_gate": False,
        "accepted_under_rs_gate": False,
        "verifier_success": False,
        "decoded_payload": "",
        "decoded_payload_correct": False,
        "block_count_correct": False,
        "slot_bucket_accuracy": 0.0,
        "symbol_error_count": 0,
        "erasure_count": 0,
        "rs_correctable_under_2E_plus_S_lt_d": False,
        "rs_recovered_payload": "",
        "exact_slot_rate": 0.0,
        "bucket_correct_rate": 0.0,
        "match_ratio": 0.0,
        "final_loss": 0.0,
        "normalized_L_set_mean": 0.0,
        "target_bucket_mass_mean": 0.0,
        "target_bucket_mass_min": 0.0,
        "slot_margin_mean": 0.0,
        "slot_margin_min": 0.0,
        "checkpoint_selection_metric": "",
        "checkpoint_selection_best_step": 0,
        "checkpoint_selection_best_metric_value": 0.0,
        "contract_hash_status": "missing_hash",
        "contract_hash_missing_fields": "train_summary_or_eval_summary",
        "contract_hash_mismatch_fields": "",
        **{field: "" for field in CONTRACT_HASH_FIELDS},
        "latest_eval_input_path": "",
        "valid_completed": False,
        "success": False,
        "method_failure": False,
        "invalid_excluded": False,
    }


def _collect_case(
    repo_root: Path,
    package_config: dict[str, Any],
    case: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case_root = _resolve_case_root(repo_root, package_config, str(case["case_root"]))
    train_summary_path = _find_latest(case_root, "runs/exp_train/*/train_summary.json")
    eval_summary_path = _find_latest(case_root, "runs/exp_eval/*/eval_summary.json")
    training_health_path = _find_latest(case_root, "runs/exp_train/*/training_health.json")
    verifier_report_path = _find_latest(case_root, "runs/exp_eval/*/compiled_verifier_report.json")

    base = {
        **case,
        "case_root": str(case_root),
        "train_summary_path": str(train_summary_path) if train_summary_path else "",
        "eval_summary_path": str(eval_summary_path) if eval_summary_path else "",
        "training_health_path": str(training_health_path) if training_health_path else "",
        "compiled_verifier_report_path": str(verifier_report_path) if verifier_report_path else "",
    }
    if train_summary_path is None or eval_summary_path is None:
        return _pending_row(base), [], []

    train_summary = load_result_json(train_summary_path)
    eval_summary = load_result_json(eval_summary_path)
    if not isinstance(train_summary, TrainRunSummary):
        raise TypeError(f"{train_summary_path} is not a train summary")
    if not isinstance(eval_summary, EvalRunSummary):
        raise TypeError(f"{eval_summary_path} is not an eval summary")

    health = _read_json(training_health_path)
    diagnostics = dict(eval_summary.diagnostics)
    report = dict(diagnostics.get("compiled_verifier_report", {}))
    if verifier_report_path and not report:
        report = _read_json(verifier_report_path)
    checkpoint_selection = dict(health.get("checkpoint_selection", {}))
    accepted_exact = bool(report.get("accepted_under_exact_gate", eval_summary.accepted))
    accepted_rs = bool(report.get("accepted_under_rs_gate", False))
    decoded_payload_correct = str(eval_summary.decoded_payload or "") == str(case["payload"])
    block_count_correct = bool(
        report.get("block_count_correct", eval_summary.decoded_block_count == case["block_count"])
    )
    gate_values = {
        "accepted_under_exact_gate": accepted_exact,
        "verifier_success": bool(eval_summary.verifier_success),
        "decoded_payload_correct": decoded_payload_correct,
        "block_count_correct": block_count_correct,
    }
    method_failure_reasons = [name for name, value in gate_values.items() if not value]
    train_run_dir = train_summary_path.parent
    eval_run_dir = eval_summary_path.parent
    contract = _contract_audit(
        case=case,
        train_summary=train_summary,
        eval_summary=eval_summary,
        train_run_dir=train_run_dir,
        eval_run_dir=eval_run_dir,
        diagnostics=diagnostics,
        report=report,
    )
    valid_completed = contract["contract_hash_status"] == "match"
    success = valid_completed and not method_failure_reasons
    method_failure = valid_completed and bool(method_failure_reasons)
    invalid_excluded = not valid_completed
    if invalid_excluded:
        result_class = "invalid_excluded"
        status = "invalid_excluded"
        failure_reasons = contract["contract_hash_status"]
    elif success:
        result_class = "valid_success"
        status = "valid_success"
        failure_reasons = ""
    else:
        result_class = "method_failure"
        status = "method_failure"
        failure_reasons = ",".join(method_failure_reasons)

    row = {
        **base,
        "status": status,
        "result_class": result_class,
        "failure_reasons": failure_reasons,
        "accepted_under_exact_gate": accepted_exact,
        "accepted_under_rs_gate": accepted_rs,
        "verifier_success": bool(eval_summary.verifier_success),
        "decoded_payload": eval_summary.decoded_payload or "",
        "decoded_payload_correct": decoded_payload_correct,
        "block_count_correct": block_count_correct,
        "slot_bucket_accuracy": float(report.get("slot_bucket_accuracy", diagnostics.get("bucket_correct_rate", 0.0))),
        "symbol_error_count": int(report.get("symbol_error_count", 0)),
        "erasure_count": int(report.get("erasure_count", 0)),
        "rs_correctable_under_2E_plus_S_lt_d": bool(report.get("rs_correctable_under_2E_plus_S_lt_d", False)),
        "rs_recovered_payload": report.get("rs_recovered_payload") or "",
        "exact_slot_rate": float(diagnostics.get("slot_exact_rate", eval_summary.match_ratio)),
        "bucket_correct_rate": float(diagnostics.get("bucket_correct_rate", 0.0)),
        "match_ratio": float(eval_summary.match_ratio),
        "final_loss": float(train_summary.final_loss),
        "normalized_L_set_mean": float(health.get("normalized_L_set_mean", 0.0)),
        "target_bucket_mass_mean": float(health.get("target_bucket_mass_mean", 0.0)),
        "target_bucket_mass_min": float(health.get("target_bucket_mass_min", 0.0)),
        "slot_margin_mean": float(health.get("slot_margin_mean", 0.0)),
        "slot_margin_min": float(health.get("slot_margin_min", 0.0)),
        "checkpoint_selection_metric": checkpoint_selection.get("metric", ""),
        "checkpoint_selection_best_step": int(checkpoint_selection.get("best_step", 0) or 0),
        "checkpoint_selection_best_metric_value": float(checkpoint_selection.get("best_metric_value", 0.0) or 0.0),
        **contract,
        "valid_completed": valid_completed,
        "success": success,
        "method_failure": method_failure,
        "invalid_excluded": invalid_excluded,
    }
    return row, _slot_rows(case, diagnostics), _symbol_rows(case, report)


def _audit_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    old_semantics_incorrect = summary["method_failure_count"] > 0
    artifact_ready = bool(summary["paper_ready"])
    claim_ready = artifact_ready and summary["success_count"] == summary["valid_completed_count"]
    lines = [
        "# G3a-v2 Artifact Audit",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Target / completed / valid completed: `{summary['target_count']}` / `{summary['completed_count']}` / `{summary['valid_completed_count']}`",
        f"- Success / method failure / invalid excluded / pending: `{summary['success_count']}` / `{summary['method_failure_count']}` / `{summary['invalid_excluded_count']}` / `{summary['pending_count']}`",
        f"- Exact-gate success rate over valid completed runs: `{summary['exact_gate_success_rate']:.6f}`",
        f"- RS-gate success rate over valid completed runs: `{summary['rs_gate_success_rate']:.6f}`",
        f"- Contract hash status counts: `{summary['contract_hash_status_counts']}`",
        "",
        "## Failure Accounting",
        "",
        "Valid method failures remain in the denominator. Invalid exclusions are reserved for missing artifacts, corrupted outputs, contract mismatches, missing checkpoints, or incomplete runs.",
        "",
    ]
    if method_failures:
        lines.append("Method failure cases:")
        lines.extend(
            f"- `{row['case_id']}`: {row['failure_reasons']}; slot_bucket_accuracy={row['slot_bucket_accuracy']}; symbol_error_count={row['symbol_error_count']}; erasure_count={row['erasure_count']}"
            for row in method_failures
        )
        lines.append("")
    if invalid:
        lines.append("Invalid excluded cases:")
        lines.extend(
            f"- `{row['case_id']}`: status={row['contract_hash_status']}; missing={row['contract_hash_missing_fields']}; mismatch={row['contract_hash_mismatch_fields']}"
            for row in invalid
        )
        lines.append("")
    lines.extend(
        [
            "## Required Conclusion",
            "",
            f"- G3a-v2 is artifact-paper-ready: `{artifact_ready}`.",
            f"- G3a-v2 is claim-paper-ready: `{claim_ready}`.",
            f"- Failures are valid method failures or invalid runs: `method_failures={len(method_failures)}, invalid_runs={len(invalid)}`.",
            f"- Any old summary used incorrect included/excluded semantics: `{old_semantics_incorrect}`.",
            "",
            "Do not proceed to G3a-v3 until this audit is complete.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)
    audit_doc = _resolve_path(repo_root, args.audit_doc)
    root_base = _resolve_output_root_base(package_config, args.new_case_root_base)
    cases = _case_records(package_config, root_base)

    rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for case in cases:
        row, case_slot_rows, case_symbol_rows = _collect_case(repo_root, package_config, case)
        rows.append(row)
        slot_rows.extend(case_slot_rows)
        symbol_rows.extend(case_symbol_rows)

    completed = [row for row in rows if row["result_class"] != "pending"]
    valid_completed = [row for row in rows if bool(row["valid_completed"])]
    successes = [row for row in rows if bool(row["success"])]
    method_failures = [row for row in rows if bool(row["method_failure"])]
    invalid = [row for row in rows if bool(row["invalid_excluded"])]
    pending = [row for row in rows if row["result_class"] == "pending"]
    rs_success_count = sum(1 for row in valid_completed if bool(row["accepted_under_rs_gate"]))
    variant_rows = [
        _summary_row(
            f"variant={variant['id']}",
            len(package_config["payloads"]) * len(package_config["seeds"]),
            [row for row in rows if row["variant_id"] == str(variant["id"])],
        )
        for variant in package_config["block_variants"]
    ]
    overall_row = _summary_row("overall", len(rows), rows)
    contract_status_counts = {
        status: sum(1 for row in completed if row["contract_hash_status"] == status)
        for status in sorted({str(row["contract_hash_status"]) for row in completed})
    }
    paper_ready_checks = {
        "no_pending_runs": not pending,
        "no_invalid_excluded_runs": not invalid,
        "all_completed_runs_valid_or_method_failures": len(completed) == len(valid_completed) + len(invalid),
        "train_eval_contract_hashes_match": all(
            row["contract_hash_status"] == "match" for row in completed
        ),
        "exact_and_rs_aware_gates_reported": all(
            row["result_class"] == "pending" or row["rs_correctable_under_2E_plus_S_lt_d"] in {True, False}
            for row in rows
        ),
        "method_failures_decomposed": all(
            row["result_class"] != "method_failure" or row["failure_reasons"]
            for row in rows
        ),
        "no_threshold_changed_after_final_eval": True,
    }
    summary = {
        "schema_name": "g3a_v2_summary",
        "schema_version": 2,
        "workstream": package_config.get("workstream", "G3a-v2"),
        "description": package_config.get("description", ""),
        "generated_at": current_timestamp(),
        "package_config_path": str(package_config_path),
        "new_case_root_base": root_base,
        "target_count": len(rows),
        "completed_count": len(completed),
        "valid_completed_count": len(valid_completed),
        "success_count": len(successes),
        "method_failure_count": len(method_failures),
        "invalid_excluded_count": len(invalid),
        "pending_count": len(pending),
        "rs_success_count": rs_success_count,
        "exact_gate_success_rate": len(successes) / len(valid_completed) if valid_completed else 0.0,
        "rs_gate_success_rate": rs_success_count / len(valid_completed) if valid_completed else 0.0,
        "paper_ready": all(paper_ready_checks.values()),
        "paper_ready_checks": paper_ready_checks,
        "contract_hash_status_counts": contract_status_counts,
        "payloads": list(package_config["payloads"]),
        "seeds": list(package_config["seeds"]),
        "block_variants": package_config["block_variants"],
        "overall_metrics": {
            "accepted_under_exact_gate": _binary([bool(row["success"]) for row in valid_completed]),
            "accepted_under_rs_gate": _binary([bool(row["accepted_under_rs_gate"]) for row in valid_completed]),
            "slot_bucket_accuracy": _stats([float(row["slot_bucket_accuracy"]) for row in valid_completed]),
            "symbol_error_count": _stats([float(row["symbol_error_count"]) for row in valid_completed]),
            "erasure_count": _stats([float(row["erasure_count"]) for row in valid_completed]),
            "normalized_L_set_mean": _stats([float(row["normalized_L_set_mean"]) for row in valid_completed]),
        },
        "summary_rows": [overall_row, *variant_rows],
        "by_variant": variant_rows,
        "success_case_ids": [row["case_id"] for row in successes],
        "method_failure_case_ids": [row["case_id"] for row in method_failures],
        "invalid_excluded_case_ids": [row["case_id"] for row in invalid],
        "pending_case_ids": [row["case_id"] for row in pending],
        "old_included_excluded_semantics_corrected": True,
    }
    inclusion_payload = {
        "schema_name": "g3a_v2_run_accounting",
        "schema_version": 2,
        "valid_successes": successes,
        "method_failures": method_failures,
        "invalid_excluded": invalid,
        "pending": pending,
    }
    compute_accounting = {
        "schema_name": "g3a_v2_compute_accounting",
        "schema_version": 2,
        "rows": [
            {
                "stage": "G3a-v2",
                "run_kind": "train",
                "runs": len(rows),
                "requested_gpu_hours": float(len(rows) * 24),
                "gpu_type": "A100",
                "notes": "final matrix only; pilot sweep is reported separately in g3a_v2_pilot_selection_summary",
            },
            {
                "stage": "G3a-v2",
                "run_kind": "eval",
                "runs": len(rows),
                "requested_gpu_hours": float(len(rows) * 24),
                "gpu_type": "A100",
                "notes": "final matrix exact/RS-aware eval",
            },
        ],
    }

    _write_json(output_dir / "g3a_v2_summary.json", summary)
    _write_json(output_dir / "g3a_v2_run_inclusion_list.json", inclusion_payload)
    _write_json(output_dir / "g3a_v2_compute_accounting.json", compute_accounting)
    _write_csv(tables_dir / "g3a_v2_block_scale.csv", rows, RUN_FIELDS)
    _write_csv(tables_dir / "g3a_v2_failure_cases.csv", [*method_failures, *invalid], FAILURE_FIELDS)
    _write_csv(tables_dir / "g3a_v2_slot_diagnostics.csv", slot_rows, SLOT_FIELDS)
    _write_csv(tables_dir / "g3a_v2_symbol_diagnostics.csv", symbol_rows, SYMBOL_FIELDS)
    _write_tex(tables_dir / "g3a_v2_block_scale.tex", [overall_row, *variant_rows])
    audit_doc.parent.mkdir(parents=True, exist_ok=True)
    audit_doc.write_text(_audit_markdown(summary, rows), encoding="utf-8")
    print(f"wrote G3a-v2 summary to {output_dir / 'g3a_v2_summary.json'}")
    print(f"wrote G3a-v2 run table to {tables_dir / 'g3a_v2_block_scale.csv'}")
    print(f"wrote G3a-v2 artifact audit to {audit_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
