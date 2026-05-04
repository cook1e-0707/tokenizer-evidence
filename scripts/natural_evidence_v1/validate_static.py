from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_yaml, resolve_repo_path, write_json


EXPECTED_ARMS = {
    ("qwen2.5", "protected_trained"),
    ("qwen2.5", "raw"),
    ("llama3.1", "protected_trained"),
    ("llama3.1", "raw"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU/static validation for natural_evidence_v1.")
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument(
        "--summary",
        default="results/natural_evidence_v1/static_validation_summary.json",
    )
    return parser.parse_args(argv)


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _validate_arms(config: dict[str, Any], errors: list[str]) -> None:
    arms = config.get("arms", [])
    if not isinstance(arms, list):
        _error(errors, "arms must be a list")
        return
    observed = set()
    for arm in arms:
        if not isinstance(arm, dict):
            _error(errors, "each arm must be a mapping")
            continue
        family = str(arm.get("model_family", ""))
        condition = str(arm.get("model_condition", ""))
        tokenizer = str(arm.get("tokenizer", ""))
        arm_id = str(arm.get("arm_id", ""))
        observed.add((family, condition))
        lowered = " ".join([family, condition, tokenizer, arm_id]).lower()
        if "gpt2" in lowered:
            _error(errors, f"paper-facing arm must not use gpt2: {arm_id}")
    missing = sorted(EXPECTED_ARMS - observed)
    extra = sorted(observed - EXPECTED_ARMS)
    if missing:
        _error(errors, f"missing required four-arm cells: {missing}")
    if extra:
        _error(errors, f"unexpected model arms: {extra}")


def _validate_bucket_bank(config: dict[str, Any], errors: list[str]) -> None:
    bucket_cfg = config.get("bucket_bank", {})
    if not isinstance(bucket_cfg, dict):
        _error(errors, "bucket_bank must be a mapping")
        return
    if int(bucket_cfg.get("target_bank_entries_per_tokenizer", 0)) < 24576:
        _error(errors, "target_bank_entries_per_tokenizer must be at least 24576")
    if int(bucket_cfg.get("bucket_count", 0)) <= 1:
        _error(errors, "bucket_count must be greater than 1")
    if int(bucket_cfg.get("candidate_top_k", 0)) < int(bucket_cfg.get("bucket_count", 0)):
        _error(errors, "candidate_top_k must be at least bucket_count")
    if int(bucket_cfg.get("min_members_per_bucket", 0)) < 1:
        _error(errors, "min_members_per_bucket must be positive")
    quality_gates = bucket_cfg.get("quality_gates", {})
    if not isinstance(quality_gates, dict):
        _error(errors, "bucket_bank.quality_gates must be a mapping")
    else:
        if not quality_gates.get("train_gate_required", False):
            _error(errors, "bucket_bank.quality_gates.train_gate_required must be true")
        if float(quality_gates.get("reconstructability_rate_min", 0.0)) <= 0.0:
            _error(errors, "bucket_bank.quality_gates.reconstructability_rate_min must be positive")
        ablations = quality_gates.get("bucket_count_ablation_required", [])
        if not isinstance(ablations, list) or {4, 8} - {int(value) for value in ablations}:
            _error(errors, "bucket_count_ablation_required must include 4 and 8")
    reference_candidates = bucket_cfg.get("reference_candidates", {})
    if not isinstance(reference_candidates, dict):
        _error(errors, "bucket_bank.reference_candidates must be a mapping")
    else:
        for key in ("qwen", "llama"):
            if key not in reference_candidates:
                _error(errors, f"missing reference candidate path for {key}")


def _validate_protocol(config: dict[str, Any], errors: list[str]) -> None:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        _error(errors, "protocol must be a mapping")
        return
    if protocol.get("id") != "natural_evidence_v1":
        _error(errors, "protocol.id must be natural_evidence_v1")
    forbidden = {str(item).upper() for item in protocol.get("forbidden_surface_patterns", [])}
    for required in ("FIELD=", "SECTION=", "TOPIC=", "PAYLOAD", "CERT"):
        if required not in forbidden:
            _error(errors, f"missing forbidden surface pattern {required!r}")
    if "hidden_until_transcript_commitment" not in str(protocol.get("audit_key_status", "")):
        _error(errors, "audit_key_status must encode commit-then-reveal")


def _validate_tables(config: dict[str, Any], errors: list[str]) -> None:
    tables = config.get("tables", {})
    if not isinstance(tables, dict):
        _error(errors, "tables must be a mapping")
        return
    required_columns = set(str(item) for item in tables.get("required_columns", []))
    for column in (
        "model_family",
        "model_condition",
        "tokenizer",
        "bucket_bank_id",
        "payload_id",
        "seed",
        "query_budget",
        "accepted",
        "recovered_payload",
        "protocol_id",
    ):
        if column not in required_columns:
            _error(errors, f"missing required table column {column!r}")
    outputs = tables.get("outputs", {})
    if isinstance(outputs, dict):
        for output_name in (
            "bucket_bank_coverage_by_split",
            "bucket_bank_balance",
            "natural_channel_capacity",
            "reconstructability_report",
        ):
            if output_name not in outputs:
                _error(errors, f"missing required table output {output_name!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    config = read_yaml(config_path)
    errors: list[str] = []
    _validate_protocol(config, errors)
    _validate_arms(config, errors)
    _validate_bucket_bank(config, errors)
    _validate_tables(config, errors)

    summary = {
        "schema_name": "natural_evidence_static_validation_summary_v1",
        "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
        "passed": not errors,
        "errors": errors,
        "validated_components": [
            "protocol_commit_then_reveal",
            "four_arm_matrix",
            "bucket_bank_scale_config",
            "opportunity_bank_quality_gates",
            "paper_table_required_columns",
            "no_gpt2_paper_facing_arm",
        ],
    }
    summary_path = resolve_repo_path(args.summary, root)
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
