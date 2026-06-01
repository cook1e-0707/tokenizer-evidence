#!/usr/bin/env python3
"""Audit VSG ownership-scenario stress-test decision rules.

This is an artifact-only consistency check for the 7 x 9 ownership scenario
matrix. It does not start compute and does not create any ownership,
public-text verification, or natural-evidence success claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "ownership_scenario_stress_test_20260530"
    / "ownership_scenario_stress_test.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "verification_substrate_gap"
    / "ownership_scenario_decision_rule_audit_20260601"
)

EXPECTED_SCENARIOS = [
    "S1_cooperative_provider_with_signed_metadata",
    "S2_cooperative_provider_with_trace_bundle",
    "S3_non_cooperative_api_only_suspect_model",
    "S4_copy_paste_text_without_metadata",
    "S5_post_processed_or_rewritten_output",
    "S6_wrapper_model_proxying_another_model",
    "S7_distilled_or_fine_tuned_descendant_model",
]

EXPECTED_METHOD_FAMILIES = [
    "statistical_watermark",
    "publicly_detectable_watermark",
    "tee_or_2pc_public_watermark_protocol",
    "zk_inference_proof",
    "signed_metadata",
    "model_fingerprint_or_trigger",
    "provider_side_trace",
    "first_divergence_diagnostic",
    "public_deterministic_text_predicate",
]

ALLOWED_STATUSES = {
    "CONDITIONAL_METADATA_SUBSTRATE_NOT_ASSUMED",
    "CONDITIONAL_TRACE_SUBSTRATE_NOT_ASSUMED",
    "FAILS_METADATA_STRIPPED",
    "FAILS_NO_API_OR_MODEL_ACCESS",
    "FAILS_NO_FINAL_TEXT_SUBSTRATE",
    "FAILS_NO_PROTOCOL_SUBSTRATE",
    "FAILS_NO_TRACE_SUBSTRATE",
    "FAILS_PUBLIC_PREDICATE_SPOOFABLE",
    "SUPPORTED_TRACE_BOUND_DIAGNOSTIC",
    "UNTESTED_AND_AT_RISK_UNDER_CURRENT_SPOOFING_RECORD",
    "UNTESTED_API_DEPENDENT",
    "UNTESTED_NEEDS_SIGNATURE_PROTOCOL",
}

TRACE_METHODS = {"provider_side_trace", "first_divergence_diagnostic"}


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    audited: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], int] = {}

    scenario_set = {row.get("scenario_id", "") for row in rows}
    method_set = {row.get("method_family", "") for row in rows}
    if scenario_set != set(EXPECTED_SCENARIOS):
        failures.append(f"scenario set mismatch: {sorted(scenario_set)}")
    if method_set != set(EXPECTED_METHOD_FAMILIES):
        failures.append(f"method family set mismatch: {sorted(method_set)}")
    expected_row_count = len(EXPECTED_SCENARIOS) * len(EXPECTED_METHOD_FAMILIES)
    if len(rows) != expected_row_count:
        failures.append(f"row count mismatch: expected {expected_row_count}, got {len(rows)}")

    for row in rows:
        pair = (row.get("scenario_id", ""), row.get("method_family", ""))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    missing_pairs = [
        (scenario, method)
        for scenario in EXPECTED_SCENARIOS
        for method in EXPECTED_METHOD_FAMILIES
        if pair_counts.get((scenario, method), 0) == 0
    ]
    duplicate_pairs = [pair for pair, count in pair_counts.items() if count > 1]
    if missing_pairs:
        failures.append(f"missing scenario/method pairs: {missing_pairs[:5]}")
    if duplicate_pairs:
        failures.append(f"duplicate scenario/method pairs: {duplicate_pairs[:5]}")

    for row in rows:
        row_failures = validate_row(row)
        failures.extend(f"{row.get('scenario_id')}::{row.get('method_family')}: {failure}" for failure in row_failures)
        audited.append({**row, "row_rule_status": "PASS" if not row_failures else "FAIL", "row_rule_failures": "; ".join(row_failures)})

    supported_rows = [
        row
        for row in rows
        if row.get("current_assessment") == "SUPPORTED_TRACE_BOUND_DIAGNOSTIC"
    ]
    supported_pairs = {(row["scenario_id"], row["method_family"]) for row in supported_rows}
    expected_supported = {
        ("S2_cooperative_provider_with_trace_bundle", "provider_side_trace"),
        ("S2_cooperative_provider_with_trace_bundle", "first_divergence_diagnostic"),
    }
    if supported_pairs != expected_supported:
        failures.append(f"supported trace-bound pairs mismatch: {sorted(supported_pairs)}")

    public_text_success_rows = [
        row
        for row in rows
        if row.get("method_family") == "public_deterministic_text_predicate"
        and row.get("current_assessment", "").startswith("SUPPORTED")
    ]
    if public_text_success_rows:
        failures.append("public deterministic text predicate has supported rows")

    return audited, failures


def validate_row(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    scenario = row.get("scenario_id", "")
    method = row.get("method_family", "")
    status = row.get("current_assessment", "")
    substrate_available = row.get("substrate_available", "")
    claim_scope = row.get("claim_scope", "")

    if status not in ALLOWED_STATUSES:
        failures.append(f"unknown status {status!r}")
    if "no paper-facing positive claim" not in claim_scope:
        failures.append("claim scope does not preserve no-positive-claim boundary")

    if status == "SUPPORTED_TRACE_BOUND_DIAGNOSTIC":
        if scenario != "S2_cooperative_provider_with_trace_bundle":
            failures.append("supported trace-bound diagnostic outside S2")
        if method not in TRACE_METHODS:
            failures.append("supported diagnostic is not a trace method")
        if substrate_available != "yes":
            failures.append("supported diagnostic requires substrate_available=yes")

    if status in {"FAILS_NO_TRACE_SUBSTRATE", "FAILS_NO_PROTOCOL_SUBSTRATE", "FAILS_METADATA_STRIPPED"}:
        if substrate_available != "no":
            failures.append(f"{status} requires substrate_available=no")

    if status.startswith("CONDITIONAL_") and substrate_available != "maybe":
        failures.append(f"{status} requires substrate_available=maybe")

    if status == "FAILS_PUBLIC_PREDICATE_SPOOFABLE":
        if substrate_available != "yes":
            failures.append("spoofable public-predicate failure assumes visible final-text substrate")
        if "codeword recovered blocks total = 0" not in row.get("current_evidence", ""):
            failures.append("spoofable public-predicate row must cite zero codeword recovery")

    if method == "public_deterministic_text_predicate" and status.startswith("SUPPORTED"):
        failures.append("public deterministic text predicate cannot be supported in current VSG scope")

    return failures


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VSG Ownership Scenario Decision-Rule Audit",
        "",
        "This artifact-only audit checks the 7 x 9 ownership scenario matrix",
        "for schema completeness, supported-row boundaries, status-code rules,",
        "and claim-scope discipline. It does not start compute or create an",
        "ownership-proof claim.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['row_count']}`",
        f"Scenarios: `{summary['scenario_count']}`",
        f"Method families: `{summary['method_family_count']}`",
        f"Failures: `{summary['failure_count']}`",
        f"Supported trace-bound rows: `{summary['supported_trace_bound_row_count']}`",
        f"Supported public final-text rows: `{summary['supported_public_text_row_count']}`",
        "",
        "## Supported Rows",
        "",
    ]
    for pair in summary["supported_trace_bound_pairs"]:
        lines.append(f"- `{pair}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Trace-bound support is restricted to the cooperative trace-bundle scenario.",
            "- Public deterministic final-text predicates have zero supported rows.",
            "- The matrix remains a stress test and does not claim ownership proof.",
        ]
    )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(input_csv: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_rows(input_csv)
    audited_rows, failures = validate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "decision_rule_audit_rows.csv"
    summary_json = output_dir / "decision_rule_audit_summary.json"
    report_md = output_dir / "decision_rule_audit_report.md"
    manifest_json = output_dir / "decision_rule_audit_manifest.json"
    write_csv(rows_csv, audited_rows)

    supported_trace_rows = [
        row for row in rows if row.get("current_assessment") == "SUPPORTED_TRACE_BOUND_DIAGNOSTIC"
    ]
    supported_public_rows = [
        row
        for row in rows
        if row.get("method_family") == "public_deterministic_text_predicate"
        and row.get("current_assessment", "").startswith("SUPPORTED")
    ]
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "schema_name": "verification_substrate_gap_ownership_decision_rule_audit_v1",
        "input_csv": repo_rel(input_csv),
        "output_dir": repo_rel(output_dir),
        "row_count": len(rows),
        "scenario_count": len({row.get("scenario_id", "") for row in rows}),
        "method_family_count": len({row.get("method_family", "") for row in rows}),
        "failure_count": len(failures),
        "failures": failures,
        "supported_trace_bound_row_count": len(supported_trace_rows),
        "supported_public_text_row_count": len(supported_public_rows),
        "supported_trace_bound_pairs": [
            f"{row['scenario_id']}::{row['method_family']}" for row in supported_trace_rows
        ],
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "ownership_proof_claimed": False,
        "public_text_only_verification_claimed": False,
    }
    write_json(summary_json, summary)
    write_report(report_md, summary)
    manifest = {
        "status": summary["status"],
        "schema_name": "verification_substrate_gap_ownership_decision_rule_audit_manifest_v1",
        "files": [
            {"path": repo_rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [rows_csv, summary_json, report_md]
        ],
    }
    write_json(manifest_json, manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build(args.input_csv, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
