from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.infrastructure.paths import current_timestamp, discover_repo_root


FIELDS = [
    "method_id",
    "display_name",
    "method_family",
    "baseline_role",
    "fidelity_grade",
    "main_table_status",
    "paper_usage",
    "source_summary",
    "source_table",
    "target_count",
    "valid_completed_count",
    "success_count",
    "method_failure_count",
    "pending_count",
    "success_rate",
    "paper_ready",
    "adaptation_label",
    "required_label",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing baseline registry.")
    parser.add_argument(
        "--summary-out",
        default="results/processed/paper_stats/baseline_paper_registry_summary.json",
    )
    parser.add_argument("--table-out", default="results/tables/baseline_paper_registry.csv")
    parser.add_argument("--doc-out", default="docs/baseline_paper_registry.md")
    return parser.parse_args()


def _read_json(repo_root: Path, path: str) -> dict[str, Any]:
    full_path = repo_root / path
    if not full_path.exists():
        return {}
    payload = json.loads(full_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _method_row(summary: dict[str, Any], method_slug: str) -> dict[str, Any]:
    for row in summary.get("summary_rows", []):
        if row.get("method_slug") == method_slug:
            return row if isinstance(row, dict) else {}
    return {}


def _count(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(value or 0)


def _rate(row: dict[str, Any], key: str = "success_rate") -> float:
    value = row.get(key, 0.0)
    return float(value or 0.0)


def _paper_ready(value: Any) -> bool:
    return bool(value)


def _registry_row(
    *,
    method_id: str,
    display_name: str,
    method_family: str,
    baseline_role: str,
    fidelity_grade: str,
    main_table_status: str,
    paper_usage: str,
    source_summary: str,
    source_table: str,
    target_count: int,
    valid_completed_count: int,
    success_count: int,
    method_failure_count: int,
    pending_count: int,
    success_rate: float,
    paper_ready: bool,
    adaptation_label: str = "",
    required_label: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "display_name": display_name,
        "method_family": method_family,
        "baseline_role": baseline_role,
        "fidelity_grade": fidelity_grade,
        "main_table_status": main_table_status,
        "paper_usage": paper_usage,
        "source_summary": source_summary,
        "source_table": source_table,
        "target_count": target_count,
        "valid_completed_count": valid_completed_count,
        "success_count": success_count,
        "method_failure_count": method_failure_count,
        "pending_count": pending_count,
        "success_rate": success_rate,
        "paper_ready": paper_ready,
        "adaptation_label": adaptation_label,
        "required_label": required_label,
        "notes": notes,
    }


def _matched_budget_rows(matched: dict[str, Any]) -> list[dict[str, Any]]:
    source_summary = "results/processed/paper_stats/baseline_summary.json"
    source_table = "results/tables/matched_budget_baselines.csv"
    specs = [
        (
            "fixed_representative",
            "Fixed representative",
            "internal_ablation",
            "internal_ownership_ablation",
            "D",
            "appendix_or_internal_ablation",
            "Internal ablation, not an external active ownership baseline.",
        ),
        (
            "uniform_bucket",
            "Uniform bucket",
            "internal_ablation",
            "internal_ownership_ablation",
            "D",
            "appendix_or_internal_ablation",
            "Internal ablation, not an external active ownership baseline.",
        ),
        (
            "english_random_active_fingerprint",
            "English-random active fingerprint",
            "weak_proxy",
            "negative_diagnostic_baseline",
            "D",
            "appendix_negative_diagnostic",
            "Weak natural-language proxy; valid failures remain in denominator.",
        ),
    ]
    rows = []
    for method_id, display_name, family, role, grade, status, notes in specs:
        method = _method_row(matched, method_id)
        rows.append(
            _registry_row(
                method_id=method_id,
                display_name=display_name,
                method_family=family,
                baseline_role=role,
                fidelity_grade=grade,
                main_table_status=status,
                paper_usage="appendix_or_ablation",
                source_summary=source_summary,
                source_table=source_table,
                target_count=_count(method, "target_count"),
                valid_completed_count=_count(method, "valid_completed_count"),
                success_count=_count(method, "success_count"),
                method_failure_count=_count(method, "method_failure_count"),
                pending_count=_count(method, "pending_count"),
                success_rate=_rate(method),
                paper_ready=_paper_ready(matched.get("paper_ready")),
                notes=notes,
            )
        )
    kgw = _method_row(matched, "kgw_provenance_control")
    rows.append(
        _registry_row(
            method_id="kgw_provenance_control",
            display_name="KGW provenance control",
            method_family="task_mismatched_provenance_control",
            baseline_role="provenance_control",
            fidelity_grade="F_for_ownership_baseline",
            main_table_status="excluded_task_mismatched_control",
            paper_usage="related_work_or_control_only",
            source_summary=source_summary,
            source_table=source_table,
            target_count=_count(kgw, "target_count"),
            valid_completed_count=_count(kgw, "valid_completed_count"),
            success_count=_count(kgw, "success_count"),
            method_failure_count=_count(kgw, "method_failure_count"),
            pending_count=_count(kgw, "pending_count"),
            success_rate=_rate(kgw),
            paper_ready=False,
            required_label="task-mismatched provenance control",
            notes="Text provenance watermarking is not a primary model-ownership payload-recovery baseline.",
        )
    )
    return rows


def _official_perinucleus_row(summary: dict[str, Any]) -> dict[str, Any]:
    source_summary = "results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json"
    source_table = "results/tables/baseline_perinucleus_official_qwen_final.csv"
    return _registry_row(
        method_id="scalable_fingerprinting_perinucleus_official_qwen_final",
        display_name="Scalable Fingerprinting / Perinucleus official Qwen adaptation",
        method_family="external_active_ownership_baseline",
        baseline_role="primary_external_baseline",
        fidelity_grade="A_adapted",
        main_table_status="eligible_with_adaptation_label",
        paper_usage="main_external_baseline",
        source_summary=source_summary,
        source_table=source_table,
        target_count=_count(summary, "target_count"),
        valid_completed_count=_count(summary, "valid_completed_count"),
        success_count=_count(summary, "success_count"),
        method_failure_count=_count(summary, "method_failure_count"),
        pending_count=_count(summary, "pending_count"),
        success_rate=_rate(summary),
        paper_ready=_paper_ready(summary.get("paper_ready")),
        adaptation_label="official-code Qwen LoRA adaptation",
        required_label="Qwen-adapted official Scalable/Perinucleus baseline",
        notes=(
            "Passed forensic replay, single-fingerprint overfit, Llama anchor, Qwen capacity selection, "
            "candidate utility sanity, and 48/48 final exact-verifier cases."
        ),
    )


def _chain_hash_row(summary: dict[str, Any]) -> dict[str, Any]:
    source_summary = "results/processed/paper_stats/baseline_chain_hash_summary.json"
    source_table = "results/tables/baseline_chain_hash.csv"
    return _registry_row(
        method_id="chain_hash_qwen_v1",
        display_name="Chain&Hash-style Qwen package",
        method_family="external_active_ownership_baseline",
        baseline_role="external_baseline_candidate",
        fidelity_grade="C_pending",
        main_table_status="not_eligible_pending_execution",
        paper_usage="protocol_pending_or_appendix_only",
        source_summary=source_summary,
        source_table=source_table,
        target_count=_count(summary, "target_count"),
        valid_completed_count=_count(summary, "valid_completed_count"),
        success_count=_count(summary, "success_count"),
        method_failure_count=_count(summary, "method_failure_count"),
        pending_count=_count(summary, "pending_count"),
        success_rate=_rate(summary, "clean_verification_success_rate"),
        paper_ready=_paper_ready(summary.get("paper_ready")),
        notes="Prepared package only; no final rows, calibration, utility, or anchor evidence in current artifacts.",
    )


def _excluded_diagnostic_rows(legacy_perinucleus: dict[str, Any]) -> list[dict[str, Any]]:
    source_summary = "results/processed/paper_stats/diagnostics/baseline_perinucleus_legacy_diagnostic_summary.json"
    source_table = "results/tables/diagnostics/baseline_perinucleus_legacy_diagnostic.csv"
    return [
        _registry_row(
            method_id="perinucleus_no_train_diagnostic",
            display_name="Perinucleus no-train / legacy diagnostic",
            method_family="failed_or_incomplete_external_diagnostic",
            baseline_role="excluded_diagnostic",
            fidelity_grade="F_for_scalable_claim",
            main_table_status="excluded_do_not_use_for_scalable_claim",
            paper_usage="internal_diagnostic_only",
            source_summary=source_summary,
            source_table=source_table,
            target_count=_count(legacy_perinucleus, "target_count"),
            valid_completed_count=_count(legacy_perinucleus, "valid_completed_count"),
            success_count=_count(legacy_perinucleus, "success_count"),
            method_failure_count=_count(legacy_perinucleus, "method_failure_count"),
            pending_count=_count(legacy_perinucleus, "pending_count"),
            success_rate=_rate(legacy_perinucleus, "exact_gate_success_rate"),
            paper_ready=False,
            required_label="not Scalable Fingerprinting",
            notes=(
                "Retained only as a quarantined diagnostic artifact. Do not merge into the official Perinucleus row; "
                "use baseline_perinucleus_official_qwen_final_* for paper-facing Scalable/Perinucleus claims."
            ),
        )
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Paper Registry",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Decision",
        "",
        (
            "The paper now has one paper-ready external active ownership baseline: "
            "`scalable_fingerprinting_perinucleus_official_qwen_final`. It must be reported with the "
            "`Qwen-adapted official Scalable/Perinucleus baseline` label."
        ),
        "",
        "The legacy adapted `baseline_perinucleus` artifacts remain excluded from Scalable Fingerprinting claims and are quarantined under `results/.../diagnostics/`.",
        "",
        "## Registry",
        "",
        "| method_id | main_table_status | target | success | rate | paper_ready | required_label |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method_id} | {main_table_status} | {target_count} | {success_count} | {success_rate:.3f} | {paper_ready} | {required_label} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not use `results/tables/baseline_perinucleus.csv` or the quarantined diagnostic copy as the successful Scalable/Perinucleus result.",
            "- Use `results/tables/baseline_perinucleus_official_qwen_final.csv` for the official Qwen-adapted Perinucleus result.",
            "- Keep valid method failures in denominators; do not convert failures into exclusions.",
            "- KGW/PostMark-style provenance controls must stay task-mismatched controls, not ownership baselines.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    matched = _read_json(repo_root, "results/processed/paper_stats/baseline_summary.json")
    official_perinucleus = _read_json(
        repo_root, "results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json"
    )
    chain_hash = _read_json(repo_root, "results/processed/paper_stats/baseline_chain_hash_summary.json")
    legacy_perinucleus = _read_json(
        repo_root, "results/processed/paper_stats/diagnostics/baseline_perinucleus_legacy_diagnostic_summary.json"
    )
    rows = [
        *_matched_budget_rows(matched),
        _official_perinucleus_row(official_perinucleus),
        _chain_hash_row(chain_hash),
        *_excluded_diagnostic_rows(legacy_perinucleus),
    ]
    main_external = [
        row["method_id"]
        for row in rows
        if row["paper_usage"] == "main_external_baseline"
        and row["paper_ready"]
        and row["main_table_status"] == "eligible_with_adaptation_label"
    ]
    summary = {
        "schema_name": "baseline_paper_registry_summary",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "method_count": len(rows),
        "paper_ready_external_baseline_count": len(main_external),
        "main_external_baselines": main_external,
        "main_table_allowed_with_labels": [
            {
                "method_id": row["method_id"],
                "required_label": row["required_label"],
                "source_table": row["source_table"],
                "success_count": row["success_count"],
                "target_count": row["target_count"],
                "success_rate": row["success_rate"],
            }
            for row in rows
            if row["method_id"] in main_external
        ],
        "appendix_or_diagnostic_methods": [
            row["method_id"]
            for row in rows
            if row["paper_usage"] in {"appendix_or_ablation", "appendix_negative_diagnostic", "internal_diagnostic_only"}
        ],
        "blocked_or_pending_methods": [
            row["method_id"]
            for row in rows
            if row["main_table_status"].startswith("not_eligible")
            or row["main_table_status"].startswith("excluded")
        ],
        "guardrails": [
            "Official Perinucleus final results are distinct from legacy adapted baseline_perinucleus artifacts.",
            "Report official Perinucleus as a Qwen-adapted official-code baseline, not as an unmodified full fine-tune reproduction.",
            "Do not report task-mismatched provenance controls as primary ownership baselines.",
        ],
        "sources": {
            "matched_budget_baselines": "results/processed/paper_stats/baseline_summary.json",
            "official_perinucleus_qwen_final": "results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json",
            "chain_hash": "results/processed/paper_stats/baseline_chain_hash_summary.json",
        },
        "rows": rows,
    }
    summary_path = repo_root / args.summary_out
    table_path = repo_root / args.table_out
    doc_path = repo_root / args.doc_out
    _write_json(summary_path, summary)
    _write_csv(table_path, rows)
    _write_doc(doc_path, summary, rows)
    print(f"wrote baseline paper registry summary to {summary_path}")
    print(f"wrote baseline paper registry table to {table_path}")
    print(f"wrote baseline paper registry doc to {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
