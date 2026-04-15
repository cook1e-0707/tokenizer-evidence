from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.core.catalog_freeze import (
    build_catalog_remediation_review,
    load_catalog_freeze_report,
    save_catalog_remediation_review,
    save_catalog_remediation_table,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a remediation review for a failed catalog freeze attempt."
    )
    parser.add_argument("--audit-report", required=True, help="Path to tokenizer_audit_report JSON.")
    parser.add_argument("--change-log", required=True, help="Path to the existing markdown change log.")
    parser.add_argument(
        "--output-table",
        required=True,
        help="Output path for the machine-readable remediation table (.json or .yaml).",
    )
    parser.add_argument(
        "--output-review",
        required=True,
        help="Output path for the markdown remediation review.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    audit_report_path = _resolve_path(repo_root, args.audit_report)
    change_log_path = _resolve_path(repo_root, args.change_log)
    output_table_path = _resolve_path(repo_root, args.output_table)
    output_review_path = _resolve_path(repo_root, args.output_review)

    report_payload = load_catalog_freeze_report(audit_report_path)
    change_log_text = change_log_path.read_text(encoding="utf-8")
    review = build_catalog_remediation_review(
        report_payload=report_payload,
        change_log_text=change_log_text,
        change_log_path=change_log_path,
    )
    save_catalog_remediation_table(review, output_table_path)
    save_catalog_remediation_review(review, output_review_path)

    print(f"output_table={output_table_path}")
    print(f"output_review={output_review_path}")
    print(f"fields={len(review.fields)}")
    print(f"blocked_fields={','.join(review.blocked_fields) if review.blocked_fields else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
