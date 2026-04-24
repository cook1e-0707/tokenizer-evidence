import csv
import json
import subprocess
import sys
from pathlib import Path

from src.infrastructure.paths import discover_repo_root


def test_diagnose_g3a_v1_failures_writes_non_overwriting_diagnostics(tmp_path: Path) -> None:
    repo_root = discover_repo_root(Path(__file__).parent)
    output_dir = tmp_path / "paper_stats"
    tables_dir = tmp_path / "tables"
    docs_dir = tmp_path / "docs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_g3a_v1_failures.py",
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
            "--docs-dir",
            str(docs_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "wrote diagnostic summary" in completed.stdout
    summary = json.loads((output_dir / "g3a_v1_diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_name"] == "g3a_v1_diagnostic_summary"
    assert summary["completed_case_count"] == 36
    assert summary["included_case_count"] == 29
    assert summary["excluded_case_count"] == 7
    assert summary["run_diagnostic_count"] == 36
    assert summary["slot_diagnostic_count"] == 168
    assert summary["conclusion"] == "ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required"

    run_rows = list(csv.DictReader((tables_dir / "g3a_v1_run_diagnostics.csv").open()))
    failure_rows = list(csv.DictReader((tables_dir / "g3a_v1_failure_cases.csv").open()))
    slot_rows = list(csv.DictReader((tables_dir / "g3a_v1_slot_diagnostics.csv").open()))
    assert len(run_rows) == 36
    assert len(failure_rows) == 7
    assert len(slot_rows) == 168
    assert {row["case_id"] for row in failure_rows} == {
        "B1_U03_s23",
        "B1_U12_s23",
        "B1_U15_s23",
        "B4_U00_s29",
        "B4_U03_s29",
        "B4_U12_s29",
        "B4_U15_s29",
    }
    assert all(row["decoded_block_count_correct"] == "True" for row in failure_rows)
    assert all(row["parser_success"] == "True" for row in failure_rows)
    assert "ROOT_CAUSE_NOT_CONFIRMED: additional instrumentation required" in (
        docs_dir / "g3a_v1_failure_analysis.md"
    ).read_text(encoding="utf-8")
