from __future__ import annotations

import json

from scripts.natural_evidence_v2.build_r4_transfer_gap_repair_package import (
    PROMPT_SCAFFOLDS,
    build_package,
    main,
)
from scripts.natural_evidence_v2.validate_r4_transfer_gap_repair_plan import DEFAULT_CONFIG, load_yaml


def test_transfer_gap_repair_package_passes() -> None:
    summary = build_package(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY"
    assert summary["prompt_scaffold_count"] == 6
    assert summary["prefix_family_count"] == 2
    assert summary["current_compute_unlocked"] is False
    assert summary["future_compute_conditionally_authorized_after_prerequisites"] is True


def test_prompt_scaffolds_avoid_disallowed_public_structures() -> None:
    haystack = "\n".join(str(row["template"]) for row in PROMPT_SCAFFOLDS).lower()

    for term in ("step ", "exactly 16", "slot", "bucket", "fingerprint", "watermark", "payload", "secret key"):
        assert term not in haystack


def test_transfer_gap_package_writer_outputs_expected_files(tmp_path, monkeypatch, capsys) -> None:
    output_dir = tmp_path / "repair_package"
    monkeypatch.setattr(
        "sys.argv",
        ["build_r4_transfer_gap_repair_package.py", "--output-dir", str(output_dir)],
    )

    assert main() == 0

    expected = {
        "prompt_scaffold_templates.jsonl",
        "prefix_family_policy.json",
        "forbidden_matcher_policy.json",
        "structural_leakage_policy.json",
        "future_route_constraints.json",
        "repair_package_summary.json",
        "repair_package_report.md",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    summary = json.loads((output_dir / "repair_package_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY"
    assert "PASS_R4_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY" in capsys.readouterr().out
