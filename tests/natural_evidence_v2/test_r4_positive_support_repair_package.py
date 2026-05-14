from __future__ import annotations

import json
from pathlib import Path

from scripts.natural_evidence_v2.build_r4_positive_support_repair_package import (
    DEFAULT_CONFIG,
    build_package,
)


def test_build_support_repair_package_static_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "support_repair"

    summary = build_package(DEFAULT_CONFIG, output_dir)

    assert summary["status"] == "PASS_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
    assert summary["event_window_rows"] >= 48
    assert summary["toy_positive_accept"] is True
    assert summary["wrong_key_accept"] is False
    assert summary["wrong_payload_accept"] is False
    assert summary["post_hoc_phrase_mining_allowed"] is False
    assert summary["generation_allowed"] is False

    contract = json.loads((output_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract["contract_id"] == "r4_positive_support_repair_v2"
    assert contract["unchanged_resubmission_allowed"] is False

    bank = json.loads((output_dir / "event_window_bank.json").read_text(encoding="utf-8"))
    assert all(row["source_policy"] == "independent_static_taxonomy_not_859277_transcripts" for row in bank)
    assert all(row["structural_marker"] is False for row in bank)
    assert all(row["technical_literal"] is False for row in bank)

