from __future__ import annotations

import json

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (
    DEFAULT_CONFIG,
    build_package,
    build_surface_bank,
    load_yaml,
    validate_package,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import validate_contract


def test_positive_event_bank_precommit_builds_without_compute(tmp_path) -> None:
    summary = build_package(DEFAULT_CONFIG, tmp_path)

    assert summary["status"] == "PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE"
    assert summary["artifact_only"] is True
    assert summary["slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["training_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["surface_count"] >= 96
    assert summary["surface_family_count"] >= 4
    assert summary["distinct_coordinate_count"] >= 20
    assert summary["key_material_exposed"] is False

    for name in (
        "surface_bank.json",
        "coordinate_mapping.jsonl",
        "codebook.json",
        "decoder_spec.json",
        "dev_gate.json",
        "precommit_manifest.json",
        "package_summary.json",
        "package_report.md",
    ):
        assert (tmp_path / name).exists()


def test_precommit_manifest_does_not_expose_key_material(tmp_path) -> None:
    build_package(DEFAULT_CONFIG, tmp_path)
    manifest = json.loads((tmp_path / "precommit_manifest.json").read_text(encoding="utf-8"))
    manifest_text = json.dumps(manifest, sort_keys=True)

    assert manifest["key_material_exposed"] is False
    assert "r4_positive_event_bank_dev_mapping_key_static_v1" not in manifest_text
    assert "r4_positive_event_bank_wrong_dev_mapping_key_static_v1" not in manifest_text
    assert manifest["audit_key_commitment"]
    assert manifest["wrong_audit_key_commitment"]


def test_surface_bank_rejects_forbidden_public_literals() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    contract = load_yaml(DEFAULT_CONFIG.parent / "r4_positive_evidence_contract_redesign.yaml")
    surfaces = build_surface_bank(config)
    surfaces[0] = {**surfaces[0], "canonical_phrase": "mention the hidden signal"}

    summary = validate_package(
        config=config,
        contract_summary=validate_contract(contract),
        surface_bank=surfaces,
        coordinate_mapping=[
            {"coordinate_id": index % 32, "polarity": 1, "surface_id": surface["surface_id"]}
            for index, surface in enumerate(surfaces)
        ],
    )

    assert summary["status"].startswith("FAIL")
    assert any("forbidden literal" in error for error in summary["errors"])


def test_surface_family_concentration_is_capped() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    contract = load_yaml(DEFAULT_CONFIG.parent / "r4_positive_evidence_contract_redesign.yaml")
    surfaces = build_surface_bank(config)
    concentrated = [{**surface, "surface_family": "one_family"} for surface in surfaces]

    summary = validate_package(
        config=config,
        contract_summary=validate_contract(contract),
        surface_bank=concentrated,
        coordinate_mapping=[
            {"coordinate_id": index % 32, "polarity": 1, "surface_id": surface["surface_id"]}
            for index, surface in enumerate(concentrated)
        ],
    )

    assert summary["status"].startswith("FAIL")
    assert "distinct surface families below configured minimum" in summary["errors"]
    assert "single surface family fraction exceeds cap" in summary["errors"]


def test_package_summary_hashes_are_consistent(tmp_path) -> None:
    summary = build_package(DEFAULT_CONFIG, tmp_path)
    saved = json.loads((tmp_path / "package_summary.json").read_text(encoding="utf-8"))

    assert saved["precommit_hash"] == summary["precommit_hash"]
    assert saved["surface_bank_sha256"] == summary["surface_bank_sha256"]
    assert saved["coordinate_mapping_sha256"] == summary["coordinate_mapping_sha256"]
