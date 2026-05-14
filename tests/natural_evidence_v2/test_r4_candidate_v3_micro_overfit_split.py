from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.natural_evidence_v2.build_r4_candidate_v3_micro_overfit_split import (
    DEFAULT_SOURCE,
    EXPECTED_SOURCE_SHA256,
    build_split,
    read_jsonl,
    row_key,
    select_split,
    sha256_file,
    validate_rows,
)
from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import parse_args, weight_for_stratum


def test_candidate_v3_source_hash_is_the_reviewed_hash() -> None:
    assert sha256_file(DEFAULT_SOURCE) == EXPECTED_SOURCE_SHA256


def test_candidate_v3_rows_pass_contract_validation() -> None:
    rows = read_jsonl(DEFAULT_SOURCE)
    summary = validate_rows(rows)

    assert len(rows) == 8192
    assert summary["duplicate_key_count"] == 0
    assert summary["row_error_count"] == 0


def test_split_is_exact_disjoint_and_balanced() -> None:
    rows = read_jsonl(DEFAULT_SOURCE)
    train_rows, heldout_rows = select_split(rows, train_count=512, heldout_count=512)

    train_keys = {row_key(row) for row in train_rows}
    heldout_keys = {row_key(row) for row in heldout_rows}
    assert len(train_rows) == 512
    assert len(heldout_rows) == 512
    assert not (train_keys & heldout_keys)

    train_strata = {(row["prefix_family_id"], row["coordinate_id"], row["target_bit"]) for row in train_rows}
    heldout_strata = {(row["prefix_family_id"], row["coordinate_id"], row["target_bit"]) for row in heldout_rows}
    assert len(train_strata) == 32
    assert len(heldout_strata) == 32


def test_build_split_writes_no_compute_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "split"
    summary = build_split(
        source_path=DEFAULT_SOURCE,
        output_dir=output_dir,
        expected_source_sha256=EXPECTED_SOURCE_SHA256,
        train_count=64,
        heldout_count=64,
    )

    assert summary["status"] == "ARTIFACT_ONLY_R4_CANDIDATE_V3_MICRO_OVERFIT_SPLIT_BUILT_NO_COMPUTE"
    assert summary["train_row_count"] == 64
    assert summary["heldout_row_count"] == 64
    assert summary["score_row_count"] == 8192
    assert summary["training_allowed"] is False
    assert summary["slurm_submission_allowed"] is False

    first_train = json.loads((output_dir / "train_rows.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_train["micro_overfit_split"] == "train"
    assert first_train["source_candidate_v3_rows_sha256"] == EXPECTED_SOURCE_SHA256

    with pytest.raises(FileExistsError):
        build_split(
            source_path=DEFAULT_SOURCE,
            output_dir=output_dir,
            expected_source_sha256=EXPECTED_SOURCE_SHA256,
            train_count=64,
            heldout_count=64,
        )


def test_r4_row_mode_is_disabled_by_default_but_parseable(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_wp5_micro_slot_lora.py",
            "--train-rows",
            "rows.jsonl",
            "--output-dir",
            "out",
            "--arm",
            "protected",
        ],
    )
    assert parse_args().row_mode == "wp5_micro_slot"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_wp5_micro_slot_lora.py",
            "--train-rows",
            "rows.jsonl",
            "--output-dir",
            "out",
            "--arm",
            "protected",
            "--row-mode",
            "r4_prefix_native_surface",
        ],
    )
    assert parse_args().row_mode == "r4_prefix_native_surface"


def test_composite_stratum_uses_reviewed_max_component_weight() -> None:
    stratum = "assistant_prefix_model_text:A useful action is:|target_surface:Prepare questions"
    assert weight_for_stratum(stratum, "r4_candidate_v3_failed_surface", 3.0) == 3.0
