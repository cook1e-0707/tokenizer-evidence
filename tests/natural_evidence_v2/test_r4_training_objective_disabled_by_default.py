from __future__ import annotations

import json

from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import main
from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import parse_args


def test_wp5_repair_knobs_disabled_by_default(monkeypatch) -> None:
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

    args = parse_args()

    assert args.target_mass_floor == 0.0
    assert args.target_mass_floor_lambda == 0.0
    assert args.target_mass_ceiling == 0.0
    assert args.target_mass_ceiling_lambda == 0.0
    assert args.surface_margin_loss_mode == "mass_relu"
    assert args.stratum_weighting_mode == "none"
    assert args.task_ce_weight == 1.0


def test_wp5_dry_run_summary_records_task_ce_weight(monkeypatch, tmp_path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"row_id": "row-0"}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_wp5_micro_slot_lora.py",
            "--train-rows",
            str(rows),
            "--output-dir",
            str(out),
            "--arm",
            "protected",
            "--row-mode",
            "r4_prefix_native_surface",
            "--task-ce-weight",
            "0.0",
            "--dry-run",
        ],
    )

    assert main() == 0

    summary = json.loads((out / "wp5_micro_slot_lora_train_summary.json").read_text(encoding="utf-8"))
    assert summary["task_ce_weight"] == 0.0
