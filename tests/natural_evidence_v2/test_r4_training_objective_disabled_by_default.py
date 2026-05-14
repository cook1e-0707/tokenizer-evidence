from __future__ import annotations

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
    assert args.stratum_weighting_mode == "none"
