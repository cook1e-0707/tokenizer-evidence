from __future__ import annotations

import math
from argparse import Namespace

import pytest

from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import (
    extract_training_stratum,
    target_mass_ceiling_loss,
    target_mass_floor_loss,
    validate_objective_args,
    weight_for_stratum,
    weighted_mean,
)


def test_weighted_mean_reports_raw_and_effective_counts() -> None:
    summary = weighted_mean([1.0, 3.0], [1.0, 3.0])

    assert summary == {
        "effective_weighted_count": 4.0,
        "mean": 2.5,
        "raw_count": 2.0,
    }


def test_weighted_mean_fails_closed_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        weighted_mean([1.0], [1.0, 2.0])


def test_target_mass_floor_loss_is_relu_floor_gap() -> None:
    assert target_mass_floor_loss(0.15, 0.40) == pytest.approx(0.25)
    assert target_mass_floor_loss(0.60, 0.40) == 0.0


def test_target_mass_ceiling_loss_is_relu_ceiling_gap() -> None:
    assert target_mass_ceiling_loss(0.60, 0.40) == pytest.approx(0.20)
    assert target_mass_ceiling_loss(0.15, 0.40) == 0.0
    assert target_mass_ceiling_loss(0.60, 0.0) == 0.0


def test_stratum_extraction_prefers_slot_surface_metadata() -> None:
    row = {"family_id": "F2_16_step_checklist_step_label_r1_expansion"}
    slot = {"target_surface": "Prepare a note", "bit_role": "payload"}

    assert extract_training_stratum(row, slot) == "target_surface:Prepare a note"


def test_r4_candidate_v3_failed_surface_weights_are_capped() -> None:
    assert weight_for_stratum("target_surface:Prepare a note", "r4_candidate_v3_failed_surface", 2.5) == 2.5
    assert weight_for_stratum("target_surface:Prepare questions", "r4_candidate_v3_failed_surface", 3.0) == 2.0
    assert weight_for_stratum("target_surface:Create a short summary", "r4_candidate_v3_failed_surface", 3.0) == 1.0
    assert weight_for_stratum("target_surface:Prepare a note", "none", 3.0) == 1.0


def test_objective_args_reject_invalid_floor_and_weights() -> None:
    args = Namespace(
        task_ce_weight=1.0,
        margin_lambda=5.0,
        margin_tau=0.15,
        target_mass_floor=0.4,
        target_mass_floor_lambda=10.0,
        target_mass_ceiling=0.8,
        target_mass_ceiling_lambda=1.0,
        stratum_weighting_mode="none",
        stratum_weight_max=3.0,
    )
    validate_objective_args(args)

    args.target_mass_floor_lambda = -1.0
    with pytest.raises(ValueError, match="target-mass-floor-lambda"):
        validate_objective_args(args)

    args.target_mass_floor_lambda = 10.0
    args.target_mass_floor = 1.1
    with pytest.raises(ValueError, match="target-mass-floor"):
        validate_objective_args(args)

    args.target_mass_floor = 0.4
    args.target_mass_ceiling = 1.1
    with pytest.raises(ValueError, match="target-mass-ceiling"):
        validate_objective_args(args)

    args.target_mass_ceiling = 0.3
    with pytest.raises(ValueError, match="target-mass-floor.*target-mass-ceiling"):
        validate_objective_args(args)

    args.target_mass_ceiling = 0.8
    args.stratum_weight_max = 0.0
    with pytest.raises(ValueError, match="stratum-weight-max"):
        validate_objective_args(args)


def test_objective_args_reject_non_finite_values() -> None:
    args = Namespace(
        task_ce_weight=1.0,
        margin_lambda=5.0,
        margin_tau=0.15,
        target_mass_floor=0.4,
        target_mass_floor_lambda=math.nan,
        target_mass_ceiling=0.8,
        target_mass_ceiling_lambda=1.0,
        stratum_weighting_mode="none",
        stratum_weight_max=3.0,
    )
    with pytest.raises(ValueError, match="target-mass-floor-lambda.*finite"):
        validate_objective_args(args)

    args.target_mass_floor_lambda = 10.0
    args.target_mass_ceiling_lambda = math.nan
    with pytest.raises(ValueError, match="target-mass-ceiling-lambda.*finite"):
        validate_objective_args(args)

    args.target_mass_ceiling_lambda = 1.0
    args.stratum_weight_max = math.inf
    with pytest.raises(ValueError, match="stratum-weight-max.*finite"):
        validate_objective_args(args)


def test_objective_args_reject_unknown_weighting_mode() -> None:
    args = Namespace(
        task_ce_weight=1.0,
        margin_lambda=5.0,
        margin_tau=0.15,
        target_mass_floor=0.4,
        target_mass_floor_lambda=10.0,
        target_mass_ceiling=0.8,
        target_mass_ceiling_lambda=1.0,
        stratum_weighting_mode="posthoc_mode",
        stratum_weight_max=3.0,
    )

    with pytest.raises(ValueError, match="stratum-weighting-mode"):
        validate_objective_args(args)
