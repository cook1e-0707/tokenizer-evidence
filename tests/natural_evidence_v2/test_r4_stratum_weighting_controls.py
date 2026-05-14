from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import weight_for_stratum


def test_stratum_weighting_none_is_neutral() -> None:
    assert weight_for_stratum("target_surface:Prepare a note", "none", 3.0) == 1.0


def test_reviewed_weak_strata_are_upweighted_and_capped() -> None:
    assert weight_for_stratum("target_surface:Prepare a note", "r4_candidate_v3_failed_surface", 2.5) == 2.5
    assert weight_for_stratum("target_surface:Prepare questions", "r4_candidate_v3_failed_surface", 3.0) == 2.0
    assert weight_for_stratum("target_surface:Create a short summary", "r4_candidate_v3_failed_surface", 3.0) == 1.0


def test_unknown_weighting_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        weight_for_stratum("target_surface:Prepare a note", "posthoc_mode", 3.0)
