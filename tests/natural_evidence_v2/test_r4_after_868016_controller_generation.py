from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.generate_r4_after_868016_controller_outputs import (
    FirstStepControllerLogitsProcessor,
    _controlled_scores_for_first_step,
)
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import ControllerConfig


torch = pytest.importorskip("torch")


def test_first_step_controller_increases_target_mass() -> None:
    scores = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    config = ControllerConfig(
        mode="additive",
        bonus_nats=1.0,
        penalty_nats=0.25,
        max_target_mass=0.50,
        max_kl_budget=0.50,
    )
    adjusted = _controlled_scores_for_first_step(
        scores=scores,
        row_target_ids=[[1]],
        row_other_ids=[[2]],
        controller_config=config,
    )
    base_probs = torch.softmax(scores[0], dim=-1)
    adjusted_probs = torch.softmax(adjusted[0], dim=-1)
    assert float(adjusted_probs[1]) > float(base_probs[1])
    assert float(adjusted_probs[2]) < float(base_probs[2])
    assert float(adjusted_probs[1]) <= 0.50 + 1e-6


def test_logits_processor_only_applies_at_initial_width() -> None:
    scores = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    config = ControllerConfig(
        mode="additive",
        bonus_nats=0.75,
        penalty_nats=0.0,
        max_target_mass=0.50,
        max_kl_budget=0.50,
    )
    processor = FirstStepControllerLogitsProcessor(
        initial_width=5,
        row_target_ids=[[1]],
        row_other_ids=[[2]],
        controller_config=config,
    )
    first_step = processor(torch.zeros((1, 5), dtype=torch.long), scores)
    later_step = processor(torch.zeros((1, 6), dtype=torch.long), scores)
    assert float(first_step[0, 1]) > float(scores[0, 1])
    assert torch.equal(later_step, scores)


def test_first_step_controller_rejects_target_other_overlap() -> None:
    config = ControllerConfig(
        mode="additive",
        bonus_nats=1.0,
        penalty_nats=0.0,
        max_target_mass=0.50,
        max_kl_budget=0.50,
    )
    with pytest.raises(ValueError, match="overlap"):
        _controlled_scores_for_first_step(
            scores=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            row_target_ids=[[1]],
            row_other_ids=[[1]],
            controller_config=config,
        )
