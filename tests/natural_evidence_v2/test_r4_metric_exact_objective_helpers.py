from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import logsumexp_softplus_margin_loss


def test_logsumexp_softplus_margin_loss_decreases_when_target_logit_rises() -> None:
    weak = logsumexp_softplus_margin_loss([0.0, 0.0, 0.0], [1], [2], margin_floor=1.0)
    strong = logsumexp_softplus_margin_loss([0.0, 2.0, 0.0], [1], [2], margin_floor=1.0)

    assert strong < weak


def test_logsumexp_softplus_margin_loss_increases_when_other_logit_rises() -> None:
    baseline = logsumexp_softplus_margin_loss([0.0, 1.0, 0.0], [1], [2], margin_floor=1.0)
    wrong_side = logsumexp_softplus_margin_loss([0.0, 1.0, 3.0], [1], [2], margin_floor=1.0)

    assert wrong_side > baseline


def test_logsumexp_softplus_margin_loss_supports_token_sets() -> None:
    single = logsumexp_softplus_margin_loss([0.0, 2.0, 0.0, 0.0], [1], [2, 3], margin_floor=0.5)
    expanded = logsumexp_softplus_margin_loss([0.0, 2.0, 2.0, 0.0], [1, 2], [3], margin_floor=0.5)

    assert expanded < single


def test_logsumexp_softplus_margin_loss_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        logsumexp_softplus_margin_loss([0.0, 0.0, 0.0], [1], [1], margin_floor=1.0)


def test_logsumexp_softplus_margin_loss_rejects_empty_sets() -> None:
    with pytest.raises(ValueError, match="target_ids"):
        logsumexp_softplus_margin_loss([0.0, 0.0, 0.0], [], [1], margin_floor=1.0)
    with pytest.raises(ValueError, match="other_ids"):
        logsumexp_softplus_margin_loss([0.0, 0.0, 0.0], [1], [], margin_floor=1.0)
