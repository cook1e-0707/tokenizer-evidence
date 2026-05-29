from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.generate_r4_after_868016_controller_outputs import validate_reviews


def _controller_review() -> dict[str, object]:
    return {
        "status": "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE",
        "teacher_forced_gate_pass": True,
    }


def test_validate_reviews_prefers_review_status_for_871057_prefar_review() -> None:
    validate_reviews(
        {
            "status": "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT",
            "review_status": "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_QWEN_TOKENIZER_PREFLIGHT_871057",
            "failed_row_count": 0,
        },
        _controller_review(),
    )


def test_validate_reviews_rejects_generic_tokenizer_status_without_review_status() -> None:
    with pytest.raises(ValueError, match="tokenizer review is not an allowed reviewed pass"):
        validate_reviews(
            {
                "status": "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT",
                "failed_row_count": 0,
            },
            _controller_review(),
        )
