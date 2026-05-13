from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (
    chat_prefix,
    first_token_id_after_prefix,
    r4_row_surface_contract,
    validate_qwen_tokenizer_boundary_contract,
    validate_static_boundary_contract,
)


class Encoded:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class BoundaryTokenizer:
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> Encoded:
        if add_special_tokens:
            raise AssertionError("test tokenizer expects add_special_tokens=False")
        ids_by_text = {
            "User: Prompt\nAssistant: For this update, I will": [10],
            "User: Prompt\nAssistant: For this update, I will ": [10, 20],
            "User: Prompt\nAssistant: For this update, I will create": [10, 20],
            "User: Prompt\nAssistant: For this update, I will create a checklist": [10, 101, 110],
            "User: Prompt\nAssistant: For this update, I will plan": [10, 201],
            "User: Prompt\nAssistant: For this update, I will set": [10, 202],
            " create": [101],
            " create a checklist": [101, 110],
            " plan": [201],
            " set": [202],
        }
        try:
            return Encoded(ids_by_text[text])
        except KeyError as exc:
            raise AssertionError(f"unexpected tokenization request: {text!r}") from exc


def r4_row(**overrides):
    row = {
        "prompt_id": "prompt_a",
        "prompt_text": "Prompt",
        "coordinate_id": 0,
        "assistant_prefix_before_surface": "For this update, I will ",
        "target_bit": 1,
        "target_surface": "create a checklist",
        "bucket_0_surfaces": ["plan", "set"],
        "bucket_1_surfaces": ["create", "create a checklist"],
    }
    row.update(overrides)
    return row


def test_r4_boundary_contract_moves_trailing_space_to_tokenizer_surface() -> None:
    contract = r4_row_surface_contract(r4_row())

    assert contract["assistant_prefix_model_text"] == "For this update, I will"
    assert contract["surface_prefix_text"] == " "
    assert contract["target_surface_labels"] == ["create", "create a checklist"]
    assert contract["target_tokenizer_scored_surface_texts"] == [" create", " create a checklist"]
    assert contract["target_tokenizer_scored_surface_text"] == " create a checklist"


def test_static_boundary_contract_passes_without_tokenizer_authorization() -> None:
    validation = validate_static_boundary_contract([r4_row()])

    assert validation["status"] == "PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING"
    assert validation["checked_row_count"] == 1
    assert validation["failed_row_count"] == 0


def test_tokenizer_boundary_preflight_uses_repaired_surface_text() -> None:
    tokenizer = BoundaryTokenizer()
    row = r4_row()
    broken_prefix = chat_prefix(tokenizer, row["prompt_text"], row["assistant_prefix_before_surface"])

    with pytest.raises(ValueError, match="surface produced no next token"):
        first_token_id_after_prefix(tokenizer, broken_prefix, "create")

    validation = validate_qwen_tokenizer_boundary_contract(tokenizer, [row])

    assert validation["status"] == "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT"
    assert validation["failed_row_count"] == 0
    assert validation["empty_target_id_row_count"] == 0
    assert validation["empty_other_id_row_count"] == 0
    assert validation["target_other_overlap_row_count"] == 0
