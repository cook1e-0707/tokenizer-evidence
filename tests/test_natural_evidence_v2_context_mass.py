import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.natural_evidence_v2.score_wp3_context_mass import (
    resolve_contextual_bucket_tokenization,
    validate_plan_rows,
    validate_tokenizer_boundaries,
)


class PrefixBoundaryTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def encode(self, text, add_special_tokens=False):
        if text == "":
            return []
        if text == "Alpha. ":
            return [10, 20]
        merged = {
            "Alpha. also": [10, 101],
            "Alpha. plus": [10, 102],
            "Alpha. too": [10, 103],
            "Alpha. again": [10, 104],
            "Alpha. exact": [10, 20, 105],
            "also": [201],
            "plus": [202],
            "too": [203],
            "again": [204],
        }
        if text in merged:
            return merged[text]
        raise AssertionError(f"unexpected tokenization request: {text!r}")


def plan_row(**overrides):
    row = {
        "plan_id": "unit_plan",
        "plan_row_id": "row_a",
        "candidate_bank_id": "discourse_marker_additive_v0",
        "casing_variant": "lowercase",
        "bucket_surfaces": {"0": ["also", "plus"], "1": ["too", "again"]},
        "prefix_before_candidate": "Alpha. ",
        "prefix_before_candidate_sha256": "sha",
        "template_preflight_only": True,
    }
    row.update(overrides)
    return row


def test_contextual_tokenization_repairs_shared_prefix_boundary():
    resolved = resolve_contextual_bucket_tokenization(PrefixBoundaryTokenizer(), plan_row())

    assert resolved["prefix_boundary_policy"] == "longest_common_token_prefix_boundary_repair"
    assert resolved["prefix_boundary_adjusted"] is True
    assert resolved["scoring_prefix_ids"] == [10]
    assert resolved["prefix_boundary_trimmed_token_count"] == 1
    assert resolved["bucket_token_ids"] == {"0": [101, 102], "1": [103, 104]}


def test_tokenizer_boundary_validation_is_model_free():
    row = plan_row()
    validation = validate_plan_rows([row])
    tokenizer_validation = validate_tokenizer_boundaries(
        tokenizer=PrefixBoundaryTokenizer(),
        plan_rows=[row],
        validation=validation,
        skip_invalid=False,
    )

    assert tokenizer_validation["status"] == "PASS_CONTEXT_MASS_TOKENIZER_BOUNDARY_VALIDATION"
    assert tokenizer_validation["model_scoring_started"] is False
    assert tokenizer_validation["prefix_boundary_adjusted_rows"] == 1
    assert tokenizer_validation["prefix_boundary_adjusted_surfaces"] == 4


def test_contextual_tokenization_rejects_mixed_scoring_prefixes():
    row = plan_row(bucket_surfaces={"0": ["also"], "1": ["exact"]})

    with pytest.raises(ValueError, match="do not share one scoring prefix"):
        resolve_contextual_bucket_tokenization(PrefixBoundaryTokenizer(), row)
