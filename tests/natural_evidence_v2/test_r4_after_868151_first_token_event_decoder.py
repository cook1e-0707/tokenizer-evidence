from __future__ import annotations

from collections import Counter

import pytest

from scripts.natural_evidence_v2.decode_r4_after_868151_first_token_event_channel import (
    checksum_valid,
    classify_first_token_event,
    contextual_technical_literal_hits,
    decode_generated_rows,
    first_lexical_event_after_exact_prefix,
)


def _score_row(*, coord: int, target_bit: int) -> dict[str, object]:
    return {
        "prompt_id": "p0",
        "coordinate_id": coord,
        "prefix_family_id": "pf",
        "target_surface": "target surface",
        "target_bit": target_bit,
        "assistant_prefix_before_surface": "A useful action is to ",
        "bucket_0_surfaces": ["review the notes", "check the notes"],
        "bucket_1_surfaces": ["clarify the notes", "summarize the notes"],
    }


def _generated_row(*, coord: int, target_bit: int, response: str = "A useful action is to clarify now") -> dict[str, object]:
    return {
        "prompt_id": "p0",
        "coordinate_id": coord,
        "prefix_family_id": "pf",
        "target_surface": "target surface",
        "target_bit": target_bit,
        "generation_condition": "protected",
        "replicate_group_id": "shard_00",
        "response_text": response,
        "response_text_sha256": f"hash-{coord}",
    }


def test_first_lexical_event_requires_exact_prefix() -> None:
    assert first_lexical_event_after_exact_prefix("A useful action is to , clarify now", "A useful action is to ") == "clarify"
    assert first_lexical_event_after_exact_prefix("Different prefix clarify now", "A useful action is to ") == ""


def test_token_id_trace_is_required_unless_old_text_fallback_is_explicit() -> None:
    row = _generated_row(coord=6, target_bit=1)
    score = _score_row(coord=6, target_bit=1)

    with pytest.raises(ValueError, match="no first_generated_token_id"):
        classify_first_token_event(
            generated_row=row,
            score_row=score,
            allow_text_fallback_for_old_transcripts=False,
        )


def test_text_fallback_maps_target_and_other_through_target_bit() -> None:
    score = _score_row(coord=6, target_bit=1)
    target_row = _generated_row(coord=6, target_bit=1, response="A useful action is to clarify now")
    other_row = _generated_row(coord=6, target_bit=1, response="A useful action is to review now")

    target = classify_first_token_event(
        generated_row=target_row,
        score_row=score,
        allow_text_fallback_for_old_transcripts=True,
    )
    other = classify_first_token_event(
        generated_row=other_row,
        score_row=score,
        allow_text_fallback_for_old_transcripts=True,
    )

    assert target["event_source"] == "text_fallback_old_transcript"
    assert target["event_status"] == "target"
    assert target["vote_bit"] == "1"
    assert other["event_status"] == "other"
    assert other["vote_bit"] == "0"


def test_token_id_trace_maps_target_and_other_through_target_bit() -> None:
    score = _score_row(coord=6, target_bit=0)
    row = _generated_row(coord=6, target_bit=0)
    row.update(
        {
            "first_generated_token_id": 101,
            "first_generated_token_text": "review",
            "target_first_token_ids": [101],
            "other_first_token_ids": [202],
        }
    )

    event = classify_first_token_event(
        generated_row=row,
        score_row=score,
        allow_text_fallback_for_old_transcripts=False,
    )

    assert event["event_source"] == "token_id_trace"
    assert event["event_status"] == "target"
    assert event["vote_bit"] == "0"


def test_token_id_overlap_fails_closed() -> None:
    score = _score_row(coord=6, target_bit=0)
    row = _generated_row(coord=6, target_bit=0)
    row.update(
        {
            "first_generated_token_id": 101,
            "target_first_token_ids": [101],
            "other_first_token_ids": [101],
        }
    )

    with pytest.raises(ValueError, match="overlap"):
        classify_first_token_event(
            generated_row=row,
            score_row=score,
            allow_text_fallback_for_old_transcripts=False,
        )


def test_decode_generated_rows_accepts_clean_token_id_trace() -> None:
    expected_bits = [1, 0, 1, 0, 0, 1, 0, 1]
    codebook = {
        "expected_codeword_bits": expected_bits,
        "pair_to_bit_mapping": [{"bit_index": index, "coordinates": [index]} for index in range(8)],
    }
    score_rows = {}
    generated_rows = []
    for coord, bit in enumerate(expected_bits):
        score = _score_row(coord=coord, target_bit=bit)
        key = ("p0", coord, "pf", "target surface")
        score_rows[key] = score
        row = _generated_row(coord=coord, target_bit=bit)
        row.update(
            {
                "first_generated_token_id": 1000 + coord,
                "first_generated_token_text": "target",
                "target_first_token_ids": [1000 + coord],
                "other_first_token_ids": [2000 + coord],
                "response_text": f"clean response {coord}",
                "response_text_sha256": f"unique-hash-{coord}",
            }
        )
        generated_rows.append(row)

    decoded_rows, event_rows = decode_generated_rows(
        generated_rows=generated_rows,
        score_rows=score_rows,
        codebook=codebook,
        expected_controls={},
        allow_text_fallback_for_old_transcripts=False,
    )

    protected = [row for row in decoded_rows if row["arm"] == "protected"][0]
    assert protected["accept"] is True
    assert protected["decoded_bits"] == "10100101"
    assert checksum_valid(expected_bits) is True
    assert Counter(row["event_status"] for row in event_rows) == Counter({"target": 8})


def test_contextual_literal_policy_allows_ordinary_coordinate_but_flags_technical_use() -> None:
    policy = {
        "hard_forbid_literals": ["bucket", "payload"],
        "contextual_literals": {
            "coordinate": {
                "ordinary_domain_allowed": True,
                "technical_cues": ["coordinate id", "slot", "payload"],
            }
        },
    }

    assert contextual_technical_literal_hits("Coordinate the room setup calmly.", policy) == []
    assert contextual_technical_literal_hits("Use coordinate id 4 for this payload.", policy) == [
        "coordinate",
        "payload",
    ]
    assert contextual_technical_literal_hits("The bucket value is visible.", policy) == ["bucket"]
