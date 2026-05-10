from __future__ import annotations

from scripts.natural_evidence_v2.build_wp5_teacher_forced_launch_plan import (
    normalize_action_phrase,
    parse_response_lines,
    payload_bits_from_wp4,
    repaired_line,
)


def test_wp5_payload_bits_concatenate_payload_and_checksum() -> None:
    contract = {
        "payload": {
            "payload_bits_msb_first": [1, 0, 1, 0, 0, 1, 0, 1],
            "checksum_bits_msb_first": [0, 1, 0, 1, 1, 1, 1, 0],
        }
    }

    assert payload_bits_from_wp4(contract) == [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0]


def test_wp5_repaired_line_avoids_double_terminal_punctuation() -> None:
    line = repaired_line(
        13,
        "Prepare",
        "Use",
        'calming phrases like "I understand" and "Let me help you.".',
    )

    assert line == 'Step 13: Prepare a simple way to use calming phrases like "I understand" and "Let me help you".'
    assert not line.endswith('.".')


def test_wp5_normalize_action_phrase_keeps_abbreviations_inside_sentence() -> None:
    phrase = normalize_action_phrase("Use", "a simple rating system (e.g., thumbs up/thumbs down) to gauge interest.")

    assert phrase == "use a simple rating system (e.g., thumbs up/thumbs down) to gauge interest"


def test_wp5_parse_response_lines_requires_step_labels() -> None:
    response = "\n".join(
        [
            "Step 1: Start with one action.",
            "Step 2: Plan the next action.",
            "Not a step line.",
        ]
    )

    parsed = parse_response_lines(response)

    assert [row["step_index"] for row in parsed] == [1, 2]
    assert parsed[0]["original_first_word"] == "Start"
