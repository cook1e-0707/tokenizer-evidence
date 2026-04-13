from pathlib import Path

from src.core.synthetic_examples import build_synthetic_smoke_example, replace_field_value
from src.core.verifier import VerificationConfig, verify_fixture, verify_structured_text


def test_verifier_recovers_payload_on_clean_synthetic_example() -> None:
    example = build_synthetic_smoke_example(payload=b"OK")
    result = verify_structured_text(
        text=example.rendered_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
        config=VerificationConfig(require_all_fields=True, decode_as_bytes=True),
    )
    assert result.success is True
    assert result.decoded_units == tuple(example.payload)


def test_same_bucket_substitution_preserves_bucket_decode_and_payload() -> None:
    example = build_synthetic_smoke_example(payload=b"OK")
    original_result = verify_structured_text(
        text=example.rendered_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
    )
    same_bucket_replacement = example.layout.get_field_spec("FIELD_A").bucket_members(
        example.encoding.bucket_tuples[0][0]
    )[1]
    mutated_text = replace_field_value(
        example.rendered_text,
        block_index=0,
        field_name="FIELD_A",
        new_value=same_bucket_replacement,
    )
    mutated_result = verify_structured_text(
        text=mutated_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
    )
    assert mutated_result.success is True
    assert mutated_result.decoded_bucket_tuples == original_result.decoded_bucket_tuples
    assert mutated_result.decoded_units == original_result.decoded_units


def test_cross_bucket_substitution_changes_decode_outcome() -> None:
    example = build_synthetic_smoke_example(payload=b"OK")
    original_bucket_id = example.encoding.bucket_tuples[0][0]
    replacement_bucket_id = (original_bucket_id + 1) % example.layout.get_field_spec("FIELD_A").bucket_count
    different_bucket_replacement = example.layout.get_field_spec("FIELD_A").bucket_members(
        replacement_bucket_id
    )[0]
    mutated_text = replace_field_value(
        example.rendered_text,
        block_index=0,
        field_name="FIELD_A",
        new_value=different_bucket_replacement,
    )
    mutated_result = verify_structured_text(
        text=mutated_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
    )
    assert mutated_result.success is False
    assert mutated_result.decoded_units != tuple(example.payload)


def test_missing_field_produces_verification_failure() -> None:
    example = build_synthetic_smoke_example(payload=b"O")
    first_line = example.rendered_text.splitlines()[0]
    segments = [segment.strip() for segment in first_line.split(";") if segment.strip()]
    kept_segments = [segment for segment in segments if not segment.startswith("FIELD_B=")]
    missing_field_text = "; ".join(kept_segments)
    result = verify_structured_text(
        text=missing_field_text,
        bucket_layout=example.layout,
        payload_codec=example.codec,
        expected_payload=example.payload,
    )
    assert result.success is False
    assert any(item.endswith("FIELD_B") for item in result.unresolved_fields)


def test_legacy_fixture_path_still_verifies() -> None:
    fixture_path = Path(__file__).parent / "data" / "synthetic_evidence.json"
    result = verify_fixture(
        fixture_path,
        config=VerificationConfig(min_score=0.5, min_match_ratio=1.0, scan_windows=True),
    )
    assert result.accepted is True
    assert result.recovered_symbols == ("TOK_A", "TOK_B", "TOK_C")
