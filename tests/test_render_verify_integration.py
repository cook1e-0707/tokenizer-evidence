from pathlib import Path

from src.core.bucket_mapping import load_bucket_layout
from src.core.parser import parse_canonical_rendered_text
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import render_bucket_tuples, render_config_from_name
from src.core.synthetic_examples import replace_field_value
from src.core.verifier import VerificationConfig, verify_canonical_rendered_text
from src.infrastructure.paths import discover_repo_root


def _layout():
    repo_root = discover_repo_root(Path(__file__).parent)
    return load_bucket_layout(repo_root / "configs" / "data" / "real_pilot_catalog.yaml")


def _render_payload(payload: bytes = b"OK"):
    layout = _layout()
    codec = BucketPayloadCodec(layout.radices)
    encoding = codec.encode_bytes(payload)
    rendered = render_bucket_tuples(
        layout,
        encoding.bucket_tuples,
        config=render_config_from_name("canonical_v1"),
    )
    return layout, codec, encoding, rendered


def test_canonical_render_parse_round_trip_is_stable() -> None:
    layout, _, encoding, rendered = _render_payload(b"OK")
    blocks = parse_canonical_rendered_text(rendered.text, layout, format_name="canonical_v1")
    parsed_tuples = tuple(block.bucket_tuple(layout.field_names) for block in blocks)
    assert parsed_tuples == encoding.bucket_tuples


def test_real_mode_verifier_recovers_payload_on_rendered_example() -> None:
    layout, codec, _, rendered = _render_payload(b"OK")
    result = verify_canonical_rendered_text(
        text=rendered.text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=b"OK",
        config=VerificationConfig(verification_mode="canonical_render"),
    )
    assert result.success is True
    assert result.decoded_payload == "OK"
    assert result.match_ratio == 1.0


def test_same_bucket_substitution_preserves_decoded_bucket_layer_in_real_mode() -> None:
    layout, codec, encoding, rendered = _render_payload(b"OK")
    field_name = layout.field_names[0]
    same_bucket_value = layout.get_field_spec(field_name).bucket_members(encoding.bucket_tuples[0][0])[1]
    mutated_text = replace_field_value(rendered.text, 0, field_name, same_bucket_value)

    original = verify_canonical_rendered_text(
        text=rendered.text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=b"OK",
        config=VerificationConfig(verification_mode="canonical_render"),
    )
    mutated = verify_canonical_rendered_text(
        text=mutated_text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=b"OK",
        config=VerificationConfig(verification_mode="canonical_render"),
    )

    assert mutated.success is True
    assert mutated.decoded_bucket_tuples == original.decoded_bucket_tuples
    assert mutated.decoded_units == original.decoded_units


def test_cross_bucket_substitution_changes_decode_outcome_in_real_mode() -> None:
    layout, codec, encoding, rendered = _render_payload(b"OK")
    field_name = layout.field_names[0]
    original_bucket_id = encoding.bucket_tuples[0][0]
    replacement_bucket_id = (original_bucket_id + 1) % layout.get_field_spec(field_name).bucket_count
    cross_bucket_value = layout.get_field_spec(field_name).bucket_members(replacement_bucket_id)[0]
    mutated_text = replace_field_value(rendered.text, 0, field_name, cross_bucket_value)

    mutated = verify_canonical_rendered_text(
        text=mutated_text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=b"OK",
        config=VerificationConfig(verification_mode="canonical_render"),
    )
    assert mutated.success is False
    assert mutated.decoded_units != (79, 75)


def test_missing_field_is_reported_cleanly_in_real_mode() -> None:
    layout, codec, _, rendered = _render_payload(b"O")
    first_line = rendered.text.splitlines()[0]
    segments = [segment.strip() for segment in first_line.split(";") if segment.strip()]
    missing_field_text = "; ".join(segment for segment in segments if not segment.startswith("TONE="))
    result = verify_canonical_rendered_text(
        text=missing_field_text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=b"O",
        config=VerificationConfig(verification_mode="canonical_render"),
    )
    assert result.success is False
    assert any(item.endswith("TONE") for item in result.unresolved_fields)
