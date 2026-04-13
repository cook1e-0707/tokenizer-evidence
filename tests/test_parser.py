from src.core.parser import parse_structured_carrier_text
from src.core.synthetic_examples import build_synthetic_smoke_example


def test_parser_extracts_clean_synthetic_fields_correctly() -> None:
    example = build_synthetic_smoke_example(payload=b"O")
    blocks = parse_structured_carrier_text(example.rendered_text, example.layout)
    assert len(blocks) == 1
    assert blocks[0].bucket_tuple(example.layout.field_names) == example.encoding.bucket_tuples[0]


def test_parser_surfaces_malformed_and_unresolved_fields() -> None:
    example = build_synthetic_smoke_example(payload=b"O")
    malformed_text = (
        "FIELD_A=A0_0; FIELD_B=B0_0; MALFORMED; FIELD_C=C0_0; FIELD_D=UNKNOWN_VALUE"
    )
    blocks = parse_structured_carrier_text(malformed_text, example.layout)
    statuses = [carrier.parse_status for carrier in blocks[0].carriers]
    assert "malformed" in statuses
    assert "unresolved_carrier" in statuses
