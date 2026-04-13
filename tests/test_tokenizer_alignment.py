from src.core.synthetic_examples import build_synthetic_bucket_layout
from src.core.tokenizer_utils import MockTokenizer, audit_carriers


def test_tokenizer_audit_distinguishes_single_and_multi_token_carriers() -> None:
    tokenizer = MockTokenizer(
        single_token_map={"A0_0": 1, "B0_0": 2, "alias": 3},
        multi_token_map={"two words": (4, 5)},
    )
    result = audit_carriers(["A0_0", "two words", "B0_0"], tokenizer=tokenizer)
    assert result.num_total == 3
    assert result.num_single_token == 2
    assert result.num_multi_token == 1


def test_tokenizer_audit_catches_duplicates_invalid_forms_and_collisions() -> None:
    layout = build_synthetic_bucket_layout(fields=("FIELD_A",), bucket_count=2, members_per_bucket=2)
    tokenizer = MockTokenizer(
        single_token_map={"dup": 11, "dup ": 11, "same_token_alias": 11},
    )
    result = audit_carriers(
        ["dup", "dup ", "", "same_token_alias"],
        tokenizer=tokenizer,
        bucket_layout=layout,
    )
    assert result.num_invalid >= 2
    assert result.num_duplicates >= 2
    collision_items = [item for item in result.diagnostics if item.token_collision]
    assert collision_items
