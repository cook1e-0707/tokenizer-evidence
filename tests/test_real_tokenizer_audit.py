from pathlib import Path

from src.core.bucket_mapping import BucketLayout, load_bucket_layout
from src.core.tokenizer_utils import MockTokenizer, audit_carriers, load_carriers_and_layout
from src.infrastructure.paths import discover_repo_root


def _catalog_path() -> Path:
    repo_root = discover_repo_root(Path(__file__).parent)
    return repo_root / "configs" / "data" / "real_pilot_catalog.yaml"


def _single_token_map(layout: BucketLayout, extra_tokens: tuple[str, ...] = ()) -> dict[str, int]:
    carriers = list(layout.all_carriers()) + list(extra_tokens)
    return {carrier: index + 1 for index, carrier in enumerate(carriers)}


def test_real_catalog_loads_with_field_metadata() -> None:
    layout = load_bucket_layout(_catalog_path())
    assert layout.catalog_name == "real-pilot-catalog"
    assert layout.field_names == ("SECTION", "TONE", "TOPIC", "REGION")
    assert layout.get_field_spec("SECTION").field_type == "category"
    assert layout.get_field_spec("TOPIC").disallowed_carriers == ("multi word",)


def test_allowed_real_catalog_carriers_can_pass_alignment_audit() -> None:
    carriers, layout = load_carriers_and_layout(
        bucket_spec_path=_catalog_path(),
        include_disallowed=False,
    )
    tokenizer = MockTokenizer(single_token_map=_single_token_map(layout))
    result = audit_carriers(carriers, tokenizer=tokenizer, bucket_layout=layout)

    assert result.is_alignment_safe is True
    assert result.rejected_carriers == ()
    assert set(result.field_summaries) == set(layout.field_names)
    assert all(summary["passed"] for summary in result.field_summaries.values())


def test_disallowed_and_multi_token_carriers_are_reported() -> None:
    carriers, layout = load_carriers_and_layout(
        bucket_spec_path=_catalog_path(),
        include_disallowed=True,
    )
    tokenizer = MockTokenizer(
        single_token_map=_single_token_map(layout, extra_tokens=("multi", "\t", "\n")),
        multi_token_map={"two words": (9001, 9002), "multi word": (9003, 9004)},
    )
    result = audit_carriers(carriers, tokenizer=tokenizer, bucket_layout=layout)

    assert result.is_alignment_safe is False
    assert result.num_invalid >= 3
    assert any(item.carrier == "two words" for item in result.rejected_carriers)
    assert any(item.carrier == "multi word" for item in result.rejected_carriers)
