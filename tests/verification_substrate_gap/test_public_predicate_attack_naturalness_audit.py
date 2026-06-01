from scripts.verification_substrate_gap.audit_public_predicate_attack_naturalness import (
    audit_text_pair,
    token_jaccard,
)


def test_token_jaccard_uses_normalized_word_sets() -> None:
    left = "A calm way forward is to define roles."
    right = "A calm way forward is to clarify roles."

    assert 0 < token_jaccard(left, right) < 1


def test_readable_pair_passes_proxy_checks() -> None:
    original = "A helpful start is to define roles and keep notes for the group."
    rewrite = (
        "A low-risk start is to define roles and keep notes for the group. "
        "This keeps the plan easy to follow."
    )

    result = audit_text_pair(original, rewrite)

    assert result["proxy_quality_status"] == "PASS_PROXY_READABILITY"
    assert result["proxy_quality_fail_reasons"] == ""


def test_broken_graft_pair_fails_proxy_checks() -> None:
    original = "A helpful start is to standardize templates for common responses."
    rewrite = (
        "A low-risk start is to organize a meeting related to bike m "
        "Overlapping pages c A helpful start is to standardize templates."
    )

    result = audit_text_pair(original, rewrite)

    assert result["proxy_quality_status"] == "FAIL_PROXY_READABILITY"
    assert "isolated_single_letter_fragment" in result["proxy_quality_fail_reasons"]
    assert "known_broken_graft_marker" in result["proxy_quality_fail_reasons"]
