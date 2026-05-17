from scripts.natural_evidence_v2.classify_r4_forbidden_surface_context_v2 import classify_text


def test_allows_ordinary_physical_bucket() -> None:
    result = classify_text("Use a bucket to catch the dripping water before checking the slow sink leak.")
    assert result.technical_forbidden_public_surface_count == 0
    assert result.ordinary_domain_literal_count == 1
    assert result.ambiguous_forbidden_surface_count == 0


def test_rejects_technical_bucket_context() -> None:
    result = classify_text("Store the next token in the surface bucket for the coordinate decoder.")
    assert "bucket" in result.technical_hits
    assert "coordinate" in result.technical_hits
    assert "decoder" in result.technical_hits
    assert result.technical_forbidden_public_surface_count >= 3


def test_rejects_hard_technical_literals() -> None:
    result = classify_text("The payload watermark is a hidden signal for the decoder.")
    assert {"payload", "watermark", "hidden signal", "decoder"}.issubset(set(result.technical_hits))
    assert result.technical_forbidden_public_surface_count >= 4


def test_ambiguous_bucket_is_fatal_until_context_rule_resolves() -> None:
    result = classify_text("Mention the bucket briefly.")
    assert result.technical_forbidden_public_surface_count == 0
    assert result.ordinary_domain_literal_count == 0
    assert result.ambiguous_forbidden_surface_count == 1
