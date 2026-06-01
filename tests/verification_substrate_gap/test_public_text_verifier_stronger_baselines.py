from scripts.verification_substrate_gap.evaluate_public_text_verifier import (
    _features_for_variant,
    _score_nb,
    _threshold_at_raw_fpr,
    _train_nb,
)


def test_stronger_public_predicate_feature_families_are_distinct() -> None:
    text = "A calm way forward is to define roles clearly."

    word_bigram = _features_for_variant(text, "P2_learned_shallow_public_predicate")
    char_ngram = _features_for_variant(text, "P4_char_ngram_public_predicate")
    word_trigram = _features_for_variant(text, "P5_word_trigram_public_predicate")
    hybrid = _features_for_variant(text, "P6_hybrid_char_word_public_predicate")

    assert any(feature.startswith("u:") for feature in word_bigram)
    assert any(feature.startswith("b:") for feature in word_bigram)
    assert not any(feature.startswith("c3:") for feature in word_bigram)
    assert any(feature.startswith("c3:") for feature in char_ngram)
    assert any(feature.startswith("t:") for feature in word_trigram)
    assert any(feature.startswith("u:") for feature in hybrid)
    assert any(feature.startswith("c4:") for feature in hybrid)


def test_variant_specific_naive_bayes_scores_prefer_matching_class() -> None:
    rows = [
        ("A calm way forward is to organize roles clearly.", "protected"),
        ("A calm way forward is to clarify responsibilities.", "protected"),
        ("Random unrelated cooking advice about soup.", "raw"),
        ("Random unrelated gardening note about soil.", "raw"),
    ]

    counts, totals, vocab = _train_nb(rows, variant_id="P5_word_trigram_public_predicate")
    protected_score = _score_nb(
        "A calm way forward is to organize responsibilities.",
        counts,
        totals,
        vocab,
        variant_id="P5_word_trigram_public_predicate",
    )
    raw_score = _score_nb(
        "Random unrelated cooking note about soil.",
        counts,
        totals,
        vocab,
        variant_id="P5_word_trigram_public_predicate",
    )

    assert protected_score > raw_score


def test_threshold_at_raw_fpr_uses_raw_scores_only() -> None:
    scored_train = [
        (10.0, "protected"),
        (9.0, "protected"),
        (3.0, "raw"),
        (2.0, "raw"),
        (1.0, "raw"),
    ]

    assert _threshold_at_raw_fpr(scored_train, 0.0) == 3.0
    assert _threshold_at_raw_fpr(scored_train, 0.5) == 2.0
