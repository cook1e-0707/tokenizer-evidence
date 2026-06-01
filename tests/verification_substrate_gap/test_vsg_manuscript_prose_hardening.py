from pathlib import Path


MANUSCRIPT = Path("manuscripts/69db2644566dcc36c9da320e")

ACTIVE_PROSE_FILES = [
    MANUSCRIPT / "main.tex",
    MANUSCRIPT / "section_01_introduction.tex",
    MANUSCRIPT / "section_02_related_work.tex",
    MANUSCRIPT / "section_03_problem_setup.tex",
    MANUSCRIPT / "section_04_tokenizer_alignment.tex",
    MANUSCRIPT / "section_05_bucket_level_injection.tex",
    MANUSCRIPT / "section_06_deterministic_verification.tex",
    MANUSCRIPT / "section_07_experiments.tex",
    MANUSCRIPT / "section_08_discussion_limitations.tex",
    MANUSCRIPT / "section_09_conclusion.tex",
    MANUSCRIPT / "appendix/proofs.tex",
    MANUSCRIPT / "appendix/formal_substrate_gap.tex",
    MANUSCRIPT / "appendix/attack_examples.tex",
    MANUSCRIPT / "appendix/extended_related_work.tex",
    MANUSCRIPT / "appendix/reproducibility.tex",
    MANUSCRIPT / "appendix/asset_licenses.tex",
]

INTERNAL_REVIEW_PHRASES = [
    "artifact-only placeholder",
    "placeholder draft",
    "visual draft",
    "Do not claim",
    "We do not claim",
    "we do not claim",
    "do not claim",
    "this draft",
    "the draft",
    "not paper-facing",
    "paper-facing positive",
    "Immediate To-Do",
    "canonical phase",
    "claim-scope lint",
    "claim lint",
    "allowlist entries",
    "Slurm",
]

REQUIRED_SUBSTRATE_POSITIONING_KEYS = {
    "fairoze2023publicly",
    "isler2023puppy",
    "duan2025pvmark",
    "luan2026vow",
    "c2pa2025spec",
    "sun2024zkllm",
    "namazi2025zkprov",
    "zhang2023watermarkssand",
    "gu2023learnability",
    "an2025ditto",
}


def test_active_manuscript_prose_has_no_internal_review_language() -> None:
    findings = []
    for path in ACTIVE_PROSE_FILES:
        text = path.read_text(encoding="utf-8")
        for phrase in INTERNAL_REVIEW_PHRASES:
            if phrase in text:
                findings.append((path.as_posix(), phrase))

    assert findings == []


def test_expert_requested_related_work_keys_are_cited_and_defined() -> None:
    related = (MANUSCRIPT / "section_02_related_work.tex").read_text(encoding="utf-8")
    extended = (MANUSCRIPT / "appendix/extended_related_work.tex").read_text(encoding="utf-8")
    references = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")

    missing_citations = [
        key for key in sorted(REQUIRED_SUBSTRATE_POSITIONING_KEYS) if key not in related + extended
    ]
    missing_bib_entries = [
        key for key in sorted(REQUIRED_SUBSTRATE_POSITIONING_KEYS) if f"{{{key}," not in references
    ]

    assert missing_citations == []
    assert missing_bib_entries == []


def test_active_manuscript_uses_rendered_png_figures_not_placeholder_svgs() -> None:
    active_text = "\n".join(path.read_text(encoding="utf-8") for path in ACTIVE_PROSE_FILES)
    required_figures = [
        "figures/figure_1_verification_substrate_map.png",
        "figures/figure_2_first_divergence_diagnostic.png",
        "figures/figure_3_controllability_vs_observability.png",
        "figures/figure_4_public_predicate_attack_ladder.png",
        "figures/figure_5_ownership_scenario_heatmap.png",
    ]

    for figure in required_figures:
        assert figure in active_text
        assert (MANUSCRIPT / figure).is_file()

    assert ".svg" not in active_text
