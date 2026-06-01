from pathlib import Path

from scripts.verification_substrate_gap.lint_claim_scope import lint_text


def test_unqualified_public_verifier_claim_fails() -> None:
    text = "This route provides a public verifier for ownership proof."
    violations = lint_text(Path("doc.md"), text)
    assert len(violations) == 2
    assert {v.phrase for v in violations} == {"public verifier", "ownership proof"}


def test_trace_bound_qualifier_passes() -> None:
    text = (
        "This is a trace-bound provider-side public verifier diagnostic, "
        "not public text-only verification."
    )
    assert lint_text(Path("doc.md"), text) == []


def test_allow_file_directive_skips_guardrail_file() -> None:
    text = "<!-- claim-lint: allow-file -->\n\npublic verifier\nownership proof\n"
    assert lint_text(Path("guardrails.md"), text) == []


def test_code_fences_are_ignored_by_default() -> None:
    text = "Safe paragraph.\n\n```text\npublic verifier\n```\n"
    assert lint_text(Path("doc.md"), text) == []


def test_code_fences_can_be_scanned_when_requested() -> None:
    text = "Safe paragraph.\n\n```text\npublic verifier\n```\n"
    violations = lint_text(Path("doc.md"), text, include_code_fences=True)
    assert len(violations) == 1
    assert violations[0].phrase == "public verifier"


def test_machine_readable_false_claim_fields_pass() -> None:
    text = '{"paper_claim_allowed": false, "not_claimed": ["ownership proof"]}'
    assert lint_text(Path("summary.json"), text) == []
