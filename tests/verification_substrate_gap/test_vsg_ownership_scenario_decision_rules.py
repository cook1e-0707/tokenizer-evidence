from copy import deepcopy
from pathlib import Path

from scripts.verification_substrate_gap import audit_vsg_ownership_scenario_decision_rules as audit


def _minimal_valid_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scenario in audit.EXPECTED_SCENARIOS:
        for method in audit.EXPECTED_METHOD_FAMILIES:
            status = "FAILS_NO_FINAL_TEXT_SUBSTRATE"
            substrate = "no"
            evidence = "public final-text codeword recovered blocks total = 0"
            blocker = "not supported in current artifacts"
            if method in {"tee_or_2pc_public_watermark_protocol", "zk_inference_proof"}:
                status = "FAILS_NO_PROTOCOL_SUBSTRATE"
                blocker = "scenario lacks protocol transcript"
            elif method in audit.TRACE_METHODS:
                status = "FAILS_NO_TRACE_SUBSTRATE"
                blocker = "provider-side trace unavailable in this scenario"
            elif method == "public_deterministic_text_predicate":
                status = "FAILS_PUBLIC_PREDICATE_SPOOFABLE"
                substrate = "yes"
                blocker = "public predicate does not recover codeword and is searchable"
            if scenario == "S2_cooperative_provider_with_trace_bundle" and method in audit.TRACE_METHODS:
                status = "SUPPORTED_TRACE_BOUND_DIAGNOSTIC"
                substrate = "yes"
                blocker = "not portable to final text without provider-side trace"
            rows.append(
                {
                    "scenario_id": scenario,
                    "scenario_title": scenario,
                    "method_family": method,
                    "evidence_substrate": method,
                    "current_assessment": status,
                    "substrate_available": substrate,
                    "current_evidence": evidence,
                    "primary_blocker": blocker,
                    "claim_scope": "scenario stress test only; no paper-facing positive claim",
                    "next_experiment": "none",
                }
            )
    return rows


def test_current_ownership_scenario_matrix_passes() -> None:
    rows = audit.read_rows(audit.DEFAULT_INPUT)

    _, failures = audit.validate_rows(rows)

    assert failures == []


def test_valid_minimal_matrix_has_only_trace_bound_support() -> None:
    rows = _minimal_valid_rows()

    _, failures = audit.validate_rows(rows)

    assert failures == []


def test_public_text_supported_row_fails() -> None:
    rows = _minimal_valid_rows()
    bad_rows = deepcopy(rows)
    for row in bad_rows:
        if row["method_family"] == "public_deterministic_text_predicate":
            row["current_assessment"] = "SUPPORTED_TRACE_BOUND_DIAGNOSTIC"
            break

    _, failures = audit.validate_rows(bad_rows)

    assert any("public deterministic text predicate" in failure for failure in failures)


def test_duplicate_scenario_method_pair_fails() -> None:
    rows = _minimal_valid_rows()
    bad_rows = rows + [dict(rows[0])]

    _, failures = audit.validate_rows(bad_rows)

    assert any("row count mismatch" in failure for failure in failures)
    assert any("duplicate scenario/method pairs" in failure for failure in failures)


def test_build_writes_artifact_only_summary(tmp_path: Path) -> None:
    summary = audit.build(audit.DEFAULT_INPUT, tmp_path)

    assert summary["status"] == "PASS"
    assert summary["row_count"] == 63
    assert summary["supported_trace_bound_row_count"] == 2
    assert summary["supported_public_text_row_count"] == 0
    assert summary["new_slurm_started"] is False
    assert summary["ownership_proof_claimed"] is False
    assert (tmp_path / "decision_rule_audit_rows.csv").is_file()
    assert (tmp_path / "decision_rule_audit_summary.json").is_file()
