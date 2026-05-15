from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml
from scripts.natural_evidence_v2.validate_r4_positive_selectivity_pressure_pivot_package import (
    DEFAULT_CONFIG,
    validate_package,
)


def test_pressure_pivot_package_passes_static_validation() -> None:
    summary = validate_package(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
    assert summary["current_compute_unlocked"] is False
    assert summary["slurm_job_submitted"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["generation_started"] is False


def test_pressure_pivot_package_rejects_generation_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["current_permissions"]["generation_allowed"] = True

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false" in summary["errors"]


def test_pressure_pivot_package_requires_no_transcript_mining() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["reuse_policy"]["forbid_post_hoc_phrase_mining_from_859491"] = False

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "reuse_policy.forbid_post_hoc_phrase_mining_from_859491 must be true" in summary["errors"]


def test_pressure_pivot_package_requires_all_failure_jobs() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["source_failed_diagnostics"] = config["source_failed_diagnostics"][:-1]

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "source_failed_diagnostics job set mismatch: ['857795', '858019', '859277']" in summary["errors"]
