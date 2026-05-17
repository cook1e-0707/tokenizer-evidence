from pathlib import Path

from scripts.natural_evidence_v2.validate_r4_after_868260_quality_gate_repair_package import (
    read_json,
    validate_package,
)


def test_r4_after_868260_quality_gate_repair_package_validates() -> None:
    failure = read_json(
        Path(
            "results/natural_evidence_v2/status/"
            "r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/"
            "failure_analysis_summary.json"
        )
    )
    summary = validate_package(
        failure=failure,
        package_dir=Path("results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517"),
    )
    assert summary["status"] == "PASS_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT"
    assert summary["reclassifies_868260"] is False
    assert summary["slurm_allowed"] is False


def test_r4_after_868260_quality_gate_repair_package_rejects_hard_forbid_bucket() -> None:
    failure = dict(
        read_json(
            Path(
                "results/natural_evidence_v2/status/"
                "r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/"
                "failure_analysis_summary.json"
            )
        )
    )
    package_dir = Path("results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517")
    summary = validate_package(failure=failure, package_dir=package_dir)
    assert not summary["errors"]

    policy_path = package_dir / "contextual_forbidden_surface_policy_v2.json"
    policy = read_json(policy_path)
    policy = dict(policy)
    policy["hard_forbid_literals"] = [*policy["hard_forbid_literals"], "bucket"]

    tmp_dir = Path("/tmp/r4_after_868260_invalid_quality_gate_repair_package")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("duplicate_safe_generation_policy.json", "repair_manifest.json"):
        (tmp_dir / name).write_text((package_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    import json

    (tmp_dir / "contextual_forbidden_surface_policy_v2.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejected = validate_package(failure=failure, package_dir=tmp_dir)
    assert rejected["status"] == "FAIL_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT"
    assert any("bucket" in error for error in rejected["errors"])
