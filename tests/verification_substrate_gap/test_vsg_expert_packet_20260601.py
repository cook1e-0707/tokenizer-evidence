import argparse
import json
import shutil
import zipfile
from pathlib import Path

from scripts.verification_substrate_gap.audit_vsg_expert_handoff_20260601 import audit
from scripts.verification_substrate_gap.verify_vsg_expert_review_packet_20260601 import (
    DEFAULT_EXTERNAL_README,
    DEFAULT_PACKET_DIR,
    DEFAULT_ZIP_PATH,
    DEFAULT_ZIP_SHA_PATH,
    sha256_file,
    verify,
)


def _args(packet_dir: Path, zip_path: Path, zip_sha_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        packet_dir=str(packet_dir),
        zip_path=str(zip_path),
        zip_sha_path=str(zip_sha_path),
        external_readme=str(DEFAULT_EXTERNAL_README),
    )


def _write_zip_and_sha(packet_dir: Path, zip_path: Path, zip_sha_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(packet_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(packet_dir).as_posix())
    zip_sha_path.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")


def _copy_packet_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    packet_dir = tmp_path / "packet"
    shutil.copytree(DEFAULT_PACKET_DIR, packet_dir)
    zip_path = tmp_path / "packet.zip"
    zip_sha_path = tmp_path / "packet.zip.sha256"
    _write_zip_and_sha(packet_dir, zip_path, zip_sha_path)
    return packet_dir, zip_path, zip_sha_path


def test_current_20260601_expert_packet_verifier_passes() -> None:
    result = verify(_args(DEFAULT_PACKET_DIR, DEFAULT_ZIP_PATH, DEFAULT_ZIP_SHA_PATH))

    assert result["status"] == "PASS"
    assert result["packet_total_file_count"] == 87
    assert result["hashed_file_count"] == result["packet_total_file_count"] - 1
    assert result["claim_lint_status"] == "PASS"
    assert result["claim_lint_violation_count"] == 0
    assert result["hardening_summary"]["public_text_stronger_baseline"]["codeword_recovered_blocks_total"] == 0
    assert result["hardening_summary"]["ownership_decision_rule_audit"]["supported_public_text_row_count"] == 0


def test_20260601_packet_contains_hardening_outputs() -> None:
    manifest = json.loads((DEFAULT_PACKET_DIR / "packet_manifest.json").read_text(encoding="utf-8"))
    paths = {row["path"] for row in manifest["files"]}

    assert "validation/hardening_summary.json" in paths
    assert (
        "evidence/hardening/public_text_verifier_stronger_local_pilot_20260601/public_text_verifier_summary.json"
        in paths
    )
    assert (
        "evidence/hardening/public_predicate_attack_naturalness_audit_20260601/attack_naturalness_proxy_summary.json"
        in paths
    )
    assert (
        "evidence/hardening/reproducibility_release_inventory_20260601/release_inventory_summary.json"
        in paths
    )
    assert (
        "evidence/hardening/ownership_scenario_decision_rule_audit_20260601/decision_rule_audit_summary.json"
        in paths
    )
    assert (
        "evidence/hardening/manuscript_figure_quality_audit_20260601/figure_quality_summary.json"
        in paths
    )


def test_20260601_packet_verifier_fails_when_required_hardening_file_is_missing(tmp_path: Path) -> None:
    packet_dir, zip_path, zip_sha_path = _copy_packet_fixture(tmp_path)
    required_file = (
        packet_dir
        / "evidence"
        / "hardening"
        / "manuscript_figure_quality_audit_20260601"
        / "figure_quality_summary.json"
    )
    required_file.unlink()
    _write_zip_and_sha(packet_dir, zip_path, zip_sha_path)

    result = verify(_args(packet_dir, zip_path, zip_sha_path))

    assert result["status"] == "FAIL"
    assert any("missing required files" in failure for failure in result["failures"])


def test_current_20260601_expert_handoff_audit_passes() -> None:
    result = audit()

    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["packet_verifier"]["status"] == "PASS"
    assert result["objective_only_scope"]["no_new_experiments"] is True
    assert result["objective_only_scope"]["overleaf_push_not_performed"] is True
