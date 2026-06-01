import json
from pathlib import Path

from scripts.verification_substrate_gap import build_vsg_reproducibility_release_inventory as inv


def test_text_flags_detect_private_paths_and_secret_terms(tmp_path: Path) -> None:
    path = tmp_path / "record.md"
    path.write_text(
        "source=/hpcstor6/scratch01/g/example\n"
        "local=/Users/example/project\n"
        "field=binding_hmac\n",
        encoding="utf-8",
    )

    private_hit, secret_hit = inv.text_flags(path)

    assert private_hit is True
    assert secret_hit is True


def test_inventory_marker_definitions_are_not_self_flagged(tmp_path: Path) -> None:
    path = tmp_path / "build_vsg_reproducibility_release_inventory.py"
    path.write_text(
        'PRIVATE_PATH_MARKERS = [\n    "/Users/",\n    "/hpcstor",\n]\n'
        'SECRET_TERM_MARKERS = [\n    "secret_key",\n]\n'
        "def main():\n    return 0\n",
        encoding="utf-8",
    )

    private_hit, secret_hit = inv.text_flags(path)

    assert private_hit is False
    assert secret_hit is False


def test_sha256_file_records_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("vsg release inventory\n", encoding="utf-8")

    assert inv.sha256_file(path) == "f8fefd6d0b8f3845c6fef107ca0707bfeaa9dcbd26202721d9c4ec0ac0560918"


def test_release_inventory_build_is_artifact_only_and_manifest_excludes_self(tmp_path: Path) -> None:
    summary = inv.build(tmp_path)

    assert summary["status"] == "PASS_VSG_REPRODUCIBILITY_RELEASE_INVENTORY_RECORDED_REVIEW_REQUIRED"
    assert summary["new_slurm_started"] is False
    assert summary["generation_started"] is False
    assert summary["model_scoring_started"] is False
    assert summary["training_started"] is False
    assert summary["public_text_only_verification_claimed"] is False
    assert summary["ownership_proof_claimed"] is False

    manifest_path = tmp_path / "release_inventory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = {row["path"] for row in manifest["files"]}

    assert "release_inventory_manifest.json" not in {Path(path).name for path in manifest_paths}
    assert any(Path(path).name == "release_inventory.csv" for path in manifest_paths)
    assert any(Path(path).name == "release_inventory_summary.json" for path in manifest_paths)
    assert any(Path(path).name == "release_inventory_report.md" for path in manifest_paths)


def test_planned_artifacts_include_current_hardening_outputs() -> None:
    planned_paths = {row["path"] for row in inv.planned_artifacts()}

    assert "scripts/verification_substrate_gap/evaluate_public_text_verifier.py" in planned_paths
    assert "scripts/verification_substrate_gap/audit_public_predicate_attack_naturalness.py" in planned_paths
    assert "results/verification_substrate_gap/VSG_CURRENT_HANDOFF_STATE_20260601.md" in planned_paths
