import argparse
import json
import shutil
import zipfile
from pathlib import Path

from scripts.verification_substrate_gap.audit_vsg_expert_handoff import audit
from scripts.verification_substrate_gap.verify_vsg_expert_review_packet import (
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


def test_current_expert_packet_verifier_passes() -> None:
    result = verify(_args(DEFAULT_PACKET_DIR, DEFAULT_ZIP_PATH, DEFAULT_ZIP_SHA_PATH))

    assert result["status"] == "PASS"
    assert result["packet_total_file_count"] == 60
    assert result["hashed_file_count"] == 59
    assert result["claim_lint_status"] == "PASS"
    assert result["claim_lint_violation_count"] == 0


def test_packet_verifier_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    packet_dir, zip_path, zip_sha_path = _copy_packet_fixture(tmp_path)
    required_file = packet_dir / "manuscript_source" / "appendix" / "formal_substrate_gap.tex"
    required_file.unlink()
    _write_zip_and_sha(packet_dir, zip_path, zip_sha_path)

    result = verify(_args(packet_dir, zip_path, zip_sha_path))

    assert result["status"] == "FAIL"
    assert any("missing required files" in failure for failure in result["failures"])


def test_packet_verifier_fails_when_manifest_self_hash_flag_is_false(tmp_path: Path) -> None:
    packet_dir, zip_path, zip_sha_path = _copy_packet_fixture(tmp_path)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_self_hash_excluded"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_zip_and_sha(packet_dir, zip_path, zip_sha_path)

    result = verify(_args(packet_dir, zip_path, zip_sha_path))

    assert result["status"] == "FAIL"
    assert "manifest_self_hash_excluded is not true" in result["failures"]


def test_current_expert_handoff_audit_passes() -> None:
    result = audit()

    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["packet_verifier"]["status"] == "PASS"
    assert result["objective_only_scope"]["no_new_experiments"] is True
