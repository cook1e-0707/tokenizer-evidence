#!/usr/bin/env python3
"""Verify the VSG expert review packet manifest, zip, and review-scope hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_DIR = ROOT / "results" / "verification_substrate_gap" / "expert_review_packet_20260531"
DEFAULT_ZIP_PATH = ROOT / "results" / "verification_substrate_gap" / "vsg_expert_review_packet_20260531.zip"
DEFAULT_ZIP_SHA_PATH = ROOT / "results" / "verification_substrate_gap" / "vsg_expert_review_packet_20260531.zip.sha256"
DEFAULT_EXTERNAL_README = ROOT / "results" / "verification_substrate_gap" / "vsg_expert_review_packet_20260531_README.txt"

REQUIRED_FILES = {
    "EXPERT_REVIEW_SCOPE_20260531.md",
    "OBJECTIVE_FACTS_20260531.md",
    "README_FOR_EXPERT_REVIEW_20260531.md",
    "manuscript/VSG_manuscript_snapshot_20260531.pdf",
    "manuscript_source/main.tex",
    "manuscript_source/section_01_introduction.tex",
    "manuscript_source/section_09_conclusion.tex",
    "manuscript_source/appendix/formal_substrate_gap.tex",
    "manuscript_source/appendix/attack_examples.tex",
    "manuscript_source/appendix/reproducibility_commands.tex",
    "evidence/figure_data/trace_bound_accepts.csv",
    "evidence/figure_data/public_text_verifier_baselines.csv",
    "evidence/figure_data/attack_ladder_summary.csv",
    "evidence/figure_data/ownership_scenario_heatmap.csv",
    "evidence/visual_drafts/table_1_claim_ledger.csv",
    "validation/claim_scope_lint_report.json",
    "validation/latex_build_summary.json",
    "validation/latex_log_scan.json",
    "validation/git_snapshot.json",
    "packet_manifest.json",
}

FORBIDDEN_PACKET_PATHS = {
    "manuscript_source/checklist_support.md",
}

FORBIDDEN_REVIEW_TEXT = [
    "Immediate To-Do",
    "0146795 Record VSG section-order review gate",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(reason: str, failures: list[str]) -> None:
    failures.append(reason)


def verify(args: argparse.Namespace) -> dict:
    packet_dir = Path(args.packet_dir)
    zip_path = Path(args.zip_path)
    zip_sha_path = Path(args.zip_sha_path)
    external_readme = Path(args.external_readme)
    failures: list[str] = []

    if not packet_dir.is_dir():
        fail(f"packet_dir missing: {packet_dir}", failures)
    if not zip_path.is_file():
        fail(f"zip_path missing: {zip_path}", failures)
    if not zip_sha_path.is_file():
        fail(f"zip sha file missing: {zip_sha_path}", failures)
    if not external_readme.is_file():
        fail(f"external README missing: {external_readme}", failures)
    if failures:
        return {"status": "FAIL", "failures": failures}

    actual_zip_sha = sha256_file(zip_path)
    expected_zip_sha = zip_sha_path.read_text(encoding="utf-8").split()[0]
    if actual_zip_sha != expected_zip_sha:
        fail(f"zip sha mismatch: expected {expected_zip_sha}, actual {actual_zip_sha}", failures)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            fail(f"zip test failed at entry: {bad}", failures)
        zip_entries = {info.filename for info in zf.infolist() if not info.is_dir()}

    packet_files = {
        path.relative_to(packet_dir).as_posix()
        for path in packet_dir.rglob("*")
        if path.is_file()
    }
    if zip_entries != packet_files:
        fail(
            f"zip entries differ from packet dir: missing={sorted(packet_files - zip_entries)[:10]}, extra={sorted(zip_entries - packet_files)[:10]}",
            failures,
        )

    missing_required = sorted(REQUIRED_FILES - packet_files)
    if missing_required:
        fail(f"missing required files: {missing_required}", failures)
    forbidden_present = sorted(FORBIDDEN_PACKET_PATHS & packet_files)
    if forbidden_present:
        fail(f"forbidden packet paths present: {forbidden_present}", failures)

    manifest = load_json(packet_dir / "packet_manifest.json")
    if manifest.get("status") != "PASS_PACKET_ASSEMBLED_ARTIFACT_ONLY_OBJECTIVE_FACTS":
        fail(f"unexpected manifest status: {manifest.get('status')}", failures)
    if not manifest.get("manifest_self_hash_excluded"):
        fail("manifest_self_hash_excluded is not true", failures)
    if manifest.get("packet_total_file_count") != len(packet_files):
        fail(
            f"packet_total_file_count mismatch: manifest={manifest.get('packet_total_file_count')} actual={len(packet_files)}",
            failures,
        )
    hashed_files = manifest.get("files", [])
    if manifest.get("hashed_file_count") != len(hashed_files):
        fail(
            f"hashed_file_count mismatch: manifest={manifest.get('hashed_file_count')} actual={len(hashed_files)}",
            failures,
        )
    listed_paths = {row.get("path") for row in hashed_files}
    expected_hashed_paths = packet_files - {"packet_manifest.json"}
    if listed_paths != expected_hashed_paths:
        fail(
            f"manifest listed paths mismatch: missing={sorted(expected_hashed_paths - listed_paths)[:10]}, extra={sorted(listed_paths - expected_hashed_paths)[:10]}",
            failures,
        )
    for row in hashed_files:
        rel = row["path"]
        path = packet_dir / rel
        if not path.is_file():
            fail(f"manifest-listed file missing: {rel}", failures)
            continue
        if path.stat().st_size != row["bytes"]:
            fail(f"byte count mismatch for {rel}: manifest={row['bytes']} actual={path.stat().st_size}", failures)
        actual = sha256_file(path)
        if actual != row["sha256"]:
            fail(f"sha256 mismatch for {rel}: manifest={row['sha256']} actual={actual}", failures)

    lint = load_json(packet_dir / "validation" / "claim_scope_lint_report.json")
    if lint.get("status") != "PASS" or lint.get("violation_count") != 0 or lint.get("checked_files") != 17:
        fail(f"claim lint summary unexpected: {lint}", failures)
    latex = load_json(packet_dir / "validation" / "latex_build_summary.json")
    if latex.get("status") != "PASS" or latex.get("pdf_sha256") != manifest.get("latex_build_summary", {}).get("pdf_sha256"):
        fail("latex build summary missing PASS or pdf sha does not match manifest", failures)
    log_scan = load_json(packet_dir / "validation" / "latex_log_scan.json")
    if log_scan.get("status") != "PASS" or log_scan.get("overfull_hbox_warning_count") != 0:
        fail(f"latex log scan unexpected: {log_scan}", failures)

    git_snapshot = manifest.get("git_snapshot", {})
    if not str(git_snapshot.get("manuscript_repository_head", "")).startswith("64510b9"):
        fail(f"unexpected manuscript head: {git_snapshot.get('manuscript_repository_head')}", failures)
    if git_snapshot.get("manuscript_git_status_short") != "":
        fail(f"manuscript status is not clean: {git_snapshot.get('manuscript_git_status_short')}", failures)

    review_text_paths = [
        external_readme,
        packet_dir / "README_FOR_EXPERT_REVIEW_20260531.md",
        packet_dir / "EXPERT_REVIEW_SCOPE_20260531.md",
        packet_dir / "OBJECTIVE_FACTS_20260531.md",
    ]
    for path in review_text_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for phrase in FORBIDDEN_REVIEW_TEXT:
            if phrase in text:
                fail(f"forbidden review text phrase in {path}: {phrase}", failures)

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "packet_dir": str(packet_dir),
        "zip_path": str(zip_path),
        "zip_sha256": actual_zip_sha,
        "packet_total_file_count": len(packet_files),
        "hashed_file_count": len(hashed_files),
        "manifest_status": manifest.get("status"),
        "manuscript_head": git_snapshot.get("manuscript_repository_head"),
        "root_head_at_packet_build": git_snapshot.get("root_repository_head"),
        "claim_lint_status": lint.get("status"),
        "claim_lint_violation_count": lint.get("violation_count"),
        "latex_log_scan_status": log_scan.get("status"),
        "overfull_hbox_warning_count": log_scan.get("overfull_hbox_warning_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--zip-sha-path", default=str(DEFAULT_ZIP_SHA_PATH))
    parser.add_argument("--external-readme", default=str(DEFAULT_EXTERNAL_README))
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = verify(args)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
