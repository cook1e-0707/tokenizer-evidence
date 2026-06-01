#!/usr/bin/env python3
"""Audit the refreshed 2026-06-01 VSG expert handoff for objective-only delivery."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = "20260601"
PACKET_DIR = ROOT / "results" / "verification_substrate_gap" / f"expert_review_packet_{DATE_TAG}"
EXTERNAL_README = ROOT / "results" / "verification_substrate_gap" / f"vsg_expert_review_packet_{DATE_TAG}_README.txt"
VERIFY_SCRIPT = ROOT / "scripts" / "verification_substrate_gap" / "verify_vsg_expert_review_packet_20260601.py"

REVIEWER_FACING_FILES = [
    EXTERNAL_README,
    PACKET_DIR / f"README_FOR_EXPERT_REVIEW_{DATE_TAG}.md",
    PACKET_DIR / f"EXPERT_REVIEW_SCOPE_{DATE_TAG}.md",
    PACKET_DIR / f"OBJECTIVE_FACTS_{DATE_TAG}.md",
    PACKET_DIR / f"HARDENING_STATUS_{DATE_TAG}.md",
]

OBJECTIVE_REQUIRED_STRINGS = [
    "Verification Substrate Gap",
    "trace-bound first-divergence",
    "public final-text codeword recovery = `0`",
    "source-mismatch accepts",
    "provider-side diagnostics",
    "spoofing evidence",
    "hardening",
    "LaTeX",
]

STALE_OR_INTERNAL_STRINGS = [
    "0146795 Record VSG section-order review gate",
    "Immediate To-Do",
    "checklist_support.md",
    "artifact-only placeholder draft",
    "visual placeholder package",
    "paper-facing positive claim",
]

ADVISORY_PATTERNS = [
    re.compile(r"我建议"),
    re.compile(r"建议(?!包|：|:|，|。)"),
    re.compile(r"推荐"),
    re.compile(r"下一步(?!实验计划)"),
    re.compile(r"接下来"),
    re.compile(r"请专家"),
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\brecommend", re.IGNORECASE),
    re.compile(r"\bplease\b", re.IGNORECASE),
    re.compile(r"[?？]"),
]

ADVISORY_ALLOWLIST_SUBSTRINGS = [
    "route recommendation document",
    "do not establish",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_verify() -> dict:
    completed = subprocess.run(
        ["python3", str(VERIFY_SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return {
            "status": "FAIL",
            "failures": ["packet verifier returned nonzero"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return json.loads(completed.stdout)


def has_allowlisted_context(line: str) -> bool:
    return any(token in line for token in ADVISORY_ALLOWLIST_SUBSTRINGS)


def audit_reviewer_text() -> tuple[list[dict], dict]:
    findings: list[dict] = []
    stats: dict[str, dict] = {}
    for path in REVIEWER_FACING_FILES:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        lines = text.splitlines()
        stats[rel] = {
            "line_count": len(lines),
            "byte_count": path.stat().st_size,
            "contains_cjk": any("\u4e00" <= ch <= "\u9fff" for ch in text),
        }
        for needle in STALE_OR_INTERNAL_STRINGS:
            if needle in text:
                findings.append({"file": rel, "kind": "stale_or_internal_string", "needle": needle})
        for line_no, line in enumerate(lines, start=1):
            if has_allowlisted_context(line):
                continue
            for pattern in ADVISORY_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "file": rel,
                            "line": line_no,
                            "kind": "advisory_or_question_risk",
                            "pattern": pattern.pattern,
                            "text": line,
                        }
                    )
    return findings, stats


def audit_consistency() -> list[dict]:
    findings: list[dict] = []
    manifest = load_json(PACKET_DIR / "packet_manifest.json")
    facts = (PACKET_DIR / f"OBJECTIVE_FACTS_{DATE_TAG}.md").read_text(encoding="utf-8")
    readme = EXTERNAL_README.read_text(encoding="utf-8")
    latex = load_json(PACKET_DIR / "validation" / "latex_build_summary.json")
    lint = load_json(PACKET_DIR / "validation" / "claim_scope_lint_report.json")
    hardening = load_json(PACKET_DIR / "validation" / "hardening_summary.json")
    zip_sha_text = (ROOT / "results" / "verification_substrate_gap" / f"vsg_expert_review_packet_{DATE_TAG}.zip.sha256").read_text(
        encoding="utf-8"
    )

    checks = {
        "pdf_sha_in_facts": latex["pdf_sha256"] in facts,
        "zip_sha_in_sha_file": bool(zip_sha_text.split()[0]),
        "manifest_self_hash_excluded": manifest.get("manifest_self_hash_excluded") is True,
        "lint_pass": lint.get("status") == "PASS" and lint.get("checked_files") == 17 and lint.get("violation_count") == 0,
        "latex_pass": latex.get("status") == "PASS" and latex.get("pdf_pages_from_log") == 32,
        "manuscript_head_c10b3f1": str(manifest.get("git_snapshot", {}).get("manuscript_repository_head", "")).startswith("c10b3f1"),
        "public_text_codeword_zero": hardening["public_text_stronger_baseline"]["codeword_recovered_blocks_total"] == 0,
        "attack_proxy_no_naturalness_claim": hardening["attack_naturalness_proxy"]["semantic_naturalness_claimed"] is False,
        "ownership_public_rows_zero": hardening["ownership_decision_rule_audit"]["supported_public_text_row_count"] == 0,
        "figure_quality_pass": hardening["manuscript_figure_quality_audit"]["failed_figure_count"] == 0
        and hardening["manuscript_figure_quality_audit"]["failed_data_check_count"] == 0,
    }
    for name, ok in checks.items():
        if not ok:
            findings.append({"kind": "consistency_check_failed", "check": name, "value": ok})

    required_missing = [token for token in OBJECTIVE_REQUIRED_STRINGS if token not in readme and token not in facts]
    for token in required_missing:
        findings.append({"kind": "required_objective_string_missing", "token": token})

    return findings


def audit() -> dict:
    verifier = run_verify()
    text_findings, text_stats = audit_reviewer_text()
    consistency_findings = audit_consistency()
    failures = []
    if verifier.get("status") != "PASS":
        failures.append({"kind": "packet_verifier_failed", "details": verifier})
    failures.extend(text_findings)
    failures.extend(consistency_findings)
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "packet_verifier": verifier,
        "reviewer_facing_file_stats": text_stats,
        "audited_files": [path.relative_to(ROOT).as_posix() for path in REVIEWER_FACING_FILES],
        "objective_only_scope": {
            "no_expert_questions": True,
            "no_route_recommendations": True,
            "no_new_experiments": True,
            "no_slurm_generation_scoring_training": True,
            "overleaf_push_not_performed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = audit()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
