#!/usr/bin/env python3
"""Lint active docs for unqualified verification claims.

The linter is intentionally conservative and text-only. It looks for phrases
that are unsafe in the substrate-gap route unless the same paragraph contains a
scope qualifier such as "trace-bound", "provider-side", or "not paper-facing".
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UNSAFE_PHRASES = [
    "public verifier",
    "text-only verification",
    "ownership proof",
    "cryptographic provenance",
    "signed trace",
    "natural evidence success",
    "watermark success",
    "phrase-decoder success",
    "llama transfer claim",
    "far claim",
    "sanitizer robustness",
    "payload diversity",
]

QUALIFIERS = [
    "trace-bound",
    "trace bound",
    "provider-side",
    "provider side",
    "authorized-verifier only",
    "authorized verifier only",
    "oracle diagnostic",
    "diagnostic only",
    "not public text-only",
    "not_public_text_only",
    "not public",
    "not_public",
    "not text-only",
    "not_text_only",
    "not signed",
    "not hmac",
    "not_hmac",
    "not paper-facing",
    "not_paper_facing",
    "not paper facing",
    "not claimed",
    "not_claimed",
    "unsupported",
    "unsupported axes",
    "outside evidence scope",
    "outside the evidence scope",
    "not allowed",
    "not unlock",
    "cannot unlock",
    "do not unlock",
    "do not make",
    "do not claim",
    "do not use it to claim",
    "do not treat",
    "spoofing",
    "spoof",
    "attack",
    "not protected success",
    "not codeword recovery",
    "not make",
    "report-only",
    "report only",
    "remain gated",
    "gated",
    'claim_allowed": false',
    "claim_allowed': false",
    'payload_diversity_tested": false',
    "historical",
    "failure",
    "failed",
    "no paper claim",
    "paper_claim_allowed: false",
    "without using trace/key",
]

ALLOW_FILE_DIRECTIVE = "claim-lint: allow-file"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    phrase: str
    excerpt: str


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _iter_paragraphs(text: str) -> Iterable[tuple[int, str]]:
    start_line = 1
    lines: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not lines:
                start_line = idx
            lines.append(line)
            continue
        if lines:
            yield start_line, "\n".join(lines)
            lines = []
    if lines:
        yield start_line, "\n".join(lines)


def _has_qualifier(paragraph: str) -> bool:
    lowered = paragraph.lower()
    return any(q in lowered for q in QUALIFIERS)


def lint_text(path: Path, text: str, *, include_code_fences: bool = False) -> list[Violation]:
    if ALLOW_FILE_DIRECTIVE in "\n".join(text.splitlines()[:10]):
        return []
    lintable_text = text if include_code_fences else _strip_code_fences(text)
    violations: list[Violation] = []
    for line, paragraph in _iter_paragraphs(lintable_text):
        lowered = paragraph.lower()
        if _has_qualifier(paragraph):
            continue
        for phrase in UNSAFE_PHRASES:
            if phrase in lowered:
                excerpt = re.sub(r"\s+", " ", paragraph).strip()
                violations.append(
                    Violation(
                        path=str(path),
                        line=line,
                        phrase=phrase,
                        excerpt=excerpt[:280],
                    )
                )
    return violations


def _expand_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}
            )
        elif path.is_file():
            expanded.append(path)
    return expanded


def lint_paths(paths: Iterable[Path], *, include_code_fences: bool = False) -> dict:
    files = _expand_paths(paths)
    all_violations: list[Violation] = []
    skipped_unreadable: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped_unreadable.append(str(path))
            continue
        all_violations.extend(lint_text(path, text, include_code_fences=include_code_fences))
    return {
        "status": "PASS" if not all_violations else "FAIL",
        "checked_files": len(files),
        "skipped_unreadable": skipped_unreadable,
        "violation_count": len(all_violations),
        "violations": [v.__dict__ for v in all_violations],
        "unsafe_phrases": UNSAFE_PHRASES,
        "qualifiers": QUALIFIERS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("docs/verification_substrate_gap"),
            Path("docs/natural_evidence_v2/CURRENT_STATE.md"),
        ],
        help="Files or directories to scan.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--include-code-fences",
        action="store_true",
        help="Scan fenced code blocks as claim text. Default strips them.",
    )
    args = parser.parse_args()

    report = lint_paths(args.paths, include_code_fences=args.include_code_fences)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
