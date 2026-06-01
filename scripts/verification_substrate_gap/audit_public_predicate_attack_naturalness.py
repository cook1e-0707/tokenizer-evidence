#!/usr/bin/env python3
"""Artifact-only readability audit for public-predicate attack examples.

This audit uses existing source-mismatch guided rewrite/graft examples. It
does not call a model, does not score semantic naturalness, and does not turn
the attack into protected success or codeword recovery. The goal is to make the
known attack-output quality risk measurable instead of leaving it as prose.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUS = "PASS_PUBLIC_PREDICATE_ATTACK_NATURALNESS_PROXY_AUDIT_RECORDED_NO_CLAIMS"

INPUT_DEFAULT = Path(
    "results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/"
    "surrogate_guided_rewrite_examples.csv"
)
OUTPUT_DEFAULT = Path(
    "results/verification_substrate_gap/public_predicate_attack_naturalness_audit_20260601"
)

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")
BROKEN_GRAFT_RE = re.compile(
    r"\b(?:overlapping pages|useful next actio|bike m\b|maintenanc\b|communication-guideline|template-standardization)\b",
    re.IGNORECASE,
)

ROW_FIELDS = [
    "source_id",
    "model_family",
    "candidate_arm",
    "rank_index",
    "transform_id",
    "query_cost_to_here",
    "original_score",
    "rewrite_score",
    "threshold",
    "score_delta",
    "original_token_count",
    "rewrite_token_count",
    "length_ratio",
    "token_jaccard",
    "starts_uppercase",
    "ends_with_sentence_punctuation",
    "isolated_fragment_count",
    "broken_graft_marker_count",
    "rewrite_opener_present",
    "rewrite_phrase_present",
    "proxy_quality_status",
    "proxy_quality_fail_reasons",
    "claim_scope",
]

GROUP_FIELDS = [
    "source_id",
    "model_family",
    "candidate_arm",
    "rows",
    "proxy_pass_rows",
    "proxy_fail_rows",
    "proxy_pass_rate",
    "mean_length_ratio",
    "mean_token_jaccard",
    "mean_isolated_fragment_count",
    "mean_broken_graft_marker_count",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def isolated_fragment_count(text: str) -> int:
    return sum(1 for token in tokens(text) if len(token) == 1 and token not in {"a", "i"})


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def audit_text_pair(original: str, rewrite: str) -> dict[str, Any]:
    original_norm = normalize_space(original)
    rewrite_norm = normalize_space(rewrite)
    original_tokens = tokens(original_norm)
    rewrite_tokens = tokens(rewrite_norm)
    length_ratio = len(rewrite_tokens) / max(len(original_tokens), 1)
    isolated_fragments = isolated_fragment_count(rewrite_norm)
    broken_markers = len(BROKEN_GRAFT_RE.findall(rewrite_norm))
    starts_upper = bool(rewrite_norm and rewrite_norm[0].isupper())
    ends_sentence = bool(SENTENCE_END_RE.search(rewrite_norm))
    jaccard = token_jaccard(original_norm, rewrite_norm)
    fail_reasons: list[str] = []
    if len(rewrite_tokens) < 12:
        fail_reasons.append("rewrite_too_short")
    if not (0.5 <= length_ratio <= 2.5):
        fail_reasons.append("length_ratio_outside_proxy_range")
    if jaccard < 0.08:
        fail_reasons.append("low_original_rewrite_token_overlap")
    if not starts_upper:
        fail_reasons.append("does_not_start_uppercase")
    if not ends_sentence:
        fail_reasons.append("does_not_end_with_sentence_punctuation")
    if isolated_fragments > 0:
        fail_reasons.append("isolated_single_letter_fragment")
    if broken_markers > 0:
        fail_reasons.append("known_broken_graft_marker")
    return {
        "original_token_count": len(original_tokens),
        "rewrite_token_count": len(rewrite_tokens),
        "length_ratio": round(length_ratio, 6),
        "token_jaccard": round(jaccard, 6),
        "starts_uppercase": starts_upper,
        "ends_with_sentence_punctuation": ends_sentence,
        "isolated_fragment_count": isolated_fragments,
        "broken_graft_marker_count": broken_markers,
        "proxy_quality_status": "PASS_PROXY_READABILITY" if not fail_reasons else "FAIL_PROXY_READABILITY",
        "proxy_quality_fail_reasons": ";".join(fail_reasons),
    }


def build(input_csv: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = _read_csv(input_csv)
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        metrics = audit_text_pair(row.get("original_excerpt", ""), row.get("best_rewrite_excerpt", ""))
        original_score = float(row.get("original_score", "0") or 0)
        rewrite_score = float(row.get("best_rewrite_score", "0") or 0)
        audit_rows.append(
            {
                "source_id": row.get("source_id", ""),
                "model_family": row.get("model_family", ""),
                "candidate_arm": row.get("candidate_arm", ""),
                "rank_index": row.get("rank_index", ""),
                "transform_id": row.get("best_transform_id", ""),
                "query_cost_to_here": row.get("public_predicate_query_cost_to_here", ""),
                "original_score": original_score,
                "rewrite_score": rewrite_score,
                "threshold": row.get("threshold", ""),
                "score_delta": round(rewrite_score - original_score, 6),
                "rewrite_opener_present": bool(row.get("rewrite_opener", "")),
                "rewrite_phrase_present": bool(row.get("rewrite_phrase", "")),
                "claim_scope": row.get("claim_scope", ""),
                **metrics,
            }
        )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        groups[(row["source_id"], row["model_family"], row["candidate_arm"])].append(row)

    group_rows: list[dict[str, Any]] = []
    fail_reason_counts: Counter[str] = Counter()
    for row in audit_rows:
        for reason in str(row["proxy_quality_fail_reasons"]).split(";"):
            if reason:
                fail_reason_counts[reason] += 1

    for (source_id, model_family, candidate_arm), rows in sorted(groups.items()):
        pass_rows = [row for row in rows if row["proxy_quality_status"] == "PASS_PROXY_READABILITY"]
        group_rows.append(
            {
                "source_id": source_id,
                "model_family": model_family,
                "candidate_arm": candidate_arm,
                "rows": len(rows),
                "proxy_pass_rows": len(pass_rows),
                "proxy_fail_rows": len(rows) - len(pass_rows),
                "proxy_pass_rate": round(len(pass_rows) / len(rows), 6) if rows else 0,
                "mean_length_ratio": round(sum(float(row["length_ratio"]) for row in rows) / len(rows), 6),
                "mean_token_jaccard": round(sum(float(row["token_jaccard"]) for row in rows) / len(rows), 6),
                "mean_isolated_fragment_count": round(
                    sum(int(row["isolated_fragment_count"]) for row in rows) / len(rows), 6
                ),
                "mean_broken_graft_marker_count": round(
                    sum(int(row["broken_graft_marker_count"]) for row in rows) / len(rows), 6
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    row_csv = output_dir / "attack_naturalness_proxy_rows.csv"
    group_csv = output_dir / "attack_naturalness_proxy_by_group.csv"
    summary_json = output_dir / "attack_naturalness_proxy_summary.json"
    report_md = output_dir / "attack_naturalness_proxy_report.md"
    manifest_json = output_dir / "attack_naturalness_proxy_manifest.json"
    _write_csv(row_csv, audit_rows, ROW_FIELDS)
    _write_csv(group_csv, group_rows, GROUP_FIELDS)

    pass_rows = [row for row in audit_rows if row["proxy_quality_status"] == "PASS_PROXY_READABILITY"]
    summary = {
        "status": STATUS,
        "schema_name": "verification_substrate_gap_public_predicate_attack_naturalness_proxy_v1",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "rows": len(audit_rows),
        "groups": len(group_rows),
        "proxy_pass_rows": len(pass_rows),
        "proxy_fail_rows": len(audit_rows) - len(pass_rows),
        "proxy_pass_rate": round(len(pass_rows) / len(audit_rows), 6) if audit_rows else 0,
        "fail_reason_counts": dict(sorted(fail_reason_counts.items())),
        "semantic_naturalness_claimed": False,
        "human_evaluation_performed": False,
        "model_evaluation_performed": False,
        "source_mismatch_rows_only": True,
        "protected_success_claimed": False,
        "codeword_recovery_claimed": False,
        "public_text_only_verification_claimed": False,
        "claim_scope": "proxy readability audit only; not semantic naturalness and not protected success",
        "new_slurm_started": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
    }
    _write_json(summary_json, summary)
    write_report(report_md, summary, group_rows)
    manifest = {
        "status": STATUS,
        "schema_name": "verification_substrate_gap_public_predicate_attack_naturalness_proxy_manifest_v1",
        "files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in [row_csv, group_csv, summary_json, report_md]
        ],
    }
    _write_json(manifest_json, manifest)
    return summary


def write_report(path: Path, summary: dict[str, Any], group_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Public-Predicate Attack Naturalness Proxy Audit",
        "",
        "This is an artifact-only proxy audit over existing guided rewrite/graft",
        "source-mismatch examples. It checks surface readability signals such as",
        "length ratio, token overlap, punctuation, isolated fragments, and known",
        "broken-graft markers. It is not a semantic naturalness evaluation.",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: `{summary['rows']}`",
        f"Proxy pass rows: `{summary['proxy_pass_rows']}`",
        f"Proxy fail rows: `{summary['proxy_fail_rows']}`",
        f"Proxy pass rate: `{summary['proxy_pass_rate']}`",
        "",
        "## By Group",
        "",
        "| Source | Model | Arm | Rows | Proxy pass | Proxy fail | Pass rate | Mean token overlap | Broken markers |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['source_id']} | {row['model_family']} | {row['candidate_arm']} | {row['rows']} | "
            f"{row['proxy_pass_rows']} | {row['proxy_fail_rows']} | {row['proxy_pass_rate']} | "
            f"{row['mean_token_jaccard']} | {row['mean_broken_graft_marker_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Passing this proxy audit would not prove naturalness. Failing it records",
            "that the current public-predicate guided rewrite/graft examples still",
            "carry visible surface-quality risks. In all cases, accepted rows remain",
            "source-mismatch spoofing evidence only, not protected success and not",
            "codeword recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.input_csv, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
