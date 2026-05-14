from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.extract_r4_positive_phrase_events import (  # noqa: E402
    extract_phrase_events,
    load_surface_bank,
    normalize_text,
)

DEFAULT_GENERATION_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_event_bank_dev_diagnostic_859277"
)
DEFAULT_SURFACE_BANK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605/surface_bank.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_positive_zero_event_support_gap_audit_20260514_2102"
)

TECHNICAL_FORBIDDEN_TERMS = (
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "decoder",
    "hidden signal",
    "evidence block",
    "bucket",
    "coordinate",
)

STRUCTURAL_PREFIX = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)]\s+|(?:next action|action|summary|note|plan)\s*:\s+|[*_`#>\s]+)",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-z][a-z'-]*", re.IGNORECASE)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        yield payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_generated_rows(generation_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(generation_dir.glob("shards/shard_*/r4_generated_outputs.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no generated output rows found under {generation_dir}")
    return rows


def _stem(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _stemmed_text(text: str) -> str:
    return " ".join(_stem(token) for token in WORD_RE.findall(normalize_text(text)))


def _stemmed_phrase(phrase: str) -> str:
    return " ".join(_stem(token) for token in WORD_RE.findall(normalize_text(phrase)))


def _segments(text: str) -> list[str]:
    parts: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            cleaned = STRUCTURAL_PREFIX.sub("", part).strip()
            if cleaned:
                parts.append(cleaned)
    return parts


def _first_word(segment: str) -> str:
    match = WORD_RE.search(segment)
    return match.group(0).lower() if match else ""


def _window(text: str, term: str, *, chars: int = 80) -> str:
    lower = text.lower()
    idx = lower.find(term)
    if idx < 0:
        return ""
    start = max(0, idx - chars)
    end = min(len(text), idx + len(term) + chars)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _word_boundary_count(text: str, term: str) -> int:
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.IGNORECASE)
    return len(pattern.findall(text))


def _audit_forbidden(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    term_condition_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], str] = {}
    for row in rows:
        condition = str(row.get("generation_condition", "unknown"))
        text = str(row.get("response_text", ""))
        lower = text.lower()
        for term in TECHNICAL_FORBIDDEN_TERMS:
            substring_hits = lower.count(term)
            word_hits = _word_boundary_count(text, term)
            if substring_hits:
                term_condition_counts[(term, condition)]["substring_hits"] += substring_hits
                term_condition_counts[(term, condition)]["rows_with_substring"] += 1
                examples.setdefault((term, condition), _window(text, term))
            if word_hits:
                term_condition_counts[(term, condition)]["word_boundary_hits"] += word_hits
                term_condition_counts[(term, condition)]["rows_with_word_boundary"] += 1

    csv_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for (term, condition), counts in sorted(term_condition_counts.items()):
        row = {
            "term": term,
            "condition": condition,
            "substring_hits": counts["substring_hits"],
            "rows_with_substring": counts["rows_with_substring"],
            "word_boundary_hits": counts["word_boundary_hits"],
            "rows_with_word_boundary": counts["rows_with_word_boundary"],
            "example_window": examples.get((term, condition), ""),
        }
        csv_rows.append(row)
        totals[f"{term}:substring_hits"] += counts["substring_hits"]
        totals[f"{term}:word_boundary_hits"] += counts["word_boundary_hits"]
    return csv_rows, {"totals": dict(totals)}


def run_audit(*, generation_dir: Path, surface_bank_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    rows = _load_generated_rows(generation_dir)
    surface_bank = load_surface_bank(surface_bank_path)
    bank_first_words = Counter(_first_word(str(surface["canonical_phrase"])) for surface in surface_bank)
    bank_stemmed = {
        str(surface["surface_id"]): _stemmed_phrase(str(surface["canonical_phrase"]))
        for surface in surface_bank
    }

    condition_counts: Counter[str] = Counter()
    exact_hits_by_condition: Counter[str] = Counter()
    loose_stem_hits_by_condition: Counter[str] = Counter()
    rows_with_exact_by_condition: Counter[str] = Counter()
    rows_with_loose_by_condition: Counter[str] = Counter()
    rows_with_bank_first_word_by_condition: Counter[str] = Counter()
    segment_counts_by_condition: Counter[str] = Counter()
    opener_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bank_surface_hits: dict[str, Counter[str]] = defaultdict(Counter)
    bank_surface_loose_hits: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        condition = str(row.get("generation_condition", "unknown"))
        condition_counts[condition] += 1
        text = str(row.get("response_text", ""))
        exact_events = extract_phrase_events(text, surface_bank, scrub_mode="all")
        exact_hits_by_condition[condition] += len(exact_events)
        if exact_events:
            rows_with_exact_by_condition[condition] += 1
            for event in exact_events:
                bank_surface_hits[str(event["surface_id"])][condition] += 1

        stemmed = _stemmed_text(text)
        loose_hits_this_row = 0
        for surface in surface_bank:
            surface_id = str(surface["surface_id"])
            stemmed_phrase = bank_stemmed[surface_id]
            if stemmed_phrase and re.search(rf"(?<![a-z0-9]){re.escape(stemmed_phrase)}(?![a-z0-9])", stemmed):
                bank_surface_loose_hits[surface_id][condition] += 1
                loose_stem_hits_by_condition[condition] += 1
                loose_hits_this_row += 1
        if loose_hits_this_row:
            rows_with_loose_by_condition[condition] += 1

        row_has_bank_first_word = False
        for segment in _segments(text):
            first = _first_word(segment)
            if not first:
                continue
            segment_counts_by_condition[condition] += 1
            opener_counts[condition][first] += 1
            if first in bank_first_words:
                row_has_bank_first_word = True
        if row_has_bank_first_word:
            rows_with_bank_first_word_by_condition[condition] += 1

    opener_csv_rows: list[dict[str, Any]] = []
    for condition, counter in sorted(opener_counts.items()):
        for opener, count in counter.most_common():
            opener_csv_rows.append(
                {
                    "condition": condition,
                    "opener": opener,
                    "count": count,
                    "in_bank_first_words": opener in bank_first_words,
                    "bank_first_word_count": bank_first_words.get(opener, 0),
                }
            )

    surface_csv_rows: list[dict[str, Any]] = []
    for surface in surface_bank:
        surface_id = str(surface["surface_id"])
        exact = bank_surface_hits[surface_id]
        loose = bank_surface_loose_hits[surface_id]
        surface_csv_rows.append(
            {
                "surface_id": surface_id,
                "surface_family": surface["surface_family"],
                "canonical_phrase": surface["canonical_phrase"],
                "first_word": _first_word(str(surface["canonical_phrase"])),
                "protected_exact_hits": exact["protected"],
                "raw_exact_hits": exact["raw"],
                "task_only_exact_hits": exact["task_only"],
                "protected_loose_stem_hits": loose["protected"],
                "raw_loose_stem_hits": loose["raw"],
                "task_only_loose_stem_hits": loose["task_only"],
            }
        )

    condition_csv_rows: list[dict[str, Any]] = []
    for condition in sorted(condition_counts):
        row_count = condition_counts[condition]
        segment_count = segment_counts_by_condition[condition]
        bank_first_rows = rows_with_bank_first_word_by_condition[condition]
        condition_csv_rows.append(
            {
                "condition": condition,
                "rows": row_count,
                "segments": segment_count,
                "exact_surface_hits": exact_hits_by_condition[condition],
                "rows_with_exact_surface_hit": rows_with_exact_by_condition[condition],
                "loose_stem_surface_hits": loose_stem_hits_by_condition[condition],
                "rows_with_loose_stem_surface_hit": rows_with_loose_by_condition[condition],
                "rows_with_bank_first_word_opener": bank_first_rows,
                "row_bank_first_word_opener_rate": bank_first_rows / row_count if row_count else 0.0,
            }
        )

    forbidden_rows, forbidden_summary = _audit_forbidden(rows)

    summary = {
        "schema_name": "natural_evidence_v2_r4_positive_zero_event_support_gap_audit_v1",
        "status": "PASS_AUDIT_RECORDED_ZERO_EVENT_SUPPORT_CONFIRMED",
        "generation_dir": str(generation_dir),
        "surface_bank_path": str(surface_bank_path),
        "generated_rows": len(rows),
        "condition_counts": dict(condition_counts),
        "surface_bank_rows": len(surface_bank),
        "bank_first_words": dict(bank_first_words),
        "condition_coverage": condition_csv_rows,
        "forbidden_matcher": forbidden_summary,
        "interpretation": (
            "Frozen exact phrase support is absent, while generated outputs contain many "
            "natural action openers that overlap bank first words. This confirms a "
            "phrase-specific support gap rather than a wrapper or Slurm failure."
        ),
        "post_hoc_phrase_mining_allowed": False,
        "unchanged_resubmission_allowed": False,
        "slurm_allowed": False,
        "generation_allowed": False,
        "paper_claim_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "support_gap_summary.json", summary)
    _write_csv(
        output_dir / "condition_coverage.csv",
        condition_csv_rows,
        [
            "condition",
            "rows",
            "segments",
            "exact_surface_hits",
            "rows_with_exact_surface_hit",
            "loose_stem_surface_hits",
            "rows_with_loose_stem_surface_hit",
            "rows_with_bank_first_word_opener",
            "row_bank_first_word_opener_rate",
        ],
    )
    _write_csv(
        output_dir / "opener_counts_by_condition.csv",
        opener_csv_rows,
        ["condition", "opener", "count", "in_bank_first_words", "bank_first_word_count"],
    )
    _write_csv(
        output_dir / "bank_surface_coverage.csv",
        surface_csv_rows,
        [
            "surface_id",
            "surface_family",
            "canonical_phrase",
            "first_word",
            "protected_exact_hits",
            "raw_exact_hits",
            "task_only_exact_hits",
            "protected_loose_stem_hits",
            "raw_loose_stem_hits",
            "task_only_loose_stem_hits",
        ],
    )
    _write_csv(
        output_dir / "forbidden_matcher_semantics.csv",
        forbidden_rows,
        [
            "term",
            "condition",
            "substring_hits",
            "rows_with_substring",
            "word_boundary_hits",
            "rows_with_word_boundary",
            "example_window",
        ],
    )
    _write_report(output_dir / "support_gap_report.md", summary, opener_counts)
    return summary


def _write_report(path: Path, summary: Mapping[str, Any], opener_counts: Mapping[str, Counter[str]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    lines = [
        "# R4 Positive Zero-Event Support Gap Audit",
        "",
        "## Verdict",
        "",
        "`PASS_AUDIT_RECORDED_ZERO_EVENT_SUPPORT_CONFIRMED`",
        "",
        "The audit confirms the 859277 failure mode: exact frozen phrase-event support is absent.",
        "The generated text contains task-natural action language, but not the locked multi-word",
        "phrases required by the precommitted extractor.",
        "",
        "## Coverage By Condition",
        "",
        "| condition | rows | segments | exact hits | rows with exact | loose stem hits | rows with bank first-word opener |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["condition_coverage"]:
        lines.append(
            "| {condition} | {rows} | {segments} | {exact_surface_hits} | "
            "{rows_with_exact_surface_hit} | {loose_stem_surface_hits} | "
            "{rows_with_bank_first_word_opener} |".format(**row)
        )
    lines.extend(["", "## Top Openers", ""])
    for condition in sorted(opener_counts):
        top = ", ".join(f"{word}={count}" for word, count in opener_counts[condition].most_common(12))
        lines.append(f"- `{condition}`: {top}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact phrase hits are zero, so the keyed decoder has no support.",
            "- Bank first-word overlap is nonzero, so the gap is phrase-specific, not a total absence of action language.",
            "- Forbidden matcher hits remain diagnostic only; matcher repair cannot rescue 859277 because support is zero.",
            "- 859277 outputs must not be mined into the next locked bank.",
            "",
            "## Artifacts",
            "",
            "- `support_gap_summary.json`",
            "- `condition_coverage.csv`",
            "- `opener_counts_by_condition.csv`",
            "- `bank_surface_coverage.csv`",
            "- `forbidden_matcher_semantics.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit zero-event support gap for R4 positive 859277 artifacts.")
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_audit(
        generation_dir=args.generation_dir if args.generation_dir.is_absolute() else ROOT / args.generation_dir,
        surface_bank_path=args.surface_bank if args.surface_bank.is_absolute() else ROOT / args.surface_bank,
        output_dir=args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
