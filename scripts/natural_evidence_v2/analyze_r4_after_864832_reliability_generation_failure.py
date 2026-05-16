from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    match_surfaces,
    normalize_text,
    read_json,
    read_jsonl,
    segment_units,
    write_csv_new,
    write_json_new,
    write_text_new,
)

DEFAULT_GENERATION_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621"
)
DEFAULT_REVIEW_SUMMARY = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_review/review_summary.json"
)
DEFAULT_SURFACE_BANK = (
    ROOT
    / "results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_failure_analysis_20260516"
)

CANDIDATE_V3_VISIBLE_PHRASES = (
    "create a plan",
    "prepare a schedule",
    "prepare a budget",
    "create a budget plan",
    "develop a plan",
    "prepare a plan",
    "create a plan for the volunteers",
    "create a plan for the team",
)


def _read_generation_rows(generation_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(generation_dir.glob("shards/shard_*/r4_generated_outputs.jsonl")):
        for row in read_jsonl(path):
            row["_source_path"] = str(path)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    write_csv_new(path, rows, fieldnames)


def _top_ngrams(texts: Iterable[str], n: int, top_k: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = normalize_text(text).split()
        for index in range(0, max(0, len(tokens) - n + 1)):
            counts[" ".join(tokens[index : index + n])] += 1
    return [{"ngram": key, "count": value} for key, value in counts.most_common(top_k)]


def _sentence_repeat_stats(text: str) -> dict[str, Any]:
    units = [normalize_text(unit) for unit in segment_units(text)]
    units = [unit for unit in units if unit]
    counts = Counter(units)
    repeated_units = sum(count for count in counts.values() if count > 1)
    return {
        "unit_count": len(units),
        "unique_unit_count": len(counts),
        "duplicate_unit_rows": repeated_units,
        "max_unit_repeat": max(counts.values(), default=0),
        "top_repeated_unit": counts.most_common(1)[0][0] if counts else "",
        "top_repeated_unit_count": counts.most_common(1)[0][1] if counts else 0,
    }


def _surface_head_counts(surface_bank: Mapping[str, Any], rows_by_condition: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    head_counts: Counter[str] = Counter()
    phrase_count = 0
    for entry in surface_bank.get("entries", []):
        phrase = normalize_text(str(entry.get("canonical_lemma_or_phrase", "")))
        if not phrase:
            continue
        phrase_count += 1
        head_counts[phrase.split()[0]] += 1

    output_word_counts: dict[str, Counter[str]] = {}
    for condition, rows in rows_by_condition.items():
        counter: Counter[str] = Counter()
        for row in rows:
            counter.update(normalize_text(str(row.get("response_text", ""))).split())
        output_word_counts[condition] = counter

    output_rows: list[dict[str, Any]] = []
    for head, bank_count in head_counts.most_common():
        payload = {"surface_head": head, "bank_phrase_count": bank_count, "bank_phrase_fraction": bank_count / phrase_count}
        for condition in ("protected", "raw", "task_only"):
            payload[f"{condition}_output_word_count"] = output_word_counts.get(condition, Counter()).get(head, 0)
        output_rows.append(payload)
    return output_rows


def _candidate_phrase_rows(rows_by_condition: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for phrase in CANDIDATE_V3_VISIBLE_PHRASES:
        normalized_phrase = f" {normalize_text(phrase)} "
        row = {"phrase": phrase}
        for condition in ("protected", "raw", "task_only"):
            count = 0
            rows_with_phrase = 0
            for payload in rows_by_condition.get(condition, []):
                text = f" {normalize_text(str(payload.get('response_text', '')))} "
                hits = text.count(normalized_phrase)
                count += hits
                rows_with_phrase += int(hits > 0)
            row[f"{condition}_count"] = count
            row[f"{condition}_rows_with_phrase"] = rows_with_phrase
        output_rows.append(row)
    return output_rows


def _condition_summary(rows_by_condition: Mapping[str, list[Mapping[str, Any]]], surface_bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for condition in ("protected", "raw", "task_only"):
        rows = rows_by_condition.get(condition, [])
        response_hashes = Counter(str(row.get("response_text_sha256", "")) for row in rows)
        response_lengths = [len(str(row.get("response_text", "")).split()) for row in rows]
        exact_surface_match_rows = 0
        exact_surface_match_count = 0
        repeat_stats = [_sentence_repeat_stats(str(row.get("response_text", ""))) for row in rows]
        for row in rows:
            matches = match_surfaces(str(row.get("response_text", "")), surface_bank, scrub_mode="all")
            exact_surface_match_count += len(matches)
            exact_surface_match_rows += int(bool(matches))
        summaries.append(
            {
                "condition": condition,
                "rows": len(rows),
                "unique_response_hashes": len(response_hashes),
                "duplicate_response_hash_rows": sum(count for count in response_hashes.values() if count > 1),
                "max_response_hash_count": max(response_hashes.values(), default=0),
                "mean_response_words": statistics.fmean(response_lengths) if response_lengths else 0.0,
                "median_response_words": statistics.median(response_lengths) if response_lengths else 0.0,
                "surface_match_rows": exact_surface_match_rows,
                "surface_match_count": exact_surface_match_count,
                "mean_max_repeated_unit_count": statistics.fmean(
                    int(item["max_unit_repeat"]) for item in repeat_stats
                )
                if repeat_stats
                else 0.0,
                "max_repeated_unit_count": max((int(item["max_unit_repeat"]) for item in repeat_stats), default=0),
                "rows_with_repeated_units": sum(1 for item in repeat_stats if int(item["max_unit_repeat"]) > 1),
            }
        )
    return summaries


def _example_rows(rows_by_condition: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for condition in ("protected", "raw", "task_only"):
        rows = rows_by_condition.get(condition, [])
        ranked = sorted(
            rows,
            key=lambda row: (
                _sentence_repeat_stats(str(row.get("response_text", "")))["max_unit_repeat"],
                len(str(row.get("response_text", "")).split()),
            ),
            reverse=True,
        )
        for row in ranked[:5]:
            stats = _sentence_repeat_stats(str(row.get("response_text", "")))
            examples.append(
                {
                    "condition": condition,
                    "generation_id": row.get("generation_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "response_text_sha256": row.get("response_text_sha256", ""),
                    "max_unit_repeat": stats["max_unit_repeat"],
                    "top_repeated_unit_count": stats["top_repeated_unit_count"],
                    "top_repeated_unit": stats["top_repeated_unit"],
                    "response_excerpt": str(row.get("response_text", ""))[:700],
                }
            )
    return examples


def run_analysis(
    *,
    generation_dir: Path,
    review_summary_path: Path,
    surface_bank_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")

    generation_rows = _read_generation_rows(generation_dir)
    review_summary = read_json(review_summary_path)
    surface_bank = read_json(surface_bank_path)
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        rows_by_condition[str(row.get("generation_condition", "unknown"))].append(row)

    condition_summary = _condition_summary(rows_by_condition, surface_bank)
    candidate_rows = _candidate_phrase_rows(rows_by_condition)
    surface_head_rows = _surface_head_counts(surface_bank, rows_by_condition)
    examples = _example_rows(rows_by_condition)
    top_ngrams: list[dict[str, Any]] = []
    for condition in ("protected", "raw", "task_only"):
        texts = [str(row.get("response_text", "")) for row in rows_by_condition.get(condition, [])]
        for n in (2, 3, 4):
            for row in _top_ngrams(texts, n=n, top_k=20):
                top_ngrams.append({"condition": condition, "n": n, **row})

    protected_summary = next(item for item in condition_summary if item["condition"] == "protected")
    gate = review_summary.get("gate", {})
    root_cause = (
        "free_generation_transfer_failure_surface_absent"
        if int(protected_summary["surface_match_count"]) == 0
        else "free_generation_transfer_failure_surface_insufficient"
    )
    status = "FAILURE_ANALYSIS_RECORDED_NO_RERUN"
    summary = {
        "schema_name": "r4_after_864832_reliability_dev_generation_867621_failure_analysis_v1",
        "status": status,
        "job_id": review_summary.get("job_id", "867621"),
        "root_cause": root_cause,
        "generated_rows": len(generation_rows),
        "conditions": {condition: len(rows_by_condition.get(condition, [])) for condition in ("protected", "raw", "task_only")},
        "protected_accepts_format_scrub_all": gate.get("protected_accepts_format_scrub_all", 0),
        "protected_gate_required": gate.get("protected_gate_required", ">=26/32"),
        "control_gate_pass": bool(gate.get("control_gate_pass", False)),
        "protected_surface_match_count": int(protected_summary["surface_match_count"]),
        "protected_surface_match_rows": int(protected_summary["surface_match_rows"]),
        "protected_duplicate_response_hash_rows": int(protected_summary["duplicate_response_hash_rows"]),
        "protected_max_response_hash_count": int(protected_summary["max_response_hash_count"]),
        "protected_rows_with_repeated_units": int(protected_summary["rows_with_repeated_units"]),
        "condition_summary": condition_summary,
        "candidate_v3_visible_phrase_counts": candidate_rows,
        "top_surface_heads": surface_head_rows[:12],
        "next_allowed_action": (
            "Record a new reviewed repair/pivot route before any rerun, training, Llama, sanitizer, FAR, "
            "payload diversity, or paper-facing claim."
        ),
        "slurm_submitted": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }

    _write_csv(
        output_dir / "condition_summary.csv",
        condition_summary,
        [
            "condition",
            "rows",
            "unique_response_hashes",
            "duplicate_response_hash_rows",
            "max_response_hash_count",
            "mean_response_words",
            "median_response_words",
            "surface_match_rows",
            "surface_match_count",
            "mean_max_repeated_unit_count",
            "max_repeated_unit_count",
            "rows_with_repeated_units",
        ],
    )
    _write_csv(
        output_dir / "candidate_v3_visible_phrase_counts.csv",
        candidate_rows,
        [
            "phrase",
            "protected_count",
            "protected_rows_with_phrase",
            "raw_count",
            "raw_rows_with_phrase",
            "task_only_count",
            "task_only_rows_with_phrase",
        ],
    )
    _write_csv(
        output_dir / "surface_head_coverage.csv",
        surface_head_rows,
        [
            "surface_head",
            "bank_phrase_count",
            "bank_phrase_fraction",
            "protected_output_word_count",
            "raw_output_word_count",
            "task_only_output_word_count",
        ],
    )
    _write_csv(output_dir / "top_ngrams_by_condition.csv", top_ngrams, ["condition", "n", "ngram", "count"])
    _write_csv(
        output_dir / "degenerate_examples.csv",
        examples,
        [
            "condition",
            "generation_id",
            "prompt_id",
            "response_text_sha256",
            "max_unit_repeat",
            "top_repeated_unit_count",
            "top_repeated_unit",
            "response_excerpt",
        ],
    )
    write_json_new(output_dir / "failure_analysis_summary.json", summary)

    report = [
        "# R4 Reliability Dev Generation 867621 Failure Analysis",
        "",
        f"Status: `{status}`",
        "",
        "## Root Cause",
        "",
        "The reviewed H200 job completed cleanly, but the protected free-generation channel did not emit the",
        "precommitted coordinate-unique reliability surface bank. This is a free-generation transfer failure,",
        "not a Slurm completion failure, tokenizer-boundary failure, decoder oracle failure, or null-control failure.",
        "",
        "## Primary Evidence",
        "",
        f"- protected accepts, `format_scrub=all`: `{summary['protected_accepts_format_scrub_all']}/32`",
        f"- protected gate: `{summary['protected_gate_required']}`",
        f"- null/control gate pass: `{summary['control_gate_pass']}`",
        f"- protected surface matches against coordinate-unique bank: `{summary['protected_surface_match_count']}`",
        f"- protected rows with any coordinate-unique bank surface: `{summary['protected_surface_match_rows']}`",
        f"- protected duplicate response hash rows: `{summary['protected_duplicate_response_hash_rows']}`",
        f"- protected max duplicate response hash count: `{summary['protected_max_response_hash_count']}`",
        f"- protected rows with repeated sentence/clause units: `{summary['protected_rows_with_repeated_units']}`",
        "",
        "## Visible Pattern",
        "",
        "The protected adapter still strongly biases old candidate-v3 continuation language such as",
        "`Create a plan` / `Prepare a schedule`, but the reliability decoder surfaces are longer",
        "coordinate-identifiable phrases such as ordinary review/check/confirm/update continuations.",
        "The old visible phrase pressure therefore does not transfer into the frozen reliability bank.",
        "",
        "## Control Decision",
        "",
        "Do not rerun this route unchanged. Do not lower gates or add 867621-observed phrases to the bank.",
        "A new reviewed repair/pivot route is required before any further Slurm, generation, training,",
        "Llama, sanitizer, FAR, payload-diversity, or paper-facing claim work.",
        "",
        "## Artifacts",
        "",
        "- `condition_summary.csv`",
        "- `candidate_v3_visible_phrase_counts.csv`",
        "- `surface_head_coverage.csv`",
        "- `top_ngrams_by_condition.csv`",
        "- `degenerate_examples.csv`",
        "- `failure_analysis_summary.json`",
    ]
    write_text_new(output_dir / "failure_analysis.md", "\n".join(report) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Artifact-only failure analysis for R4 reliability dev generation 867621."
    )
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--review-summary", type=Path, default=DEFAULT_REVIEW_SUMMARY)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_analysis(
        generation_dir=args.generation_dir,
        review_summary_path=args.review_summary,
        surface_bank_path=args.surface_bank,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
