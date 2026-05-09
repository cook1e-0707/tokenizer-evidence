from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import stable_hash_hex, token_surface_allowed, write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_v1_local_suffix_repair_dry_run_v1"
ROW_SCHEMA = "natural_evidence_v1_local_suffix_repair_dry_run_row_v1"

ROW_FIELDS = [
    "row_id",
    "dry_run_status",
    "model_condition",
    "expected_payload_id",
    "seed",
    "prompt_id",
    "prompt_slot",
    "match_policy",
    "drift_reason",
    "target_token_text",
    "observed_token_text",
    "replacement_mode",
    "observed_occurrence_count",
    "local_suffix_window_chars",
    "frame_index",
    "frame_digit_index",
    "paper_claim_allowed",
    "training_started",
    "generation_started",
    "e2e_eval_started",
]

READINESS_FIELDS = ["gate", "status", "evidence", "next_action"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an artifact-only local-suffix repair dry-run from prepared "
            "branch-aware/regenerated-suffix inputs. This performs only "
            "deterministic text-level feasibility checks. It does not load a "
            "model/tokenizer, regenerate suffixes, train, run E2E, or claim FAR."
        )
    )
    parser.add_argument("--repair-examples-jsonl", required=True)
    parser.add_argument("--branch-plan-jsonl", required=True)
    parser.add_argument("--prepared-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix-window-chars", type=int, default=160)
    parser.add_argument("--max-markdown-examples", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value) == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _choose_target_token(tokens: Any) -> tuple[str, str]:
    if not isinstance(tokens, list):
        return "", "no_target_tokens"
    candidates = [token for token in tokens if isinstance(token, dict)]
    for token in candidates:
        text = str(token.get("token_text", ""))
        if token_surface_allowed(text):
            return text, "first_surface_allowed_target_token"
    for token in candidates:
        text = str(token.get("token_text", ""))
        if text.strip():
            return text, "first_nonempty_target_token_surface_flagged"
    return "", "no_nonempty_target_token"


def _candidate_needles(observed: str) -> list[tuple[str, str]]:
    needles: list[tuple[str, str]] = []
    if observed:
        needles.append((observed, "observed_token_text_exact"))
    stripped = observed.strip()
    if stripped and stripped != observed:
        needles.append((stripped, "observed_token_text_stripped"))
    return needles


def _replace_once(response: str, observed: str, target: str, suffix_window_chars: int) -> dict[str, Any]:
    if not response:
        return {
            "dry_run_status": "UNREPAIRABLE_EMPTY_RESPONSE_TEXT",
            "replacement_mode": "none",
            "observed_occurrence_count": 0,
            "repaired_response_text": "",
            "local_suffix_window_after_observed": "",
            "prefix_before_observed": "",
            "observed_match_text": "",
        }
    if not observed.strip():
        return {
            "dry_run_status": "UNREPAIRABLE_EMPTY_OBSERVED_TOKEN_TEXT",
            "replacement_mode": "none",
            "observed_occurrence_count": 0,
            "repaired_response_text": response,
            "local_suffix_window_after_observed": "",
            "prefix_before_observed": "",
            "observed_match_text": "",
        }
    if not target.strip():
        return {
            "dry_run_status": "UNREPAIRABLE_EMPTY_TARGET_TOKEN_TEXT",
            "replacement_mode": "none",
            "observed_occurrence_count": 0,
            "repaired_response_text": response,
            "local_suffix_window_after_observed": "",
            "prefix_before_observed": "",
            "observed_match_text": "",
        }
    for needle, mode in _candidate_needles(observed):
        occurrence_count = response.count(needle)
        index = response.find(needle)
        if index < 0:
            continue
        replacement = target if needle[:1].isspace() else target.strip()
        before = response[:index]
        after = response[index + len(needle) :]
        return {
            "dry_run_status": "REPAIR_DRY_RUN_TEXT_SUBSTITUTION_READY_NOT_REGENERATED",
            "replacement_mode": mode,
            "observed_occurrence_count": occurrence_count,
            "repaired_response_text": before + replacement + after,
            "local_suffix_window_after_observed": after[:suffix_window_chars],
            "prefix_before_observed": before[-suffix_window_chars:],
            "observed_match_text": needle,
        }
    return {
        "dry_run_status": "NEEDS_TOKENIZER_ALIGNED_OR_BRANCH_REGEN_OBSERVED_TEXT_NOT_FOUND",
        "replacement_mode": "none",
        "observed_occurrence_count": 0,
        "repaired_response_text": response,
        "local_suffix_window_after_observed": "",
        "prefix_before_observed": "",
        "observed_match_text": "",
    }


def _load_plan_by_row_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        rows[str(row.get("row_id", ""))] = row
    return rows


def _make_row(
    example: Mapping[str, Any],
    plan_row: Mapping[str, Any] | None,
    suffix_window_chars: int,
) -> dict[str, Any]:
    target_token_text, target_choice_status = _choose_target_token(example.get("target_bucket_tokens", []))
    repair = _replace_once(
        str(example.get("original_response_text", "")),
        str(example.get("observed_token_text", "")),
        target_token_text,
        suffix_window_chars,
    )
    row_id = str(example.get("row_id", "")) or stable_hash_hex(
        [
            "local_suffix_repair",
            example.get("model_condition", ""),
            example.get("expected_payload_id", ""),
            example.get("prompt_id", ""),
            example.get("prompt_slot", ""),
            example.get("match_policy", ""),
        ]
    )[:24]
    return {
        "schema_name": ROW_SCHEMA,
        "row_id": row_id,
        "dry_run_status": repair["dry_run_status"],
        "target_choice_status": target_choice_status,
        "model_condition": str(example.get("model_condition", "")),
        "expected_payload_id": str(example.get("expected_payload_id", "")),
        "seed": str(example.get("seed", "")),
        "prompt_id": str(example.get("prompt_id", "")),
        "prompt_slot": _as_int(example.get("prompt_slot", 0)),
        "match_policy": str(example.get("match_policy", "")),
        "drift_reason": str(example.get("drift_reason", "")),
        "target_bucket": str(example.get("target_bucket", "")),
        "target_token_text": target_token_text,
        "observed_token_text": str(example.get("observed_token_text", "")),
        "replacement_mode": repair["replacement_mode"],
        "observed_occurrence_count": repair["observed_occurrence_count"],
        "local_suffix_window_chars": suffix_window_chars,
        "prompt": str(example.get("prompt", "")),
        "user_probe": str(example.get("user_probe", "")),
        "original_response_text": str(example.get("original_response_text", "")),
        "repaired_response_text": repair["repaired_response_text"],
        "prefix_before_observed": repair["prefix_before_observed"],
        "observed_match_text": repair["observed_match_text"],
        "local_suffix_window_after_observed": repair["local_suffix_window_after_observed"],
        "token_index": example.get("token_index", ""),
        "frame_index": "" if plan_row is None else plan_row.get("frame_index", ""),
        "frame_digit_index": "" if plan_row is None else plan_row.get("frame_digit_index", ""),
        "dry_run_only": True,
        "model_loading_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "result_claim": "local_suffix_repair_dry_run_not_regenerated_not_training_not_far",
    }


def _counter_rows(counter: Counter[str], field_name: str) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items()):
        rows.append({field_name: key, "rows": count, "fraction": count / total if total else 0.0})
    return rows


def _readiness_rows(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    ready = summary["repair_ready_rows"]
    total = summary["repair_example_rows"]
    return [
        {
            "gate": "local_suffix_repair_dry_run",
            "status": summary["status"],
            "evidence": f"ready_text_substitution_rows={ready}/{total}",
            "next_action": "Use as dry-run evidence only; do not train from this artifact.",
        },
        {
            "gate": "branch_aware_model_scoring",
            "status": "NEEDS_RESULTS",
            "evidence": "dry-run is text-level only and not compatibility scoring",
            "next_action": "Run Slurm-scored branch-aware compatibility diagnostic if proceeding.",
        },
        {
            "gate": "protected_task_only_comparison",
            "status": "BLOCKED" if summary["model_condition_counts"] == {"raw": total} else "NEEDS_REVIEW",
            "evidence": f"model_condition_counts={summary['model_condition_counts']}",
            "next_action": "Export/select protected and task-only rows before ownership-signal conclusions.",
        },
        {
            "gate": "training_or_e2e",
            "status": "BLOCKED",
            "evidence": "training_started=false; e2e_eval_started=false",
            "next_action": "No training or E2E rerun.",
        },
    ]


def _write_markdown(path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], max_examples: int) -> None:
    lines = [
        "# Local-Suffix Repair Dry-Run",
        "",
        "This is an artifact-only dry-run. It performs deterministic text-level local substitutions only. It does not regenerate suffixes, score branch-aware compatibility, train, run E2E, claim payload recovery, or estimate FAR.",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "## Counts",
        "",
        f"- repair examples: `{summary['repair_example_rows']}`",
        f"- text-substitution-ready rows: `{summary['repair_ready_rows']}`",
        f"- needs tokenizer-aligned/branch regeneration rows: `{summary['needs_tokenizer_aligned_or_branch_regen_rows']}`",
        f"- model condition counts: `{summary['model_condition_counts']}`",
        "",
        "## Limitation",
        "",
        "This dry-run uses approximate text replacement around the observed token text. It is not tokenizer-aligned regeneration and cannot establish compatibility, protected lift, payload recovery, or FAR.",
        "",
        "## Examples",
        "",
    ]
    for row in rows[:max_examples]:
        lines.extend(
            [
                f"### {row['row_id']}",
                "",
                f"- status: `{row['dry_run_status']}`",
                f"- observed -> target: `{row['observed_token_text']}` -> `{row['target_token_text']}`",
                f"- condition: `{row['model_condition']}`",
                "",
                "Original:",
                "",
                "```text",
                str(row.get("original_response_text", ""))[:1200],
                "```",
                "",
                "Dry-run repaired:",
                "",
                "```text",
                str(row.get("repaired_response_text", ""))[:1200],
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dry_run(
    *,
    repair_examples_jsonl: Path,
    branch_plan_jsonl: Path,
    prepared_summary_json: Path,
    output_dir: Path,
    suffix_window_chars: int,
    max_markdown_examples: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "local_suffix_repair_dry_run_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing local-suffix dry-run: {summary_path}")
    prepared_summary = _read_json(prepared_summary_json)
    repair_examples = list(_iter_jsonl(repair_examples_jsonl))
    plan_by_row_id = _load_plan_by_row_id(branch_plan_jsonl)
    rows = [
        _make_row(example, plan_by_row_id.get(str(example.get("row_id", ""))), suffix_window_chars)
        for example in repair_examples
    ]
    status_counts = Counter(str(row["dry_run_status"]) for row in rows)
    condition_counts = Counter(str(row["model_condition"]) for row in rows)
    drift_counts = Counter(str(row["drift_reason"]) for row in rows)
    ready_rows = status_counts["REPAIR_DRY_RUN_TEXT_SUBSTITUTION_READY_NOT_REGENERATED"]
    needs_regen_rows = status_counts["NEEDS_TOKENIZER_ALIGNED_OR_BRANCH_REGEN_OBSERVED_TEXT_NOT_FOUND"]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "model_loading_started": False,
            "model_scoring_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "local_suffix_repair_dry_run_not_regenerated_not_training_not_far",
        },
        "inputs": {
            "repair_examples_jsonl": str(repair_examples_jsonl),
            "branch_plan_jsonl": str(branch_plan_jsonl),
            "prepared_summary_json": str(prepared_summary_json),
            "prepared_status": prepared_summary.get("status", ""),
        },
        "repair_example_rows": len(rows),
        "repair_ready_rows": ready_rows,
        "needs_tokenizer_aligned_or_branch_regen_rows": needs_regen_rows,
        "dry_run_status_counts": dict(sorted(status_counts.items())),
        "model_condition_counts": dict(sorted(condition_counts.items())),
        "drift_reason_counts": dict(sorted(drift_counts.items())),
        "suffix_window_chars": suffix_window_chars,
        "limitation": (
            "Approximate text-level local substitution only. This is not "
            "tokenizer-aligned regeneration, branch-aware model scoring, "
            "payload recovery, or FAR."
        ),
        "next_allowed_action": (
            "Run Slurm-scored branch-aware compatibility diagnostic from the "
            "prepared scoring plan or export protected/task-only examples for "
            "a richer repair dry-run. Training remains forbidden."
        ),
        "training_allowed": False,
        "e2e_rerun_allowed": False,
    }
    write_json(summary_path, summary)
    write_jsonl(output_dir / "local_suffix_repair_dry_run_rows.jsonl", rows)
    write_csv(output_dir / "local_suffix_repair_dry_run_rows.csv", rows, ROW_FIELDS)
    write_csv(output_dir / "local_suffix_repair_dry_run_by_status.csv", _counter_rows(status_counts, "dry_run_status"), ["dry_run_status", "rows", "fraction"])
    write_csv(output_dir / "local_suffix_repair_dry_run_by_condition.csv", _counter_rows(condition_counts, "model_condition"), ["model_condition", "rows", "fraction"])
    write_csv(output_dir / "local_suffix_repair_dry_run_by_drift_reason.csv", _counter_rows(drift_counts, "drift_reason"), ["drift_reason", "rows", "fraction"])
    write_csv(output_dir / "local_suffix_repair_dry_run_readiness.csv", _readiness_rows(summary), READINESS_FIELDS)
    _write_markdown(output_dir / "local_suffix_repair_dry_run_examples.md", summary, rows, max_markdown_examples)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_dry_run(
        repair_examples_jsonl=_resolve(args.repair_examples_jsonl),
        branch_plan_jsonl=_resolve(args.branch_plan_jsonl),
        prepared_summary_json=_resolve(args.prepared_summary_json),
        output_dir=_resolve(args.output_dir),
        suffix_window_chars=int(args.suffix_window_chars),
        max_markdown_examples=int(args.max_markdown_examples),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
