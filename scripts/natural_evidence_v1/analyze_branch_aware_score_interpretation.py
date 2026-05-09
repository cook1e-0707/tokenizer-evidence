from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import stable_hash_hex, write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_v1_branch_aware_score_interpretation_v1"
CANDIDATE_SCHEMA = "natural_evidence_v1_repaired_target_mass_probe_candidate_v1"

SLICE_FIELDS = [
    "group_kind",
    "group_value",
    "rows",
    "branch_aware_proxy_pass_rows",
    "branch_aware_proxy_pass_rate",
    "primary_probe_candidate_rows",
    "primary_probe_candidate_rate",
    "secondary_candidate_rows",
    "rejected_rows",
    "mean_response_delta_nll_per_token",
    "mean_suffix_delta_nll_per_token",
    "decision",
]

CONTROL_FIELDS = [
    "expected_payload_id",
    "seed",
    "match_policy",
    "drift_reason",
    "observed_token_class",
    "protected_rows",
    "protected_branch_pass_rate",
    "protected_primary_candidates",
    "task_only_rows",
    "task_only_branch_pass_rate",
    "task_only_primary_candidates",
    "raw_rows",
    "raw_branch_pass_rate",
    "raw_primary_candidates",
    "protected_minus_task_only_branch_pass_rate",
    "protected_minus_raw_branch_pass_rate",
    "decision",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "source_row_id",
    "candidate_tier",
    "candidate_reason",
    "probe_role",
    "model_condition",
    "expected_payload_id",
    "payload_id",
    "seed",
    "prompt_id",
    "prompt_slot",
    "match_policy",
    "drift_reason",
    "observed_token_class",
    "target_bucket",
    "observed_token_text",
    "target_token_text",
    "response_delta_nll_per_token",
    "suffix_delta_nll_per_token",
    "frame_index",
    "frame_digit_index",
    "token_index",
    "paper_claim_allowed",
    "training_started",
    "generation_started",
    "e2e_eval_started",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only branch-aware score interpretation and repaired "
            "training-target preflight. Reads branch-aware proxy score rows and "
            "prepared/dry-run rows, reports repairable slices, and exports "
            "candidate rows for a future teacher-forced repaired target-mass "
            "probe. Does not train, generate, run E2E, recover payloads, or "
            "estimate FAR."
        )
    )
    parser.add_argument("--score-rows-jsonl", required=True)
    parser.add_argument("--score-summary-json", required=True)
    parser.add_argument("--scoring-plan-jsonl", required=True)
    parser.add_argument("--dry-run-rows-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-response-delta-primary", type=float, default=0.35)
    parser.add_argument("--max-suffix-delta-primary", type=float, default=0.5)
    parser.add_argument("--min-slice-rows", type=int, default=4)
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Sequence[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_group_value(parts: Sequence[Any]) -> str:
    return "|".join(str(part) if str(part) else "<empty>" for part in parts)


def _load_joined_rows(
    *,
    score_rows_jsonl: Path,
    scoring_plan_jsonl: Path,
    dry_run_rows_jsonl: Path,
) -> list[dict[str, Any]]:
    plan_by_id = {str(row.get("row_id", "")): row for row in _iter_jsonl(scoring_plan_jsonl)}
    dry_by_id = {str(row.get("row_id", "")): row for row in _iter_jsonl(dry_run_rows_jsonl)}
    rows: list[dict[str, Any]] = []
    for score_row in _iter_jsonl(score_rows_jsonl):
        row_id = str(score_row.get("row_id", ""))
        merged = {**plan_by_id.get(row_id, {}), **dry_by_id.get(row_id, {}), **score_row}
        merged["branch_aware_proxy_pass"] = _as_bool(merged.get("branch_aware_proxy_pass"))
        merged["response_naturalness_proxy_pass"] = _as_bool(merged.get("response_naturalness_proxy_pass"))
        merged["suffix_preserving_proxy_pass"] = _as_bool(merged.get("suffix_preserving_proxy_pass"))
        merged["response_delta_nll_per_token"] = _as_float(merged.get("response_delta_nll_per_token"))
        merged["suffix_delta_nll_per_token"] = _as_float(merged.get("suffix_delta_nll_per_token"))
        rows.append(merged)
    return rows


def classify_probe_candidate(
    row: Mapping[str, Any],
    *,
    max_response_delta_primary: float,
    max_suffix_delta_primary: float,
) -> tuple[str, str, bool, bool]:
    if not _as_bool(row.get("branch_aware_proxy_pass")):
        return (
            "REJECT_BRANCH_AWARE_PROXY_FAIL",
            "branch-aware proxy failed response or suffix threshold",
            False,
            False,
        )
    token_class = str(row.get("observed_token_class", ""))
    drift_reason = str(row.get("drift_reason", ""))
    response_delta = _as_float(row.get("response_delta_nll_per_token"))
    suffix_delta = _as_float(row.get("suffix_delta_nll_per_token"))
    if token_class == "punctuation":
        return (
            "SECONDARY_PUNCTUATION_ABLATION_ONLY",
            "punctuation replacements are proxy-compatible but not clean primary natural-token targets",
            False,
            True,
        )
    primary_delta = response_delta <= max_response_delta_primary and suffix_delta <= max_suffix_delta_primary
    if drift_reason == "compatible_non_target" and primary_delta:
        return (
            "PRIMARY_COMPATIBLE_NON_TARGET_LOW_DELTA",
            "compatible non-target drift with low response and suffix NLL deltas",
            True,
            False,
        )
    if drift_reason == "compatible_non_target":
        return (
            "PRIMARY_COMPATIBLE_NON_TARGET_PROXY_PASS",
            "compatible non-target drift passed proxy thresholds but needs delta review",
            True,
            False,
        )
    if drift_reason == "observed_token_not_candidate_set" and primary_delta:
        return (
            "PRIMARY_OUT_OF_CANDIDATE_SET_LOW_DELTA",
            "out-of-candidate observed token can be locally repaired with low NLL deltas",
            True,
            False,
        )
    if drift_reason == "observed_token_not_candidate_set":
        return (
            "SECONDARY_OUT_OF_CANDIDATE_SET_PROXY_PASS",
            "out-of-candidate repair passed proxy thresholds but has higher repair risk",
            False,
            True,
        )
    if drift_reason == "observed_bucket_not_compatible":
        return (
            "SECONDARY_BUCKET_POLICY_REVIEW",
            "observed bucket was incompatible; use for bucket-policy review before training targets",
            False,
            True,
        )
    return (
        "SECONDARY_UNKNOWN_REPAIR_CLASS",
        "proxy-compatible row with unrecognized drift reason",
        False,
        True,
    )


def _probe_role(row: Mapping[str, Any]) -> str:
    condition = str(row.get("model_condition", ""))
    if condition == "protected_trained":
        return "protected_repaired_target_mass_probe_candidate"
    if condition == "task_only_lora":
        return "task_only_control_repaired_target_mass_probe_candidate"
    if condition == "raw":
        return "raw_null_repaired_target_mass_probe_candidate"
    return "unknown_condition_probe_candidate"


def _candidate_id(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("row_id", "")),
        str(row.get("model_condition", "")),
        str(row.get("expected_payload_id", "")),
        str(row.get("seed", "")),
        str(row.get("match_policy", "")),
    ]
    return "cand_" + stable_hash_hex(parts)[:16]


def _annotate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_response_delta_primary: float,
    max_suffix_delta_primary: float,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        tier, reason, primary, secondary = classify_probe_candidate(
            row,
            max_response_delta_primary=max_response_delta_primary,
            max_suffix_delta_primary=max_suffix_delta_primary,
        )
        output = {
            **dict(row),
            "candidate_id": _candidate_id(row),
            "source_row_id": row.get("row_id", ""),
            "candidate_tier": tier,
            "candidate_reason": reason,
            "probe_role": _probe_role(row),
            "primary_probe_candidate": primary,
            "secondary_candidate": secondary,
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "result_claim": "repaired_target_mass_probe_candidate_preflight_not_training_not_far",
        }
        annotated.append(output)
    return annotated


def _slice_decision(rows: Sequence[Mapping[str, Any]], min_slice_rows: int) -> str:
    total = len(rows)
    branch_pass = sum(1 for row in rows if _as_bool(row.get("branch_aware_proxy_pass")))
    primary = sum(1 for row in rows if row.get("primary_probe_candidate"))
    secondary = sum(1 for row in rows if row.get("secondary_candidate"))
    if total < min_slice_rows:
        return "LOW_N_SLICE_FOR_EXAMPLES_ONLY"
    if primary > 0 and _rate(primary, total) >= 0.5:
        return "PRIMARY_REPAIRABLE_SLICE"
    if primary > 0:
        return "MIXED_REPAIRABLE_SLICE"
    if branch_pass > 0 and secondary > 0:
        return "SECONDARY_OR_ABLATION_ONLY"
    return "NOT_REPAIRABLE_UNDER_PROXY"


def _summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_kind: str,
    group_value: str,
    min_slice_rows: int,
) -> dict[str, Any]:
    total = len(rows)
    branch_pass = sum(1 for row in rows if _as_bool(row.get("branch_aware_proxy_pass")))
    primary = sum(1 for row in rows if row.get("primary_probe_candidate"))
    secondary = sum(1 for row in rows if row.get("secondary_candidate"))
    return {
        "group_kind": group_kind,
        "group_value": group_value,
        "rows": total,
        "branch_aware_proxy_pass_rows": branch_pass,
        "branch_aware_proxy_pass_rate": _rate(branch_pass, total),
        "primary_probe_candidate_rows": primary,
        "primary_probe_candidate_rate": _rate(primary, total),
        "secondary_candidate_rows": secondary,
        "rejected_rows": total - primary - secondary,
        "mean_response_delta_nll_per_token": _mean(
            [_as_float(row.get("response_delta_nll_per_token")) for row in rows]
        ),
        "mean_suffix_delta_nll_per_token": _mean(
            [_as_float(row.get("suffix_delta_nll_per_token")) for row in rows]
        ),
        "decision": _slice_decision(rows, min_slice_rows),
    }


def build_slice_summary(rows: Sequence[Mapping[str, Any]], *, min_slice_rows: int) -> list[dict[str, Any]]:
    group_specs = [
        ("condition", ("model_condition",)),
        ("drift_reason", ("drift_reason",)),
        ("token_class", ("observed_token_class",)),
        ("payload_seed", ("expected_payload_id", "seed")),
        ("condition_drift_reason", ("model_condition", "drift_reason")),
        ("condition_token_class", ("model_condition", "observed_token_class")),
        ("condition_payload_seed", ("model_condition", "expected_payload_id", "seed")),
        ("drift_reason_token_class", ("drift_reason", "observed_token_class")),
        (
            "condition_drift_reason_token_class",
            ("model_condition", "drift_reason", "observed_token_class"),
        ),
        ("payload_seed_drift_reason", ("expected_payload_id", "seed", "drift_reason")),
        ("payload_seed_token_class", ("expected_payload_id", "seed", "observed_token_class")),
    ]
    output: list[dict[str, Any]] = []
    for group_kind, fields in group_specs:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_safe_group_value([row.get(field, "") for field in fields])].append(row)
        for value, group_rows in sorted(grouped.items()):
            output.append(_summarize_group(group_rows, group_kind=group_kind, group_value=value, min_slice_rows=min_slice_rows))
    return output


def build_control_comparison(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = str(row.get("seed", "")) if str(row.get("model_condition", "")) != "raw" else ""
        key = (
            str(row.get("expected_payload_id", "")),
            seed,
            str(row.get("match_policy", "")),
            str(row.get("drift_reason", "")),
            str(row.get("observed_token_class", "")),
        )
        grouped[key].append(row)
    raw_index: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("model_condition", "")) == "raw":
            raw_index[
                (
                    str(row.get("expected_payload_id", "")),
                    str(row.get("match_policy", "")),
                    str(row.get("drift_reason", "")),
                    str(row.get("observed_token_class", "")),
                )
            ].append(row)

    output: list[dict[str, Any]] = []
    for (payload_id, seed, policy, drift_reason, token_class), group_rows in sorted(grouped.items()):
        if seed == "":
            continue
        protected = [row for row in group_rows if str(row.get("model_condition", "")) == "protected_trained"]
        task_only = [row for row in group_rows if str(row.get("model_condition", "")) == "task_only_lora"]
        raw = raw_index.get((payload_id, policy, drift_reason, token_class), [])
        if not protected and not task_only:
            continue

        def pass_rate(items: Sequence[Mapping[str, Any]]) -> float:
            return _rate(sum(1 for row in items if _as_bool(row.get("branch_aware_proxy_pass"))), len(items))

        protected_rate = pass_rate(protected)
        task_rate = pass_rate(task_only)
        raw_rate = pass_rate(raw)
        protected_primary = sum(1 for row in protected if row.get("primary_probe_candidate"))
        task_primary = sum(1 for row in task_only if row.get("primary_probe_candidate"))
        raw_primary = sum(1 for row in raw if row.get("primary_probe_candidate"))
        if protected_primary > 0 and (not task_only or not raw):
            decision = "LOW_N_OR_MISSING_CONTROL_FOR_PROTECTED_CANDIDATE"
        elif (
            protected
            and task_only
            and raw
            and protected_rate > task_rate + 0.1
            and protected_rate > raw_rate + 0.1
            and protected_primary > task_primary
            and protected_primary > raw_primary
        ):
            decision = "POTENTIAL_PROTECTED_SLICE_NEEDS_LOCKBOX"
        elif protected_primary > 0:
            decision = "PROBE_CANDIDATES_EXIST_NO_CLEAR_PROTECTED_SEPARATION"
        else:
            decision = "NO_PRIMARY_PROTECTED_CANDIDATE"
        output.append(
            {
                "expected_payload_id": payload_id,
                "seed": seed,
                "match_policy": policy,
                "drift_reason": drift_reason,
                "observed_token_class": token_class,
                "protected_rows": len(protected),
                "protected_branch_pass_rate": protected_rate,
                "protected_primary_candidates": protected_primary,
                "task_only_rows": len(task_only),
                "task_only_branch_pass_rate": task_rate,
                "task_only_primary_candidates": task_primary,
                "raw_rows": len(raw),
                "raw_branch_pass_rate": raw_rate,
                "raw_primary_candidates": raw_primary,
                "protected_minus_task_only_branch_pass_rate": protected_rate - task_rate,
                "protected_minus_raw_branch_pass_rate": protected_rate - raw_rate,
                "decision": decision,
            }
        )
    return output


def _candidate_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id", ""),
        "source_row_id": row.get("source_row_id", ""),
        "candidate_tier": row.get("candidate_tier", ""),
        "candidate_reason": row.get("candidate_reason", ""),
        "probe_role": row.get("probe_role", ""),
        "model_condition": row.get("model_condition", ""),
        "expected_payload_id": row.get("expected_payload_id", ""),
        "payload_id": row.get("payload_id", ""),
        "seed": row.get("seed", ""),
        "prompt_id": row.get("prompt_id", ""),
        "prompt_slot": row.get("prompt_slot", ""),
        "match_policy": row.get("match_policy", ""),
        "drift_reason": row.get("drift_reason", ""),
        "observed_token_class": row.get("observed_token_class", ""),
        "target_bucket": row.get("target_bucket", ""),
        "observed_token_text": row.get("observed_token_text", ""),
        "target_token_text": row.get("target_token_text", ""),
        "response_delta_nll_per_token": row.get("response_delta_nll_per_token", ""),
        "suffix_delta_nll_per_token": row.get("suffix_delta_nll_per_token", ""),
        "frame_index": row.get("frame_index", ""),
        "frame_digit_index": row.get("frame_digit_index", ""),
        "token_index": row.get("token_index", ""),
        "paper_claim_allowed": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
    }


def _write_markdown(
    *,
    path: Path,
    summary: Mapping[str, Any],
    top_slices: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Branch-aware score interpretation and repaired-target preflight",
        "",
        "This is an artifact-only interpretation of the 848414 branch-aware proxy scores. It does not train, generate, rerun E2E, recover payloads, or estimate FAR.",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "## Aggregate",
        "",
        f"- scored rows: `{summary['scored_rows']}`",
        f"- branch-aware proxy pass: `{summary['branch_aware_proxy_pass_rows']}/{summary['scored_rows']}` (`{summary['branch_aware_proxy_pass_rate']:.4f}`)",
        f"- primary repaired target-mass probe candidates: `{summary['primary_probe_candidate_rows']}`",
        f"- secondary or ablation candidates: `{summary['secondary_candidate_rows']}`",
        f"- rejected rows: `{summary['rejected_rows']}`",
        "",
        "## Candidate Decision",
        "",
        f"`{summary['candidate_decision']}`",
        "",
        "Primary candidates require branch-aware proxy pass, non-punctuation tokens, and a repair class that is not bucket-policy ambiguous. Secondary rows are useful for ablations or policy review but should not be treated as clean repaired training targets.",
        "",
        "## Best Repairable Slices",
        "",
        "| Slice | Rows | Primary candidates | Branch pass | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in top_slices[:12]:
        lines.append(
            "| {group_kind}:{group_value} | {rows} | {primary_probe_candidate_rows} | {branch_aware_proxy_pass_rows} ({branch_aware_proxy_pass_rate:.4f}) | {decision} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Protected-vs-Control Warning",
            "",
            "Rows can be locally repairable while still failing to show a protected-specific signal. The control comparison table should be reviewed before any future target-mass probe.",
            "",
            "| Payload | Seed | Policy | Drift | Token class | Protected pass | Task-only pass | Raw pass | Decision |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in control_rows[:12]:
        lines.append(
            "| {expected_payload_id} | {seed} | {match_policy} | {drift_reason} | {observed_token_class} | {protected_branch_pass_rate:.4f} | {task_only_branch_pass_rate:.4f} | {raw_branch_pass_rate:.4f} | {decision} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Next Allowed Action",
            "",
            "Use the primary candidate manifest only for an artifact-only repaired teacher-forced target-mass probe design. Do not start training or E2E from this result.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    score_rows_jsonl: Path,
    score_summary_json: Path,
    scoring_plan_jsonl: Path,
    dry_run_rows_jsonl: Path,
    output_dir: Path,
    max_response_delta_primary: float,
    max_suffix_delta_primary: float,
    min_slice_rows: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "branch_aware_score_interpretation_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing interpretation summary: {summary_path}")

    score_summary = _read_json(score_summary_json)
    rows = _load_joined_rows(
        score_rows_jsonl=score_rows_jsonl,
        scoring_plan_jsonl=scoring_plan_jsonl,
        dry_run_rows_jsonl=dry_run_rows_jsonl,
    )
    annotated = _annotate_rows(
        rows,
        max_response_delta_primary=max_response_delta_primary,
        max_suffix_delta_primary=max_suffix_delta_primary,
    )
    primary = [row for row in annotated if row.get("primary_probe_candidate")]
    secondary = [row for row in annotated if row.get("secondary_candidate")]
    rejected = [row for row in annotated if not row.get("primary_probe_candidate") and not row.get("secondary_candidate")]
    slice_rows = build_slice_summary(annotated, min_slice_rows=min_slice_rows)
    control_rows = build_control_comparison(annotated)
    candidate_counts = Counter(str(row.get("candidate_tier", "")) for row in annotated)
    primary_by_condition = Counter(str(row.get("model_condition", "")) for row in primary)
    primary_by_drift = Counter(str(row.get("drift_reason", "")) for row in primary)
    primary_by_token = Counter(str(row.get("observed_token_class", "")) for row in primary)
    protected_control_warning = any(
        row["decision"] == "PROBE_CANDIDATES_EXIST_NO_CLEAR_PROTECTED_SEPARATION"
        for row in control_rows
    )
    if primary and protected_control_warning:
        candidate_decision = "PRIMARY_CANDIDATES_EXIST_BUT_NO_TRAINING_GATE_PROTECTED_CONTROL_SEPARATION_WEAK"
    elif primary:
        candidate_decision = "PRIMARY_CANDIDATES_EXIST_NEEDS_CONTROL_REVIEW"
    else:
        candidate_decision = "NO_PRIMARY_REPAIRED_TARGET_MASS_PROBE_CANDIDATES"

    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING",
        "inputs": {
            "score_rows_jsonl": str(score_rows_jsonl),
            "score_summary_json": str(score_summary_json),
            "scoring_plan_jsonl": str(scoring_plan_jsonl),
            "dry_run_rows_jsonl": str(dry_run_rows_jsonl),
            "score_status": score_summary.get("status", ""),
        },
        "thresholds": {
            "max_response_delta_primary": max_response_delta_primary,
            "max_suffix_delta_primary": max_suffix_delta_primary,
            "min_slice_rows": min_slice_rows,
        },
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "branch_aware_score_interpretation_not_training_not_far",
        },
        "scored_rows": len(annotated),
        "branch_aware_proxy_pass_rows": sum(1 for row in annotated if row.get("branch_aware_proxy_pass")),
        "branch_aware_proxy_pass_rate": _rate(
            sum(1 for row in annotated if row.get("branch_aware_proxy_pass")),
            len(annotated),
        ),
        "primary_probe_candidate_rows": len(primary),
        "secondary_candidate_rows": len(secondary),
        "rejected_rows": len(rejected),
        "candidate_tier_counts": dict(sorted(candidate_counts.items())),
        "primary_candidate_counts_by_condition": dict(sorted(primary_by_condition.items())),
        "primary_candidate_counts_by_drift_reason": dict(sorted(primary_by_drift.items())),
        "primary_candidate_counts_by_token_class": dict(sorted(primary_by_token.items())),
        "candidate_decision": candidate_decision,
        "training_allowed": False,
        "e2e_rerun_allowed": False,
        "next_allowed_action": (
            "Design or run an artifact-only repaired teacher-forced target-mass "
            "probe over primary candidates; do not train or rerun E2E."
        ),
    }

    sorted_slices = sorted(
        slice_rows,
        key=lambda row: (
            int(row["primary_probe_candidate_rows"]),
            float(row["primary_probe_candidate_rate"]),
            int(row["branch_aware_proxy_pass_rows"]),
        ),
        reverse=True,
    )
    sorted_control = sorted(
        control_rows,
        key=lambda row: (
            int(row["protected_primary_candidates"]),
            float(row["protected_minus_task_only_branch_pass_rate"]),
            float(row["protected_minus_raw_branch_pass_rate"]),
        ),
        reverse=True,
    )
    candidate_rows = [
        {
            "schema_name": CANDIDATE_SCHEMA,
            **row,
            "claim_control": summary["claim_control"],
        }
        for row in primary
    ]

    write_json(summary_path, summary)
    write_csv(output_dir / "branch_aware_score_slice_summary.csv", slice_rows, SLICE_FIELDS)
    write_csv(output_dir / "branch_aware_score_protected_vs_controls.csv", control_rows, CONTROL_FIELDS)
    write_jsonl(output_dir / "repaired_target_mass_probe_candidates.jsonl", candidate_rows)
    write_csv(
        output_dir / "repaired_target_mass_probe_candidates.csv",
        [_candidate_csv_row(row) for row in primary],
        CANDIDATE_FIELDS,
    )
    write_jsonl(
        output_dir / "repaired_target_mass_probe_secondary_candidates.jsonl",
        [{"schema_name": CANDIDATE_SCHEMA, **row, "claim_control": summary["claim_control"]} for row in secondary],
    )
    _write_markdown(
        path=output_dir / "branch_aware_score_interpretation.md",
        summary=summary,
        top_slices=sorted_slices,
        control_rows=sorted_control,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        score_rows_jsonl=_resolve(args.score_rows_jsonl),
        score_summary_json=_resolve(args.score_summary_json),
        scoring_plan_jsonl=_resolve(args.scoring_plan_jsonl),
        dry_run_rows_jsonl=_resolve(args.dry_run_rows_jsonl),
        output_dir=_resolve(args.output_dir),
        max_response_delta_primary=float(args.max_response_delta_primary),
        max_suffix_delta_primary=float(args.max_suffix_delta_primary),
        min_slice_rows=int(args.min_slice_rows),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
