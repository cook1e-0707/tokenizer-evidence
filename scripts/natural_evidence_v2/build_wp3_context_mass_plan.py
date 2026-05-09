from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_template_density_audit_balanced_850278/candidate_micro_slots.jsonl"
)
DEFAULT_RESPONSES = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_template_density_responses_balanced_20260508_2331/"
    "qwen_v2_wp3_template_density_responses.jsonl"
)
DEFAULT_BUCKET_BANK = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_detector_bank_scaffold_repaired_20260508_2308/two_way_bucket_bank_scaffold.json"
)

CASE_VARIANTS = ("lowercase", "sentence_case")
PREFIX_BOUNDARY_TOKENIZATION_POLICY = "score_longest_common_token_prefix_when_candidate_retokenizes_boundary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only WP3 context-specific bucket-mass scoring "
            "plan from balanced template response detections. This extracts "
            "prefix_before_candidate at detected slots and records lowercase "
            "and sentence-case bucket variants separately. It does not score a "
            "model, train, generate text, run E2E, aggregate FAR, or make "
            "positive claims."
        )
    )
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--responses-jsonl", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--bucket-bank", type=Path, default=DEFAULT_BUCKET_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be a mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{index}")
            rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("response_text", "output_text", "model_output", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"response row has no text field: {row.get('response_id')}")


def sentence_case(text: str) -> str:
    lowered = text.strip().lower()
    return lowered[:1].upper() + lowered[1:] if lowered else lowered


def apply_case_variant(text: str, variant: str) -> str:
    if variant == "lowercase":
        return text.strip().lower()
    if variant == "sentence_case":
        return sentence_case(text)
    raise ValueError(f"unsupported case variant: {variant}")


def observed_case_variant(surface: str) -> str:
    if surface == surface.lower():
        return "lowercase"
    if surface == sentence_case(surface):
        return "sentence_case"
    return "other"


def bucket_bank_by_id(bucket_bank: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for bank in bucket_bank.get("candidate_banks", []):
        if not isinstance(bank, Mapping):
            raise ValueError("candidate_banks entries must be mappings")
        bank_id = str(bank.get("candidate_bank_id", ""))
        if not bank_id:
            raise ValueError("candidate bank missing candidate_bank_id")
        if bank_id in output:
            raise ValueError(f"duplicate candidate_bank_id: {bank_id}")
        output[bank_id] = dict(bank)
    return output


def cased_buckets(bank: Mapping[str, Any], variant: str) -> dict[str, list[str]]:
    buckets = bank.get("buckets", {})
    if not isinstance(buckets, Mapping):
        raise ValueError(f"bank buckets must be a mapping: {bank.get('candidate_bank_id')}")
    output: dict[str, list[str]] = {}
    for bucket_id, members in sorted(buckets.items()):
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
            raise ValueError(f"bank bucket must be a sequence: {bank.get('candidate_bank_id')}:{bucket_id}")
        output[str(bucket_id)] = [apply_case_variant(str(member), variant) for member in members]
    return output


def response_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        response_id = str(row.get("response_id", f"response_{index:06d}"))
        if response_id in output:
            raise ValueError(f"duplicate response_id in responses artifact: {response_id}")
        output[response_id] = row
    return output


def representative_detection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": str(row.get("prompt_id", "")),
        "response_id": str(row.get("response_id", "")),
        "family_id": str(row.get("family_id", "")),
        "slot_index": int(row.get("slot_index", 0)),
        "span_start": int(row.get("span_start", 0)),
        "span_end": int(row.get("span_end", 0)),
        "surface_text": str(row.get("surface_text", "")),
        "bucket_id": str(row.get("bucket_id", "")),
    }


def plan_row_id(*, bank_id: str, variant: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "candidate_bank_id": bank_id,
            "casing_variant": variant,
            "prefix_before_candidate": prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def build_plan_rows(
    *,
    plan_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    responses_by_id: Mapping[str, Mapping[str, Any]],
    banks_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    input_case_counts_by_bank: dict[str, Counter[str]] = defaultdict(Counter)
    input_surface_counts_by_bank: dict[str, Counter[str]] = defaultdict(Counter)
    eligible_count = 0
    skipped_counts = Counter()

    for row in candidate_rows:
        if str(row.get("eligibility_status", "")) != "CANDIDATE_NEEDS_TOKENIZER_AND_MASS_AUDIT":
            skipped_counts[str(row.get("rejection_reason", "ineligible")) or "ineligible"] += 1
            continue

        bank_id = str(row.get("candidate_bank_id", ""))
        if bank_id not in banks_by_id:
            raise ValueError(f"candidate row references unknown bank: {bank_id}")

        response_id = str(row.get("response_id", ""))
        response_row = responses_by_id.get(response_id)
        if response_row is None:
            raise ValueError(f"candidate row references missing response_id: {response_id}")

        response_text = response_text_from_row(response_row)
        response_sha = sha256_text(response_text)
        recorded_sha = str(row.get("response_sha256", ""))
        if recorded_sha and recorded_sha != response_sha:
            raise ValueError(f"response_sha256 mismatch for {response_id}")

        span_start = int(row.get("span_start", 0))
        span_end = int(row.get("span_end", 0))
        surface = str(row.get("surface_text", ""))
        if span_start < 0 or span_end < span_start or span_end > len(response_text):
            raise ValueError(f"invalid span for {response_id}:{row.get('slot_index')}")
        if response_text[span_start:span_end] != surface:
            raise ValueError(
                f"span surface mismatch for {response_id}:{row.get('slot_index')}: "
                f"{response_text[span_start:span_end]!r} != {surface!r}"
            )

        eligible_count += 1
        observed_case = observed_case_variant(surface)
        input_case_counts_by_bank[bank_id][observed_case] += 1
        input_surface_counts_by_bank[bank_id][surface] += 1
        prefix = response_text[:span_start]

        for variant in CASE_VARIANTS:
            key = (bank_id, variant, prefix)
            group = groups.get(key)
            if group is None:
                bank = banks_by_id[bank_id]
                group = {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(bank_id=bank_id, variant=variant, prefix=prefix),
                    "candidate_bank_id": bank_id,
                    "slot_type": str(bank.get("slot_type", "")),
                    "anchor_kind": str(bank.get("anchor_kind", "")),
                    "bucket_policy_id": str(row.get("bucket_policy_id", "")),
                    "casing_variant": variant,
                    "bucket_surfaces": cased_buckets(bank, variant),
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": prefix == "",
                    "prefix_boundary_tokenization_policy": PREFIX_BOUNDARY_TOKENIZATION_POLICY,
                    "prefix_boundary_repair_reason": (
                        "Slurm job 850372 showed configured-tokenizer prefix-boundary "
                        "retokenization on a context-mass row. Repaired scorer must score "
                        "a shared token prefix and require one next-token continuation per "
                        "bucket surface."
                    ),
                    "tokenizer_boundary_validation_status": "NOT_EVALUATED",
                    "source_detection_count": 0,
                    "source_response_ids": set(),
                    "source_family_counts": Counter(),
                    "observed_surface_counts": Counter(),
                    "observed_case_variant_counts": Counter(),
                    "representative_detection": representative_detection(row),
                    "structural_selection": "current_detector_eligible_template_slot",
                    "scoring_status": "PLANNED_NOT_SCORED",
                    "tokenizer_stability_status": "NOT_EVALUATED",
                    "mass_gate_status": "NOT_EVALUATED",
                    "template_preflight_only": True,
                    "model_scoring_started": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                }
                groups[key] = group

            group["source_detection_count"] += 1
            group["source_response_ids"].add(response_id)
            group["source_family_counts"][str(row.get("family_id", ""))] += 1
            group["observed_surface_counts"][surface] += 1
            group["observed_case_variant_counts"][observed_case] += 1

    plan_rows: list[dict[str, Any]] = []
    context_counters: dict[tuple[str, str], int] = defaultdict(int)
    for key in sorted(groups):
        group = groups[key]
        counter_key = (group["candidate_bank_id"], group["casing_variant"])
        context_counters[counter_key] += 1
        group["context_index_within_bank_variant"] = context_counters[counter_key] - 1
        group["source_response_count"] = len(group["source_response_ids"])
        group["source_response_ids"] = sorted(group["source_response_ids"])[:8]
        group["source_response_ids_truncated"] = group["source_response_count"] > 8
        group["source_family_counts"] = dict(sorted(group["source_family_counts"].items()))
        group["observed_surface_counts"] = dict(sorted(group["observed_surface_counts"].items()))
        group["observed_case_variant_counts"] = dict(sorted(group["observed_case_variant_counts"].items()))
        plan_rows.append(group)

    audit = {
        "eligible_detection_rows": eligible_count,
        "skipped_candidate_rows_by_reason": dict(sorted(skipped_counts.items())),
        "observed_case_counts_by_bank": {
            bank_id: dict(sorted(counter.items()))
            for bank_id, counter in sorted(input_case_counts_by_bank.items())
        },
        "observed_surface_counts_by_bank": {
            bank_id: dict(sorted(counter.items()))
            for bank_id, counter in sorted(input_surface_counts_by_bank.items())
        },
    }
    return plan_rows, audit


def summarize_plan(
    *,
    plan_id: str,
    candidate_jsonl: Path,
    responses_jsonl: Path,
    bucket_bank: Path,
    output_dir: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    rows_by_bank = Counter(str(row["candidate_bank_id"]) for row in plan_rows)
    rows_by_variant = Counter(str(row["casing_variant"]) for row in plan_rows)
    rows_by_bank_variant = Counter(
        f"{row['candidate_bank_id']}::{row['casing_variant']}" for row in plan_rows
    )
    family_counts = Counter(str(row.get("family_id", "")) for row in candidate_rows)
    template_preflight_only = bool(response_rows) and all(
        str(row.get("response_source", row.get("artifact_role", ""))).startswith(
            "template_density_preflight"
        )
        or str(row.get("artifact_role", "")).startswith("template_density_preflight")
        for row in response_rows
    )
    return {
        "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_summary_v1",
        "status": "WP3_CONTEXT_SPECIFIC_MASS_SCORE_PLAN_WRITTEN_NOT_SCORED",
        "plan_id": plan_id,
        "candidate_jsonl": str(candidate_jsonl),
        "responses_jsonl": str(responses_jsonl),
        "bucket_bank": str(bucket_bank),
        "score_plan_jsonl": str(output_dir / "qwen_v2_wp3_context_mass_score_plan.jsonl"),
        "candidate_input_rows": len(candidate_rows),
        "response_input_rows": len(response_rows),
        "template_preflight_only": template_preflight_only,
        "source_family_counts": dict(sorted(family_counts.items())),
        "eligible_detection_rows": int(audit.get("eligible_detection_rows", 0)),
        "score_plan_rows": len(plan_rows),
        "score_plan_rows_by_bank": dict(sorted(rows_by_bank.items())),
        "score_plan_rows_by_casing_variant": dict(sorted(rows_by_variant.items())),
        "score_plan_rows_by_bank_and_casing_variant": dict(sorted(rows_by_bank_variant.items())),
        "observed_case_counts_by_bank": audit.get("observed_case_counts_by_bank", {}),
        "observed_surface_counts_by_bank": audit.get("observed_surface_counts_by_bank", {}),
        "skipped_candidate_rows_by_reason": audit.get("skipped_candidate_rows_by_reason", {}),
        "casing_variants_audited_separately": list(CASE_VARIANTS),
        "prefix_source": "response_text_prefix_before_detected_candidate_span",
        "prefix_boundary_tokenization_policy": PREFIX_BOUNDARY_TOKENIZATION_POLICY,
        "prefix_boundary_repair_reason": (
            "Prepared after Slurm job 850372 failed on configured-tokenizer "
            "prefix-boundary retokenization before score artifacts were written."
        ),
        "structural_selection": "current_detector_eligible_template_slot",
        "artifact_only": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "wp4_allowed": False,
        "next_allowed_action": (
            "Review the repaired context-specific mass scoring plan/scorer and "
            "local no-model validation; do not submit Slurm scoring until the "
            "repair is explicitly allowlisted."
        ),
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Context-Specific Mass Score Plan",
            "",
            "Artifact-only scoring plan built from balanced template response detections.",
            "",
            f"status: `{summary['status']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            "",
            "Rows contain prefix_before_candidate at detected template slots.",
            "Lowercase and sentence-case bucket variants are recorded separately.",
            "Tokenizer prefix-boundary retokenization is handled by the repaired scorer with a shared longest-token-prefix policy.",
            "No model scoring, training, generation, E2E, FAR aggregation, or positive claim was started.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    candidate_jsonl = resolve_path(args.candidate_jsonl)
    responses_jsonl = resolve_path(args.responses_jsonl)
    bucket_bank_path = resolve_path(args.bucket_bank)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    candidate_rows = read_jsonl(candidate_jsonl)
    response_rows = read_jsonl(responses_jsonl)
    bucket_bank = read_json(bucket_bank_path)
    plan_id = output_dir.name
    plan_rows, audit = build_plan_rows(
        plan_id=plan_id,
        candidate_rows=candidate_rows,
        responses_by_id=response_index(response_rows),
        banks_by_id=bucket_bank_by_id(bucket_bank),
    )
    summary = summarize_plan(
        plan_id=plan_id,
        candidate_jsonl=candidate_jsonl,
        responses_jsonl=responses_jsonl,
        bucket_bank=bucket_bank_path,
        output_dir=output_dir,
        candidate_rows=candidate_rows,
        response_rows=response_rows,
        plan_rows=plan_rows,
        audit=audit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "qwen_v2_wp3_context_mass_score_plan.jsonl", plan_rows)
    write_json(output_dir / "qwen_v2_wp3_context_mass_score_plan_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "score_plan_rows": summary["score_plan_rows"],
                "casing_variants": summary["casing_variants_audited_separately"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
