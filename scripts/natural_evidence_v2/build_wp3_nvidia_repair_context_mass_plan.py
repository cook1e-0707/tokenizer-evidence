from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPOSALS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "nvidia_assisted_context_repair_20260508_2021/nvidia_assisted_design_parsed_proposals.json"
)
FORBIDDEN_PUBLIC_SURFACES = (
    "FIELD=",
    "SECTION=",
    "TOPIC=",
    "PAYLOAD",
    "CERT",
    "EVIDENCE",
    "CARRIER",
    "OWNER",
    "fingerprint",
    "watermark",
    "bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only WP3 context-mass score plan from NVIDIA "
            "design-assistant repair proposals. This writes candidate rows for "
            "later base-Qwen scoring through Chimera Slurm only. It does not "
            "score a model, train, generate text, run E2E, FAR, or claims."
        )
    )
    parser.add_argument("--proposals-json", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Optional allowlist of proposal model ids to include. Defaults to all parsed models.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be a mapping: {path}")
    return payload


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


def public_text_has_forbidden(text: str) -> list[str]:
    upper = text.upper()
    return [term for term in FORBIDDEN_PUBLIC_SURFACES if term.upper() in upper]


def normalized_prefix_shape(prefix: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    value = str(prefix)
    if value == "Step N:":
        value = "Step 1:"
        notes.append("placeholder_N_normalized_to_1")
    elif value == "N.":
        value = "1."
        notes.append("placeholder_N_normalized_to_1")
    if value and value[-1] in ":,;.":
        value = value + " "
        notes.append("trailing_space_added_after_punctuation")
    return value, notes


def proposal_public_texts(proposal: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("side0", "side1", "prefix_shapes"):
        values = proposal.get(key, [])
        if isinstance(values, list):
            texts.extend(str(value) for value in values)
    return texts


def proposal_quality_filter(proposal: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    side0 = [str(value) for value in proposal.get("side0", []) if str(value)]
    side1 = [str(value) for value in proposal.get("side1", []) if str(value)]
    prefixes = [str(value) for value in proposal.get("prefix_shapes", []) if str(value)]
    if not side0 or not side1:
        reasons.append("empty_side")
    if not prefixes:
        reasons.append("empty_prefix_shapes")
    hits: list[str] = []
    for text in proposal_public_texts(proposal):
        hits.extend(public_text_has_forbidden(text))
    if hits:
        reasons.append("forbidden_public_surface:" + ",".join(sorted(set(hits))))
    if str(proposal.get("new_bank_id", "")) == "sentence_opener_sequence_v1":
        reasons.append("dropped_mixed_sentence_opener_prefix_candidate")
    if str(proposal.get("new_bank_id", "")) == "function_word_preposition_v1":
        reasons.append("dropped_bad_prefix_surface_join_candidate")
    return not reasons, reasons


def plan_row_id(*, plan_id: str, bank_id: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "plan_id": plan_id,
            "candidate_bank_id": bank_id,
            "prefix_before_candidate": prefix,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:20]


def rows_from_proposals(
    *,
    payload: Mapping[str, Any],
    proposals_json: Path,
    plan_id: str,
    include_models: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    source_model_counts = Counter()
    candidate_bank_counts = Counter()

    for proposal_block in payload.get("proposals", []):
        if not isinstance(proposal_block, Mapping):
            continue
        model = str(proposal_block.get("model", ""))
        if include_models and model not in include_models:
            continue
        proposal = proposal_block.get("proposal", {})
        if not isinstance(proposal, Mapping):
            continue
        for index, repair in enumerate(proposal.get("repair_proposals", [])):
            if not isinstance(repair, Mapping):
                continue
            keep, reasons = proposal_quality_filter(repair)
            if not keep:
                dropped.append(
                    {
                        "model": model,
                        "new_bank_id": str(repair.get("new_bank_id", "")),
                        "drop_reasons": reasons,
                    }
                )
                continue
            side0 = [str(value) for value in repair.get("side0", []) if str(value)]
            side1 = [str(value) for value in repair.get("side1", []) if str(value)]
            bank_id = str(repair.get("new_bank_id", f"repair_{index:02d}"))
            for prefix_index, raw_prefix in enumerate(repair.get("prefix_shapes", [])):
                prefix, normalization_notes = normalized_prefix_shape(str(raw_prefix))
                if not prefix:
                    dropped.append(
                        {
                            "model": model,
                            "new_bank_id": bank_id,
                            "drop_reasons": ["empty_prefix_after_normalization"],
                        }
                    )
                    continue
                row = {
                    "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_row_v1",
                    "plan_id": plan_id,
                    "plan_row_id": plan_row_id(plan_id=plan_id, bank_id=bank_id, prefix=prefix),
                    "candidate_bank_id": bank_id,
                    "source_bank_id": str(repair.get("source_bank_id", "")),
                    "source_proposal_model": model,
                    "source_proposals_json": str(proposals_json),
                    "proposal_index": index,
                    "prefix_shape_index": prefix_index,
                    "slot_type": str(repair.get("slot_type", "")),
                    "anchor_kind": str(repair.get("anchor_kind", "")),
                    "bucket_policy_id": "qwen_v2_wp3_nvidia_assisted_repair_two_way",
                    "casing_variant": str(repair.get("case_variant", "")) or "proposal_defined",
                    "bucket_surfaces": {"0": side0, "1": side1},
                    "prefix_before_candidate": prefix,
                    "prefix_before_candidate_sha256": sha256_text(prefix),
                    "prefix_is_empty": prefix == "",
                    "source_detection_count": 1,
                    "source_response_count": 0,
                    "source_family_counts": {},
                    "observed_surface_counts": {},
                    "observed_case_variant_counts": {},
                    "template_preflight_only": False,
                    "structural_selection": "nvidia_design_assistant_prefix_shape_not_qwen_gate",
                    "normalization_notes": normalization_notes,
                    "proposal_decision": str(repair.get("decision", "")),
                    "proposal_reason": str(repair.get("why_qwen_mass_may_improve", "")),
                    "proposal_risk": str(repair.get("risk", "")),
                    "scoring_status": "PLANNED_NOT_SCORED",
                    "tokenizer_stability_status": "NOT_EVALUATED",
                    "mass_gate_status": "NOT_EVALUATED",
                    "artifact_only": True,
                    "not_qwen_gate": True,
                    "model_scoring_started": False,
                    "training_started": False,
                    "generation_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                }
                rows.append(row)
                source_model_counts[model] += 1
                candidate_bank_counts[bank_id] += 1

    dedup: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = str(row["plan_row_id"])
        if key in dedup:
            duplicates += 1
            continue
        dedup[key] = row
    rows = [dedup[key] for key in sorted(dedup)]
    summary = {
        "schema_name": "natural_evidence_v2_wp3_nvidia_repair_context_mass_plan_summary_v1",
        "status": "WP3_NVIDIA_REPAIR_CONTEXT_MASS_SCORE_PLAN_WRITTEN_NOT_SCORED",
        "plan_id": plan_id,
        "proposals_json": str(proposals_json),
        "score_plan_jsonl": "",
        "proposal_source_models_included": sorted(source_model_counts),
        "score_plan_rows": len(rows),
        "score_plan_rows_by_source_model": dict(sorted(source_model_counts.items())),
        "score_plan_rows_by_candidate_bank": dict(sorted(candidate_bank_counts.items())),
        "dropped_proposals": dropped,
        "duplicate_rows_dropped": duplicates,
        "artifact_only_design_assistance": True,
        "not_qwen_gate": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "next_allowed_action": (
            "Validate this plan structurally, then score it with base Qwen only "
            "through the allowlisted Chimera Slurm context-mass scorer."
        ),
    }
    return rows, summary


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# NVIDIA-Assisted WP3 Repair Context-Mass Plan",
            "",
            "Artifact-only plan derived from external design-assistant proposals.",
            "Rows are not Qwen gates until scored by base Qwen through Chimera Slurm.",
            "",
            f"status: `{summary['status']}`",
            f"score_plan_rows: `{summary['score_plan_rows']}`",
            "",
            "No training, generation, E2E, FAR aggregation, or paper claim was started.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    proposals_json = resolve_path(args.proposals_json)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    payload = read_json(proposals_json)
    plan_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_dir.name)
    rows, summary = rows_from_proposals(
        payload=payload,
        proposals_json=proposals_json,
        plan_id=plan_id,
        include_models=set(args.include_model),
    )
    if not rows:
        raise RuntimeError("no repair score-plan rows survived filtering")
    output_dir.mkdir(parents=True, exist_ok=True)
    score_plan = output_dir / "qwen_v2_wp3_nvidia_repair_context_mass_score_plan.jsonl"
    summary = dict(summary)
    summary["score_plan_jsonl"] = str(score_plan)
    write_jsonl(score_plan, rows)
    write_json(output_dir / "qwen_v2_wp3_nvidia_repair_context_mass_score_plan_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "score_plan_rows": summary["score_plan_rows"],
                "dropped_proposals": len(summary["dropped_proposals"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
