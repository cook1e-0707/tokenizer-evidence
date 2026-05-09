from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
DEFAULT_SPLIT_MANIFEST = (
    ROOT
    / "results/natural_evidence_v2/prompts/"
    "wp2_controlled_natural_prompt_family_scaffold_20260508_2123/split_manifest.json"
)

DETECTOR_ID = "qwen_v2_wp3_micro_slot_detector_v0_artifact_scaffold"
SLOT_POLICY_ID = DETECTOR_ID
BUCKET_POLICY_ID = "qwen_v2_wp3_two_way_bucket_policy_v0_artifact_scaffold"
PROMPT_SET_ID = "qwen_v2_wp2_controlled_natural_prompt_set_v1"

FORBIDDEN_ACTIONS = (
    "training",
    "generation",
    "model_transcript_generation",
    "qwen_e2e_rerun",
    "llama",
    "same_family_null",
    "sanitizer_benchmark",
    "far_aggregation",
    "positive_paper_claim",
)

BANK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "candidate_bank_id": "sentence_opener_sequence_v0",
        "slot_type": "sentence_opener",
        "anchor_kind": "sentence_boundary",
        "buckets": {
            "0": ["first", "second", "third"],
            "1": ["start", "continue", "finish"],
        },
    },
    {
        "candidate_bank_id": "step_opener_action_v0",
        "slot_type": "bullet_or_step_opener",
        "anchor_kind": "line_or_step_boundary",
        "buckets": {
            "0": ["check", "review", "list"],
            "1": ["choose", "set", "make"],
        },
    },
    {
        "candidate_bank_id": "discourse_marker_additive_v0",
        "slot_type": "discourse_marker",
        "anchor_kind": "clause_or_sentence_start",
        "buckets": {
            "0": ["also", "plus"],
            "1": ["too", "again"],
        },
    },
    {
        "candidate_bank_id": "optional_hedge_frequency_v0",
        "slot_type": "optional_hedge",
        "anchor_kind": "local_modifier",
        "buckets": {
            "0": ["often", "usually"],
            "1": ["sometimes", "typically", "normally"],
        },
    },
    {
        "candidate_bank_id": "transition_word_plain_v0",
        "slot_type": "transition_word",
        "anchor_kind": "sentence_or_clause_transition",
        "buckets": {
            "0": ["instead", "then", "now"],
            "1": ["however", "finally", "otherwise"],
        },
    },
    {
        "candidate_bank_id": "function_word_conjunction_v0",
        "slot_type": "function_word_alternative",
        "anchor_kind": "local_function_word",
        "buckets": {
            "0": ["and", "or"],
            "1": ["but", "so"],
        },
    },
    {
        "candidate_bank_id": "function_word_preposition_v0",
        "slot_type": "function_word_alternative",
        "anchor_kind": "local_function_word",
        "buckets": {
            "0": ["with", "for"],
            "1": ["by", "from"],
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a natural_evidence_v2 WP3 artifact-only detector contract "
            "and two-way bucket-bank audit scaffold. This does not run model "
            "generation, training, E2E, FAR aggregation, or paper claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--responses-jsonl",
        type=Path,
        help=(
            "Optional fixed response artifact to run the scaffold detector on. "
            "Rows may contain response_text, output_text, model_output, or text."
        ),
    )
    parser.add_argument("--max-response-rows", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows and index >= max_rows:
                break
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{index + 1}")
                rows.append(payload)
    return rows


def flatten_bank_members(bank: Mapping[str, Any]) -> list[str]:
    buckets = bank.get("buckets", {})
    if not isinstance(buckets, Mapping):
        raise ValueError(f"bank buckets must be a mapping: {bank.get('candidate_bank_id')}")
    members: list[str] = []
    for bucket_id in ("0", "1"):
        raw_members = buckets.get(bucket_id, [])
        if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
            raise ValueError(f"bank bucket {bucket_id} must be a sequence")
        members.extend(str(member) for member in raw_members)
    return members


def validate_bank_specs(banks: Sequence[Mapping[str, Any]], allowed_slot_types: Sequence[str]) -> None:
    bank_ids: set[str] = set()
    global_members: dict[str, str] = {}
    for bank in banks:
        bank_id = str(bank.get("candidate_bank_id", ""))
        if not bank_id:
            raise ValueError("candidate_bank_id is required")
        if bank_id in bank_ids:
            raise ValueError(f"duplicate candidate_bank_id: {bank_id}")
        bank_ids.add(bank_id)

        slot_type = str(bank.get("slot_type", ""))
        if slot_type not in allowed_slot_types:
            raise ValueError(f"{bank_id}: slot_type not in config primary_slot_types: {slot_type}")

        buckets = bank.get("buckets", {})
        if not isinstance(buckets, Mapping) or set(buckets) != {"0", "1"}:
            raise ValueError(f"{bank_id}: every primary bank must have bucket ids 0 and 1")

        seen_members: set[str] = set()
        for bucket_id in ("0", "1"):
            members = [str(member) for member in buckets[bucket_id]]
            if not members:
                raise ValueError(f"{bank_id}: bucket {bucket_id} is empty")
            if any(not member.strip() for member in members):
                raise ValueError(f"{bank_id}: bucket {bucket_id} contains an empty member")
            overlap = seen_members.intersection(members)
            if overlap:
                raise ValueError(f"{bank_id}: overlapping members: {sorted(overlap)}")
            seen_members.update(members)
            for member in members:
                normalized_member = member.strip().lower()
                if normalized_member in global_members:
                    raise ValueError(
                        f"{bank_id}: duplicate normalized member {member!r} also appears in "
                        f"{global_members[normalized_member]}"
                    )
                global_members[normalized_member] = bank_id


def detector_contract(config: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    slot_policy = config.get("slot_policy", {})
    if not isinstance(slot_policy, Mapping):
        slot_policy = {}
    return {
        "schema_name": "natural_evidence_v2_wp3_detector_contract_v1",
        "status": "WP3_DETECTOR_CONTRACT_SCAFFOLDED_NOT_AUDITED",
        "protocol_id": str(protocol.get("id", "")),
        "prompt_set_id": str(split_manifest.get("prompt_set_id", PROMPT_SET_ID)),
        "detector_id": DETECTOR_ID,
        "slot_policy_id": SLOT_POLICY_ID,
        "bucket_policy_id": BUCKET_POLICY_ID,
        "response_local": True,
        "slot_order": "stable_response_order_by_span_then_slot_type",
        "eligible_slot_types": list(slot_policy.get("primary_slot_types", [])),
        "row_schema": [
            "schema_name",
            "protocol_id",
            "prompt_set_id",
            "prompt_id",
            "split",
            "family_id",
            "response_id",
            "response_sha256",
            "slot_policy_id",
            "slot_index",
            "slot_type",
            "anchor_kind",
            "span_start",
            "span_end",
            "surface_text",
            "left_context",
            "right_context",
            "candidate_bank_id",
            "bucket_policy_id",
            "bucket_id",
            "eligibility_status",
            "rejection_reason",
        ],
        "density_gate_status": "NOT_EVALUATED",
        "mass_gate_status": "NOT_EVALUATED",
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def bucket_bank_scaffold(config: Mapping[str, Any]) -> dict[str, Any]:
    bucket_policy = config.get("bucket_policy", {})
    if not isinstance(bucket_policy, Mapping):
        bucket_policy = {}
    slot_policy = config.get("slot_policy", {})
    if not isinstance(slot_policy, Mapping):
        slot_policy = {}
    allowed_slot_types = [str(item) for item in slot_policy.get("primary_slot_types", [])]
    banks = [dict(bank) for bank in BANK_SPECS]
    validate_bank_specs(banks, allowed_slot_types)
    return {
        "schema_name": "natural_evidence_v2_wp3_two_way_bucket_bank_scaffold_v1",
        "status": "WP3_TWO_WAY_BUCKET_BANK_SCAFFOLDED_NOT_TOKENIZER_OR_MASS_AUDITED",
        "bucket_policy_id": BUCKET_POLICY_ID,
        "bucket_count": int(bucket_policy.get("primary_bucket_count", 2)),
        "required_bucket_ids": [0, 1],
        "disallowed_primary_bucket_counts": [
            int(bucket_policy.get("ablation_bucket_count", 4)),
            int(bucket_policy.get("forbidden_bucket_count", 8)),
        ],
        "entries_claim_boundary": (
            "bucket-bank entries are natural next-token measurable opportunity/catalog entries"
        ),
        "candidate_banks": banks,
        "candidate_bank_count": len(banks),
        "candidate_surface_count": sum(len(flatten_bank_members(bank)) for bank in banks),
        "tokenizer_stability_status": "NOT_EVALUATED",
        "mass_gate_status": "NOT_EVALUATED",
        "min_bucket_mass_required": float(bucket_policy.get("min_bucket_mass", 0.005)),
        "max_mass_ratio_required": float(bucket_policy.get("max_mass_ratio", 5.0)),
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def forbidden_hits(text: str, forbidden_terms: Sequence[str]) -> list[str]:
    upper_text = text.upper()
    return [term for term in forbidden_terms if str(term).upper() in upper_text]


def context_window(text: str, start: int, end: int, width: int = 24) -> tuple[str, str]:
    return text[max(0, start - width):start], text[end:min(len(text), end + width)]


def response_text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("response_text", "output_text", "model_output", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("response rows must include response_text, output_text, model_output, or text")


def response_id_from_row(row: Mapping[str, Any], index: int) -> str:
    for key in ("response_id", "completion_id", "row_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    prompt_id = str(row.get("prompt_id", f"row_{index:05d}"))
    return f"{prompt_id}_response"


def member_to_bank_map(banks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for bank in banks:
        bank_id = str(bank["candidate_bank_id"])
        slot_type = str(bank["slot_type"])
        anchor_kind = str(bank["anchor_kind"])
        buckets = bank["buckets"]
        for bucket_id, members in buckets.items():
            for member in members:
                mapping[str(member).lower()] = {
                    "candidate_bank_id": bank_id,
                    "slot_type": slot_type,
                    "anchor_kind": anchor_kind,
                    "bucket_id": str(bucket_id),
                }
    return mapping


def detect_candidates(
    *,
    row: Mapping[str, Any],
    index: int,
    banks: Sequence[Mapping[str, Any]],
    forbidden_terms: Sequence[str],
    protocol_id: str,
) -> list[dict[str, Any]]:
    response_text = response_text_from_row(row)
    response_sha = sha256_text(response_text)
    response_id = response_id_from_row(row, index)
    term_map = member_to_bank_map(banks)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in sorted(term_map, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    candidates: list[dict[str, Any]] = []
    for match in pattern.finditer(response_text):
        surface = match.group(0)
        normalized = surface.lower()
        bank_info = term_map[normalized]
        hits = forbidden_hits(surface, forbidden_terms)
        rejection_reason = ""
        eligibility_status = "CANDIDATE_NEEDS_TOKENIZER_AND_MASS_AUDIT"
        if hits:
            eligibility_status = "REJECTED"
            rejection_reason = "public_forbidden_surface_term"
        elif not surface.isalpha():
            eligibility_status = "REJECTED"
            rejection_reason = "non_alpha_surface"

        left_context, right_context = context_window(response_text, match.start(), match.end())
        candidates.append(
            {
                "schema_name": "natural_evidence_v2_wp3_micro_slot_candidate_v1",
                "protocol_id": protocol_id,
                "prompt_set_id": str(row.get("prompt_set_id", PROMPT_SET_ID)),
                "prompt_id": str(row.get("prompt_id", "")),
                "split": str(row.get("split", "")),
                "family_id": str(row.get("family_id", "")),
                "response_id": response_id,
                "response_sha256": response_sha,
                "slot_policy_id": SLOT_POLICY_ID,
                "slot_index": len(candidates),
                "slot_type": bank_info["slot_type"],
                "anchor_kind": bank_info["anchor_kind"],
                "span_start": match.start(),
                "span_end": match.end(),
                "surface_text": surface,
                "left_context": left_context,
                "right_context": right_context,
                "candidate_bank_id": bank_info["candidate_bank_id"],
                "bucket_policy_id": BUCKET_POLICY_ID,
                "bucket_id": bank_info["bucket_id"],
                "eligibility_status": eligibility_status,
                "rejection_reason": rejection_reason,
            }
        )
    return sorted(candidates, key=lambda item: (item["span_start"], item["span_end"], item["slot_type"]))


def density_accounting(candidate_rows: Sequence[Mapping[str, Any]], response_count: int) -> dict[str, Any]:
    slot_counts_by_response = Counter(str(row["response_id"]) for row in candidate_rows)
    slot_counts_by_type = Counter(str(row["slot_type"]) for row in candidate_rows)
    rejected = Counter(str(row["rejection_reason"]) for row in candidate_rows if row["rejection_reason"])
    eligible_count = sum(
        1 for row in candidate_rows if row["eligibility_status"] == "CANDIDATE_NEEDS_TOKENIZER_AND_MASS_AUDIT"
    )
    responses_with_any_slot = len(slot_counts_by_response)
    average = 0.0 if response_count == 0 else len(candidate_rows) / response_count
    return {
        "schema_name": "natural_evidence_v2_wp3_density_accounting_scaffold_v1",
        "status": "DIAGNOSTIC_ONLY_NOT_GATE_RESULT",
        "total_responses": response_count,
        "responses_with_any_slot": responses_with_any_slot,
        "prompt_coverage": 0.0 if response_count == 0 else responses_with_any_slot / response_count,
        "average_micro_slots_per_response": average,
        "candidate_slot_rows": len(candidate_rows),
        "candidate_rows_needing_tokenizer_and_mass_audit": eligible_count,
        "slot_counts_by_type": dict(sorted(slot_counts_by_type.items())),
        "rejected_candidate_counts_by_reason": dict(sorted(rejected.items())),
        "density_gate_status": "NOT_EVALUATED",
        "tokenizer_stability_status": "NOT_EVALUATED",
        "mass_gate_status": "NOT_EVALUATED",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def summary_payload(
    *,
    config: Mapping[str, Any],
    split_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    output_dir: Path,
    contract: Mapping[str, Any],
    bank_scaffold: Mapping[str, Any],
    responses_jsonl: Path | None,
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    return {
        "schema_name": "natural_evidence_v2_wp3_detector_bank_scaffold_summary_v1",
        "status": "WP3_DETECTOR_AND_TWO_WAY_BUCKET_BANK_SCAFFOLDED_NOT_AUDITED",
        "action_scope": "artifact_only_detector_contract_and_bucket_bank_scaffold",
        "protocol_id": str(protocol.get("id", "")),
        "prompt_set_id": str(split_manifest.get("prompt_set_id", PROMPT_SET_ID)),
        "detector_id": str(contract["detector_id"]),
        "slot_policy_id": str(contract["slot_policy_id"]),
        "bucket_policy_id": str(bank_scaffold["bucket_policy_id"]),
        "output_dir": str(output_dir),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "detector_contract_json": str(output_dir / "wp3_detector_contract.json"),
        "bucket_bank_scaffold_json": str(output_dir / "two_way_bucket_bank_scaffold.json"),
        "candidate_bank_count": int(bank_scaffold["candidate_bank_count"]),
        "candidate_surface_count": int(bank_scaffold["candidate_surface_count"]),
        "responses_jsonl": None if responses_jsonl is None else str(responses_jsonl),
        "candidate_micro_slot_rows": len(candidate_rows),
        "density_gate_status": "NOT_EVALUATED",
        "mass_gate_status": "NOT_EVALUATED",
        "tokenizer_stability_status": "NOT_EVALUATED",
        "precommit_status": "not_committed",
        "wp4_allowed": False,
        "gates_unlocked": [],
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "forbidden_actions_confirmed": list(FORBIDDEN_ACTIONS),
        "next_allowed_action": (
            "WP3 artifact-only tokenizer/density/mass audit implementation on fixed artifacts only; "
            "no transcript generation, training, E2E, Llama, same-family null, sanitizer, FAR, or positive claim."
        ),
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Detector And Bucket-Bank Scaffold",
            "",
            "Artifact-only scaffold for the v2 micro-slot detector contract and two-way bucket-bank catalog.",
            "",
            f"status: `{summary['status']}`",
            f"detector_id: `{summary['detector_id']}`",
            f"bucket_policy_id: `{summary['bucket_policy_id']}`",
            "",
            "Density, tokenizer stability, and mass gates are not evaluated by this artifact.",
            "No model calls, training, E2E evaluation, FAR aggregation, or positive paper claim were started.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    config_path = resolve_path(args.config)
    split_manifest_path = resolve_path(args.split_manifest)
    config = load_yaml(config_path)
    split_manifest = load_json(split_manifest_path)
    forbidden_terms = [str(item) for item in config.get("forbidden_surface_terms", [])]

    contract = detector_contract(config, split_manifest)
    bank_scaffold = bucket_bank_scaffold(config)
    candidate_rows: list[dict[str, Any]] = []
    density: dict[str, Any] | None = None
    responses_jsonl: Path | None = None

    if args.responses_jsonl is not None:
        responses_jsonl = resolve_path(args.responses_jsonl)
        response_rows = read_jsonl(responses_jsonl, max_rows=max(0, int(args.max_response_rows)))
        protocol_id = str(contract["protocol_id"])
        for index, row in enumerate(response_rows):
            candidate_rows.extend(
                detect_candidates(
                    row=row,
                    index=index,
                    banks=bank_scaffold["candidate_banks"],
                    forbidden_terms=forbidden_terms,
                    protocol_id=protocol_id,
                )
            )
        density = density_accounting(candidate_rows, len(response_rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "wp3_detector_contract.json", contract)
    write_json(output_dir / "two_way_bucket_bank_scaffold.json", bank_scaffold)
    if candidate_rows:
        write_jsonl(output_dir / "candidate_micro_slots.jsonl", candidate_rows)
    if density is not None:
        write_json(output_dir / "density_accounting_scaffold.json", density)

    summary = summary_payload(
        config=config,
        split_manifest_path=split_manifest_path,
        split_manifest=split_manifest,
        output_dir=output_dir,
        contract=contract,
        bank_scaffold=bank_scaffold,
        responses_jsonl=responses_jsonl,
        candidate_rows=candidate_rows,
    )
    write_json(output_dir / "wp3_detector_bank_scaffold_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "candidate_bank_count": summary["candidate_bank_count"],
                "candidate_surface_count": summary["candidate_surface_count"],
                "density_gate_status": summary["density_gate_status"],
                "mass_gate_status": summary["mass_gate_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
